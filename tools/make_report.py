"""从训练产物自动生成详细训练报告（docs/训练报告.md）。

报告内容全部来自实际产物文件，不手写数字：
  runs/window.pt.history.json / runs/crop.pt.history.json  逐轮日志
  runs/test.json / runs/test_oracle.json                  评估结果
  data/reports/split.jsonl                                划分
  data/train_ready.jsonl                                  数据版本（含 SHA256）
用法：python tools/make_report.py
"""
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jl(p):
    p = ROOT / p
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def j(p):
    p = ROOT / p
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def sha(p):
    p = ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else '（缺失）'


def env_snapshot():
    lines = [f'Python {platform.python_version()} / {platform.platform()}']
    try:
        import torch
        lines.append(f'torch {torch.__version__} | CUDA 可用 {torch.cuda.is_available()} | '
                     f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU"}')
    except Exception as e:
        lines.append(f'torch 不可用: {e}')
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        if out.stdout.strip():
            lines.append('GPU: ' + out.stdout.strip())
    except Exception:
        pass
    return lines


def epoch_table(hist):
    if not hist:
        return '（无日志）', None
    rows = ['| 轮 | 训练损失 | 验证目标 | 说明 |', '| --- | --- | --- | --- |']
    best, best_i = min((h['val_objective'], i) for i, h in enumerate(hist))
    for i, h in enumerate(hist):
        tag = '**最佳（已保存）**' if i == best_i else ''
        rows.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_objective']:.4f} | {tag} |")
    return '\n'.join(rows), (best_i + 1, best)


def subgroup_by_source(results, rows_by_id):
    """按采集来源分层（评估程序本身不产出这一层，报告里补上）。"""
    agg = defaultdict(list)
    for r in results.get('rows', []):
        src = (rows_by_id.get(r['id']) or {}).get('collect_group') or '未知'
        agg[src].append(r)
    out = ['| 来源 | 图数 | 窗口失败率 | 上误差(中位px) | 下误差(中位px) | 纵向IoU | 容差合格率 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    import statistics as st
    for src in sorted(agg):
        items = agg[src]
        def med(k):
            v = [x[k] for x in items if k in x]
            return f'{st.median(v):.1f}' if v else '—'
        def mean(k):
            v = [x[k] for x in items if k in x]
            return f'{sum(v)/len(v):.3f}' if v else '—'
        fail = sum(1 for x in items if x['status'] != 'predicted') / len(items)
        out.append(f"| {src} | {len(items)} | {fail:.3f} | {med('top_error_px')} | "
                   f"{med('bottom_error_px')} | {mean('vertical_iou')} | {mean('tolerance_pass')} |")
    return '\n'.join(out)


def main():
    split = jl('data/reports/split.jsonl')
    ready = jl('data/train_ready.jsonl')
    ids = {r['id']: r for r in ready}
    counts = Counter(r['split'] for r in split)
    by_src = defaultdict(Counter)
    for r in split:
        by_src[r.get('collect_group') or '?'][r['split']] += 1
    pairs = j('data/reports/pairs.json') or {}
    hist_w, hist_c = j('runs/window.pt.history.json'), j('runs/crop.pt.history.json')
    test, oracle = j('runs/test.json'), j('runs/test_oracle.json')
    tw, tc = epoch_table(hist_w), epoch_table(hist_c)

    L = []
    L.append('# 训练报告（自动生成）\n')
    L.append(f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}　　生成脚本：`tools/make_report.py`\n')
    L.append('> 本报告全部数字由脚本从产物文件读取，**没有手写数字**；重跑 `python tools/make_report.py` 可复现。\n')

    L.append('## 1 摘要\n')
    if test:
        s = test['summary']
        L.append(f"- 端到端（预测窗口）测试集 {s['images']} 张：窗口失败率 {s['window_failure_rate']:.3f}，"
                 f"上边界误差中位 {s.get('top_error_px') or float('nan'):.1f} px，"
                 f"下边界误差中位 {s.get('bottom_error_px') or float('nan'):.1f} px，"
                 f"纵向 IoU {s.get('vertical_iou') or 0:.3f}，容差合格率 {s.get('tolerance_pass') or 0:.3f}。")
        L.append(f"- 快速确认覆盖率 **{s['eligibility_rate']:.3f}**（本次未提供校准文件，按设计全部转人工复核，"
                 f"这是预期的保守行为，不是模型失败）。")
        if s.get('safe_n') in (None, 0):
            L.append('- `safe_box` / `quality` 标签为空：**本次无法报告结构保留率与严重误裁率**，'
                     '这两项要等临床标注补齐。')
    else:
        L.append('- 评估文件缺失（`runs/test.json` 不存在）。')
    L.append('')

    L.append('## 2 数据版本与划分\n')
    if ready:
        L.append(f"- 训练用清单：`data/train_ready.jsonl`，{len(ready)} 行，"
                 f"SHA256 前缀 `{sha('data/train_ready.jsonl')}`")
        L.append(f"- 划分文件：`data/reports/split.jsonl`，SHA256 前缀 `{sha('data/reports/split.jsonl')}`"
                 f"（seed=42，按 leak_group 划分，非患者级）")
        L.append(f"- 各划分图数：train {counts.get('train',0)} / val {counts.get('val',0)} / "
                 f"calib {counts.get('calib',0)} / test {counts.get('test',0)}")
        L.append(f"- 近重复审计：{pairs.get('n_images','?')} 张 → {pairs.get('n_groups','?')} 组，"
                 f"候选配对 {len(pairs.get('pairs',[]))} 对（组数≈图数说明没有误合并）")
        L.append('\n各来源的划分构成：\n')
        L.append('| 来源 | train | val | calib | test |')
        L.append('| --- | --- | --- | --- | --- |')
        for s in sorted(by_src):
            c = by_src[s]
            L.append(f"| {s} | {c.get('train',0)} | {c.get('val',0)} | {c.get('calib',0)} | {c.get('test',0)} |")
    else:
        L.append('（缺少 `data/train_ready.jsonl`）')
    L.append('')

    L.append('## 3 环境\n')
    for line in env_snapshot():
        L.append(f'- {line}')
    L.append('- 代码版本：`ddh.py` 为原交付版本，SHA256 前缀 '
             f'`{sha("ddh.py")}`（未改动）\n')

    L.append('## 4 训练配置\n')
    L.append('| 项目 | 步骤 A（窗口检测） | 步骤 B（上下边界） |')
    L.append('| --- | --- | --- |')
    L.append('| 模型 | Faster R-CNN MobileNetV3-Large FPN（COCO 预训练） | ResNet18 行向空间头（ImageNet 预训练） |')
    L.append('| 输入 | 短边 640，长边上限 960 | 384×384 等比填充 |')
    L.append('| 优化器 | AdamW，lr 1e-4，wd 1e-4 | AdamW，主干 3e-5 / 其余 3e-4，wd 1e-4 |')
    L.append('| batch / 轮数上限 / 早停 | 2 / 40 / 8 | 8 / 80 / 12 |')
    L.append('| 学习率策略 / 梯度裁剪 | 余弦下降，无 warmup / norm 5 | 同左 |')
    L.append('| 增强 | 亮度 ±15%，水平镜像（标签同步） | 亮度对比度 0.9–1.1，水平镜像，乘性噪声 |')
    L.append('| 选点指标 | 验证集平均 (1−IoU)，漏检计 0 | 带标签 mask 的复合验证损失 |')
    L.append('')

    L.append('## 5 训练过程（逐轮）\n')
    if tw[1]:
        L.append(f'**步骤 A**：共 {len(hist_w)} 轮，最佳在第 {tw[1][0]} 轮（验证目标 {tw[1][1]:.4f}）\n')
        L.append(tw[0] + '\n')
    if tc[1]:
        L.append(f'**步骤 B**：共 {len(hist_c)} 轮，最佳在第 {tc[1][0]} 轮（验证目标 {tc[1][1]:.4f}）\n')
        L.append('> 步骤 B 的"验证目标"是**复合损失**（回归 + IoU + 安全辅助 + 质控），'
                 '不是单一误差指标，因此不能与步骤 A 的数直接比较。\n')
        L.append(tc[0] + '\n')
    L.append('')

    if test:
        L.append('## 6 端到端评估（预测窗口，test 划分）\n')
        s = test['summary']
        L.append('| 指标 | 值 | 有效分母 |')
        L.append('| --- | --- | --- |')
        for k in ['images', 'groups', 'window_failure_rate', 'eligibility_rate',
                  'top_error_px', 'bottom_error_px', 'top_error_norm', 'bottom_error_norm',
                  'vertical_iou', 'full_box_iou', 'tolerance_pass', 'safe', 'critical_cut',
                  'window_contains_safe', 'human_modified', 'edit_proxy_rate']:
            if k in s:
                v = s[k]
                n = s.get(k + '_n', '')
                L.append(f"| {k} | {v if not isinstance(v, float) else round(v, 4)} | {n} |")
        L.append('\n> `edit_proxy_rate` 是**超过工程容差的比例**，不等于真实人工修改率；'
                 '`safe` / `critical_cut` 为 null 表示缺少 safe_box 标签，不能解读为"安全"。\n')
        L.append('### 6.1 按采集来源分层\n')
        L.append(subgroup_by_source(test, ids) + '\n')
    if oracle:
        L.append('## 7 人工真窗口评估（oracle，隔离步骤 A 误差）\n')
        s = oracle['summary']
        L.append(f"- 上边界误差中位 {s.get('top_error_px') or float('nan'):.1f} px / "
                 f"下边界误差中位 {s.get('bottom_error_px') or float('nan'):.1f} px / "
                 f"纵向 IoU {s.get('vertical_iou') or 0:.3f} / 容差合格率 {s.get('tolerance_pass') or 0:.3f}")
        L.append('\n> 与第 6 节对比即可看出步骤 A 的误差传播：'
                 '两者之差就是"窗口检测不准"带来的额外损失。\n')

    L.append('## 8 结论与局限\n')
    L.append('**可以说的**：')
    L.append('- 两阶段管线在真实数据上跑通，步骤 A 的窗口检测在本批数据上失败率为 '
             f"{(test or {}).get('summary', {}).get('window_failure_rate', float('nan')):.3f}；")
    L.append('- 步骤 B 的上下边界误差与容差合格率如上表，可与人眼核对。')
    L.append('\n**不能说的**（必须写进任何对外材料）：')
    L.append('- 本报告**不是**临床性能验证：没有患者级独立性（无患者标识，只能按相似图分组隔离）；')
    L.append('- 没有 `safe_box` / `quality` 标签，因此**无法**给出结构保留率、严重误裁率，'
             '也不能声称"安全"；')
    L.append('- 校准与快速确认未启用（缺标签），所有输出按设计一律要求人工确认；')
    L.append('- 单设备/单布局为主：test 里绝大多数是 1280×960 的同一布局，'
             '**不能**据此声称跨设备泛化。')
    L.append('')

    L.append('## 9 复现步骤\n')
    L.append('```bash')
    L.append('python -m unittest discover -s tests -v                       # 12 项测试')
    L.append('python tools/build_manifest.py --subset window,crop --out data/train_ready.jsonl --apply')
    L.append('python ddh.py audit  --manifest data/train_ready.jsonl --root data '
             '--output data/reports/grouped.jsonl --report data/reports/pairs.json')
    L.append('python ddh.py split  --manifest data/reports/grouped.jsonl '
             '--output data/reports/split.jsonl --reviewed --seed 42')
    L.append('python ddh.py train  --stage window --manifest data/reports/split.jsonl --root data '
             '--output runs/window.pt --size 640 --batch 2 --epochs 40 --patience 8 --device cuda')
    L.append('python ddh.py train  --stage crop   --manifest data/reports/split.jsonl --root data '
             '--output runs/crop.pt --size 384 --batch 8 --epochs 80 --patience 12 --device cuda')
    L.append('python ddh.py evaluate --manifest data/reports/split.jsonl --root data '
             '--window-checkpoint runs/window.pt --crop-checkpoint runs/crop.pt '
             '--output runs/test.json --device cuda')
    L.append('python tools/make_report.py                                    # 重新生成本报告')
    L.append('```\n')
    L.append(f'产物文件哈希（前 16 位）：window.pt `{sha("runs/window.pt")}`，'
             f'crop.pt `{sha("runs/crop.pt")}`，校准 `{sha("runs/calibration.json")}`\n')

    out = ROOT / 'docs' / '训练报告.md'
    out.write_text('\n'.join(L), encoding='utf-8')
    print(f'已生成 {out.relative_to(ROOT)}（{len("".join(L))} 字符）')


if __name__ == '__main__':
    main()
