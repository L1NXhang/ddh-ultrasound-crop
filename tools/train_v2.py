"""新口径训练流水线：合并标注 → 构建清单 → 划分 → 训练 → 评估 → 出报告。

设计要点：
  1. 窗口标签没变（还是那个模板），所以**步骤 A 复用已有的 runs/window.pt**，
     只重训步骤 B——既省 16 分钟，又让"新旧口径对比"只差在裁剪目标上，更干净。
  2. 训练前做**口径一致性闸门**：三个人标的数据如果有系统性偏移（中位数差 >25px），
     默认拒绝训练（可以用 --force 跳过）。这是防止再出现"阳那批"那种混口径训练。
  3. 全程产物落在 data/reports_v2/ 与 runs_v2/，不覆盖旧口径的结果，便于对比。

用法：
  python tools/train_v2.py --dry-run        # 只检查标注与一致性，不训练
  python tools/train_v2.py                  # 全流程
  python tools/train_v2.py --force          # 跳过一致性闸门
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LABEL_FILES = ['data/human_A.jsonl', 'data/human_B.jsonl', 'data/human_C.jsonl',
               'data/annotations_human.jsonl']
MEDIAN_GATE = 25.0     # 三人上边距中位数之差超过这个值就拒绝训练


def sh(cmd):
    print('  $', ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f'命令失败：{cmd}')


def load(p):
    p = ROOT / p
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--epochs-b', type=int, default=80)
    ap.add_argument('--patience-b', type=int, default=12)
    args = ap.parse_args()

    print('=== 0/6 自动修正窗口（逐图边缘检测，不依赖模板）===')
    src = 'data/train_ready_v2.jsonl' if (ROOT / 'data/train_ready_v2.jsonl').exists() else 'data/train_ready.jsonl'
    base_file = src
    if (ROOT / 'data/train_ready_v2.jsonl').exists():
        print(f'  使用已修正的窗口清单 {src}')
    else:
        print(f'  未找到修正清单，先跑：python tools/auto_window.py --apply（当前用 {src}）')

    print('=== 1/6 检查标注文件 ===')
    present = [(f, len(load(f))) for f in LABEL_FILES if (ROOT / f).exists()]
    for f, n in present:
        print(f'  {f}: {n} 条')
    if not present:
        raise SystemExit('没有任何标注文件')
    merged_ids = set()
    for f, _ in present:
        merged_ids |= {r['id'] for r in load(f)}
    print(f'  去重后共 {len(merged_ids)} 张有新口径裁剪标签')

    print('=== 2/6 口径一致性闸门 ===')
    base = {r['id']: r for r in load(base_file)}
    per = {}
    for f, _ in present:
        recs = [r for r in load(f) if r.get('crop') and base.get(r['id'], {}).get('window')]
        tm = [r['crop'][0] - base[r['id']]['window'][1] for r in recs]
        if tm:
            per[Path(f).stem] = (len(tm), st.median(tm))
    for name, (n, m) in per.items():
        print(f'  {name:24s} {n:4d} 条   上边距中位 {m:.0f} px')
    if len(per) > 1:
        vals = [m for _, m in per.values()]
        spread = max(vals) - min(vals)
        print(f'  各文件之间中位数之差：{spread:.0f} px（阈值 {MEDIAN_GATE:.0f}）')
        if spread > MEDIAN_GATE and not args.force:
            print('  ✗ 口径不一致：可能是有人没按同一标准标。')
            print('    建议：让偏差大的那位重标/抽查，或确认后用 --force 强制继续。')
            raise SystemExit(2)
        print('  ✓ 口径一致')

    if args.dry_run:
        print('\n（--dry-run：只检查，不训练）')
        return

    print('=== 3/6 合并并构建新口径清单 ===')
    (ROOT / 'data' / 'reports_v2').mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs_v2').mkdir(parents=True, exist_ok=True)
    sh([PY, 'tools/merge_labels.py', '--inputs'] + [f for f, _ in present]
       + ['--out', 'data/labels_v2.jsonl', '--apply'])
    sh([PY, 'tools/build_manifest.py',
        '--crop-from', ','.join(f for f, _ in present),
        '--subset', 'window,crop',
        '--out', 'data/train_v2.jsonl', '--apply'])

    print('=== 4/6 近重复审计与划分（seed 42，与旧口径一致）===')
    sh([PY, 'ddh.py', 'audit', '--manifest', 'data/train_v2.jsonl', '--root', 'data',
        '--output', 'data/reports_v2/grouped.jsonl', '--report', 'data/reports_v2/pairs.json'])
    sh([PY, 'ddh.py', 'split', '--manifest', 'data/reports_v2/grouped.jsonl',
        '--output', 'data/reports_v2/split.jsonl', '--reviewed', '--seed', '42'])

    print('=== 5/6 只重训步骤 B（步骤 A 复用 runs/window.pt）===')
    t0 = time.time()
    sh([PY, 'ddh_v2.py', 'train', '--stage', 'crop', '--manifest', 'data/reports_v2/split.jsonl',
        '--root', 'data', '--output', 'runs_v2/crop.pt', '--size', '384', '--batch', '8',
        '--epochs', str(args.epochs_b), '--patience', str(args.patience_b),
        '--device', 'cuda', '--safe-mode', 'derived', '--delta', '0.04'])
    print(f'  训练用时 {(time.time()-t0)/60:.1f} 分钟')

    print('=== 6/6 评估与报告 ===')
    sh([PY, 'ddh_v2.py', 'evaluate', '--manifest', 'data/reports_v2/split.jsonl', '--root', 'data',
        '--window-checkpoint', 'runs/window.pt', '--crop-checkpoint', 'runs_v2/crop.pt',
        '--output', 'runs_v2/test.json', '--device', 'cuda', '--split', 'test'])
    sh([PY, 'ddh_v2.py', 'evaluate', '--manifest', 'data/reports_v2/split.jsonl', '--root', 'data',
        '--window-checkpoint', 'runs/window.pt', '--crop-checkpoint', 'runs_v2/crop.pt',
        '--output', 'runs_v2/test_oracle.json', '--device', 'cuda', '--split', 'test',
        '--oracle-window'])

    # 旧模型 vs 新模型，在同一批新口径标注上的对比
    print('=== 附：新旧模型在测试集上的上边界误差对比 ===')
    for tag, ckpt in [('旧口径模型', 'runs/crop.pt'), ('新口径模型', 'runs_v2/crop.pt')]:
        out = f'runs_v2/cmp_{Path(ckpt).parent.name}.json'
        sh([PY, 'ddh_v2.py', 'evaluate', '--manifest', 'data/reports_v2/split.jsonl', '--root', 'data',
            '--window-checkpoint', 'runs/window.pt', '--crop-checkpoint', ckpt,
            '--output', out, '--device', 'cuda', '--split', 'test'])
        s = json.loads((ROOT / out).read_text(encoding='utf-8'))['summary']
        print(f'  {tag}: 上边界误差中位 {s["top_error_px"]:.1f} px，'
              f'容差合格率 {s["tolerance_pass"]:.3f}，纵向 IoU {s["vertical_iou"]:.3f}')

    print('\n完成。产物：data/train_v2.jsonl、data/reports_v2/、runs_v2/')


if __name__ == '__main__':
    main()
