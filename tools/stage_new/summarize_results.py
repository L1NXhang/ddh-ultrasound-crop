"""汇总四组实验结果 → 表格 + 柱状图（B/C 项核心图表）。

只读 runs/20261103_resnet18_vs_50/ 下的 metrics.json，不产生任何虚构数字。
若某组尚未训练完成，明确标注「未完成」而不是跳过或补 0。

用法：python tools/stage_new/summarize_results.py
"""
import json, csv, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = Path(__file__).resolve().parents[2]
# 支持命令行指定运行目录（默认用正确标签的 corrected 版本）
_default = PROJ / 'runs' / '20261103_resnet18_vs_50_corrected'
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else _default
if not RUN.is_absolute():
    RUN = PROJ / RUN
FIG = RUN / 'figures'; FIG.mkdir(exist_ok=True, parents=True)
REP = RUN / 'reports'; REP.mkdir(exist_ok=True, parents=True)

CONFIGS = [
    ('r18_l3', 'resnet18', 'ResNet18\n仅Layer3'),
    ('r50_l3', 'resnet50', 'ResNet50\n仅Layer3'),
    ('r18_l123', 'resnet18', 'ResNet18\nL1+L2+L3融合'),
    ('r50_l123', 'resnet50', 'ResNet50\nL1+L2+L3融合'),
    # 层级验证（若已完成会自动纳入）
    ('r18_l4', 'resnet18', 'ResNet18\n仅Layer4'),
    ('r18_l1234', 'resnet18', 'ResNet18\nL1+L2+L3+L4'),
]
# 图 6 专用：冷色调（蓝-青-靛），与图 7 的暖色调形成区分
COLORS = {'r18_l3': '#2B6CB0', 'r50_l3': '#1B7A8C',
          'r18_l123': '#7FB2E0', 'r50_l123': '#6FBFC9',
          'r18_l4': '#5B4B9E', 'r18_l1234': '#A7B4C8'}


def load():
    out = {}
    for cfg, sub, _ in CONFIGS:
        p = RUN / sub / cfg / 'metrics.json'
        out[cfg] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    return out


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    data = load()
    done = [c for c, _, _ in CONFIGS if data[c]]
    missing = [c for c, _, _ in CONFIGS if not data[c]]
    if not done:
        print('还没有任何一组训练完成'); return
    if missing:
        print(f'[注意] 尚未完成: {missing}（图中不会为它们画任何柱子）')

    # ---------- 表：B 项（18 vs 50，仅 Layer3）+ C 项（融合消融）合在一张 CSV ----------
    rows = []
    for cfg, sub, label in CONFIGS:
        m = data[cfg]
        if not m:
            rows.append({'config': cfg, 'status': '未完成'}); continue
        t = m['test']
        rows.append({
            'config': cfg, 'status': 'done', 'backbone': m['backbone'], 'fusion': m['fusion'],
            'params_total': m['params_total'], 'checkpoint_mb': m['checkpoint_mb'],
            'train_time_sec': m['train_time_sec'], 'epochs_run': m['epochs_run'],
            'best_epoch': m['best_epoch'], 'peak_gpu_mem_mb': m['peak_gpu_mem_mb'],
            'n_test': m['n_test'],
            'top_err_px_mean': round(t['top_err_px_mean'], 2),
            'top_err_px_median': round(t['top_err_px_median'], 2),
            'bot_err_px_mean': round(t['bot_err_px_mean'], 2),
            'bot_err_px_median': round(t['bot_err_px_median'], 2),
            'vert_iou_mean': round(t['vert_iou_mean'], 4),
            'pass_2pct': round(t['pass_2pct'], 4),
            'pass_3pct': round(t['pass_3pct'], 4),
            'pass_5pct': round(t['pass_5pct'], 4),
        })
    if any(r['status'] == 'done' for r in rows):
        keys = list(rows[0].keys())
        with open(REP / 'results_all_configs.csv', 'w', newline='', encoding='utf-8-sig') as f:
            wr = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, '') for k in keys})
    (REP / 'results_all_configs.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    # ---------- 图3：核心指标柱状图（B 项主图） ----------
    metrics = [('top_err_px_mean', '上边界平均误差 (px)\n↓越低越好'),
               ('bot_err_px_mean', '下边界平均误差 (px)\n↓越低越好'),
               ('vert_iou_mean', '纵向 IoU\n↑越高越好'),
               ('pass_3pct', '3% 容差合格率\n↑越高越好')]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2))
    labels = [l for c, _, l in CONFIGS if data[c]]
    keys = [c for c, _, _ in CONFIGS if data[c]]
    x = np.arange(len(labels))
    for ax, (k, title) in zip(axes.ravel(), metrics):
        vals = [data[c]['test'][k] for c in keys]
        bars = ax.bar(x, vals, color=[COLORS[c] for c in keys], width=.65)
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.3f}' if v < 1.5 else f'{v:.1f}', (b.get_x() + b.get_width() / 2, v),
                        textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.6, rotation=30, ha='right')
        ax.grid(axis='y', alpha=.3)
        # IoU 与合格率不从 0 起，避免视觉夸大微小差异；但明确标注轴范围
        if k == 'vert_iou_mean':
            lo = min(vals) - 0.012
            ax.set_ylim(lo, max(vals) + 0.016)
            ax.set_title(f'{title}\n（纵轴起点 {lo:.3f}，非 0）', fontsize=10)
        elif k == 'pass_3pct':
            ax.set_ylim(0, max(vals) * 1.22)
    pass
    fig.tight_layout()
    save(fig, 'fig3_main_metrics_bar')

    # ---------- 图4：相对成本倍数（以 r18_l3 = 1.0 为基准，对数轴） ----------
    # 用"倍数"而非绝对值：读者一眼能读出"贵了几倍"，比看三组绝对值更直观。
    # 训练时长因本机 GPU 被其他程序争抢，不具可比性，故不纳入。
    base = 'r18_l3'
    cost = [('params_total', '参数量'),
            ('checkpoint_mb', '模型文件大小'),
            ('peak_gpu_mem_mb', '训练显存峰值')]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (k, title) in zip(axes, cost):
        b0 = data[base].get(k)
        ratios = [(data[c].get(k) / b0) if (data[c].get(k) and b0) else 0 for c in keys]
        ypos = list(range(len(keys)))[::-1]
        ax.barh(ypos, ratios, color=[COLORS[c] for c in keys], height=.62)
        for y, r in zip(ypos, ratios):
            ax.annotate(f'{r:.2f}×', (r, y), textcoords='offset points',
                        xytext=(5, 0), va='center', fontsize=9)
        ax.axvline(1.0, color='#555', lw=1.2, ls='--', zorder=0)
        ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f'{title}（相对 r18_l3）', fontsize=10.5)
        ax.set_xlim(0, max(ratios) * 1.30)
        ax.grid(axis='x', alpha=.3)
    pass
    fig.tight_layout()
    save(fig, 'fig4_cost_bar')

    # ---------- 图5：训练曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for cfg, sub, label in CONFIGS:
        hp = RUN / sub / cfg / 'history.json'
        if not hp.exists():
            continue
        h = json.loads(hp.read_text(encoding='utf-8'))
        ep = [r['epoch'] for r in h]
        axes[0].plot(ep, [r['train_loss'] for r in h], color=COLORS[cfg],
                     label=label.replace('\n', ' '), lw=1.8)
        axes[1].plot(ep, [r['val_objective'] for r in h], color=COLORS[cfg],
                     label=label.replace('\n', ' '), lw=1.8)
    axes[0].set_title('训练损失'); axes[0].set_xlabel('epoch'); axes[0].legend(fontsize=9)
    axes[1].set_title('验证目标（1 − 加权IoU，↓越低越好）'); axes[1].set_xlabel('epoch'); axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.3)
    pass
    fig.tight_layout()
    save(fig, 'fig5_training_curves')

    print('figures written:', sorted(p.name for p in FIG.glob('fig*.png')))
    for r in rows:
        if r['status'] == 'done':
            print(f"{r['config']:10s} top={r['top_err_px_mean']:6.2f} bot={r['bot_err_px_mean']:6.2f} "
                  f"iou={r['vert_iou_mean']:.4f} pass3={r['pass_3pct']:.3f} params={r['params_total']}")


if __name__ == '__main__':
    main()
