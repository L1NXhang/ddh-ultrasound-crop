"""E 项出图：三组 × 50 张的推理成本柱状图。

主指标采用 **CPU 推理时间**，理由见 bench_cpu.py 文档：
  GPU 计时被本机桌面应用干扰（离散度最高 214%，出现"融合比基线快 10 倍"的不可能结果），
  而 CPU 测量稳定（组间波动 ±3%），且项目本身提供 CPU 安装包，具有真实部署意义。
GPU 数据仍保留在 reports/bench_inference_3groups.csv 中作为参考，但**不用于结论**。

⚠️ 这些图**没有任何准确率柱子** —— 所用 150 张来自「未使用（未标注）」目录，
没有人工标注，无法计算准确率。图标题中明确写死这一点。
"""
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = Path(__file__).resolve().parents[2]
RUN = PROJ / 'runs' / '20261103_resnet18_vs_50_corrected'
REP, FIG = RUN / 'reports', RUN / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

CONFIGS = ['r18_l3', 'r50_l3', 'r18_l123', 'r50_l123', 'r18_l4', 'r18_l1234']
# 图 7 专用：暖色调（橙-红-棕），与图 6 的冷色调形成区分
COLORS = {'r18_l3': '#D9741E', 'r50_l3': '#C0392B',
          'r18_l123': '#F0B268', 'r50_l123': '#E28C7A',
          'r18_l4': '#8C4A2F', 'r18_l1234': '#BFA58C'}
GROUPS = ['A', 'B', 'C']


def main():
    p = REP / 'bench_cpu_3groups.csv'
    if not p.exists():
        raise SystemExit(f'缺少 {p}，请先运行 bench_cpu.py')
    rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
    by = {(r['config'], r['group']): r for r in rows}
    configs = [c for c in CONFIGS if (c, 'A') in by]

    x = np.arange(len(GROUPS)); w = 0.8 / len(configs)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    # 左：CPU 单张推理时间
    ax = axes[0]
    for i, cfg in enumerate(configs):
        vals = [float(by[(cfg, g)]['cpu_infer_ms_median']) for g in GROUPS]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=cfg, color=COLORS[cfg])
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.1f}', (b.get_x() + b.get_width() / 2, v),
                        textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'样本组 {g}\n(50 张)' for g in GROUPS])
    ax.set_ylabel('单张前向耗时 (ms)')
    ax.set_title('CPU 单张推理时间（稳定，组间波动 ±3%）', fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

    # 右：模型大小 + 参数量（与样本无关，取 A 组代表的常量）
    ax = axes[1]
    sizes = [float(by[(c, 'A')]['checkpoint_mb']) for c in configs]
    params = [int(by[(c, 'A')]['params_total']) / 1e6 for c in configs]
    xc = np.arange(len(configs))
    bars = ax.bar(xc, sizes, 0.55, color=[COLORS[c] for c in configs])
    for b, v, pm in zip(bars, sizes, params):
        ax.annotate(f'{v:.1f} MB\n({pm:.1f}M 参数)',
                    (b.get_x() + b.get_width() / 2, v), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=8)
    ax.set_xticks(xc); ax.set_xticklabels(configs, rotation=15, fontsize=9)
    ax.set_ylabel('模型文件大小 (MB)')
    ax.set_title('模型大小与参数量', fontsize=10)
    ax.grid(axis='y', alpha=.3)
    ax.set_ylim(0, max(sizes) * 1.28)

    pass
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig7_three_groups_cost.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 图8：CPU 时间的相对倍率（更直观）
    fig, ax = plt.subplots(figsize=(8, 4.4))
    base = float(by[('r18_l3', 'A')]['cpu_infer_ms_median'])
    ratios = [np.mean([float(by[(c, g)]['cpu_infer_ms_median']) for g in GROUPS]) / base
              for c in configs]
    bars = ax.bar(range(len(configs)), ratios, color=[COLORS[c] for c in configs], width=.6)
    for b, v, c in zip(bars, ratios, configs):
        ms = np.mean([float(by[(c, g)]['cpu_infer_ms_median']) for g in GROUPS])
        ax.annotate(f'{v:.2f}×\n({ms:.0f} ms)', (b.get_x() + b.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
    ax.axhline(1.0, ls='--', c='k', lw=1)
    ax.set_xticks(range(len(configs))); ax.set_xticklabels(configs, rotation=15)
    ax.set_ylabel('相对 ResNet18/仅L3 的 CPU 推理耗时倍率')
    ax.set_title('图8  CPU 推理耗时倍率（基线 = ResNet18/仅L3）')
    ax.set_ylim(0, max(ratios) * 1.25); ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig8_params_by_config.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print('bench figures written')
    for c in configs:
        ms = np.mean([float(by[(c, g)]['cpu_infer_ms_median']) for g in GROUPS])
        print(f'  {c:10s} CPU {ms:6.1f} ms/img  比值 {ms/base:.2f}x  '
              f'ckpt {by[(c,"A")]["checkpoint_mb"]} MB')


if __name__ == '__main__':
    main()
