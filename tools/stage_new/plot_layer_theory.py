"""层级选择论证图（修正版）。

⚠️ 第一版基于"量化误差硬下界"理论，已被 r18_l4 实验证伪，本版重画。

新版三张子图：
  ① 感受野 vs 输入高度 —— L3 是第一个"看得全"的层（这一条依然成立）
  ② 实测精度对比 L3 vs L4 —— 无显著差异（证伪旧理论）
  ③ 选型决策链：为什么停在 L3（够用即止）

数据来源：
  - 感受野：runs/20261103_resnet18_vs_50/reports/resnet_structure_raw.json（hook 实测）
  - 精度：runs/20261103_resnet18_vs_50_corrected/{r18_l3,r18_l4}/metrics.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = Path(__file__).resolve().parents[2]
RUN = PROJ / 'runs' / '20261103_resnet18_vs_50_corrected'
OLD = PROJ / 'runs' / '20261103_resnet18_vs_50'
FIG = RUN / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

IN = 384
STAGES = ['layer1', 'layer2', 'layer3', 'layer4']
LBL = ['Layer1', 'Layer2', 'Layer3', 'Layer4']


def main():
    raw = json.loads((OLD / 'reports' / 'resnet_structure_raw.json').read_text(encoding='utf-8'))
    r18 = raw['resnet18']
    r50 = raw['resnet50']

    l3 = json.loads((RUN / 'resnet18' / 'r18_l3' / 'metrics.json').read_text(encoding='utf-8'))['test']
    l4 = json.loads((RUN / 'resnet18' / 'r18_l4' / 'metrics.json').read_text(encoding='utf-8'))['test']

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    # ---------- ① 感受野 vs 输入尺寸（结论仍成立）----------
    ax = axes[0]
    x = np.arange(4)
    rf18 = [r18['receptive_field'][s]['rf_px'] for s in STAGES]
    rf50 = [r50['receptive_field'][s]['rf_px'] for s in STAGES]
    ax.plot(x, rf18, 'o-', lw=2.2, ms=9, color='#3b7dd8', label='ResNet18')
    ax.plot(x, rf50, 's--', lw=2, ms=7, color='#e07b39', label='ResNet50')
    ax.axhline(IN, ls=':', color='#c0392b', lw=2)
    ax.text(0.05, IN * 1.18, f'输入高度 H = {IN} px', fontsize=9.5, color='#c0392b')
    ax.axhspan(IN, 6000, color='#27ae60', alpha=.09)
    ax.text(0.15, 2600, '可看全图区域', fontsize=10, color='#1e8449', fontweight='bold')
    ax.scatter([2], [rf18[2]], s=320, facecolor='none', edgecolor='#27ae60', lw=3, zorder=5)
    for i, a in enumerate(rf18):
        ax.annotate(f'{a}', (i, a), textcoords='offset points', xytext=(-17, -16), fontsize=9)
    for i, b in enumerate(rf50):
        ax.annotate(f'{b}', (i, b), textcoords='offset points', xytext=(6, 5), fontsize=9)
    ax.annotate('L3 首次覆盖全图\n(483 > 384)', xy=(2, 483), xytext=(0.4, 1600),
                fontsize=9.5, color='#1e8449', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.8))
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(LBL)
    ax.set_ylabel('理论感受野 (px, log)')
    ax.set_title('① 浅层看不到全图\nL1/L2 感受野远小于输入 384 px', fontsize=11)
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=.3)

    # ---------- ② 实测精度：L3 vs L4 ----------
    ax = axes[1]
    metrics = ['上边界\n中位', '上边界\n均值', '下边界\n中位', '下边界\n均值']
    v3 = [l3['top_err_px_median'], l3['top_err_px_mean'],
          l3['bot_err_px_median'], l3['bot_err_px_mean']]
    v4 = [l4['top_err_px_median'], l4['top_err_px_mean'],
          l4['bot_err_px_median'], l4['bot_err_px_mean']]
    xx = np.arange(len(metrics)); w = 0.36
    b1 = ax.bar(xx - w / 2, v3, w, label='仅用 Layer3', color='#3b7dd8')
    b2 = ax.bar(xx + w / 2, v4, w, label='仅用 Layer4', color='#c0392b')
    for bs in (b1, b2):
        for b in bs:
            ax.annotate(f'{b.get_height():.1f}', (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8.5)
    ax.axhline(16, ls='--', color='#7f8c8d', lw=1.8)
    ax.text(3.45, 16.4, '旧理论预测的 L4 下界 16 px（已被实验证伪）',
            fontsize=8, color='#7f8c8d', ha='right', style='italic')
    ax.set_xticks(xx); ax.set_xticklabels(metrics)
    ax.set_ylabel('边界误差 (px)')
    ax.set_title('② L3 与 L4 精度基本持平\nL4 实测 12.0 px，远低于预测下界', fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=.3)
    ax.set_ylim(0, 21)

    # ---------- ③ 选型决策 ----------
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    boxes = [
        (9.0, '输入 384×384 需要全局语义\nL1(43px)、L2(131px) 看不到全图 → 不满足', '#e74c3c'),
        (6.9, 'L3 感受野 483 px ≥ 384 px\n首次能看全整张图 → 满足', '#27ae60'),
        (4.5, '继续加深到 L4？\n实测中位 12.0 px vs L3 的 11.2 px\n无显著差异 (p = 0.87) → 无收益', '#e67e22'),
        (2.0, '决定：使用 Layer3\n（够用即止，而非"L4 不可用"）', '#2c3e50'),
    ]
    for (cy, txt, col) in boxes:
        ax.add_patch(plt.Rectangle((0.9, cy - 0.95), 8.2, 1.9,
                                   facecolor=col, alpha=.13, edgecolor=col, lw=2))
        ax.text(5.0, cy, txt, ha='center', va='center', fontsize=9.2, color='#2c3e50')
    for y0, y1 in [(8.05, 7.85), (5.95, 5.45), (3.55, 3.0)]:
        ax.annotate('', xy=(5, y1), xytext=(5, y0),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    ax.set_title('③ 选型决策链', fontsize=11)

    pass
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig_layer_selection_theory.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print('fig_layer_selection_theory (修正版) 已生成')
    print(f"  L3 上中位 {v3[0]:.1f} 上均值 {v3[1]:.2f} | L4 上中位 {v4[0]:.1f} 上均值 {v4[1]:.2f}")


if __name__ == '__main__':
    main()
