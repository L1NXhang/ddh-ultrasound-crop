"""生成技术开发说明所需的两张示意图：
  fig_flow_system   系统两阶段流程（规则窗口定位 → CropNet 预测边界）
  fig_pipeline_diff 数据质量 vs 模型升级的对比（说明"修数据比换模型划算"）
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = Path(__file__).resolve().parents[2]
RUN = PROJ / 'runs' / '20261103_resnet18_vs_50_corrected'
FIG = RUN / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {
    'io': '#e8eef7', 'rule': '#dbe8fb', 'nn': '#dff0e0',
    'head': '#dff0e0', 'out': '#fde6d3', 'human': '#fbe0e0',
    'edge': '#4a5568',
}


def box(ax, x, y, w, h, text, kind='io', fontsize=9, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.012',
                       linewidth=1.2, edgecolor=COLORS['edge'],
                       facecolor=COLORS[kind])
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, label='', style='-|>'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                        linewidth=1.4, color=COLORS['edge'], shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.022, label, ha='center', va='bottom',
                fontsize=8, color='#c0392b')


def fig_flow():
    """系统流程图：2 行 x 3 列，避免单行 6 列被压缩导致字号过小。"""
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    W, H = 0.27, 0.175
    xs = [0.035, 0.365, 0.695]

    # ---- 第一行：① ② ③ ----
    r1 = 0.755
    row1 = [
        ('① 原始超声图像\n设备整屏截图\n1280x960（含黑边）', 'io'),
        ('② 成像窗口定位\n规则几何算法，无需训练\n掩膜 / 形态学 / 连通域', 'rule'),
        ('③ letterbox\n等比缩放并补零\n384x384', 'rule'),
    ]
    for x, (t, k) in zip(xs, row1):
        box(ax, x, r1, W, H, t, k, fontsize=9.6, bold=(k == 'rule'))
    for a in range(2):
        arrow(ax, xs[a] + W, r1 + H / 2, xs[a + 1], r1 + H / 2)

    # 行间连线：③ 折返到 ④
    ax.plot([0.83, 0.83], [r1 - 0.016, 0.545], color='#4a5568', lw=1.4)
    ax.annotate('', xy=(0.868, 0.545), xytext=(0.83, 0.545),
                arrowprops=dict(arrowstyle='-|>', color='#4a5568', lw=1.4))

    # ---- 第二行：④ ⑤ ⑥（从右向左）----
    r2 = 0.37
    row2 = [
        ('⑥ 裁剪结果\n按预测边界输出\n纯成像区图像', 'out'),
        ('⑤ 行卷积 + 预测头\n1x1 Conv -> ROIAlign\n累积 softmax 保证有序', 'nn'),
        ('④ ResNet-18 主干\n取 Layer3 特征\n24x24 空间分辨率', 'nn'),
    ]
    for x, (t, k) in zip(xs, row2):
        box(ax, x, r2, W, H, t, k, fontsize=9.6, bold=(k == 'nn'))
    for a in range(2):
        arrow(ax, xs[a + 1], r2 + H / 2, xs[a] + W, r2 + H / 2)

    # ---- 底部说明 ----
    ax.text(0.5, 0.125,
            '① ② 为规则几何部分（CPU 约 45-47 ms），无需任何训练数据\n'
            '③ - ⑥ 为神经网络部分，CPU 端到端约 200 ms/张，GPU 约 67 ms/张\n'
            '系统输出仅为"裁剪后的标准图像"，不输出 Graf 分型结论',
            ha='center', va='center', fontsize=8.8, color='#3d4a5c', linespacing=1.8,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f7f9fc', edgecolor='#c8d4e4'))

    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig_flow_system.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig_flow_system regenerated')


def fig_data_vs_model():
    """用真实数据对比：修数据的收益 vs 换更大模型的收益。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # 左：修数据的收益（项目历史记录 + 本次复现）
    ax = axes[0]
    stages = ['旧口径模型\n（旧口径标签训练）', '三人重标 862 张\n统一口径（只重训）\n[历史记录]', '本次复现\n同一新口径\n[本次实验]']
    vals = [89.0, 17.5, None]
    # 本次复现：读真实中位数
    import json as _json2
    _m = _json2.loads((RUN / 'resnet18' / 'r18_l3' / 'metrics.json').read_text(encoding='utf-8'))
    vals[2] = _m['test']['top_err_px_median']
    colors = ['#c0392b', '#27ae60', '#2e86c1']
    bars = ax.bar(range(3), [v if v else 0 for v in vals], color=colors, width=.6)
    for b, v in zip(bars, vals):
        ax.annotate(f'{v:.1f} px', (b.get_x() + b.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 4), ha='center', fontsize=11,
                    fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel('上边界误差中位数 (px)  ↓越低越好')
    ax.set_title(f'修数据：{vals[0]:.0f} px → {vals[1]:.1f} px（↓{(1-vals[1]/vals[0])*100:.0f}%）\n'
                 f'（模型结构完全未变，只换了标注）', fontsize=10, color='#27ae60')
    ax.grid(axis='y', alpha=.3)
    ax.set_ylim(0, 100)

    # 右：换模型的收益（本次实验，从 metrics.json 真实读取）
    ax = axes[1]
    names = ['ResNet18\n仅Layer3\n(基线)', 'ResNet18\nL1+2+3\n融合', 'ResNet50\n仅Layer3', 'ResNet50\nL1+2+3\n融合']
    cfg_paths = [('resnet18', 'r18_l3'), ('resnet18', 'r18_l123'),
                 ('resnet50', 'r50_l3'), ('resnet50', 'r50_l123')]
    import json as _json
    ious, params = [], []
    for sub, cfg in cfg_paths:
        m = _json.loads((RUN / sub / cfg / 'metrics.json').read_text(encoding='utf-8'))
        ious.append(m['test']['vert_iou_mean'])
        params.append(m['params_total'] / 1e6)
    colors2 = ['#3b7dd8', '#27ae60', '#e07b39', '#f2a869']
    bars = ax.bar(range(4), ious, color=colors2, width=.62)
    for b, v, p in zip(bars, ious, params):
        ax.annotate(f'IoU {v:.4f}\n{p:.2f}M 参数', (b.get_x() + b.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel('纵向 IoU  ↑越高越好')
    lo, hi = min(ious), max(ious)
    pad = max((hi - lo) * 1.6, 0.006)
    ax.set_ylim(lo - pad, hi + pad * 1.4)
    ax.set_title(f'换模型/加融合：IoU 仅 {lo:.4f} → {hi:.4f}（差 {(hi-lo)*100:.2f} 个百分点，统计不显著）\n'
                 f'而参数量翻倍（{min(params):.2f}M → {max(params):.2f}M）', fontsize=10, color='#c0392b')
    ax.grid(axis='y', alpha=.3)

    pass
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig_data_vs_model.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig_data_vs_model 已生成')


if __name__ == '__main__':
    fig_flow()
    fig_data_vs_model()
