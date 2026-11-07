"""A 项出图：ResNet18 vs ResNet50 层级结构图（分辨率 / 通道数 / 感受野）。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RUN = Path(__file__).resolve().parents[2] / 'runs' / '20261103_resnet18_vs_50'
RAW = json.loads((RUN / 'reports' / 'resnet_structure_raw.json').read_text(encoding='utf-8'))
FIG = RUN / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

STAGES = ['conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
LABEL = {'conv1': 'conv1', 'maxpool': 'maxpool', 'layer1': 'Layer1',
         'layer2': 'Layer2', 'layer3': 'Layer3', 'layer4': 'Layer4'}


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    r18, r50 = RAW['resnet18'], RAW['resnet50']

    # ---------- 图1：三联图（分辨率 / 通道数 / 感受野） ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(STAGES))
    w = 0.38

    res18 = [r18['stages'][s]['h'] for s in STAGES]
    res50 = [r50['stages'][s]['h'] for s in STAGES]
    ch18 = [r18['stages'][s]['channels'] for s in STAGES]
    ch50 = [r50['stages'][s]['channels'] for s in STAGES]
    rf18 = [r18['receptive_field'][s]['rf_px'] for s in STAGES]
    rf50 = [r50['receptive_field'][s]['rf_px'] for s in STAGES]

    ax = axes[0]
    # 分辨率两网络完全相同，重合会看不清 -> 画一条折线 + 数值标注
    ax.plot(x, res18, 'o-', color='#3b7dd8', lw=2, ms=7, label='ResNet18 = ResNet50（完全重合）')
    for i, v in enumerate(res18):
        ax.annotate(f'{v}', (i, v), textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=9)
    ax.set_yscale('log'); ax.set_ylabel('特征图边长 (px, log)')
    ax.set_title('① 空间分辨率逐层下降（两网络一致）')
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in STAGES], rotation=30, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    ax = axes[1]
    ax.bar(x - w / 2, ch18, w, label='ResNet18', color='#3b7dd8')
    ax.bar(x + w / 2, ch50, w, label='ResNet50', color='#e07b39')
    ax.set_yscale('log'); ax.set_ylabel('通道数 (log)')
    ax.set_title('② 通道数逐层增加')
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in STAGES], rotation=30, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    ax = axes[2]
    ax.bar(x - w / 2, rf18, w, label='ResNet18', color='#3b7dd8')
    ax.bar(x + w / 2, rf50, w, label='ResNet50', color='#e07b39')
    ax.axhline(384, ls='--', c='k', lw=1)
    ax.text(0.02, 420, '输入边长 384 px', fontsize=9)
    ax.set_yscale('log'); ax.set_ylabel('理论感受野 (px, log)')
    ax.set_title('③ 感受野逐层增大（越大=语义越强，定位越粗）')
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in STAGES], rotation=30, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    fig.suptitle('图1  ResNet18 vs ResNet50 层级特征变化（输入 384×384，实测 hook 读数）', fontsize=13)
    fig.tight_layout()
    save(fig, 'fig1_resnet_structure_compare')

    # ---------- 图2：特征图空间尺寸示意（方格图） ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (tag, d) in zip(axes, [('ResNet18', r18), ('ResNet50', r50)]):
        ax.set_title(f'{tag} 各层特征图空间尺寸')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 384); ax.set_ylim(-0.5, 384)
        for i, s in enumerate(STAGES[2:]):
            n = d['stages'][s]['h']
            size = 384 * (n / 384)
            ax.add_patch(plt.Rectangle((0, i * 20), size, 16, fill=False,
                                       edgecolor=plt.cm.viridis(i / 4), lw=2))
            ax.text(4, i * 20 + 5, f"{LABEL[s]}  {n}×{n}  ch={d['stages'][s]['channels']}",
                    fontsize=9, color=plt.cm.viridis(i / 4))
        ax.set_yticks([])
        ax.invert_yaxis()
    fig.suptitle('图2  分辨率 × 通道数的权衡：越深越"宽"越"粗"', fontsize=13)
    fig.tight_layout()
    save(fig, 'fig2_resolution_channel_tradeoff')

    # ---------- 表 ----------
    lines = []
    for s in STAGES:
        lines.append(dict(stage=LABEL[s],
                          r18_res=f"{r18['stages'][s]['h']}×{r18['stages'][s]['w']}",
                          r18_ch=r18['stages'][s]['channels'],
                          r18_rf=r18['receptive_field'][s]['rf_px'],
                          r50_res=f"{r50['stages'][s]['h']}×{r50['stages'][s]['w']}",
                          r50_ch=r50['stages'][s]['channels'],
                          r50_rf=r50['receptive_field'][s]['rf_px'],
                          r18_params=r18['params_per_stage'][s],
                          r50_params=r50['params_per_stage'][s]))
    (RUN / 'reports' / 'resnet_structure_table.json').write_text(
        json.dumps(lines, ensure_ascii=False, indent=2), encoding='utf-8')

    # CSV（给 PPT/文档直接用）
    import csv
    with open(RUN / 'reports' / 'resnet_structure_table.csv', 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(lines[0]))
        wr.writeheader(); wr.writerows(lines)
    print('figures + table written')


if __name__ == '__main__':
    main()
