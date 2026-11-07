"""生成项目背景对比图：未裁剪原图 → 本系统裁剪结果 → Graf 分型示意图。

用途：技术开发说明第一章的"设计目标"配图，说明
  "我们的工作是把 左图 自动变成 中图，从而让 右图 的分析变得便捷可靠"。

输出：
  figures/fig_bg_compare.png/.pdf    三栏对比图（左：原图，中：裁剪结果，右：Graf 分型示意）
  figures/fig_crop_demo.png/.pdf     单独一张"裁剪前后"对照（大图，供正文使用）
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
from PIL import Image

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = Path(__file__).resolve().parents[2]
RUN = PROJ / 'runs' / '20261103_resnet18_vs_50_corrected'
FIG = RUN / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# 选一张测试集里预测准确的样例（案例 01 的 id）
CASE_ID = '47356990'


def load_case(case_id):
    """从 test_per_image.jsonl 取该图的预测与真值，以及原图路径。"""
    rows = [json.loads(l) for l in (RUN / 'resnet18' / 'r18_l3' / 'test_per_image.jsonl')
            .read_text(encoding='utf-8').splitlines() if l.strip()]
    rec = next(r for r in rows if str(r['id']) == case_id)
    metas = {}
    for l in (PROJ / 'data' / 'reports_v2' / 'split.jsonl').read_text(encoding='utf-8').splitlines():
        if not l.strip():
            continue
        m = json.loads(l)
        metas[str(m['id'])] = m
    return rec, metas[case_id]


def draw_graf_on_crop(ax, crop):
    """在真实裁剪图上叠加 Graf 标注（参照临床文献的标注风格自绘，非复制原图）。

    标注元素与文献一致：
      - 基准线 baseline（蓝色虚线，沿髂骨外缘）
      - 骨顶线 L1 bony roof line（黄色虚线）
      - 软骨顶线 L2 cartilage roof line（浅橙虚线）
      - α 角（基准线与骨顶线）、β 角（基准线与软骨顶线）
    """
    import numpy as np
    ax.imshow(crop, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    h, w = crop.size[1], crop.size[0]

    # 归一化坐标 (0~1)，原点左上
    C = (0.36, 0.50)          # 髋臼顶点：两条顶线的交点
    a_deg, b_deg = 33.7, -20.1  # 骨顶线相对水平的角度；软骨顶线角度

    def pt(px, py):
        return px * w, py * h

    # ---- baseline：蓝色虚线，水平向右 ----
    ax.plot([C[0]*w, 0.97*w], [C[1]*h, C[1]*h],
            color='#1f9ed9', lw=2.0, ls=(0, (6, 3)), zorder=6)
    ax.text(0.80*w, C[1]*h - 0.035*h, 'base line  $L_B$', color='#1f9ed9',
            fontsize=9.5, fontweight='bold', ha='center', zorder=8)

    # ---- 骨顶线 L1：黄色虚线，从髋臼顶点向右上 ----
    L = 0.62
    x1 = C[0] + L*abs(np.cos(np.radians(a_deg)))
    y1 = C[1] - L*abs(np.sin(np.radians(a_deg)))
    ax.plot([C[0]*w, x1*w], [C[1]*h, y1*h],
            color='#f1c40f', lw=2.2, ls=(0, (6, 3)), zorder=6)
    ax.text(0.03*w, 0.93*h, 'bony roof line\n$L_1$', color='#f1c40f',
            fontsize=9.5, fontweight='bold', va='top', zorder=8)

    # ---- 软骨顶线 L2：浅橙虚线，从髋臼顶点向右上但更平 ----
    a2 = 55.0
    x2 = C[0] + 0.56*abs(np.cos(np.radians(a2)))
    y2 = C[1] - 0.56*abs(np.sin(np.radians(a2)))
    ax.plot([C[0]*w, x2*w], [C[1]*h, y2*h],
            color='#e8a87c', lw=2.2, ls=(0, (5, 3)), zorder=6)
    ax.text(0.52*w, 0.10*h, 'cartilage roof line\n$L_2$', color='#e8a87c',
            fontsize=9.5, fontweight='bold', va='bottom', zorder=8)

    # ---- α 角弧：baseline(水平向右) 与 L1(右上) 之间 ----
    # Arc 在数据坐标下画；宽高按图像像素比例换算，保证是正圆
    from matplotlib.patches import Arc
    R1 = 0.22 * w
    ax.add_patch(Arc((C[0]*w, C[1]*h), R1, R1,
                     theta1=-a_deg, theta2=0,
                     color='#f1c40f', lw=2.4, zorder=9))
    ax.text((C[0]+0.045)*w, (C[1]+0.055)*h, r'$\alpha$', color='#f1c40f',
            fontsize=17, fontweight='bold', zorder=10)

    # ---- β 角弧：baseline(水平向右) 与 L2(右上更陡) 之间 ----
    R2 = 0.30 * w
    ax.add_patch(Arc((C[0]*w, C[1]*h), R2, R2,
                     theta1=-a2, theta2=-a_deg,
                     color='#e8a87c', lw=2.4, zorder=9))
    ax.text((C[0]+0.145)*w, (C[1]-0.085)*h, r'$\beta$', color='#e8a87c',
            fontsize=15, fontweight='bold', zorder=10)


def main():
    rec, meta = load_case(CASE_ID)
    img = Image.open(PROJ / 'data' / meta['path']).convert('RGB')
    w, h = img.size
    W = meta['window']               # [x0,y0,x1,y1]
    gt = rec['gt_bounds']            # 归一化 [top, bot]
    pred = rec['pred_bounds']

    # 先算好裁剪结果（中栏），供排版阶段读取其真实尺寸
    wh = W[3] - W[1]
    y_top = int(round(W[1] + gt[0] * wh))
    y_bot = int(round(W[1] + gt[1] * wh))
    y_top = max(W[1], min(W[3] - 1, y_top))
    y_bot = max(y_top + 1, min(W[3], y_bot))
    crop = img.crop((W[0], y_top, W[2], y_bot))

    # ---------- 三栏排版：每栏按自身图像的真实宽高比设置坐标轴 ----------
    # 关键：不使用 aspect='auto'（那会把图拉伸填满坐标轴框）。
    # 这里让坐标轴框的宽高比 === 图像的宽高比，图像自然填满且不变形。
    graf_img = Image.open(PROJ / 'docs' / '开发文档_第2版' / 'fig_graf_user.png').convert('RGB')
    mid_img = Image.open(PROJ / 'docs' / '开发文档_第2版' / 'fig_crop_user.jpg').convert('RGB')

    W_FIG, H_FIG = 14.6, 5.2
    H_IN = 3.18                      # 三栏统一高度（英寸），宽度由各自比例推出
    asp = [w / h,
           mid_img.size[0] / mid_img.size[1],
           graf_img.size[0] / graf_img.size[1]]

    fig = plt.figure(figsize=(W_FIG, H_FIG))
    y0 = 0.145
    hf = H_IN / H_FIG
    x = 0.015
    GAP = 0.020
    boxes = []
    for a in asp:
        wf = H_IN * a / W_FIG
        boxes.append((x, y0, wf, hf))
        x += wf + GAP
    axes = [fig.add_axes(b) for b in boxes]

    # ---------- 左：未裁剪原图 ----------
    ax = axes[0]
    ax.imshow(img)
    ax.add_patch(plt.Rectangle((W[0], W[1]), W[2] - W[0], W[3] - W[1],
                               fill=False, edgecolor='#27ae60', lw=2.2, ls='--'))
    ax.text(W[0] + 8, W[1] - 14, '有效成像区', color='#27ae60', fontsize=10.5, fontweight='bold')
    ax.set_title('① 原始超声图像（未裁剪）' + chr(10) + '四周黑边 + 设备参数栏 + 深度标尺', fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, -0.075, f'{w}×{h} px', transform=ax.transAxes, fontsize=9.5, color='#7f8c8d')

    # ---------- 中：裁剪结果（保持原始宽高比） ----------
    ax = axes[1]
    ax.imshow(mid_img)
    ax.set_title('② 本系统自动裁剪结果' + chr(10) + '去除黑边与标尺，仅保留成像区', fontsize=11.5, color='#27ae60')
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, -0.075, f'{mid_img.size[0]}×{mid_img.size[1]} px',
            transform=ax.transAxes, fontsize=9.5, color='#27ae60')

    # ---------- 右：Graf 分型标注（保持原始宽高比） ----------
    ax = axes[2]
    ax.imshow(graf_img)
    ax.set_title('③ Graf 分型标注' + chr(10) + '以基准线、骨顶线、软骨顶线测 α / β 角',
                 fontsize=11.5, color='#8e44ad')
    ax.set_xticks([]); ax.set_yticks([])

    # 顶部流程箭头（落在两栏之间的空隙里）
    y_arrow = y0 + hf / 2
    for i in (0, 1):
        xa = boxes[i][0] + boxes[i][2] + 0.002
        xb = boxes[i + 1][0] - 0.002
        fig.patches.append(FancyArrowPatch((xa, y_arrow), (xb, y_arrow),
                                           transform=fig.transFigure,
                                           arrowstyle='-|>', mutation_scale=22,
                                           color='#34495e', lw=2.4))

    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig_bg_compare.{ext}', dpi=210, bbox_inches='tight')
    plt.close(fig)
    print('fig_bg_compare 已生成')

    # ---------- 单独一张裁剪前后对照 ----------
    fig2, axs = plt.subplots(1, 2, figsize=(9.6, 5.0))
    axs[0].imshow(img)
    axs[0].add_patch(plt.Rectangle((W[0], W[1]), W[2] - W[0], W[3] - W[1],
                                   fill=False, edgecolor='#27ae60', lw=2.2, ls='--'))
    axs[0].set_title(f'裁剪前  {w}×{h} px', fontsize=11)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[1].imshow(crop, aspect='auto')
    axs[1].set_title(f'裁剪后  {crop.size[0]}×{crop.size[1]} px', fontsize=11, color='#27ae60')
    axs[1].set_xticks([]); axs[1].set_yticks([])
    pass
    fig2.tight_layout()
    for ext in ('png', 'pdf'):
        fig2.savefig(FIG / f'fig_crop_demo.{ext}', dpi=210, bbox_inches='tight')
    plt.close(fig2)
    print('fig_crop_demo 已生成')
    print(f'  样例 id={CASE_ID}  原图 {w}×{h} → 裁剪 {crop.size[0]}×{crop.size[1]}')


if __name__ == '__main__':
    sys.exit(main())
