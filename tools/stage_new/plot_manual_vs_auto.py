"""人工裁剪 vs 本系统自动出图 —— 效率对比图。

数据来源（全部为真实测量，见 reports/manual_annotation_baseline.md）：
  人工：本组 3 名成员，每人手工裁剪 10 张，计时如下（单位：秒）
        第1人 2分34秒96 = 154.96 s
        第2人 2分20秒78 = 140.78 s
        第3人 2分43秒45 = 163.45 s
  系统：docs/CPU_GPU速度实测_20261015.md 中的端到端 CPU 单张耗时 199.31 ms

输出：
  figures/fig_manual_vs_auto.png/.pdf   左右并排的效率对比图
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = Path(__file__).resolve().parents[2]
RUN = PROJ / 'runs' / '20261103_resnet18_vs_50_corrected'
FIG = RUN / 'figures'
WORK = FIG / 'manual_vs_auto'

# ---------- 真实测量数据（秒 / 10 张）----------
MANUAL_SEC = [154.96, 140.78, 163.45]      # 第1/2/3 人
N_IMG = 10
SYSTEM_MS = 199.31                          # 本系统 CPU 端到端单张耗时


def main():
    manual_per = [t / N_IMG for t in MANUAL_SEC]
    m_mean = sum(manual_per) / len(manual_per)
    m_min, m_max = min(manual_per), max(manual_per)
    sys_per = SYSTEM_MS / 1000.0
    speedup = m_mean / sys_per

    fig = plt.figure(figsize=(15.4, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.235], hspace=0.10, wspace=0.035,
                          left=0.028, right=0.972, top=0.955, bottom=0.035)

    # ---------------- 左：人工裁剪 ----------------
    axL = fig.add_subplot(gs[0, 0])
    imgL = Image.open(WORK / 'left_manual_paint.png')
    axL.imshow(imgL)
    axL.set_xticks([]); axL.set_yticks([])
    for sp in axL.spines.values():
        sp.set_color('#c0392b'); sp.set_linewidth(3)
    axL.set_title('① 人工裁剪：画笔软件里手工框选、逐个另存',
                  fontsize=13, fontweight='bold', color='#c0392b', pad=12)

    # ---------------- 右：本系统 ----------------
    axR = fig.add_subplot(gs[0, 1])
    imgR = Image.open(WORK / 'right_auto_ui.png')
    axR.imshow(imgR)
    axR.set_xticks([]); axR.set_yticks([])
    for sp in axR.spines.values():
        sp.set_color('#1e8449'); sp.set_linewidth(3)
    axR.set_title('② 本系统：整文件夹导入，自动预测边界，一键导出',
                  fontsize=13, fontweight='bold', color='#1e8449', pad=12)

    # ---------------- 中间分隔：细箭头 + "改进" 文字 ----------------
    fig.patches.append(FancyArrowPatch(
        (0.478, 0.615), (0.522, 0.615), transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=22, color='#7f8c8d', lw=1.8, zorder=20))

    # ---------------- 下左：耗时数字 ----------------
    axB1 = fig.add_subplot(gs[1, 0])
    axB1.axis('off')
    axB1.set_xlim(0, 10); axB1.set_ylim(0, 10)
    axB1.add_patch(FancyBboxPatch((0.1, 0.6), 9.8, 8.6, boxstyle='round,pad=0.1',
                                  facecolor='#fdecea', edgecolor='#c0392b', lw=1.8))
    lines = [
        f'人均 {m_mean:.2f} 秒/张   （范围 {m_min:.2f}–{m_max:.2f} 秒）',
        f'3 人 × 10 张实测：154.96 / 140.78 / 163.45 秒',
        f'400 张约需 {m_mean * 400 / 60:.0f} 分钟',
    ]
    for i, t in enumerate(lines):
        axB1.text(5.0, 7.6 - i * 2.0, t, ha='center', va='center',
                  fontsize=11.5 if i == 0 else 9.8,
                  color='#c0392b', fontweight='bold' if i == 0 else 'normal')

    # ---------------- 下右：耗时数字 ----------------
    axB2 = fig.add_subplot(gs[1, 1])
    axB2.axis('off')
    axB2.set_xlim(0, 10); axB2.set_ylim(0, 10)
    axB2.add_patch(FancyBboxPatch((0.1, 0.6), 9.8, 8.6, boxstyle='round,pad=0.1',
                                  facecolor='#eafaf1', edgecolor='#1e8449', lw=1.8))
    lines2 = [
        f'{sys_per:.3f} 秒/张   （CPU 端到端实测）',
        f'全流程含设备黑边去除 + 窗口定位 + 边界预测',
        f'400 张约需 {sys_per * 400 / 60:.1f} 分钟',
    ]
    for i, t in enumerate(lines2):
        axB2.text(5.0, 7.6 - i * 2.0, t, ha='center', va='center',
                  fontsize=11.5 if i == 0 else 9.8,
                  color='#1e8449', fontweight='bold' if i == 0 else 'normal')

    # 图内不放总标题：由 LaTeX 的 \caption 统一编号，避免与正文图题重复

    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig_manual_vs_auto.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print('fig_manual_vs_auto 已生成')
    print(f'  人工人均 {m_mean:.2f} s/张（{m_min:.2f}–{m_max:.2f}）')
    print(f'  系统 {sys_per:.3f} s/张，提速 {speedup:.1f} 倍')
    print(f'  400 张：人工 {m_mean*400/60:.1f} 分钟 vs 系统 {sys_per*400/60:.1f} 分钟')


if __name__ == '__main__':
    main()
