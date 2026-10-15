"""通用成像窗口检测：逐图检测，不依赖任何模板。

设计依据（全部来自本项目实测，不是凭直觉）
────────────────────────────────────────
1) 内容判据：黑边像素恒为 0；成像区即使很暗，也总有一部分像素 > 2/255
   （实测深部暗区的「非纯黑占比」0.36~0.88）。二值判据比亮度、比局部标准差都稳。
   注意顶部有一条近乎黑的窄带（行灰度仅 5~16），块阈值必须取得足够低才能保留。

2) 粗框：内容掩膜 → 水平开运算 → 闭运算 → 取最大连通域的外接框。
   开运算在闭运算之前，是为了先移除深度标尺刻度点这类 1~2 块宽的竖细结构；
   否则闭运算会把它们膨胀成横贯的桥，把内容与右侧界面元素连成一片
   （实测：最大连通域右边界被拖到 x=984，而真实内容止于 x=864）。

3) 左右边界：走廊 + 像素级吸附，见 _corridor_edge 的说明。

4) 上下边界：从粗框边缘向内扫描，见 _scan_row 的说明。

5) 拒识：面积占比异常、贴图像边界、存在第二个大连通域（拼图/双屏）→ 标记可疑。

验收（人工确认窗口为真值，四边各自 ≤4px 命中率）：
   老设备 862 张：左 100% 上 100% 右 99% 下 95%，四边同时达标 94%
   新设备  89 张：左  99% 上  98% 右 97% 下 100%，四边同时达标 93%
   复现：python tools/validate_window.py（老设备）；新设备见 docs/批量裁剪工具说明.md

接口：
    detect(img_or_path) -> dict(box, confidence, flags, area_ratio, second_ratio)
"""
from pathlib import Path

import numpy as np
from PIL import Image

# ---- 参数（均由实测标定，改动请重跑 tools/validate_window.py）----
# 以下阈值全部由实测标定（tools/validate_window.py 可复现）：
#   黑边：像素最大值为 0，"非纯黑像素占比" = 0.000
#   成像区深部：均值为 0（暗）但非纯黑占比 0.36~0.88
# 所以判据用「非纯黑像素比例」，而不是亮度或局部标准差。
T_NONBLACK = 2     # 像素级：大于它就不算纯黑
T_BLOCK = 0.05     # 块级：4×4 块里非纯黑占比超过它 → 该块算内容
                   # 取低值保证敏感性：成像区边缘的稀疏内容带不会被误杀。
                   # 稀疏干扰元素（刻度点、TGC 条）不靠调高此阈值排除——
                   # 那样会连真实内容带一起切掉——而由细化阶段的结构判据处理。
BLOCK = 4          # 块大小
MIN_AREA = 0.05    # 内容面积占整图比例下限
MAX_AREA = 0.95    # 上限
SECOND_RATIO = 0.25  # 第二大连通域 / 最大连通域 超过此值 → 疑似多窗口

# ---- 细化参数 ----
T_LOW = 0.35       # 「走廊」判据：列占用率低于它算黑边
                   # 实测 0.35 优于 0.22：偏向早判、再由吸附拉回像素级，
                   # 两设备右边界 ≤4px 命中率 97%/99%（0.22 时 94%/99%）
G_MAX = 2          # 连续这么多列低于 T_LOW 才算走廊（允许内容内部短暗缝）
SNAP = 12          # 像素级吸附的邻域半径
W_SNAP = 4         # 吸附梯度两侧取样宽度
# 上下：边界处常是近乎黑的窄带（新设备实测行均值 4.9~16.6，窗外 0.13），
#   二值占用率在这里是噪声级（0.02~0.09），改用行灰度均值 + 连通游走，
#   从内容深处向两边走，遇到低于阈值的行就停——孤立的文字行不会把边界拉过去。
T_EDGE = 0.20      # 上下边：从粗框边缘向内的扫描判据（行占用率 > 它 → 该行有内容）
# 事后检查：窗口外若仍有一整片实心内容，说明该边切早了（如声影暗带被当成窗外走廊）。
# 用「阈值 + 连续宽度」双条件，避免把 TGC 条(19px, 0.37~0.41)、标尺线(3px, 0.5~0.54)
# 这类窄条界面元素误报成"窗外还有内容"。
T_SOLID = 0.50     # 该列/行的占用率下限
SOLID_RUN = 5      # 连续这么多列/行（4px/块 → 20px）才算"一整片"


def _box_sum(a, k):
    """k×k 滑窗求和（积分图，O(N)）"""
    pad = k // 2
    A = np.pad(a, pad, mode='edge')
    c = np.cumsum(np.cumsum(A, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = a.shape
    return (c[k:k + H, k:k + W] - c[0:H, k:k + W] - c[k:k + H, 0:W] + c[0:H, 0:W])


def content_mask(a):
    """内容掩膜：4×4 块里「非纯黑像素占比」超过 T_BLOCK 的块算内容。

    a 为原分辨率灰度图。用像素级判定再聚合到块级，而不是先缩放再判定——
    缩放会把暗区的稀疏斑点平均掉，导致深部内容被误判为黑边（这是之前的失败原因）。
    """
    nb = (a > T_NONBLACK).astype(np.float32)
    H, W = nb.shape
    Hc, Wc = H // BLOCK, W // BLOCK
    nb = nb[:Hc * BLOCK, :Wc * BLOCK].reshape(Hc, BLOCK, Wc, BLOCK)
    frac = nb.mean(axis=(1, 3))
    return frac > T_BLOCK


def close_mask(m, r=2):
    """形态学闭运算（先膨胀后腐蚀），用最大/最小滤波近似"""
    def dil(x):
        out = x.copy()
        for dy in range(-r, r + 1):
            out |= np.roll(x, dy, axis=0)
            out |= np.roll(x, dy, axis=1)
        return out

    def ero(x):
        out = x.copy()
        for dy in range(-r, r + 1):
            out &= np.roll(x, dy, axis=0)
            out &= np.roll(x, dy, axis=1)
        return out
    return ero(dil(m))


def open_x(m, r=1):
    """水平方向开运算：一个块必须在左右各 r 个块内都连续存在，才保留。

    用途：深度标尺刻度点、TGC 指示条等「竖直细长/稀疏」界面元素只有 1~2 个块宽，
    会被结构性移除；而成像区是数百块宽的整体，左右边界不受影响。
    必须在 close_mask 之前调用：闭运算会把稀疏点先膨胀连成横贯的桥，之后再也开不掉。
    """
    def ero(x):
        out = x.copy()
        for dx in range(-r, r + 1):
            out &= np.roll(x, dx, axis=1)
        return out

    def dil(x):
        out = x.copy()
        for dx in range(-r, r + 1):
            out |= np.roll(x, dx, axis=1)
        return out
    return dil(ero(m))


def components(m):
    """两遍扫描 + 并查集，返回 [(面积, y0, x0, y1, x1), ...]（按面积降序）"""
    H, W = m.shape
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    labels = np.zeros((H, W), dtype=np.int32)
    nxt = 1
    for y in range(H):
        row = m[y]
        for x in np.flatnonzero(row):
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up and left:
                labels[y, x] = min(up, left)
                union(up, left)
            elif up:
                labels[y, x] = up
            elif left:
                labels[y, x] = left
            else:
                labels[y, x] = nxt
                parent[nxt] = nxt
                nxt += 1
    # 归并
    boxes = {}
    ys, xs = np.nonzero(labels)
    for y, x in zip(ys, xs):
        root = find(int(labels[y, x]))
        b = boxes.get(root)
        if b is None:
            boxes[root] = [1, y, x, y, x]
        else:
            b[0] += 1
            if y < b[1]: b[1] = y
            if x < b[2]: b[2] = x
            if y > b[3]: b[3] = y
            if x > b[4]: b[4] = x
    out = sorted(boxes.values(), key=lambda b: -b[0])
    return [(b[0], b[1], b[2], b[3] + 1, b[4] + 1) for b in out]


def _corridor_edge(col, lo, hi, side, x_start):
    """左右边：从内容内部向外扫，遇到第一条「连续 ≥G_MAX 列低于 T_LOW」的走廊即停。

    各设备的共同结构：内容 → 黑走廊（窗外）→ 才是各种界面元素。
    走廊之后的干扰（深度标尺竖线、TGC 渐变条，实测占用率 0.37~0.54）
    根本扫不到，因此不需要靠阈值把干扰和内容分开——
    这一点很关键：老设备窗口右缘本身是渐隐的（0.36~0.57），
    与 TGC 条（0.37）数值完全重叠，任何静态阈值都分不开两者。
    允许内容内部存在短于 G_MAX 的暗缝（无回声带、测量线），不会被切断。
    """
    if len(col) >= 3:
        # 3 列中值滤波：走廊里常夹着标尺短横这类单列亮点（实测 0.29~0.57），
        # 会把「连续低列」计数打断，导致走廊起点晚判 5~8px；中值把它们拉平，
        # 而内容断崖处（1.00 → 0.08）中值几乎不动。
        col = np.median(np.stack([col, np.roll(col, 1), np.roll(col, -1)]), axis=0)
    n = len(col)
    run = 0
    coarse = None
    if side == 'right':
        for x in range(x_start, min(hi, n)):
            if col[x] < T_LOW:
                run += 1
                if run >= G_MAX:
                    coarse = x - run + 1        # 走廊起点：内容止于其左侧
                    break
            else:
                run = 0
    else:
        for x in range(x_start, max(lo, 0) - 1, -1):
            if col[x] < T_LOW:
                run += 1
                if run >= G_MAX:
                    coarse = x + run            # 走廊右界 + 1：内容的第一个列
                    break
            else:
                run = 0
    if coarse is None:
        return None
    # 像素级吸附：走廊只负责「是哪条边」，±SNAP 邻域内取跨边界梯度最大者定位。
    # 老设备渐隐带起点虽也是强梯度，但它离走廊起点 30px 以上，不会被吸附过去。
    best, best_g = coarse, None
    for x in range(max(lo, 0, coarse - SNAP), min(hi, n, coarse + SNAP) + 1):
        if x - W_SNAP < 0 or x + W_SNAP > n:
            continue
        g = float(col[x - W_SNAP:x].mean() - col[x:x + W_SNAP].mean()) if side == 'right' \
            else float(col[x:x + W_SNAP].mean() - col[x - W_SNAP:x].mean())
        if best_g is None or g > best_g:
            best, best_g = x, g
    return best


def _scan_row(a, box, side):
    """上下边：从粗框边缘向内扫，取第一条「非纯黑占比 > T_EDGE」的行。

    方向很重要：从粗框边缘向内扫，才能跨过窗口底部的暗淡区（实测老设备
    窗口下缘 130px 内行灰度只有 0.6~7.8，从内容深处向外游走会被暗区截断），
    同时窗前/窗后的文字行占用率只有 0.03~0.08，低于 T_EDGE 不会被选中。
    x 范围用粗框全域（含界面侧栏）：新设备实测顶部边界行恰是全宽占用率
    越过 0.2 的那一行（0.201），与人工口径一致。
    """
    x0, y0, x1, y1 = box
    rng = range(y0, y1) if side == 'top' else range(y1 - 1, y0 - 1, -1)
    for y in rng:
        row = a[y, max(0, x0):x1]
        if row.size and (row > T_NONBLACK).mean() > T_EDGE:
            return y if side == 'top' else y + 1
    return None


def detect(src, scale=4, refine=True):
    """返回 {'box':[x0,y0,x1,y1], 'confidence':0~1, 'flags':[...], ...}"""
    im = Image.open(src) if isinstance(src, (str, Path)) else src
    im = im.convert('L')
    W, H = im.size
    a_full = np.asarray(im, dtype=np.float32)
    m = close_mask(open_x(content_mask(a_full)))
    a_s = m          # 后续用块坐标，换算回原图时乘 BLOCK
    comps = components(m)
    flags = []
    if not comps:
        return {'box': None, 'confidence': 0.0, 'flags': ['no_content']}
    area, y0, x0, y1, x1 = comps[0]
    total = a_s.size
    ratio = area / total
    if ratio < MIN_AREA:
        flags.append('area_too_small')
    if ratio > MAX_AREA:
        flags.append('area_too_large')
    if len(comps) > 1 and comps[1][0] / max(area, 1) > SECOND_RATIO:
        flags.append('multiple_blocks')
    if x0 <= 0 or y0 <= 0 or x1 >= a_s.shape[1] or y1 >= a_s.shape[0]:
        flags.append('touches_border')
    box = [x0 * BLOCK, y0 * BLOCK, min(W, x1 * BLOCK), min(H, y1 * BLOCK)]
    if refine:
        # 左右：断崖+守卫（抗侧边栏窄条）；上下：行灰度游走（抗稀疏文字/刻度）
        col = (a_full[box[1]:box[3], :] > T_NONBLACK).mean(axis=0)
        x_start = int(np.argmax(col[box[0]:box[2]])) + box[0]   # 内容内部（最密列）
        l = _corridor_edge(col, box[0], box[2], 'left', x_start)
        r = _corridor_edge(col, box[0], box[2], 'right', x_start)
        l = box[0] if l is None else l
        r = box[2] if r is None else r
        if r - l < 40:                          # 结果退化则保留粗框
            l, r = box[0], box[2]
        t = _scan_row(a_full, box, 'top')
        bt = _scan_row(a_full, box, 'bottom')
        box = [l, box[1] if t is None else t, r, box[3] if bt is None else bt]
    # 事后检查：窗口之外是否还有「一整片实心内容」。只加标记，不改变边界，
    # 因此不会影响已验证的边界精度。
    def _solid_run(prof, lo, hi):
        run = best = 0
        for v in prof[lo:hi]:
            run = run + 1 if v > T_SOLID else 0
            best = max(best, run)
        return best

    x0, y0, x1, y1 = box
    col = (a_full[y0:y1, :] > T_NONBLACK).mean(axis=0)
    row = (a_full[:, x0:x1] > T_NONBLACK).mean(axis=1)
    if x1 < W and _solid_run(col, x1, W) >= SOLID_RUN:
        flags.append('content_beyond_right')
    if x0 > 0 and _solid_run(col, 0, x0) >= SOLID_RUN:
        flags.append('content_beyond_left')
    if y1 < H and _solid_run(row, y1, H) >= SOLID_RUN:
        flags.append('content_beyond_bottom')
    if y0 > 0 and _solid_run(row, 0, y0) >= SOLID_RUN:
        flags.append('content_beyond_top')
    conf = 1.0
    if 'no_content' in flags or 'area_too_small' in flags:
        conf = 0.2
    elif flags:
        conf = 0.6
    return {'box': [int(v) for v in box], 'confidence': round(conf, 2), 'flags': flags,
            'area_ratio': round(ratio, 3),
            'second_ratio': round(comps[1][0] / max(area, 1), 3) if len(comps) > 1 else 0.0}


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        d = detect(p)
        print(p, '→', d)
