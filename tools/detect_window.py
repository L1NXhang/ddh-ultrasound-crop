"""通用成像窗口检测：逐图检测，不依赖任何模板。

设计依据（全部来自本项目实测，不是凭直觉）
────────────────────────────────────────
1) 判据：黑边与成像区的本质差别是"有没有斑点纹理"，不只是亮暗。
     黑边：灰度 0~8，局部标准差 ≈ 0~1（纯色）
     成像区深部：亮度可能同样低，但仍有斑点噪声，局部标准差 > 3
   所以内容掩膜 =（局部标准差 > t_std）∪（灰度 > t_gray）。
   只用亮度阈值会在暗区断开——那正是必须保留的内容。

2) 连通域：界面上的参数文字、深度标尺、人体示意图都是独立小块，
   取"最大连通域"的包围盒可自动把它们排除；这是比阈值更本质的判据。

3) 闭运算：成像区中间的无回声暗带会把连通域切开，先做闭运算补上。

4) 细化：粗尺度（1/4）只能到 ±4px，回到原分辨率沿四条边用同一判据收紧。

5) 拒识：面积占比异常、贴图像边界、存在第二个大连通域（拼图/双屏）→ 标记可疑。

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
T_BLOCK = 0.05     # 块级：4×4 块里非纯黑占比超过它 → 该块算内容（黑边精确为 0，仍有安全余量）
T_EDGE = 0.20      # 边界细化：一条边(行/列)的非纯黑占比超过它 → 该边有内容
BLOCK = 4          # 块大小
MIN_AREA = 0.05    # 内容面积占整图比例下限
MAX_AREA = 0.95    # 上限
SECOND_RATIO = 0.25  # 第二大连通域 / 最大连通域 超过此值 → 疑似多窗口


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


def _refine_edge(a, box, side):
    """在原分辨率上把某条边收紧到像素级。

    判据 = 「这一列/行里有多大比例的像素不是纯黑」（阈值取得极低 T_EDGE=4）：
      成像区（含深部暗区）：几乎整条都不是纯黑   → 通过
      黑边                ：整条都是纯黑          → 不通过
      深度刻度/文字（稀疏）：只有 1~5% 不是纯黑    → 不通过  ← 之前误判的根源
    实测：用「标准差 > 8」会把刻度点当成内容（占用率仅 1% 却有几个亮点）；
          改用「非纯黑比例」后，既能容下暗内容，又能排除稀疏元素。
    """
    x0, y0, x1, y1 = box
    if side in ('top', 'bottom'):
        rng = range(y0, y1) if side == 'top' else range(y1 - 1, y0 - 1, -1)
        for y in rng:
            row = a[y, max(0, x0):x1]
            if row.size and (row > T_NONBLACK).mean() > T_EDGE:
                return y
    else:
        rng = range(x0, x1) if side == 'left' else range(x1 - 1, x0 - 1, -1)
        for x in rng:
            col = a[max(0, y0):y1, x]
            if col.size and (col > T_NONBLACK).mean() > T_EDGE:
                return x
    return None


def detect(src, scale=4, refine=True):
    """返回 {'box':[x0,y0,x1,y1], 'confidence':0~1, 'flags':[...], ...}"""
    im = Image.open(src) if isinstance(src, (str, Path)) else src
    im = im.convert('L')
    W, H = im.size
    a_full = np.asarray(im, dtype=np.float32)
    m = close_mask(content_mask(a_full))
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
        a = np.asarray(im, dtype=np.float32)
        t = _refine_edge(a, box, 'top')
        b = _refine_edge(a, box, 'bottom')
        l = _refine_edge(a, box, 'left')
        r = _refine_edge(a, box, 'right')
        box = [l if l is not None else box[0], t if t is not None else box[1],
               (r + 1) if r is not None else box[2], (b + 1) if b is not None else box[3]]
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
