"""把人工裁切图反向定位回原图，自动得到 crop 的 top/bottom（原图像素）。

依据方案文档 §5.1：只有人工裁剪图但没有边界时，可用匹配算法辅助寻找原图位置，
但必须人工核对；匹配不唯一时要重新处理，不能默默采用。

方法：灰度化后做 FFT 互相关求 SSD 曲面，先在同尺度匹配，残差大再扫尺度
（应对裁切时被缩放过的情况）。输出置信度与"次优峰"边距，模糊的一律标记出来。

输出仍是建议：data/crop_from_manual.jsonl；确认无误后再合并进 annotations。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def integral_sumsq(a, h, w):
    """窗口平方和，返回 (H-h+1, W-w+1)。"""
    s = np.cumsum(np.cumsum(a ** 2, 0), 1)
    s = np.pad(s, ((1, 0), (1, 0)))
    return s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]


def ssd_map(O, T):
    """O: (H,W) 原图；T: (h,w) 模板。返回 SSD 曲面 (H-h+1, W-w+1)。"""
    H, W = O.shape
    h, w = T.shape
    if h > H or w > W:
        return None
    shape = (H, W)
    corr = np.fft.irfft2(np.fft.rfft2(O, shape) * np.conj(np.fft.rfft2(T, shape)), shape)
    corr = corr[:H - h + 1, :W - w + 1]
    return integral_sumsq(O.astype(np.float64), h, w) + float((T ** 2).sum()) - 2 * corr


def best_offset(O, T, radius=16):
    """返回 (ssd_best, (y,x), prominence, (y2,x2))。

    prominence = (次优 ssd − 最优 ssd) / 模板能量，衡量峰有多突出：
    完全相同的重复内容会让次优峰和最优峰一样低，此时 prominence≈0，判为不唯一。
    """
    m = ssd_map(O, T)
    if m is None:
        return None
    y0, x0 = np.unravel_index(np.argmin(m), m.shape)
    ssd = float(m[y0, x0])
    mm = m.copy()
    mm[max(0, y0 - radius):y0 + radius + 1, max(0, x0 - radius):x0 + radius + 1] = np.inf
    y1, x1 = np.unravel_index(np.argmin(mm), mm.shape)
    energy = float((T ** 2).sum()) + 1e-6
    return ssd, (int(y0), int(x0)), float((float(mm[y1, x1]) - ssd) / energy), (int(y1), int(x1))


def match_one(orig, crop, scales):
    """返回 dict：scale、box、rmse、prominence。

    1.0 尺度优先：JPEG 重编码本身就会抬高残差，所以不能靠残差判断有没有被缩放，
    而是看多尺度搜索里 1.0 是不是和最优差不多好。
    """
    O = np.asarray(orig.convert('L'), dtype=np.float32)
    C = np.asarray(crop.convert('L'), dtype=np.float32)
    Ho, Wo = O.shape
    Hc, Wc = C.shape
    results = []
    for s in scales:                      # s = 裁切像素 / 原图像素
        tw, th = int(round(Wc / s)), int(round(Hc / s))
        if th < 8 or tw < 8 or th > Ho or tw > Wo:
            continue
        T = np.asarray(Image.fromarray(C.astype(np.uint8)).resize((tw, th), Image.Resampling.BILINEAR),
                       dtype=np.float32)
        got = best_offset(O, T)
        if got is None:
            continue
        ssd, (y, x), prom, _ = got
        rmse = float(np.sqrt(max(ssd, 0) / T.size))
        results.append({'scale': round(s, 3), 'box': [x, y, x + tw, y + th],
                        'rmse': round(rmse, 3), 'prominence': round(prom, 3), 'crop_wh': [Wc, Hc]})
        if s == 1.0 and rmse < 2.0:
            break
    if not results:
        return None
    best = min(results, key=lambda r: r['rmse'])
    unit = next((r for r in results if r['scale'] == 1.0), None)
    if unit and unit['rmse'] <= 1.25 * best['rmse']:
        best = unit
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/annotations.todo.jsonl')
    ap.add_argument('--root', default='data')
    ap.add_argument('--cropped', required=True, help='人工裁切图所在目录')
    ap.add_argument('--out', default='data/crop_from_manual.jsonl')
    ap.add_argument('--prom-min', type=float, default=0.05, help='峰值突出度下限；低于此值判为匹配不唯一')
    args = ap.parse_args()
    root, C = Path(args.root), Path(args.cropped)
    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    byid = {r['id']: r for r in rows}

    scales = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.7, 1.3, 0.6, 1.4, 0.5]
    out, n_ok, n_bad, n_missing = [], 0, 0, 0
    for f in sorted(C.glob('*.jpg')) + sorted(C.glob('*.png')):
        rid = f.stem.split('_')[0]
        r = byid.get(rid)
        if r is None:
            n_missing += 1
            out.append({'id': f.stem, 'status': '没有对应原图'})
            continue
        m = match_one(Image.open(root / r['path']), Image.open(f), scales)
        if m is None:
            n_bad += 1
            out.append({'id': rid, 'status': '匹配失败'})
            continue
        if m['prominence'] < args.prom_min:
            status = '匹配不唯一，需人工核对'
        elif m['scale'] != 1.0:
            status = '裁切图被缩放过，位置可靠但请核对'
        else:
            status = 'ok'
        ok = status == 'ok'
        n_ok += ok
        n_bad += (not ok)
        y0, x0, x1, y1 = m['box']
        out.append({'id': rid, 'status': status, 'matched_box': m['box'],
                    'crop': [y0, y1], 'scale': m['scale'], 'rmse': m['rmse'],
                    'prominence': m['prominence'], 'crop_wh': m['crop_wh'],
                    'window_wh': [r['window'][2] - r['window'][0], r['window'][3] - r['window'][1]]
                    if r.get('window') else None})
    Path(args.out).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out), encoding='utf-8')
    scales_seen, status_seen = {}, {}
    for x in out:
        if 'scale' in x:
            scales_seen[x['scale']] = scales_seen.get(x['scale'], 0) + 1
        status_seen[x['status']] = status_seen.get(x['status'], 0) + 1
    print(json.dumps({'裁切图': len(out), '可直接采用': n_ok, '需人工核对': n_bad,
                      '无对应原图': n_missing, '尺度分布': scales_seen, '状态分布': status_seen,
                      '输出': args.out}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
