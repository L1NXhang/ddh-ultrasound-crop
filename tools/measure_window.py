"""逐图测量成像窗口（左/右用边缘检测+对称镜像，上/下用内容边界）。

为什么不用模型检测：窗口模型只在旧设备上训练过，到新设备会按旧比例给框，
导致底部切掉成像区；而裁剪线是按"窗口高度的百分比"预测的，窗口一错，线就全飘。

实测依据（用人工修正的 23 张当标准答案）：
  左边界边缘检测误差中位 -1px、±6px 命中 100%；右边界用对称镜像得到同样精度。

用法：python tools/measure_window.py --manifest data/new_batch.jsonl --sample 30
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def detect(path, thr=12, occ=0.5, run=40, row_occ=0.12):
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
    H, W = a.shape
    m = a > thr
    # 上下：整幅宽度上有内容的行（阈值低一些，避免暗区截断）
    ro = m.mean(1)
    rows = np.where(ro > row_occ)[0]
    if len(rows) < 20:
        return None
    # 取最长的连续行段
    best, cur, start = (0, 0, 0), 0, None
    for i in range(H + 1):
        v = ro[i] if i < H else 0
        if v > row_occ:
            if start is None: start = i
            cur = i - start + 1
            if cur > best[0]: best = (cur, start, i + 1)
        else:
            start = None
    top, bot = best[1], best[2]
    # 左右：在该行带内找第一个连续 run 达阈值的列
    p = m[top:bot].mean(0)
    base = float(p.mean())
    ok = p >= max(0.25, occ * base)
    left = None
    c = 0
    for x in range(max(0, int(W * 0.05)), W):
        c = c + 1 if ok[x] else 0
        if c >= run:
            left = x - run + 1
            break
    if left is None:
        return None
    right = int(round(2 * (W / 2) - left))      # 对称镜像
    return [int(left), int(top), int(right), int(bot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/new_batch.jsonl')
    ap.add_argument('--root', default='.')
    ap.add_argument('--sample', type=int, default=30, help='每个分辨率抽样多少张')
    ap.add_argument('--out', default='data/reports/new_window_measured.json')
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r['image_wh'])].append(r)
    out = {}
    for wh, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        step = max(1, len(items) // args.sample)
        got = []
        for r in items[::step][:args.sample]:
            pth = next((ROOT / 'data/images').rglob(r['id'] + '.jpg'), None) or (ROOT / r['path'])
            if not pth.exists():
                continue
            b = detect(pth)
            if b:
                got.append(b)
        if not got:
            continue
        left = int(st.median([g[0] for g in got]))
        right = int(st.median([g[2] for g in got]))
        top = int(st.median([g[1] for g in got]))
        bot = int(max(g[3] for g in got))          # 取最大：保证把所有内容都包住
        out[f'{wh[0]}x{wh[1]}'] = {
            'window': [left, top, right, bot],
            'n_sample': len(got),
            'top_range': [int(min(g[1] for g in got)), int(max(g[1] for g in got))],
            'bottom_range': [int(min(g[3] for g in got)), int(max(g[3] for g in got))],
        }
        print(f'{wh[0]}x{wh[1]}: n={len(got):3d}  窗口={[left,top,right,bot]}  '
              f'上边界范围 {out[f"{wh[0]}x{wh[1]}"]["top_range"]}  下边界范围 {out[f"{wh[0]}x{wh[1]}"]["bottom_range"]}')
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
