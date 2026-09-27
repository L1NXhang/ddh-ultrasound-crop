"""逐图自动定位窗口的左右边界（不再依赖模板）。

原理（由数据实证）：
  1. 窗口水平中心恒等于图像中心（实测 639±1.1 px）→ 只需检一条边，另一条镜像；
  2. 左右边界是"暗→亮"的持续跃变 → 用相对阈值 + 长 run 判定，比绝对阈值稳；
  3. 上下边界与裁剪训练无关（已验证）→ 沿用原值，只改左右。

输出：
  data/reports/auto_window.jsonl   每图的建议窗口与置信度
  data/review/auto_window_check.png 抽样核对图（红=原窗口，绿=自动窗口）
用法：python tools/auto_window.py [--apply]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def detect_half_width(a, win, thr=12, frac=0.5, run=60, margin=8):
    """返回 (建议左边界, 置信度)。a 为灰度图。"""
    y0, y1 = int(win[1]), int(win[3])
    if y1 - y0 < 20:
        return None, 0
    p = (a[y0:y1] > thr).mean(0)                 # 每列在窗口高度内的"有内容"占比
    inner = p[int(win[0]):int(win[2])]
    base = float(inner.mean())                   # 窗口内的平均占比，作为相对基准
    lo = max(0, int(win[0]) + margin)
    hi = int(win[2])
    ok = p >= max(0.25, frac * base)
    c = 0
    for x in range(lo, hi):
        c = c + 1 if ok[x] else 0
        if c >= run:
            left = x - run + 1
            # 置信度：左侧是否真的暗（跃变明显）
            dark = 1.0 - float(p[max(0, left - 12):left].mean()) if left > 12 else 1.0
            return left, round(min(1.0, dark * 1.2), 3)
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/train_ready.jsonl')
    ap.add_argument('--root', default='data')
    ap.add_argument('--out', default='data/reports/auto_window.jsonl')
    ap.add_argument('--sheet', default='data/review/auto_window_check.png')
    ap.add_argument('--apply', action='store_true',
                    help='把修正后的窗口写进 data/train_ready_v2.jsonl（原文件不动）')
    ap.add_argument('--tolerance', type=int, default=15,
                    help='与现有窗口差多少像素才替换。实测：差 <=10px 属正常抖动（标注者也不会改），'
                         '>15px 才是真错位；检测器在真错位的图上精度 ±1px')
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]

    out, changed, failed = [], [], 0
    for r in rows:
        w = r.get('window')
        if not w:
            failed += 1
            continue
        a = np.asarray(Image.open(ROOT / args.root / r['path']).convert('L'), dtype=np.float32)
        left, conf = detect_half_width(a, w)
        if left is None:
            failed += 1
            continue
        cx = a.shape[1] / 2
        new = [int(round(left)), w[1], int(round(2 * cx - left)), w[3]]
        diff = max(abs(new[0] - w[0]), abs(new[2] - w[2]))
        rec = {'id': r['id'], 'path': r['path'], 'old': list(w), 'new': new,
               'diff_px': diff, 'confidence': conf, 'width': new[2] - new[0],
               'change': diff > args.tolerance}
        out.append(rec)
        if rec['change']:
            changed.append(rec)

    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out), encoding='utf-8')

    # 抽样核对图：一半是"要改的"，一半是"不用改的"
    import random
    random.seed(1)
    picks = random.sample(changed, min(8, len(changed))) + \
            random.sample([x for x in out if not x['change']], min(4, sum(1 for x in out if not x['change'])))
    if picks:
        cw, ch, cols = 420, 315, 4
        rn = (len(picks) + cols - 1) // cols
        sheet = Image.new('RGB', (cw * cols, ch * rn), 'black')
        d = ImageDraw.Draw(sheet)
        for i, x in enumerate(picks):
            im = Image.open(ROOT / args.root / x['path']).convert('RGB')
            ow, oh = im.size
            im2 = im.resize((cw, int(cw * oh / ow)))
            x0, y0 = (i % cols) * cw, (i // cols) * ch
            sheet.paste(im2, (x0, y0))
            kx = cw / ow
            for box, col in ((x['old'], (255, 60, 60)), (x['new'], (60, 255, 120))):
                d.rectangle([x0 + box[0] * kx, y0 + box[1] * kx, x0 + box[2] * kx, y0 + box[3] * kx],
                            outline=col, width=2)
            d.text((x0 + 4, y0 + 2), f"{x['id']} 差{x['diff_px']}px conf={x['confidence']}", fill=(255, 255, 0))
        sheet.save(ROOT / args.sheet)

    wdist = Counter(round(x['width'] / 8) * 8 for x in out)
    print(f'共处理 {len(out)} 张，检测失败 {failed} 张')
    print(f'  与模板一致（不用改）: {len(out) - len(changed)}')
    print(f'  需要修正             : {len(changed)}（{len(changed)/max(1,len(out))*100:.0f}%）')
    print(f'  修正后的宽度分布（前 6）: {wdist.most_common(6)}')
    print(f'  低置信（conf<0.6）的  : {sum(1 for x in out if x["confidence"] < 0.6)}')
    if args.apply:
        byid = {x['id']: x for x in out}
        rows2 = []
        for r in rows:
            r2 = dict(r)
            rec = byid.get(r2['id'])
            if rec and rec['change']:
                r2['window'] = rec['new']
            rows2.append(r2)
        target = ROOT / 'data' / 'train_ready_v2.jsonl'
        target.write_text(''.join(json.dumps(x, ensure_ascii=False) + chr(10) for x in rows2), encoding='utf-8')
        print(f'已写出修正清单 → {target.relative_to(ROOT)}（修正 {len(changed)} 张，原清单未动）')
    print(f'→ {args.out}\n→ {args.sheet}')


if __name__ == '__main__':
    main()
