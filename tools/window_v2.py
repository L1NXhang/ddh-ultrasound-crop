"""逐图测量成像窗口（新设备用）。判据全部来自实测标定：

  上下：整幅宽度上的"有内容"行，阈值 0.02（低阈值才能抓住深部暗区，实测唯一稳定的判据）
  左右：在上下行带内，找第一段/最后一段连续 50 列都超过相对阈值的区域
        （不用镜像对称——实测新设备并不对称）

用法：
  python tools/window_v2.py --manifest data/new_batch.jsonl --render 6      # 看效果
  python tools/window_v2.py --manifest data/review_new.jsonl --apply        # 写入窗口
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def detect_window(path, thr=12, row_occ=0.02, col_frac=0.5, col_run=50):
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
    H, W = a.shape
    m = a > thr
    ro = m.mean(1)
    # 上下：需要连续 20 行的占用都超过 row_occ
    ok_row = ro > row_occ
    run = 20
    top = bot = None
    c = 0
    for i in range(H):
        c = c + 1 if ok_row[i] else 0
        if c >= run and top is None:
            top = i - run + 1
        if ok_row[i]:
            if top is not None:
                bot = i + 1
    if top is None or bot is None or bot - top < 40:
        return None
    # 左右：在这条行带里找列
    band = m[top:bot]
    p = band.mean(0)
    base = float(p.max())
    ok = p >= max(0.10, col_frac * base)
    L = R = None
    c = 0
    for x in range(W):
        c = c + 1 if ok[x] else 0
        if c >= col_run:
            L = x - col_run + 1
            break
    c = 0
    for x in range(W - 1, -1, -1):
        c = c + 1 if ok[x] else 0
        if c >= col_run:
            R = x + col_run - 1
            break
    if L is None or R is None or R - L < 40:
        return None
    return [int(L), int(top), int(R), int(bot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/review_new.jsonl')
    ap.add_argument('--root', default='.')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--render', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    p = ROOT / args.manifest
    rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    ok, fail = 0, 0
    for r in rows:
        pth = next((ROOT / 'data/images').rglob(r['id'] + '.jpg'), None) or (ROOT / r['path'])
        b = detect_window(pth) if pth.exists() else None
        if b:
            r['window'] = b
            ok += 1
        else:
            fail += 1
    print(f'测量成功 {ok}/{len(rows)}，失败 {fail}')
    import statistics as st
    hs = [r['window'][3] - r['window'][1] for r in rows]
    print(f'窗口高度：中位 {st.median(hs):.0f} px（模型给的是 ~470-500）')
    if args.apply:
        out = ROOT / (args.out or args.manifest)
        out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
        print('已写回', out.relative_to(ROOT))
    if args.render:
        picks = rows[:args.render]
        cw, ch = 420, 400
        sheet = Image.new('RGB', (cw * 3, ch * ((len(picks) + 2) // 3)), (25, 25, 25))
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(picks):
            pth = next((ROOT / 'data/images').rglob(r['id'] + '.jpg'), None)
            im = Image.open(pth).convert('RGB')
            ow, oh = im.size
            x0, y0 = (i % 3) * cw, (i // 3) * ch
            sheet.paste(im.resize((cw, int(cw * oh / ow))), (x0, y0))
            kx = cw / ow
            w = r['window']
            d.rectangle([x0 + w[0] * kx, y0 + w[1] * kx, x0 + w[2] * kx, y0 + w[3] * kx],
                        outline=(60, 255, 120), width=2)
            d.text((x0 + 4, y0 + 2), f"{r['image_wh'][0]}x{r['image_wh'][1]} y{w[1]}-{w[3]}", fill=(255, 255, 0))
        sp = ROOT / 'data/review/window_v2_check.png'
        sheet.save(sp)
        print('核对图 →', sp.relative_to(ROOT))


if __name__ == '__main__':
    main()
