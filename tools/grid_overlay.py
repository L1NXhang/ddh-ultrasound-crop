"""给图片叠加带刻度的坐标网格，供人工/视觉核对窗口边界。

用法：python tools/grid_overlay.py --image <图> --out <png> [--scale 1.0] [--box x0,y0,x1,y1]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--step', type=int, default=50)
    ap.add_argument('--box', default='')
    args = ap.parse_args()
    im = Image.open(ROOT / args.image).convert('RGB')
    w, h = im.size
    if args.scale != 1.0:
        im = im.resize((int(w * args.scale), int(h * args.scale)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    s = args.scale
    for x in range(0, w, args.step):
        d.line([x * s, 0, x * s, h * s], fill=(255, 60, 60), width=1)
        if x % (args.step * 2) == 0:
            d.text((x * s + 2, 2), str(x), fill=(255, 220, 0))
    for y in range(0, h, args.step):
        d.line([0, y * s, w * s, y * s], fill=(60, 160, 255), width=1)
        if y % (args.step * 2) == 0:
            d.text((2, y * s + 2), str(y), fill=(255, 220, 0))
    if args.box:
        b = [int(v) for v in args.box.split(',')]
        d.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=(0, 255, 0), width=3)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f'{args.image} {w}x{h} → {args.out}（网格 {args.step}px，红线=x，蓝线=y）')


if __name__ == '__main__':
    main()
