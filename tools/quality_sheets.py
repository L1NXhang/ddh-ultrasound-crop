"""生成"切面可用性"预筛拼图：把每张图裁到成像窗口区域再拼接，便于视觉判断。

用法：python tools/quality_sheets.py --from A --per-sheet 24 --cols 6
输出：data/review/qsheet_<来源>_<序号>.png
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', default='data/human_A.jsonl')
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--cell', type=int, default=250)
    ap.add_argument('--per-sheet', type=int, default=24)
    ap.add_argument('--prefix', default='qsheet')
    args = ap.parse_args()
    recs = []
    for f in [x.strip() for x in args.files.split(',') if x.strip()]:
        p = Path(f)
        if p.exists():
            recs += [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    recs = [r for r in recs if r.get('window')]
    recs.sort(key=lambda r: r['id'])
    cw = ch = args.cell
    rows = (args.per_sheet + args.cols - 1) // args.cols
    made = []
    for s in range(0, len(recs), args.per_sheet):
        chunk = recs[s:s + args.per_sheet]
        sheet = Image.new('RGB', (cw * args.cols, ch * rows), 'black')
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            pth = next(Path('data/images').rglob(r['id'] + '.jpg'), None)
            if not pth:
                continue
            im = Image.open(pth).convert('RGB')
            x0, y0, x1, y1 = [int(v) for v in r['window']]
            crop = im.crop((x0, y0, x1, y1))
            crop.thumbnail((cw - 6, ch - 6), Image.Resampling.LANCZOS)
            px, py = (i % args.cols) * cw + 3, (i // args.cols) * ch + 3
            sheet.paste(crop, (px, py))
            tag = f"{r['id']}"
            if r.get('quality') == 0:
                tag += ' [标0]'
            d.text((px + 2, py), tag, fill=(255, 220, 0))
        out = ROOT / 'data' / 'review' / f'{args.prefix}_{s//args.per_sheet:02d}.png'
        sheet.save(out)
        made.append(out)
    print(f'{len(recs)} 张 → {len(made)} 张拼图（每张 {args.per_sheet} 图，格 {cw}px）')
    for m in made[:30]:
        print('  ', m.relative_to(ROOT))


if __name__ == '__main__':
    main()
