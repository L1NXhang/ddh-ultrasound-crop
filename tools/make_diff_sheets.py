"""Build contact sheets comparing human crop boxes against the model prediction.

Purpose: let the owner eyeball *where* the two annotators differ from the model.

Colour legend (per tile):
  CYAN   dashed  model window   (what the rule-based window detector proposed)
  CYAN   solid   human window   (member's corrected window, when changed)
  ORANGE dashed  model crop line pair (top/bottom the model predicted)
  GREEN  solid   human crop box (final saved crop)
  MAGENTA        the delta band between model line and human line (highlight)
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
HANDOUT = ROOT / 'data/handout/新设备标注_20261016'
M2_RES = ROOT.parent / 'DDH裁剪结果/member2_placeholder'  # replaced via --member2

FONT_BOLD = r'C:\Windows\Fonts\msyhbd.ttc'
FONT_REG = r'C:\Windows\Fonts\msyh.ttc'

CYAN = (0, 229, 255)
GREEN = (0, 255, 102)
ORANGE = (255, 153, 0)
MAGENTA = (255, 0, 200)
WHITE = (255, 255, 255)
RED = (255, 60, 60)


def font(size, bold=True):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
    except OSError:
        return ImageFont.load_default()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(member, count):
    p = HANDOUT / f'成员{member}_{count}张/任务清单.jsonl'
    return [json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]


def load_member3(zip_path):
    zf = zipfile.ZipFile(zip_path)
    rows = [json.loads(l) for l in zf.read('results.jsonl').decode('utf-8').splitlines() if l.strip()]
    return rows, zf


def draw_dashed_rect(d, box, color, width=2, dash=10):
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash * 2):
        d.line([x, y0, min(x + dash, x1), y0], fill=color, width=width)
        d.line([x, y1, min(x + dash, x1), y1], fill=color, width=width)
    for y in range(y0, y1, dash * 2):
        d.line([x0, y, x0, min(y + dash, y1)], fill=color, width=width)
        d.line([x1, y, x1, min(y + dash, y1)], fill=color, width=width)


def render_tile(image, rec, scale):
    """Return an RGB tile with overlays and a caption strip."""
    W, H = image.size
    tw, th = int(W * scale), int(H * scale)
    head = 46
    tile = Image.new('RGB', (tw, th + head), (18, 24, 34))
    tile.paste(image.convert('RGB').resize((tw, th), Image.Resampling.LANCZOS), (0, head))
    d = ImageDraw.Draw(tile)
    f = font(15)

    def S(b):
        return [int(round(b[0] * scale)), int(round(head + b[1] * scale)),
                int(round(b[2] * scale)), int(round(head + b[3] * scale))]

    model_win = rec.get('model_window') or rec.get('window')
    human_win = rec.get('window')
    pred = (rec.get('prediction') or {}).get('crop_box')
    human_crop = rec.get('clean_box')

    # highlight delta band between model and human crop lines
    if pred and human_crop:
        overlay = Image.new('RGBA', tile.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for y_model, y_human in ((pred[1], human_crop[1]), (pred[3], human_crop[3])):
            if y_model != y_human:
                y0, y1 = sorted([y_model, y_human])
                od.rectangle(S([human_crop[0], y0, human_crop[2], y1]), fill=MAGENTA + (70,))
        tile = Image.alpha_composite(tile.convert('RGBA'), overlay).convert('RGB')
        d = ImageDraw.Draw(tile)

    if model_win:
        draw_dashed_rect(d, S(model_win), CYAN, 2, 8)
    if human_win and model_win and list(human_win) != list(model_win):
        d.rectangle(S(human_win), outline=CYAN, width=3)
    if pred:
        draw_dashed_rect(d, S([pred[0], pred[1], pred[2], pred[3]]), ORANGE, 3, 12)
        # model line pair as full-width dashed rules
        for y in (pred[1], pred[3]):
            yy = head + y * scale
            for x in range(0, tw, 18):
                d.line([x, yy, min(x + 10, tw), yy], fill=ORANGE, width=2)
    if human_crop:
        d.rectangle(S(human_crop), outline=GREEN, width=4)

    title = rec['_label']
    d.text((8, 4), title, font=f, fill=WHITE)
    return tile


def build_sheet(items, cols, scale, out_path, title, subtitle):
    tiles = [render_tile(img, rec, scale) for img, rec in items]
    if not tiles:
        raise SystemExit('no tiles')
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    pad = 10
    header = 96
    sheet = Image.new('RGB', (cols * tw + pad * (cols + 1),
                              header + rows * th + pad * (rows + 1)), (10, 14, 20))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), title, font=font(26), fill=WHITE)
    d.text((pad, 46), subtitle, font=font(16, False), fill=(150, 170, 190))
    # legend
    lx = pad
    ly = 70
    fl = font(14, False)
    for text, color in (('青色虚线=模型窗口', CYAN), ('青色实线=成员改后的窗口', CYAN),
                        ('橙虚线=模型裁剪线', ORANGE), ('绿框=成员最终裁剪', GREEN),
                        ('品红=两者差的那一条', MAGENTA)):
        d.rectangle([lx, ly + 4, lx + 14, ly + 14], outline=color, width=2)
        d.text((lx + 20, ly), text, font=fl, fill=WHITE)
        lx += 20 + int(d.textlength(text, font=fl)) + 22

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = header + pad + r * (th + pad)
        sheet.paste(tile, (x, y))
    sheet.save(out_path, quality=92)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member3-zip', required=True)
    ap.add_argument('--member2-results', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'data/reports/cross_annotator_20261021'))
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--scale', type=float, default=0.30)
    ap.add_argument('--per-sheet', type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m2 = manifest(2, 237)
    m2_by_sha = {file_sha(ROOT / m['original_path']): m for m in m2}
    res2 = [json.loads(l) for l in Path(args.member2_results).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    for r in res2:
        m = m2_by_sha[r['source_sha256']]
        r['_file'] = m['filename']
        r['_path'] = ROOT / m['original_path']
        r['_member'] = '成员2(你标)'

    res3, zf3 = load_member3(args.member3_zip)
    m3 = manifest(3, 237)
    m3_by_sha = {file_sha(ROOT / m['original_path']): m for m in m3}
    for r in res3:
        m = m3_by_sha[r['source_sha256']]
        r['_file'] = m['filename']
        r['_path'] = ROOT / m['original_path']
        r['_member'] = '成员3'

    def bottom_delta(r):
        pred = (r.get('prediction') or {}).get('crop_box')
        return r['clean_box'][3] - pred[3] if pred and r.get('clean_box') else 0

    def window_delta(r):
        return max(abs(a - b) for a, b in zip(r['window'], r.get('model_window') or r['window']))

    sheets = []

    # Sheet 1+2: member3 window corrections (largest first)
    wm3 = sorted([r for r in res3 if r.get('window_modified')], key=window_delta, reverse=True)
    for i, group in enumerate([wm3[:args.per_sheet], wm3[args.per_sheet:args.per_sheet * 2]], 1):
        if not group:
            continue
        items = []
        for r in group:
            r = dict(r, _label=f'{r["_file"]}  window改动 {window_delta(r)}px')
            items.append((Image.open(r['_path']), r))
        p = build_sheet(items, args.cols, args.scale, out_dir / f'成员3_改窗口_{i}.jpg',
                        f'成员3 改了成像区窗口（第{i}组，共{len(wm3)}张）',
                        '青色虚线是模型给的窗口，实线是成员3改后的。重点看模型是不是漏掉了下缘组织。')
        sheets.append((p, len(items), '成员3 改窗口'))

    # Sheet 3+4: member3 largest bottom tightening
    bt3 = sorted(res3, key=bottom_delta, reverse=True)[:args.per_sheet * 2]
    for i, group in enumerate([bt3[:args.per_sheet], bt3[args.per_sheet:]], 1):
        if not group:
            continue
        items = []
        for r in group:
            r = dict(r, _label=f'{r["_file"]}  下边界收紧 {bottom_delta(r)}px')
            items.append((Image.open(r['_path']), r))
        p = build_sheet(items, args.cols, args.scale, out_dir / f'成员3_下边界收紧_{i}.jpg',
                        f'成员3 把下边界往上收（第{i}组）',
                        '橙虚线是模型给的裁剪线，绿框是成员最终保留范围。看模型是不是裁进了组织里。')
        sheets.append((p, len(items), '成员3 下边界收紧'))

    # Sheet 5: member3 largest bottom widening
    bw3 = sorted(res3, key=bottom_delta)[:args.per_sheet]
    items = []
    for r in bw3:
        r = dict(r, _label=f'{r["_file"]}  下边界放宽 {bottom_delta(r)}px')
        items.append((Image.open(r['_path']), r))
    p = build_sheet(items, args.cols, args.scale, out_dir / '成员3_下边界放宽.jpg',
                    f'成员3 把下边界往下放（{len(bw3)}张）',
                    '这些是模型裁太紧、成员3 补回内容的情况。')
    sheets.append((p, len(items), '成员3 下边界放宽'))

    # Sheet 6: member2 window corrections (only 5)
    wm2 = sorted([r for r in res2 if r.get('window_modified')], key=window_delta, reverse=True)
    items = []
    for r in wm2:
        r = dict(r, _label=f'{r["_file"]}  window改动 {window_delta(r)}px')
        items.append((Image.open(r['_path']), r))
    if items:
        p = build_sheet(items, args.cols, args.scale, out_dir / '成员2_改窗口_全部.jpg',
                        f'成员2（你亲手标）改了窗口的全部 {len(items)} 张',
                        '对比成员3 的 34 张——同一批新设备图，你几乎不改窗口。')
        sheets.append((p, len(items), '成员2 改窗口'))

    # Sheet 7+8: member2 bottom tightening (largest)
    bt2 = sorted(res2, key=bottom_delta, reverse=True)[:args.per_sheet]
    items = []
    for r in bt2:
        r = dict(r, _label=f'{r["_file"]}  下边界收紧 {bottom_delta(r)}px')
        items.append((Image.open(r['_path']), r))
    p = build_sheet(items, args.cols, args.scale, out_dir / '成员2_下边界收紧.jpg',
                    '成员2（你亲手标）下边界收紧最多的 30 张',
                    '和成员3 的同名图集不是同一批，但同一设备同一批来源，可比尺度。')
    sheets.append((p, len(items), '成员2 下边界收紧'))

    # save the exact selection
    sel = {
        'member3_window_modified': [r['_file'] for r in wm3],
        'member3_bottom_tighten_top60': [r['_file'] for r in bt3],
        'member3_bottom_widen_top30': [r['_file'] for r in bw3],
        'member2_window_modified': [r['_file'] for r in wm2],
        'member2_bottom_tighten_top30': [r['_file'] for r in bt2],
        'sheets': [{'file': p.name, 'tiles': n, 'kind': k} for p, n, k in sheets],
    }
    (out_dir / 'selection.json').write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding='utf-8')
    for p, n, k in sheets:
        print(f'{p.relative_to(ROOT)}  ({n} 张)  {k}')


if __name__ == '__main__':
    main()