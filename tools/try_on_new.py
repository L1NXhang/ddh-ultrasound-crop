"""在新数据上试跑模型：看裁剪建议长什么样（无标签，只能视检）。

用法：python tools/try_on_new.py --n 50
输出：data/review/try_new_XX.png（每张 25 图，绿=预测上边界 蓝=预测下边界 红=检测窗口）
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ddh                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/new_batch.jsonl')
    ap.add_argument('--root', default='.')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--cols', type=int, default=5)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    # 按分辨率分层抽样，保证三种设备都覆盖
    by = {}
    for r in rows:
        by.setdefault(tuple(r['image_wh']), []).append(r)
    picks = []
    per = max(1, args.n // max(1, len(by)))
    for wh, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        picks += v[::max(1, len(v)//per)][:per]
    picks = picks[:args.n]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wm, wa = ddh.load_checkpoint(ROOT / 'runs/window.pt', device)
    bm, ba = ddh.load_checkpoint(ROOT / 'runs_v2/crop.pt', device)
    print(f'用 窗口模型 runs/window.pt + 新口径裁剪模型 runs_v2/crop.pt 跑 {len(picks)} 张新数据')

    res = []
    for r in picks:
        p = ddh.pipeline(r, str(ROOT / args.root), wm, bm, ba, device, samples=4)
        res.append((r, p))
    ok = [x for x in res if x[1].get('box')]
    print(f'  检出窗口：{len(ok)}/{len(res)}；裁剪预测：{sum(1 for _,p in res if p.get("bounds"))} 张')

    cell_w, cell_h = 300, 320
    per_sheet = args.cols * 3
    for s in range(0, len(res), per_sheet):
        chunk = res[s:s + per_sheet]
        rn = (len(chunk) + args.cols - 1) // args.cols
        sheet = Image.new('RGB', (cell_w * args.cols, cell_h * rn), (25,25,25))
        d = ImageDraw.Draw(sheet)
        for i, (r, p) in enumerate(chunk):
            pth = next((ROOT / 'data/images').rglob(r['id'] + '.jpg'), None)
            if not pth:
                continue
            im = Image.open(pth).convert('RGB')
            x0, y0, x1, y1 = p.get('box') or r['image_wh'] and [0, 0, r['image_wh'][0], r['image_wh'][1]]
            crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
            ow, oh = crop.size
            sc = min((cell_w - 6) / ow, (cell_h - 6) / oh)
            th = crop.resize((int(ow * sc), int(oh * sc)))
            px, py = (i % args.cols) * cell_w + 3, (i // args.cols) * cell_h + 3
            sheet.paste(th, (px, py))
            if p.get('bounds'):
                t, b = p['bounds']
                gy, gb = py + th.height * t, py + th.height * b
                d.rectangle([px - 2, py - 2, px + th.width + 2, py + th.height + 2], outline=(80, 80, 80), width=1)
                d.line([px, gy, px + th.width, gy], fill=(0, 255, 120), width=2)
                d.line([px, gb, px + th.width, gb], fill=(80, 200, 255), width=2)
                d.text((px + 2, py), f"{r['id'][:12]} {r['image_wh'][0]}x{r['image_wh'][1]}", fill=(255, 255, 0))
            else:
                d.text((px + 2, py), f"{r['id'][:12]} 无窗口", fill=(255, 80, 80))
        out = ROOT / 'data' / 'review' / f'try_new_{s//per_sheet:02d}.png'
        sheet.save(out)
        print('  →', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
