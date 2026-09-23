"""把检测器给出的窗口框，用客观判据验收后写成"待人工抽检的标签建议"。

判据（与老批次一致）：
  1) 分数≥阈值且恰好一个框 —— 多框/无框一律交人工；
  2) 框外一圈条带必须干净（亮像素占比 ≤ 0.10）—— 防止框切到内容；
  3) 框内必须有内容（填充率 ≥ 0.35）—— 防止框落在黑边上。
通过者写入 proposals（仍需人工抽检后才合并进清单），未通过者写入 review 队列。

用法：python tools/accept_windows.py --manifest data/new_batch.jsonl --root .
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ddh                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BELT, BELT_MAX, FILL_MIN = 8, 0.10, 0.35


def belt_and_fill(path, box, scale=4):
    im = Image.open(path).convert('L')
    w, h = im.size
    a = np.asarray(im.resize((max(1, w // scale), max(1, h // scale)),
                             Image.Resampling.BILINEAR), dtype=np.float32)
    x0, y0, x1, y1 = [v // scale for v in box]
    b = max(1, BELT // scale)
    strips = [a[max(0, y0 - b):y0, x0:x1], a[y1:y1 + b, x0:x1],
              a[y0:y1, max(0, x0 - b):x0], a[y0:y1, x1:x1 + b]]
    belt = float((np.concatenate([s.ravel() for s in strips]) > 12).mean()) if strips else 1.0
    inside = a[y0:y1, x0:x1]
    fill = float((inside > 12).mean()) if inside.size else 0.0
    return belt, fill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/new_batch.jsonl')
    ap.add_argument('--root', default='.')
    ap.add_argument('--ckpt', default='runs/window.pt')
    ap.add_argument('--threshold', type=float, default=0.8)
    ap.add_argument('--out', default='data/reports/new_window_proposals.jsonl')
    ap.add_argument('--queue', default='data/reports/new_window_queue.txt')
    ap.add_argument('--sheet', default='data/review/new_accepted_check.png')
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wm, meta = ddh.load_checkpoint(ROOT / args.ckpt, device)

    ok, bad = [], []
    for i, r in enumerate(rows):
        p = ROOT / args.root / r['path']
        im = ddh.image_for(r, ROOT / args.root)
        out = wm([ddh.to_tensor(im).to(device)])[0]
        keep = (out['scores'] >= args.threshold) & (out['labels'] == 1)
        n = int(keep.sum())
        if n != 1:
            bad.append((r['id'], f'窗口数={n}'))
            continue
        b = out['boxes'][keep][0].detach().cpu().tolist()
        w, h = im.size
        box = [max(0, int(b[0])), max(0, int(b[1])), min(w, int(round(b[2]))), min(h, int(round(b[3])))]
        if box[2] - box[0] < 32 or box[3] - box[1] < 32:
            bad.append((r['id'], '窗口过小'))
            continue
        belt, fill = belt_and_fill(p, box)
        if belt > BELT_MAX:
            bad.append((r['id'], f'框外不干净(belt={belt:.2f})'))
            continue
        if fill < FILL_MIN:
            bad.append((r['id'], f'框内几乎无内容(fill={fill:.2f})'))
            continue
        ok.append({'id': r['id'], 'path': r['path'], 'window': box,
                   'score': round(float(out['scores'][keep][0]), 4),
                   'belt': round(belt, 4), 'fill': round(fill, 3),
                   'image_wh': r.get('image_wh') or list(im.size), 'source': 'model+自动验收'})
        if (i + 1) % 100 == 0:
            print(f'  …{i+1}/{len(rows)}，通过 {len(ok)}')

    (ROOT / args.out).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in ok), encoding='utf-8')
    (ROOT / args.queue).write_text('\n'.join(f'{i}\t{why}' for i, why in bad), encoding='utf-8')

    # 抽检图：通过者里分层抽样
    import random
    random.seed(0)
    sample = random.sample(ok, min(16, len(ok))) if ok else []
    cell_w, cell_h, cols = 400, 300, 4
    rn = max(1, (len(sample) + cols - 1) // cols)
    sheet = Image.new('RGB', (cell_w * cols, cell_h * rn), 'black')
    d = ImageDraw.Draw(sheet)
    for i, x in enumerate(sample):
        im = Image.open(ROOT / args.root / x['path']).convert('RGB')
        ow, oh = im.size
        sheet.paste(im.resize((cell_w, int(cell_w * oh / ow))), ((i % cols) * cell_w, (i // cols) * cell_h))
        bx = x['window']; kx = cell_w / ow
        d.rectangle([(i % cols) * cell_w + bx[0] * kx, (i // cols) * cell_h + bx[1] * kx,
                     (i % cols) * cell_w + bx[2] * kx, (i // cols) * cell_h + bx[3] * kx],
                    outline=(0, 255, 0), width=2)
        d.text(((i % cols) * cell_w + 4, (i // cols) * cell_h + 2),
               f'{x["image_wh"][0]}x{x["image_wh"][1]} s={x["score"]:.2f}', fill=(255, 255, 0))
    sheet.save(ROOT / args.sheet)

    by_wh = Counter(f"{x['image_wh'][0]}x{x['image_wh'][1]}" for x in ok)
    reasons = Counter(w for _, w in bad)
    print(f'\n通过自动验收：{len(ok)}/{len(rows)} = {len(ok)/len(rows)*100:.0f}%')
    print(f'  按分辨率：{dict(by_wh)}')
    print(f'交人工：{len(bad)}')
    for k, v in reasons.most_common():
        print(f'    {k}: {v}')
    print(f'\n→ {args.out}\n→ {args.queue}\n→ {args.sheet}')


if __name__ == '__main__':
    main()
