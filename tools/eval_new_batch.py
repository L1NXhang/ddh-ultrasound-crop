"""把训练好的窗口检测器跑在新数据上，看它对未见设备的泛化表现。

输出：检出率、分数分布、以及抽样核对图（供人工视觉复核）。
用法：python tools/eval_new_batch.py [--per-layout 20] [--ckpt runs/window.pt]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
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
    ap.add_argument('--ckpt', default='runs/window.pt')
    ap.add_argument('--per-layout', type=int, default=20)
    ap.add_argument('--out', default='data/reports/new_batch_eval.json')
    ap.add_argument('--sheet', default='data/review/new_model_check.png')
    ap.add_argument('--threshold', type=float, default=0.8)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    layout = json.loads((ROOT / 'data' / 'reports' / 'new_batch_layouts.json').read_text(encoding='utf-8'))
    # 用布局聚类结果分组抽样
    by_layout = {}
    for name, info in layout.items():
        for s in info['samples']:
            by_layout[s] = name
    picks = defaultdict(list)
    for r in rows:
        k = by_layout.get(r['path']) or f"{r['image_wh'][0]}x{r['image_wh'][1]}#?"
        if len(picks[k]) < args.per_layout:
            picks[k].append(r)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wm, meta = ddh.load_checkpoint(ROOT / args.ckpt, device)
    print(f'检测器 {args.ckpt}（训练时短边 {meta["size"]}），设备 {device}\n')
    report, sheet_items = {}, []
    for k in sorted(picks):
        got, scores, boxes = 0, [], []
        for r in picks[k]:
            im = ddh.image_for(r, ROOT / args.root)
            out = wm([ddh.to_tensor(im).to(device)])[0]
            keep = (out['scores'] >= args.threshold) & (out['labels'] == 1)
            boxes.append(out['boxes'][keep].detach().cpu().tolist())
            scores += out['scores'][keep].detach().cpu().tolist()
            if int(keep.sum()) == 1:
                got += 1
            if len(sheet_items) < 16:
                sheet_items.append((k, r, boxes[-1]))
        report[k] = {'抽样': len(picks[k]), '恰一个高分窗口': got,
                     '恰一个比例': round(got / len(picks[k]), 3),
                     '分数中位': round(float(np.median(scores)), 3) if scores else None,
                     '平均框数': round(float(np.mean([len(b) for b in boxes])), 2)}
        print(f'{k:16s} n={len(picks[k]):3d}  恰一个窗口 {got:3d} ({got/len(picks[k])*100:3.0f}%)  '
              f'分数中位 {report[k]["分数中位"]}  平均框数 {report[k]["平均框数"]}')

    cell_w, cell_h = 400, 300
    cols = 4
    rn = (len(sheet_items) + cols - 1) // cols
    sheet = Image.new('RGB', (cell_w * cols, cell_h * rn), 'black')
    d = ImageDraw.Draw(sheet)
    for i, (k, r, bxs) in enumerate(sheet_items):
        im = Image.open(ROOT / r['path']).convert('RGB')
        ow, oh = im.size
        im2 = im.resize((cell_w, int(cell_w * oh / ow)))
        x0, y0 = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im2, (x0, y0))
        kx = cell_w / ow
        for b in bxs:
            d.rectangle([x0 + b[0] * kx, y0 + b[1] * kx, x0 + b[2] * kx, y0 + b[3] * kx],
                        outline=(0, 255, 0), width=2)
        d.text((x0 + 4, y0 + 2), f'{r["image_wh"][0]}x{r["image_wh"][1]} {len(bxs)}框', fill=(255, 255, 0))
    sheet.save(ROOT / args.sheet)
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    tot = sum(v['抽样'] for v in report.values())
    ok = sum(v['恰一个高分窗口'] for v in report.values())
    print(f'\n合计：{ok}/{tot} = {ok/tot*100:.0f}% 有唯一高分窗口')
    print(f'→ {args.out}\n→ {args.sheet}')


if __name__ == '__main__':
    main()
