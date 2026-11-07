"""D 项：50 张案例展示 —— 从 **test 集** 里筛模型表现最好的 50 张。

筛选口径（必须与 PPT 一致，不得改写）：
  1. 候选池：固定的 test 集 88 张（`data/reports_v2/split.jsonl` 中 split=='test'）。
     test 集**从未参与训练**，因此不存在数据泄漏。
  2. 排序依据：该模型在 test 上的"最差单边误差"（max(top_err_px, bot_err_px)）升序，
     取前 50 张 = 该模型表现最好的 50 张。
  3. ⚠️ 这是**按结果筛选的展示集**，只用于说明"模型能做到什么"，**不能代表总体性能**。
     总体性能一律以全部 88 张的 metrics.json 为准。

用法：python tools/stage_new/make_cases.py --config r18_l3
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = Path(__file__).resolve().parents[2]
RUN = PROJ / 'runs' / '20261103_resnet18_vs_50_corrected'
TOP_N = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='r18_l3')
    ap.add_argument('--sub', default=None)
    ap.add_argument('--per-sheet', type=int, default=10)
    args = ap.parse_args()
    sub = args.sub or ('resnet18' if args.config.startswith('r18') else 'resnet50')
    d = RUN / sub / args.config
    per = d / 'test_per_image.jsonl'
    if not per.exists():
        raise SystemExit(f'缺少 {per}，请先完成该组训练评测')

    recs = [json.loads(l) for l in per.read_text(encoding='utf-8').splitlines() if l.strip()]
    rows = [json.loads(l) for l in (PROJ / 'data/reports_v2/split.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    by_id = {r['id']: r for r in rows}
    for r in recs:
        r['max_err_px'] = max(r['top_err_px'], r['bot_err_px'])
    recs.sort(key=lambda r: r['max_err_px'])
    picked = recs[:TOP_N]

    outdir = RUN / 'figures' / f'cases_{args.config}'
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, r in enumerate(picked, 1):
        meta = by_id[r['id']]
        img_path = PROJ / 'data' / meta['path']
        im = Image.open(img_path).convert('RGB')
        x0, y0, x1, y1 = meta['window']
        win = im.crop((x0, y0, x1, y1))
        W, H = win.size
        draw = ImageDraw.Draw(win)
        # 真值（绿）来自标注 crop；预测（红）来自 test_per_image.jsonl 里的 pred_bounds（归一化）
        gt_top = meta['crop'][0] - y0
        gt_bot = meta['crop'][1] - y0
        pb = r['pred_bounds']
        pred_top = pb[0] * H
        pred_bot = pb[1] * H
        draw.line([(0, gt_top), (W, gt_top)], fill=(0, 200, 0), width=4)
        draw.line([(0, gt_bot), (W, gt_bot)], fill=(0, 200, 0), width=4)
        draw.line([(0, pred_top), (W, pred_top)], fill=(230, 30, 30), width=3)
        draw.line([(0, pred_bot), (W, pred_bot)], fill=(230, 30, 30), width=3)
        win.save(outdir / f'{rank:02d}_{r["id"]}.png')
        manifest.append({'rank': rank, 'id': r['id'], 'group': r['group'],
                         'top_err_px': round(r['top_err_px'], 1),
                         'bot_err_px': round(r['bot_err_px'], 1),
                         'max_err_px': round(r['max_err_px'], 1),
                         'vert_iou': round(r['vert_iou'], 4),
                         'file': f'{rank:02d}_{r["id"]}.png'})

    (d / 'cases_top50_manifest.json').write_text(
        json.dumps({'config': args.config,
                    'selection_rule': 'test集88张中按 max(top_err_px,bot_err_px) 升序取前50',
                    'caveat': '按结果筛选的展示集，不代表总体性能；总体性能见 metrics.json',
                    'candidates': len(recs), 'picked': len(manifest), 'items': manifest},
                   ensure_ascii=False, indent=2), encoding='utf-8')

    # 拼图：每张 sheet 放 per_sheet 张
    sheets = []
    for start in range(0, len(manifest), args.per_sheet):
        chunk = manifest[start:start + args.per_sheet]
        cols, rowsn = 5, 2
        fig, axes = plt.subplots(rowsn, cols, figsize=(cols * 2.6, rowsn * 3.4))
        axes = axes.ravel()
        for ax in axes:
            ax.axis('off')
        for ax, item in zip(axes, chunk):
            ax.imshow(Image.open(outdir / item['file']))
            ax.set_title(f"#{item['rank']} {item['id']}\n上{item['top_err_px']:.0f}px 下{item['bot_err_px']:.0f}px",
                         fontsize=8)
        sheet = outdir / f'sheet_{start // args.per_sheet + 1}.png'
        fig.suptitle(f'案例展示（按表现筛选，非总体性能） {args.config} 第{start // args.per_sheet + 1}张', fontsize=11)
        fig.tight_layout()
        fig.savefig(sheet, dpi=160, bbox_inches='tight')
        plt.close(fig)
        sheets.append(str(sheet.relative_to(PROJ)))
    print(json.dumps({'picked': len(manifest), 'sheets': sheets}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
