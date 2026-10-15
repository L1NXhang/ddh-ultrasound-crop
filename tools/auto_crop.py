"""全自动批量裁剪：逐图跑「算法检测窗口 + 模型预测裁剪线」，输出标线图与裁剪图。

与交互版工具（tools/crop_app.py）复用同一套推理与导出代码，结果完全一致，
区别只是不需要人工按 Enter——用于批量出效果、生成验收材料、跑基线。

用法：
  # 按清单跑（可过滤某一批/某人）
  python tools/auto_crop.py --manifest data/分配20261009.jsonl --where collect_group=余 --out data/auto/分配20261009_余
  # 直接给一个图片文件夹（无清单，id 取文件名）
  python tools/auto_crop.py --folder "D:/待处理" --out "D:/输出"
  # 常用参数
  --margin 0.02（额外安全边距，占窗口高度）  --device cpu|cuda  --limit N（只跑前 N 张）

输出：<out>/cropped/  <out>/annotated/  <out>/results.jsonl  <out>/gallery.html（快速翻看用）
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='')
    ap.add_argument('--where', default='', help='过滤清单，如 collect_group=余')
    ap.add_argument('--folder', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--crop-checkpoint', default='runs_v2/crop.pt')
    ap.add_argument('--margin', type=float, default=0.02)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if not args.manifest and not args.folder:
        ap.error('需要 --manifest 或 --folder 之一')

    import crop_app as tool                      # 交互工具的推理/导出，直接复用
    cp = Path(args.crop_checkpoint)
    args.crop_checkpoint = str(cp if cp.is_absolute() else ROOT / cp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tool.STATE['out'] = out
    tool.STATE['margin'] = args.margin
    tool.STATE['dry'] = False
    tool.load_models(args)

    if args.folder:
        folder = Path(args.folder)
        files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in tool.IMAGE_EXT)
        tool.STATE['folder'] = folder
        rows = [{'id': p.stem, 'file': str(p.relative_to(folder))} for p in files]
    else:
        man = [json.loads(x) for x in Path(args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
        if args.where:
            k, v = args.where.split('=', 1)
            man = [r for r in man if str(r.get(k)) == v]
        tool.STATE['folder'] = ROOT              # 清单里的 path 相对项目根
        rows = [{'id': r['id'], 'file': r['path']} for r in man]
    if args.limit:
        rows = rows[:args.limit]
    tool.STATE['rows'] = rows
    print(f'待处理 {len(rows)} 张 → {out}')

    recs, t0, n_fail = [], time.time(), 0
    for i, r in enumerate(rows, 1):
        d = tool.predict(r['id'])
        rec = dict(r)
        rec.update({k: d.get(k) for k in ('window', 'crop', 'window_flags', 'window_confidence')})
        rec['model_std'] = d.get('std')
        if d.get('crop'):
            tool.export(rec)                     # 写 cropped/ 与 annotated/
        else:
            n_fail += 1
            rec['exported'] = False
        recs.append(rec)
        if i % 20 == 0 or i == len(rows):
            print(f'  {i}/{len(rows)}  用时 {time.time()-t0:.0f}s')

    rp = out / 'results.jsonl'
    rp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in recs), encoding='utf-8')

    # 快速翻看用的页面（缩略图 → 点开原图）
    items = ''.join(
        f'<figure><a href="annotated/{Path(r["file"]).stem}.png" target="_blank">'
        f'<img src="annotated/{Path(r["file"]).stem}.png" loading="lazy"></a>'
        f'<figcaption>{Path(r["file"]).name}<br>窗口 {r.get("window")}<br>裁剪 {r.get("crop")}</figcaption></figure>'
        for r in recs if r.get('crop'))
    (out / 'gallery.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>自动裁剪结果</title>'
        '<style>body{background:#111;color:#ddd;font:13px/1.5 system-ui;margin:20px}'
        'figure{display:inline-block;margin:6px;width:320px;vertical-align:top}'
        'img{width:320px;border:1px solid #444}figcaption{color:#9ab;font-size:11px}</style>'
        f'<h2>自动裁剪结果（{len([r for r in recs if r.get("crop")])} 张）</h2>{items}',
        encoding='utf-8')

    print(f'\n完成：{len(recs)-n_fail} 张成功，{n_fail} 张失败（无窗口），用时 {time.time()-t0:.0f}s')
    print(f'  {out}/cropped/     裁好的图')
    print(f'  {out}/annotated/   标了窗口(红)与上下线(绿/蓝)的图')
    print(f'  {out}/results.jsonl  每张的窗口、裁剪、模型不确定度、质检标记')
    print(f'  {out}/gallery.html   浏览器打开可快速翻看')


if __name__ == '__main__':
    main()
