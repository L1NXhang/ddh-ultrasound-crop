"""窗口模板：给“同一种界面布局”找一个人工可核对的标准框，把标注量从 1080 降到约 140。

方法（先说清为什么这么做）：
  1. 逐图检测只能给建议，且水平边界不稳；亮度阈值无法精确定义窗口（方案文档也这样警告）。
  2. 但“同一台机器、同一视口的截图窗口位置相同”，所以按分辨率归组，取众数框当模板。
  3. 用客观判据验收模板：模板框外一圈条带的亮度占比 belt。
     belt 干净说明“这个框没有切掉内容”；belt 变亮说明该图窗口比模板大，必须人工单独框。
     注意 belt 只能证明“框不小”，所以模板取自众数，而不用覆盖率搜索（搜索会漂向超大框）。

输出都是建议，不写 annotations 的 window 字段；模板必须人工核对后把 confirmed 改成 true。
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BELT = 8          # 条带宽度（原图像素）
BELT_MAX = 0.10   # 条带亮度占比阈值
TOL = 10          # 边差在 10 像素内视为同一个框（阈值抖动只有几像素）


def belt_on_array(a, box, scale):
    h, w = a.shape
    x0, y0, x1, y1 = [min(max(v // scale, 0), lim) for v, lim in zip(box, (w, h, w, h))]
    b = max(1, BELT // scale)
    strips = [a[max(0, y0 - b):y0, x0:x1], a[y1:y1 + b, x0:x1],
              a[y0:y1, max(0, x0 - b):x0], a[y0:y1, x1:x1 + b]]
    return float((np.concatenate([s.ravel() for s in strips]) > 12).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/annotations.todo.jsonl')
    ap.add_argument('--prefill', default='data/window_prefill.jsonl')
    ap.add_argument('--root', default='data')
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--min-template-images', type=int, default=100,
                    help='该分辨率图数少于此值就不给模板，全部人工框')
    ap.add_argument('--out', default='data/window_templates.json')
    ap.add_argument('--check', default='data/window_template_check.png')
    ap.add_argument('--queue', default='data/template_review_queue.txt')
    args = ap.parse_args()
    root = Path(args.root)
    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    prop = {json.loads(x)['id']: json.loads(x)
            for x in Path(args.prefill).read_text(encoding='utf-8').splitlines() if x.strip()}

    groups = defaultdict(list)
    for r in rows:
        groups[tuple(prop.get(r['id'], {}).get('image_wh') or (0, 0))].append(r)

    templates, queue, report = {}, [], {}
    for wh, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        cache = [np.asarray(Image.open(root / r['path']).convert('L').resize(
            (max(1, wh[0] // args.scale), max(1, wh[1] // args.scale)),
            Image.Resampling.BILINEAR), dtype=np.float32) for r in items]
        boxes = [tuple(prop[r['id']]['window']) for r in items if prop.get(r['id'], {}).get('window')]
        if not boxes or len(items) < args.min_template_images:
            for r in items:
                queue.append(f"{r['id']}\t{(r.get('collect_group') or r.get('source_group') or '未标')}\t{wh[0]}x{wh[1]}\t该分辨率样本少或检测失败，请人工框")
            report[f'{wh[0]}x{wh[1]}'] = {'n': len(items), '模板': None, '说明': '样本少，全部人工框'}
            continue
        anchor = Counter(boxes).most_common(1)[0][0]
        near = [b for b in set(boxes) if all(abs(b[i] - anchor[i]) <= TOL for i in range(4))]
        template = max(near, key=lambda b: boxes.count(b))      # 邻域内出现最多的框
        use = [i for i, a in enumerate(cache) if belt_on_array(a, template, args.scale) <= BELT_MAX]
        dev = [i for i in range(len(items)) if i not in use]
        for i in dev:
            br = belt_on_array(cache[i], template, args.scale)
            queue.append(f"{items[i]['id']}\t{(items[i].get('collect_group') or '未标')}\t{wh[0]}x{wh[1]}\t窗口与模板不同(belt={br:.2f})，请人工框")
        templates[f'{wh[0]}x{wh[1]}'] = {
            'template_xyxy': list(template), 'confirmed': False,
            '适用图片数': len(use), 'n_total': len(items), '偏离图片数': len(dev),
            'applicable_ids': [items[i]['id'] for i in use],
            'sources': dict(Counter((r.get('collect_group') or r.get('source_group') or '未标') for r in items))}
        report[f'{wh[0]}x{wh[1]}'] = {'n': len(items), '模板': list(template),
                                     '模板适用': len(use), '需人工': len(dev),
                                     '偏离样例': [items[i]['id'] for i in dev[:5]]}

    Path(args.out).write_text(json.dumps(
        {'note': '模板需人工核对后把 confirmed 改成 true 才能使用；适用图片数由框外条带亮度判据得出',
         'belt_threshold': BELT_MAX, 'belt_px': BELT, 'tolerance_px': TOL,
         'templates': templates}, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.queue).write_text('\n'.join(queue), encoding='utf-8')

    picks = []
    for wh, t in templates.items():
        w0, h0 = (int(v) for v in wh.split('x'))
        pick = next((r for r in rows if tuple(prop[r['id']].get('image_wh') or ()) == (w0, h0)), None)
        if pick:
            picks.append((pick, t['template_xyxy']))
    if picks:
        cell_w, cell_h = 640, 480
        sheet = Image.new('RGB', (cell_w * len(picks), cell_h), 'black')
        d = ImageDraw.Draw(sheet)
        for k, (pick, box) in enumerate(picks):
            path = root / pick['path']
            ow, oh = Image.open(path).size
            sheet.paste(Image.open(path).convert('RGB').resize((cell_w, cell_h), Image.Resampling.BILINEAR),
                        (k * cell_w, 0))
            kx, ky = cell_w / ow, cell_h / oh
            d.rectangle([k * cell_w + box[0] * kx, box[1] * ky,
                         k * cell_w + box[2] * kx, box[3] * ky], outline=(255, 0, 0), width=3)
            d.text((k * cell_w + 6, 6), f'{k}:{pick["id"]} {ow}x{oh}', fill=(255, 255, 0))
        sheet.save(args.check)

    total_use = sum(t['适用图片数'] for t in templates.values())
    print(json.dumps({'图数': len(rows), '模板数': len(templates),
                      '模板可直接套用': total_use, '需人工': len(rows) - total_use,
                      '分组明细': report, '模板文件': args.out,
                      '人工队列': args.queue, '核对图': args.check},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
