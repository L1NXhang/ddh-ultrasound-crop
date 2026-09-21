"""窗口预填建议：自动找出截图里的超声成像矩形，输出建议供人工核对。

刻意不修改 annotations.todo.jsonl 的 window 字段：
按方案文档，window 必须是人工确认的标签，算法结果只能当标注辅助。
输出：
  data/window_prefill.jsonl     每图建议框 + 置信度
  data/window_prefill_check.png 抽样核对图（红框 = 建议）
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

THRESHOLD = 12          # 灰度阈值：截图黑底≈0，成像区有回声信号


def longest_run(occ, thr):
    best_len, best_start, start = 0, 0, None
    for i, v in enumerate(list(occ) + [0.0]):
        if v >= thr and start is None:
            start = i
        elif v < thr and start is not None:
            if i - start > best_len:
                best_len, best_start = i - start, start
            start = None
    return best_start, best_start + best_len


def detect_box(g, row_thr=0.2, col_thr=0.5):
    """在缩放灰度图上找最大的“连续且足够亮”的矩形块。

    顺序很重要：先按整幅宽度找行带（成像区只占高度的约 60%，
    所以列占用必须限制在行带之内算，否则块内的暗区会把列方向切断）。
    """
    mask = g > THRESHOLD
    r0, r1 = longest_run(mask.mean(1), row_thr)      # 行：占满整幅宽度，阈值取低
    if r1 <= r0:
        return None
    c0, c1 = longest_run(mask[r0:r1].mean(0), col_thr)  # 列：只在行带内统计
    if c1 <= c0:
        return None
    return c0, r0, c1, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/annotations.todo.jsonl')
    ap.add_argument('--root', default='data')
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--out', default='data/window_prefill.jsonl')
    ap.add_argument('--sheet', default='data/window_prefill_check.png')
    ap.add_argument('--sheet-per-source', type=int, default=8)
    args = ap.parse_args()
    root, s = Path(args.root), args.scale
    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]

    proposals, fps = [], []
    for r in rows:
        im = Image.open(root / r['path']).convert('RGB')
        w, h = im.size
        sw, sh = w // s, h // s
        g = np.asarray(im.resize((sw, sh), Image.Resampling.BILINEAR).convert('L'), dtype=np.float32)
        det = detect_box(g)
        if det is None:
            proposals.append({'id': r['id'], 'window': None, 'confidence': 0.0,
                              'reason': 'no_block_found', 'image_wh': [w, h]})
            continue
        c0, r0, c1, r1 = det
        box = [c0 * s, r0 * s, min(w, c1 * s), min(h, r1 * s)]
        inside = g[r0:r1, c0:c1]
        fill = float((inside > THRESHOLD).mean())
        col_occ = float((g[:, c0:c1] > THRESHOLD).mean(0).mean())
        row_occ = float((g[r0:r1, :] > THRESHOLD).mean(1).mean())
        touches = c0 == 0 or r0 == 0 or c1 >= sw or r1 >= sh
        conf = round(min(col_occ, row_occ) * min(1.0, fill / 0.9), 3)
        proposals.append({'id': r['id'], 'window': box, 'confidence': conf,
                          'col_occupancy': round(col_occ, 3), 'row_occupancy': round(row_occ, 3),
                          'fill': round(fill, 3), 'touches_border': bool(touches),
                          'image_wh': [w, h]})
        # 界面指纹：整图缩小后把成像区涂黑，只留界面外壳，用于聚类外观样式
        fp = np.asarray(im.resize((64, 48), Image.Resampling.BILINEAR).convert('L'),
                        dtype=np.float32) / 255.0
        xs, xe = int(round(c0 / sw * 64)), int(round(c1 / sw * 64))
        ys, ye = int(round(r0 / sh * 48)), int(round(r1 / sh * 48))
        fp = fp.copy()
        fp[ys:ye, xs:xe] = 0.0
        fps.append((r['id'], (r.get('collect_group') or r.get('source_group') or '未标'), fp.ravel()))

    with open(args.out, 'w', encoding='utf-8') as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # 外观样式聚类：只留界面外壳后，同一台机器的截图指纹应当几乎一致
    reps, labels = [], {}
    for pid, src, v in fps:
        hit = next((j for j, (rv, _) in enumerate(reps) if np.abs(rv - v).mean() < 0.02), None)
        if hit is None:
            reps.append((v, [(pid, src)])); labels[pid] = len(reps) - 1
        else:
            reps[hit][1].append((pid, src)); labels[pid] = hit
    clusters = {k: dict(Counter(src for _, src in m)) for k, (_, m) in enumerate(reps)}

    by_source = defaultdict(list)
    for p, r in zip(proposals, rows):
        by_source[(r.get('collect_group') or r.get('source_group') or '未标')].append(p)
    summary = {src: {'图数': len(ps),
                     '有建议': sum(1 for p in ps if p['window']),
                     '低置信<0.5': sum(1 for p in ps if p['window'] and p['confidence'] < 0.5),
                     '碰边': sum(1 for p in ps if p['window'] and p['touches_border']),
                     '中位置信': round(float(np.median([p['confidence'] for p in ps if p['window']] or [0])), 3)}
               for src, ps in sorted(by_source.items())}

    per = args.sheet_per_source
    picks = [(src, p) for src in sorted(by_source) for p in by_source[src][:per]]
    if picks:
        cell, cols = 240, per
        n_rows = len({src for src, _ in picks})
        sheet = Image.new('RGB', (cell * cols, cell * n_rows), 'black')
        d = ImageDraw.Draw(sheet)
        byid = {r['id']: r for r in rows}
        for k, (src, p) in enumerate(picks):
            orig = Image.open(root / byid[p['id']]['path']).convert('RGB')
            th = orig.copy(); th.thumbnail((cell - 8, cell - 8))
            x, y = (k % cols) * cell + 4, (k // cols) * cell + 4
            sheet.paste(th, (x, y))
            kx = th.width / orig.width
            d.rectangle([x + p['window'][0] * kx, y + p['window'][1] * kx,
                         x + p['window'][2] * kx, y + p['window'][3] * kx],
                        outline=(255, 40, 40), width=2)
            d.text((x + 4, y + 2), f"{src} {p['confidence']:.2f}", fill=(255, 255, 0))
        sheet.save(args.sheet)

    print(json.dumps({'图数': len(rows), '有窗口建议': sum(1 for p in proposals if p['window']),
                      '无建议': sum(1 for p in proposals if not p['window']),
                      '分来源': summary,
                      '外观样式聚类数': len(reps),
                      '每类来源构成': {f'style_{k:02d}': v for k, v in clusters.items()},
                      '建议文件': args.out, '核对图': args.sheet},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
