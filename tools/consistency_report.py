"""标注一致性报告：用解剖锚点找出"该重标"的图，而不是全量重标。

原理：把每张图的裁切上边界对齐到**解剖内容起点**（窗口内第一行有组织的行），
得到"裁切上边界距组织起点多少像素"。同一套口径下这个值应该很稳。
以人工质量最高的那一批（默认 张）的中位数为参照，偏离超过阈值的就是重标候选。

用法：python tools/consistency_report.py [--reference 张] [--tolerance 40]
输出：data/reports/consistency.json + 控制台摘要
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def content_top(a, window, thr=25, frac=0.15):
    """窗口内第一行"有组织"的位置：行内亮像素占比超过 frac。"""
    x0, y0, x1, y1 = window
    w = a[y0:y1, x0:x1]
    prof = (w > thr).mean(axis=1)
    idx = np.where(prof > frac)[0]
    return y0 + int(idx[0]) if len(idx) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/train_ready.jsonl')
    ap.add_argument('--root', default='data')
    ap.add_argument('--reference', default='张', help='作为口径参照的来源（人工质量最高那批）')
    ap.add_argument('--tolerance', type=float, default=40.0, help='偏离参照中位多少像素算重标候选')
    ap.add_argument('--out', default='data/reports/consistency.json')
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]

    per = defaultdict(list)
    for r in rows:
        a = np.asarray(Image.open(ROOT / args.root / r['path']).convert('L'), dtype=np.float32)
        ct = content_top(a, r['window'])
        if ct is None:
            continue
        per[r['collect_group']].append({
            'id': r['id'], 'd_top': r['crop'][0] - ct, 'height': r['crop'][1] - r['crop'][0],
            'crop': r['crop'], 'content_top': ct})

    ref_vals = [x['d_top'] for x in per[args.reference]]
    ref = st.median(ref_vals)
    ref_sd = st.pstdev(ref_vals)
    heights = [x['height'] for x in per[args.reference]]
    h_med, h_sd = st.median(heights), st.pstdev(heights)

    print(f'参照口径（{args.reference}，n={len(ref_vals)}）：裁切上边界距组织起点 '
          f'{ref:.0f} ± {ref_sd:.1f} px；高度 {h_med:.0f} ± {h_sd:.1f} px')
    print(f'判定：偏离 > {args.tolerance:.0f} px，或高度偏离 > {2.5*h_sd:.0f} px\n')
    print(f'{"来源":6s} {"n":>4s} {"d_top 中位":>10s} {"d_top 标准差":>12s} {"需重标":>6s} {"占比":>6s}')
    out, total_flag = {}, 0
    for s in sorted(per):
        v = per[s]
        flag = [x for x in v if abs(x['d_top'] - ref) > args.tolerance or abs(x['height'] - h_med) > 2.5 * h_sd]
        total_flag += len(flag)
        print(f'{s:6s} {len(v):4d} {st.median([x["d_top"] for x in v]):10.0f} '
              f'{st.pstdev([x["d_top"] for x in v]):12.1f} {len(flag):6d} {len(flag)/len(v)*100:5.0f}%')
        out[s] = {'n': len(v), 'd_top_median': round(st.median([x['d_top'] for x in v]), 1),
                  'd_top_pstdev': round(st.pstdev([x['d_top'] for x in v]), 1),
                  'flag': [x['id'] for x in flag]}
    print(f'\n合计重标候选：{total_flag} / {len(rows)}（{total_flag/len(rows)*100:.0f}%）')
    Path(ROOT / args.out).write_text(json.dumps(
        {'reference': args.reference, 'reference_d_top': ref, 'reference_pstdev': round(ref_sd, 1),
         'tolerance_px': args.tolerance, 'by_source': out,
         'note': 'd_top = 裁切上边界 − 组织起点；偏离参照中位超过容差即为重标候选'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('明细 →', args.out)


if __name__ == '__main__':
    main()
