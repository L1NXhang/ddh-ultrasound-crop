"""用人工标注的窗口验收通用窗口检测算法。

输入：data/train_ready_v2.jsonl（862 张，人工确认过的 window）
输出：误差分布 + 最差案例清单

用法：python tools/validate_window.py
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_window import detect                                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/train_ready_v2.jsonl')
    ap.add_argument('--root', default='data')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    rows = [r for r in rows if r.get('window')]
    if args.limit:
        rows = rows[:args.limit]

    errs, worst, flags = [], [], {}
    for r in rows:
        d = detect(ROOT / args.root / r['path'])
        if not d['box']:
            flags['检测失败'] = flags.get('检测失败', 0) + 1
            continue
        for f in d['flags']:
            flags[f] = flags.get(f, 0) + 1
        h = r['window']
        e = [d['box'][i] - h[i] for i in range(4)]          # [left, top, right, bottom]
        errs.append(e)
        m = max(abs(v) for v in e)
        worst.append((m, r['id'], e, d['box'], h))

    E = np.array(errs)
    names = ['左', '上', '右', '下']
    print(f'验收 {len(E)} 张（人工标注窗口为真值）\n')
    print(f'{"边":4s} {"中位误差":>9s} {"平均":>7s} {"95%分位":>8s} {"≤4px":>7s} {"≤8px":>7s} {"≤15px":>7s}')
    for i, n in enumerate(names):
        v = E[:, i]
        print(f'{n:4s} {st.median(v):+9.1f} {v.mean():+7.1f} {np.percentile(np.abs(v),95):8.1f} '
              f'{np.mean(np.abs(v)<=4)*100:6.0f}% {np.mean(np.abs(v)<=8)*100:6.0f}% {np.mean(np.abs(v)<=15)*100:6.0f}%')
    mx = np.abs(E).max(axis=1)
    print(f'\n四条边最大误差：≤4px {np.mean(mx<=4)*100:.0f}%  ≤8px {np.mean(mx<=8)*100:.0f}%  ≤15px {np.mean(mx<=15)*100:.0f}%  ≤30px {np.mean(mx<=30)*100:.0f}%')
    print(f'完全一致（四边全 0）: {np.mean(mx==0)*100:.0f}%')
    print(f'\n质检标记统计：{flags if flags else "无"}')
    worst.sort(reverse=True)
    print('\n最差的 8 个案例（最大边误差）：')
    for m, rid, e, got, want in worst[:8]:
        print(f'  {rid}  最大差 {m:3.0f}px  检测={got}  人工={want}  各边={e}')
    Path(ROOT / 'data/reports/window_validation.json').write_text(
        json.dumps({'n': len(E), 'per_edge_median': [round(st.median(E[:, i]), 1) for i in range(4)],
                    'within4': round(float(np.mean(np.abs(E) <= 4)), 3),
                    'within8': round(float(np.mean(np.abs(E) <= 8)), 3),
                    'maxerr_within15': round(float(np.mean(mx <= 15)), 3),
                    'flags': flags,
                    'worst': [{'id': r, 'err': e, 'got': g, 'want': w} for m, r, e, g, w in worst[:20]]},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n→ data/reports/window_validation.json')


if __name__ == '__main__':
    main()
