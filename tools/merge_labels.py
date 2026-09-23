"""把多人标注的结果合并成一份，并做冲突与一致性检查。

用法：python tools/merge_labels.py --inputs data/human_A.jsonl data/human_B.jsonl \
                                    --out data/annotations_human.jsonl
"""
import argparse
import json
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    p = ROOT / p
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', default='data/annotations_human.jsonl')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    merged, dup = {}, []
    per_person = defaultdict(list)
    for f in args.inputs:
        for r in load(f):
            if r['id'] in merged:
                dup.append((r['id'], f))
            merged[r['id']] = r
            per_person[Path(f).stem].append(r)

    print(f'合并 {len(args.inputs)} 份 → 共 {len(merged)} 条'
          + (f'（重复 id {len(dup)} 个，后写覆盖先写）' if dup else ''))
    print('\n每人标注情况（用"上边距中位"看口径是否一致）：')
    print(f'{"文件":28s} {"条数":>5s} {"上边距中位":>10s} {"标准差":>7s} {"有safe_box":>10s} {"quality=1":>9s}')
    for name, recs in per_person.items():
        tm = [r['crop'][0]-r['window'][1] for r in recs if r.get('crop') and r.get('window')]
        sb = sum(1 for r in recs if r.get('safe_box'))
        q1 = sum(1 for r in recs if r.get('quality') == 1)
        med = f'{st.median(tm):.0f}' if tm else '—'
        sd = f'{st.pstdev(tm):.1f}' if len(tm) > 1 else '—'
        print(f'{name:28s} {len(recs):5d} {med:>10s} {sd:>7s} {sb:10d} {q1:9d}')
    if len(per_person) > 1:
        meds = [st.median([r['crop'][0]-r['window'][1] for r in v if r.get('crop') and r.get('window')])
                for v in per_person.values() if any(r.get('crop') for r in v)]
        if len(meds) > 1:
            print(f'\n各人上边距中位数之差：最大 {max(meds)-min(meds):.0f} px'
                  f'（>15px 说明口径不一致，需要统一后再训练）')
    if args.apply:
        p = ROOT / args.out
        if p.exists():
            p.rename(ROOT / 'data' / f'{p.stem}.bak_{time.strftime("%Y%m%d_%H%M%S")}.jsonl')
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in merged.values()),
                     encoding='utf-8')
        print(f'\n已写出 {args.out}（{len(merged)} 条，旧文件已备份）')
    else:
        print('\n（预演，未写文件；确认后加 --apply）')


if __name__ == '__main__':
    main()
