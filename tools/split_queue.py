"""把一份清单平均拆成 N 份，供多人同时标注。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--parts', type=int, default=3)
    ap.add_argument('--names', default='A,B,C', help='每份的标签，用于文件名')
    ap.add_argument('--prefix', default='', help='输出前缀，默认 <manifest>_part')
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    names = [s.strip() for s in args.names.split(',')][:args.parts]
    n = len(rows)
    base = args.prefix or (str(Path(args.manifest).with_suffix('')) + '_part')
    buckets = [[] for _ in range(args.parts)]
    for i, r in enumerate(rows):
        buckets[i % args.parts].append(r)          # 轮转分配，保证每份的图分布均匀
    out = []
    for name, b in zip(names, buckets):
        p = ROOT / f'{base}_{name}.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in b), encoding='utf-8')
        out.append((p.name, len(b)))
    print(f'共 {n} 张 → 拆成 {len(out)} 份：')
    for name, cnt in out:
        print(f'  {name}: {cnt} 张')


if __name__ == '__main__':
    main()
