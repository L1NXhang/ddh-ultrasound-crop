"""对已保存的标注做机械性修正（需要人工确认后再跑）。

目前支持：
  --round      把坐标取整（工具早期版本漏了 safe_box 的取整）
  --flip-quality  把 quality 0↔1 对调（用于"快捷键按反了"的情况）
  --report     只报告问题，不改文件（默认行为）

用法：
  python tools/fix_labels.py --report
  python tools/fix_labels.py --round --apply
  python tools/fix_labels.py --flip-quality --apply
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='data/annotations_human.jsonl')
    ap.add_argument('--round', action='store_true')
    ap.add_argument('--flip-quality', action='store_true')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    p = ROOT / args.file
    recs = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

    n_frac = sum(1 for r in recs if any(
        any(abs(v - round(v)) > 1e-6 for v in r[k]) for k in ('window', 'crop', 'safe_box') if r.get(k)))
    viol = [r for r in recs if r.get('crop') and r.get('safe_box')
            and (r['crop'][0] > r['safe_box'][1] or r['crop'][1] < r['safe_box'][3])]
    out_x = [r for r in recs if r.get('window') and r.get('safe_box')
             and not (r['window'][0] <= r['safe_box'][0] and r['safe_box'][2] <= r['window'][2]
                      and r['window'][1] <= r['safe_box'][1] and r['safe_box'][3] <= r['window'][3])]
    from collections import Counter
    print(f'文件 {args.file}：{len(recs)} 条')
    print(f'  坐标非整数      : {n_frac} 条')
    print(f'  裁剪切进 safe_box: {len(viol)} 条')
    print(f'  safe_box 出窗口  : {len(out_x)} 条')
    print(f'  quality 分布     : {dict(Counter(str(r.get("quality")) for r in recs))}')

    if not args.apply:
        print('\n（只报告，未修改。要真正修改请加 --apply）')
        return
    bk = ROOT / 'data' / f'annotations_human.bak_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'
    shutil.copy2(p, bk)
    changed = 0
    for r in recs:
        if args.round:
            for k in ('window', 'crop', 'safe_box'):
                if r.get(k):
                    nb = [int(round(v)) for v in r[k]]
                    if nb != r[k]:
                        r[k] = nb
                        changed += 1
        if args.flip_quality and r.get('quality') in (0, 1):
            r['quality'] = 1 - r['quality']
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in recs), encoding='utf-8')
    print(f'\n已修改并写回（备份 {bk.name}）。改动计数 {changed}')


if __name__ == '__main__':
    main()
