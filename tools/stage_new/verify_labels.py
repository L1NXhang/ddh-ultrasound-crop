"""核实标签口径：split.jsonl / train_v2.jsonl / labels_v2.jsonl 三者的 crop 是否一致。

关键背景：
  本次实验用 data/reports/split.jsonl 训练与评估，
  但原项目 runs_v2/crop.pt（文档称上边界 17.5px）在同文件上实测 103.76px。
  差异可能来自"标签口径不同"，必须查清后才能写报告。
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]

FILES = ['data/reports/split.jsonl', 'data/train_v2.jsonl', 'data/labels_v2.jsonl',
         'data/train_ready_v2.jsonl', 'data/reports_v2/split.jsonl']


def load(f):
    p = PROJ / f
    if not p.exists():
        return None
    return {json.loads(l)['id']: json.loads(l)
            for l in p.read_text(encoding='utf-8').splitlines() if l.strip()}


def crop_stats(d, name):
    """返回窗口内相对位置（归一化）的统计，用于比较口径。"""
    tops, bots, hts = [], [], []
    for r in d.values():
        c, w = r.get('crop'), r.get('window')
        if not c or not w:
            continue
        y0, y1 = w[1], w[3]
        h = y1 - y0
        if h <= 0:
            continue
        tops.append((c[0] - y0) / h)
        bots.append((c[1] - y0) / h)
        hts.append((c[1] - c[0]) / h)
    if not tops:
        print(f'{name}: 无有效 crop'); return None
    s = {'n': len(tops),
         'top_rel_median': round(float(np.median(tops)), 4),
         'bot_rel_median': round(float(np.median(bots)), 4),
         'height_rel_median': round(float(np.median(hts)), 4),
         'top_abs_median': None}
    # 绝对像素
    abstop = [(r['crop'][0] - r['window'][1]) for r in d.values()
              if r.get('crop') and r.get('window')]
    s['top_abs_median'] = int(np.median(abstop)) if abstop else None
    print(f"{name:32s} n={s['n']:4d}  上边界(窗口内)中位={s['top_abs_median']:4d}px  "
          f"相对={s['top_rel_median']:.4f}  下边界相对={s['bot_rel_median']:.4f}  "
          f"裁剪高度相对={s['height_rel_median']:.4f}")
    return s


def main():
    data = {}
    for f in FILES:
        d = load(f)
        if d:
            data[f] = d
            crop_stats(d, f)
        else:
            print(f'{f:32s} 不存在')

    print('\n=== 逐图比较（同一 id 的 crop 是否相同）===')
    keys = list(data.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = data[keys[i]], data[keys[j]]
            common = set(a) & set(b)
            if not common:
                continue
            same = sum(1 for k in common
                       if a[k].get('crop') == b[k].get('crop')
                       and a[k].get('window') == b[k].get('window'))
            print(f'{keys[i]:32s} vs {keys[j]:32s}  共同 {len(common):4d}  '
                  f'完全一致 {same:4d} ({same/len(common)*100:.1f}%)')

    # 找出差异样例
    if 'data/reports/split.jsonl' in data and 'data/train_v2.jsonl' in data:
        a = data['data/reports/split.jsonl']; b = data['data/train_v2.jsonl']
        print('\n=== split.jsonl vs train_v2.jsonl 差异样例 ===')
        shown = 0
        for k in sorted(set(a) & set(b)):
            if a[k].get('crop') != b[k].get('crop') or a[k].get('window') != b[k].get('window'):
                print(f"  id={k}  split.jsonl: crop={a[k].get('crop')} window={a[k].get('window')}")
                print(f"           train_v2.jsonl: crop={b[k].get('crop')} window={b[k].get('window')}")
                shown += 1
                if shown >= 5:
                    break
        if shown == 0:
            print('  无差异（两者完全一致）')


if __name__ == '__main__':
    main()
