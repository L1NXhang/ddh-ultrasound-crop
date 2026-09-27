"""在新设备数据上预跑模型，把预测写成队列（供人工"接受/微调"，不需要画线）。

产物：data/review_new.jsonl —— window 与 crop 都是模型预测值，
      用标注工具打开后按 Enter 即表示"接受模型答案"，拖动即表示"微调"。
      human_modified 字段直接给出接受率。

用法：python tools/prepare_new_review.py --n 100
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ddh                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/new_batch.jsonl')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', default='data/review_new.jsonl')
    ap.add_argument('--offset', type=int, default=0, help='裁剪线预置偏移（0=直接用模型答案）')
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    by = {}
    for r in rows:
        by.setdefault(tuple(r['image_wh']), []).append(r)
    picks, per = [], max(1, args.n // max(1, len(by)))
    for wh, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        picks += v[::max(1, len(v) // per)][:per]
    picks = picks[:args.n]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wm, wa = ddh.load_checkpoint(ROOT / 'runs/window.pt', device)
    bm, ba = ddh.load_checkpoint(ROOT / 'runs_v2/crop.pt', device)
    out, stats = [], []
    for r in picks:
        p = ddh.pipeline(r, str(ROOT), wm, bm, ba, device, samples=4)
        rec = dict(r)
        if p.get('box') and p.get('bounds'):
            rec['window'] = list(p['box'])
            b = [max(0, p['bounds'][0]), min(1, p['bounds'][1])]
            c = ddh.raw_crop(p['box'], b)
            rec['crop'] = [c[1], c[3]]
            h = c[3] - c[1]
            stats.append((r['image_wh'][0], h, c[1] - p['box'][1], h / (p['box'][3] - p['box'][1])))
        rec['quality'] = None
        rec['safe_box'] = None
        out.append(rec)
    p_out = ROOT / args.out
    p_out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out), encoding='utf-8')
    print(f'已写出 {len(out)} 行 → {p_out.relative_to(ROOT)}（window/crop = 模型预测）')
    if stats:
        print(f'  裁剪高度 中位 {st.median([s[1] for s in stats]):.0f} px')
        print(f'  上边距   中位 {st.median([s[2] for s in stats]):.0f} px'
              f'（占窗口高 {st.median([s[2]/max(1,s[1]) for s in stats])*100:.0f}%）')
        print(f'  裁剪高度/窗口高 中位 {st.median([s[3] for s in stats])*100:.0f}%')


if __name__ == '__main__':
    main()
