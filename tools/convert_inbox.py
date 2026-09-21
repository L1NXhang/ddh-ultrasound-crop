"""把 crops_inbox 里各来源的人工裁切图批量转成 crop 标签，并给出每人一份质量报告。

质量指标（客观可算的）：
  - 能不能定位回原图（匹配不唯一/失败 = 严重问题）
  - 是否被缩放（scale != 1.0）
  - 裁切宽度 vs 该图成像窗口宽度（裁掉左右 / 留黑边）
  - 裁切高度（过窄可能切掉解剖）
  - 上下边界是否落在窗口范围内
  - 编号缺失/重复/不在清单里
另外按来源各生成一张核对图，供人工看解剖是否裁得对。

用法：python tools/convert_inbox.py --apply
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_crops import match_one                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCALES = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5]
MIN_HEIGHT = 120          # 低于这个高度视为可能裁太窄
WIDTH_TOL = 12            # 裁切宽度与窗口宽度允许的差


def load_jsonl(p):
    p = ROOT / p
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--out', default='data/crop_from_manual.jsonl')
    ap.add_argument('--sheet-per-source', type=int, default=12)
    args = ap.parse_args()
    rows = load_jsonl('data/annotations.todo.jsonl')
    byid = {r['id']: r for r in rows}
    old = {x['id']: x for x in load_jsonl(args.out)}

    sources = [d for d in sorted((ROOT / 'data' / 'crops_inbox').iterdir()) if d.is_dir()]
    results, report = {}, {}
    for src in sources:
        crops = sorted(list(src.glob('*.jpg')) + list(src.glob('*.png')))
        if not crops:
            continue
        stat = Counter()
        details = []
        for f in crops:
            rid = f.stem.split('_')[0]
            r = byid.get(rid)
            if r is None:
                stat['编号不在清单里'] += 1
                details.append({'id': rid, 'status': '编号不在清单里'})
                continue
            m = match_one(Image.open(ROOT / 'data' / r['path']), Image.open(f), SCALES)
            if m is None:
                stat['匹配失败'] += 1
                details.append({'id': rid, 'status': '匹配失败'})
                continue
            y0, x0, x1, y1 = m['box']
            h = y1 - y0
            w = x1 - x0
            win = r.get('window')
            if m['prominence'] < 0.05:
                status = '匹配不唯一，需人工核对'
            elif m['scale'] != 1.0:
                status = '裁切图被缩放过，请核对'
            else:
                status = 'ok'
            item = {'id': rid, 'status': status, 'scale': m['scale'], 'rmse': m['rmse'],
                    'prominence': m['prominence'], 'matched_box': m['box'], 'crop': [y0, y1],
                    'crop_wh': m['crop_wh']}
            if m['prominence'] < 0.05:
                stat['匹配不唯一'] += 1
            elif m['scale'] != 1.0:
                stat['被缩放'] += 1
            else:
                stat['定位正常'] += 1
            if win:
                item['window_wh'] = [win[2] - win[0], win[3] - win[1]]
                if abs(w - (win[2] - win[0])) > WIDTH_TOL:
                    stat['裁切宽度与窗口不一致'] += 1
                if not (win[1] <= y0 and y1 <= win[3]):
                    stat['上下伸出窗口'] += 1
            if h < MIN_HEIGHT:
                stat['裁切过窄(<%dpx)' % MIN_HEIGHT] += 1
            details.append(item)
        # 缺失：原图里有、这来源没有裁切图的
        have = {d['id'] for d in details if 'crop' in d}
        want = [r['id'] for r in rows if r.get('collect_group') == src.name]
        missing = [i for i in want if i not in have]
        dup = [k for k, v in Counter(d['id'] for d in details).items() if v > 1]
        heights = [d['crop'][1] - d['crop'][0] for d in details if 'crop' in d]
        report[src.name] = {
            '裁切图数': len(crops), '原图数': len(want), '缺失': len(missing), '重复编号': len(dup),
            '指标': dict(stat),
            '高度': {'最小': min(heights), '中位': int(np.median(heights)), '最大': max(heights)} if heights else None}
        report[src.name]['缺失样例'] = missing[:5]
        for d in details:
            if 'crop' in d:
                results[d['id']] = d
        # 核对图：均匀抽样，画出反推区域
        ok = [d for d in details if 'crop' in d and d.get('status') == 'ok']
        step = max(1, len(ok) // args.sheet_per_source)
        picks = ok[::step][:args.sheet_per_source]
        if picks:
            cell = 300
            cols = min(6, len(picks))
            rn = (len(picks) + cols - 1) // cols
            sheet = Image.new('RGB', (cell * cols, cell * rn), 'black')
            d = ImageDraw.Draw(sheet)
            for k, it in enumerate(picks):
                p = ROOT / 'data' / byid[it['id']]['path']
                im = Image.open(p).convert('RGB')
                ow, oh = im.size
                im2 = im.resize((cell, int(cell * oh / ow)))
                x, y = (k % cols) * cell, (k // cols) * cell
                sheet.paste(im2, (x, y))
                kx = cell / ow
                bx = it['matched_box']
                d.rectangle([x + bx[0] * kx, y + bx[1] * kx, x + bx[2] * kx, y + bx[3] * kx],
                            outline=(255, 0, 0), width=2)
                d.text((x + 4, y + 2), it['id'], fill=(255, 255, 0))
            sheet.save(ROOT / f'data/review/crops_{src.name}.png')

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.apply:
        merged = dict(old)
        for k, v in results.items():
            merged[k] = v
        target = ROOT / args.out
        target.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in merged.values()),
                          encoding='utf-8')
        print(f'\n已写出 {args.out}：共 {len(merged)} 条 crop 标签')
    else:
        print('\n（预演，未写标签文件；确认后加 --apply）')


if __name__ == '__main__':
    main()
