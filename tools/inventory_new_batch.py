"""盘点新批次的设备与界面布局：按分辨率分组 + 外壳指纹聚类。

外壳指纹 = 整图缩小后把"保守内框"涂黑，只留下界面边框区域。
同一台机器同一视口的截图，这部分几乎逐像素相同。
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default='data/images/Ⅱc+Ⅱa-b')
    ap.add_argument('--out', default='data/reports/new_batch_layouts.json')
    ap.add_argument('--tol', type=float, default=0.02, help='指纹平均绝对差小于此值归为同一布局')
    args = ap.parse_args()
    base = ROOT / args.folder
    files = sorted([p for p in base.rglob('*.jpg')])
    print(f'图片 {len(files)} 张')

    recs = []
    for p in files:
        im = Image.open(p).convert('L')
        w, h = im.size
        fp = np.asarray(im.resize((64, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        # 把中间 60% 区域涂黑，只留外壳（窗口外的界面元素）
        fp = fp.copy()
        fp[int(48 * 0.15):int(48 * 0.85), int(64 * 0.15):int(64 * 0.85)] = 0.0
        recs.append({'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'wh': (w, h),
                     'sub': p.parent.name, 'fp': fp.ravel()})

    groups = defaultdict(list)
    for r in recs:
        groups[r['wh']].append(r)
    print('\n按分辨率：')
    for wh, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        subs = Counter(x['sub'] for x in v)
        print(f'  {wh[0]}x{wh[1]}: {len(v):4d} 张   子文件夹 {dict(subs)}')

    # 每个分辨率内按指纹聚类
    out = {}
    for wh, v in groups.items():
        clusters = []
        for r in v:
            hit = None
            for c in clusters:
                if float(np.abs(c['rep'] - r['fp']).mean()) < args.tol:
                    hit = c
                    break
            if hit is None:
                clusters.append({'rep': r['fp'], 'members': [r]})
            else:
                hit['members'].append(r)
        clusters.sort(key=lambda c: -len(c['members']))
        print(f'\n=== {wh[0]}x{wh[1]}：{len(clusters)} 种布局 ===')
        for i, c in enumerate(clusters):
            m = c['members']
            subs = Counter(x['sub'] for x in m)
            print(f'  布局{i}: {len(m):4d} 张  子文件夹 {dict(subs)}  样例 {Path(m[0]["path"]).name[:40]}')
            out[f'{wh[0]}x{wh[1]}#{i}'] = {'wh': list(wh), 'n': len(m),
                                           'subs': dict(subs),
                                           'samples': [x['path'] for x in m[:3]],
                                           'interior_spread': round(float(np.mean([
                                               np.abs(c['rep'] - x['fp']).mean() for x in m])), 4)}
    Path(ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n明细 → {args.out}')


if __name__ == '__main__':
    main()
