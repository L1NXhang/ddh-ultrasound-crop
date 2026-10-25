"""Scan the WHOLE training set: predict top/bottom for every image, compare with
the human label, and rank by disagreement -> find all "blurry / hard" images.

Also computes, for every image, the high-echo-band position (owner's criterion:
the real boundary is where a distinct echogenic layer first appears) so we can
check whether that criterion tracks the human label better than the model does.
"""
import json, sys, time, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r'D:\ddhmodel\ddh_multidevice_v2')
OUT = ROOT / 'data' / 'reports' / 'blur_scan_20261024'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
import ddh

rows = [json.loads(l) for l in (ROOT/'data/experiments/merge_20261022/split.jsonl')
        .read_text(encoding='utf-8').splitlines() if l.strip()]


def human_top(r):
    """Authoritative human top line for each row."""
    if r.get('clean_box'):
        return int(r['clean_box'][1])
    if r.get('crop'):
        return int(r['crop'][0])
    return None


def band_metrics(gray, x0, x1, y, half=45):
    """Sharpness + 'is there actually a structure interface here' signals."""
    h = gray.shape[0]
    t, b = max(0, y-half), min(h, y+half)
    seg = gray[t:b, x0:x1].mean(axis=1)
    if len(seg) < 12:
        return dict(peak=0., span=1., contrast=0., snr=0.)
    g = np.abs(np.diff(seg))
    peak = float(g.max())
    i = int(np.argmax(g))
    lo = seg[max(0, i-10):i+1]; hi = seg[i+1:i+11]
    if len(lo) < 2 or len(hi) < 2:
        return dict(peak=peak, span=99., contrast=0., snr=0.)
    a0, a1 = float(lo.min()), float(hi.max())
    contrast = a1 - a0
    # 20%-80% transition width
    if contrast > 1e-6:
        ys = np.arange(max(0, i-10), i+1)
        yy = seg[max(0, i-10):i+1]
        lo_i = np.where(yy <= a0 + .2*contrast)[0]
        hi_i = np.where(hi >= a0 + .8*contrast)[0]
        span = float(abs(hi_i[0] + 1 - lo_i[-1])) if len(lo_i) and len(hi_i) else 99.
    else:
        span = 99.
    noise = float(np.std(np.diff(seg)) * 1.4826 + 1e-6)
    return dict(peak=peak, span=max(span, 1.), contrast=contrast, snr=contrast/noise)


def first_echo_band(gray, x0, x1, y_top, y_bot, min_contrast=0.06, win=6):
    """Owner's criterion: scan down from window top; return the row where a
    clearly brighter echogenic band FIRST appears above the local background.

    Implemented as: for each candidate row y, compare mean intensity of the
    band above (y-win..y-1) with a long background reference above it.
    The first row where intensity jumps by >= min_contrast is the candidate.
    """
    prof = gray[y_top:y_bot, x0:x1].mean(axis=1)
    n = len(prof)
    if n < 40:
        return None
    best = None
    for i in range(15, n-win-5):
        above = prof[max(0, i-25):i-2].mean()      # background above
        here = prof[i-win:i+win].mean()            # candidate band
        if here - above >= min_contrast:
            best = y_top + i
            break
    return best


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = ddh.load_checkpoint(ROOT/'runs_v2/crop.pt', device)
    print(f'device={device}  rows={len(rows)}', flush=True)

    out = []
    t0 = time.time()
    for k, r in enumerate(rows):
        path = ROOT / r['path']
        x0, y0, x1, y1 = r['window']
        ht = human_top(r)
        try:
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                gray = np.asarray(im.convert('L'), dtype=np.float32) / 255.
                x, roi = ddh.letterbox(rgb.crop((x0, y0, x1, y1)), 384)
            ddh.seed_all(20261024 + k)
            pred = ddh.mc_crop(model, x[None].to(device), roi[None].to(device), 4)
            bounds = pred['bounds'][0].tolist()
            std = pred.get('std')
            box = ddh.raw_crop(r['window'], bounds)
            mt, mb = int(box[1]), int(box[3])
        except Exception as e:
            print('FAIL', r.get('id'), e, flush=True)
            continue

        rec = dict(id=r['id'], path=r['path'], split=r['split'], cohort=r['cohort'],
                   author=r.get('annotation_author'), window=r['window'],
                   human_top=ht, model_top=mt, model_bottom=mb,
                   human_bottom=(int((r.get('clean_box') or [0,0,0,r['crop'][-1] if r.get('crop') else y1])[3])
                                 if r.get('clean_box') or r.get('crop') else None),
                   bounds_std=(float(torch.as_tensor(std).flatten()[0]) if std is not None else None))
        if ht is not None:
            rec['top_err'] = mt - ht
            rec['abs_top_err'] = abs(mt - ht)
        m = band_metrics(gray, x0, x1, mt)
        rec.update({f'band_{k2}': round(v, 4) for k2, v in m.items()})
        if ht is not None:
            mh = band_metrics(gray, x0, x1, ht)
            rec.update({f'humanband_{k2}': round(v, 4) for k2, v in mh.items()})
        fe = first_echo_band(gray, x0, x1, y0, y1)
        rec['first_echo_top'] = fe
        if fe is not None and ht is not None:
            rec['echo_vs_human'] = fe - ht
        out.append(rec)
        if (k+1) % 200 == 0:
            print(f'  {k+1}/{len(rows)}  {time.time()-t0:.0f}s', flush=True)

    with open(OUT/'scan.jsonl', 'w', encoding='utf-8') as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'\n完成 {len(out)} 张，{time.time()-t0:.0f}s -> {OUT/"scan.jsonl"}')

    # --- ranking ---
    with_lbl = [r for r in out if r.get('abs_top_err') is not None]
    with_lbl.sort(key=lambda r: -r['abs_top_err'])
    print(f'\n=== 误差最大的 20 张（全量 {len(with_lbl)} 张中）===')
    for r in with_lbl[:20]:
        print(f"  {r['abs_top_err']:5d}px  split={r['split']:5s} {str(r['author'] or '-'):4s} "
              f"band_peak={r['band_peak']:.4f} snr={r['band_snr']:.2f} echo_vs_human={r.get('echo_vs_human')}")

    import statistics
    errs = sorted(r['abs_top_err'] for r in with_lbl)
    print(f'\n全量 |上边界误差|: 中位={errs[len(errs)//2]:.0f}px  p90={errs[int(len(errs)*.9)]:.0f}px  最大={errs[-1]}px')
    for th in (20, 30, 40, 60):
        n = sum(1 for e in errs if e >= th)
        print(f'  >= {th}px : {n} 张 ({100*n/len(errs):.1f}%)')

    # first_echo criterion accuracy
    fe_valid = [r for r in out if r.get('echo_vs_human') is not None]
    if fe_valid:
        d = sorted(abs(r['echo_vs_human']) for r in fe_valid)
        dm = sorted(r['abs_top_err'] for r in fe_valid)
        print(f'\n=== 「首个高回声带」判据 vs 人工 ===')
        print(f'  可计算 {len(fe_valid)} 张')
        print(f'  高回声带 |误差| 中位={d[len(d)//2]:.0f}px  p90={d[int(len(d)*.9)]:.0f}px')
        print(f'  同批模型   |误差| 中位={dm[len(dm)//2]:.0f}px  p90={dm[int(len(dm)*.9)]:.0f}px')
        better = sum(1 for r in fe_valid if abs(r['echo_vs_human']) < r['abs_top_err'])
        print(f'  高回声带比模型更接近人工: {better}/{len(fe_valid)} ({100*better/len(fe_valid):.1f}%)')


if __name__ == '__main__':
    main()