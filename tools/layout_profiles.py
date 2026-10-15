"""Experimental layout matching, trained only from the experiment's train split.

Profiles are candidates, not verified device identities or automatic approvals.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from crop_geometry import validate_window
from window_lab import ROOT, read, write, write_rows, metrics

GRID = (64, 48)
BOX_TOLERANCE = 8
MIN_SUPPORT = 3
MAX_DISTANCE = .12
AMBIGUITY_GAP = .02


def fingerprint(image):
    gray = image.convert('L').resize(GRID, Image.Resampling.BOX)
    return np.asarray(gray, dtype=np.float32)/255


def outside_mask(box, size):
    width, height = size
    x, y = np.meshgrid((np.arange(GRID[0])+.5)*width/GRID[0],
                       (np.arange(GRID[1])+.5)*height/GRID[1])
    return (x < box[0]) | (x >= box[2]) | (y < box[1]) | (y >= box[3])


def fit_profiles(samples):
    groups = []
    for sample in samples:
        image = sample['image']
        box = validate_window(sample['window'], image.size)
        match = next((g for g in groups if g['size'] == list(image.size) and
                      np.max(np.ptp(np.array(g['boxes']+[box]), axis=0)) <= BOX_TOLERANCE*2), None)
        item = {'box': box, 'feature': fingerprint(image)}
        if match is None:
            groups.append({'size': list(image.size), 'boxes': [box], 'items': [item]})
        else:
            match['boxes'] = match['boxes'] + [box]
            match['items'] = match['items'] + [item]
    profiles = []
    for group in groups:
        if len(group['items']) < MIN_SUPPORT:
            continue
        box = np.rint(np.median(group['boxes'], axis=0)).astype(int).tolist()
        mask = outside_mask(box, group['size'])
        if mask.mean() < .05:
            continue
        reference = np.median([item['feature'] for item in group['items']], axis=0)
        profiles.append({'id': f'layout_{len(profiles)+1}', 'size': group['size'], 'box': box,
                         'support': len(group['items']), 'reference': reference.tolist()})
    return {'version': 1, 'max_distance': MAX_DISTANCE, 'ambiguity_gap': AMBIGUITY_GAP,
            'limitation': 'UI-only appearance matching can miss changes inside the window; candidates require review.',
            'profiles': profiles}


def route(image, bank):
    feature = fingerprint(image)
    candidates = []
    for profile in bank['profiles']:
        if profile['size'] != list(image.size):
            continue
        mask = outside_mask(profile['box'], image.size)
        if (feature[~mask] > .02).mean() < .03:
            continue
        distance = float(np.abs(feature-np.array(profile['reference']))[mask].mean())
        candidates.append(dict(profile_id=profile['id'], box=profile['box'], distance=distance,
                               support=profile['support']))
    candidates.sort(key=lambda c: c['distance'])
    unknown = {'route': 'unknown_layout', 'box': None, 'review_required': True}
    if not candidates or candidates[0]['distance'] > bank['max_distance']:
        return dict(unknown, reason='unsupported_size_or_ui_mismatch', candidates=candidates[:2])
    best = candidates[0]
    for other in candidates[1:]:
        if other['distance']-best['distance'] > bank['ambiguity_gap']:
            break
        if max(abs(a-b) for a, b in zip(best['box'], other['box'])) > BOX_TOLERANCE:
            return dict(unknown, reason='ambiguous_profiles', candidates=candidates[:2])
    return dict(best, route='known_layout_candidate', review_required=True)


def run(out):
    rows = read(out/'manifest.jsonl')
    samples = []
    for r in rows:
        if r['split'] == 'train':
            with Image.open(ROOT/r['path']) as im:
                samples.append({'image': im.convert('RGB'), 'window': r['window']})
    bank = fit_profiles(samples)
    write(out/'layout_profiles.json', bank)
    results = []
    for r in rows:
        with Image.open(ROOT/r['path']) as im:
            prediction = route(im, bank)
        results.append(dict(r, **prediction))
    summary = {'profiles': [{k: v for k, v in p.items() if k != 'reference'} for p in bank['profiles']]}
    for ds in ('old', 'recent'):
        for split in ('train', 'val', 'test'):
            subset = [r for r in results if r['dataset'] == ds and r['split'] == split]
            accepted = [r for r in subset if r['box']]
            summary[ds+'/'+split] = dict(metrics(subset), accepted=len(accepted),
                                          coverage=len(accepted)/len(subset) if subset else None,
                                          accepted_metrics=metrics(accepted))
    write_rows(out/'layout_predictions.jsonl', results)
    write(out/'layout_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/experiments/window_corrected_20261013')
    run(ROOT/parser.parse_args().out)
