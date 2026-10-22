import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import ddh
from detect_window import detect


def parameter_change(parent, candidate):
    squared_base, squared_delta, total, changed = 0., 0., 0, 0
    sections = collections.defaultdict(lambda: {'elements': 0, 'changed': 0, 'delta_squared': 0.})
    original, updated = parent.state_dict(), candidate.state_dict()
    for key in sorted(set(dict(parent.named_parameters()))):
        a, b = original[key].double().cpu(), updated[key].double().cpu()
        delta = b - a
        n = int((delta != 0).sum())
        base_sum, delta_sum = float(a.square().sum()), float(delta.square().sum())
        squared_base += base_sum
        squared_delta += delta_sum
        total += a.numel()
        changed += n
        section = sections[key.split('.')[0]]
        section['elements'] += a.numel()
        section['changed'] += n
        section['delta_squared'] += delta_sum
    return {'parameter_elements': total, 'changed_elements': changed,
            'changed_fraction': changed / total,
            'relative_l2_change': math.sqrt(squared_delta / squared_base),
            'sections': dict(sections)}


def predict(model, path, window, seed, device, margin=0.0):
    if window is None:
        return None
    ddh.seed_all(seed)
    with Image.open(path) as image:
        x, roi = ddh.letterbox(image.convert('RGB').crop(window), 384)
    result = ddh.mc_crop(model, x[None].to(device), roi[None].to(device), 4)
    bounds = result['bounds'][0].tolist()
    bounds = [max(0., bounds[0] - margin), min(1., bounds[1] + margin)]
    return ddh.raw_crop(window, bounds)


def measure(row, box):
    if row['cohort'] == 'new':
        target = row['clean_box']
    else:
        target = [row['window'][0], row['crop'][0], row['window'][2], row['crop'][1]]
    if box is None:
        return {'predicted': False, 'pass2': False, 'pass4': False}
    top, bottom = abs(box[1] - target[1]), abs(box[3] - target[3])
    height = row['window'][3] - row['window'][1]
    intersection = max(0, min(box[3], target[3]) - max(box[1], target[1]))
    union = box[3] - box[1] + target[3] - target[1] - intersection
    return {'predicted': True, 'box': box, 'top_mae_px': top, 'bottom_mae_px': bottom,
            'bottom_signed_px': box[3] - target[3], 'vertical_iou': intersection / max(1, union),
            'pass2': max(top, bottom) / height <= .02, 'pass4': max(top, bottom) / height <= .04}


def summarize(records):
    keys = ['top_mae_px', 'bottom_mae_px', 'bottom_signed_px', 'vertical_iou', 'pass2', 'pass4']
    out = {'n': len(records), 'predicted': sum(r['predicted'] for r in records)}
    for k in keys:
        vals = [r[k] for r in records if k in r]
        out[k] = statistics.mean(vals) if vals else None
    return out


def bootstrap_pairs(rows, metric):
    groups = collections.defaultdict(list)
    for row in rows:
        if row['parent']['predicted'] and row['candidate']['predicted']:
            groups[row['group']].append(row['candidate'][metric] - row['parent'][metric])
    keys = sorted(groups)
    if not keys:
        return None
    rng = np.random.default_rng(20261022)
    samples = [np.mean([v for i in rng.integers(0, len(keys), len(keys))
                        for v in groups[keys[i]]]) for _ in range(2000)]
    return {'metric': 'candidate_minus_parent_' + metric, 'groups': len(keys),
            'paired_group_bootstrap_95_percent_interval':
                np.quantile(samples, [.025, .975]).tolist(),
            'caveat': 'Exploratory interval; groups are not patient identities.'}


def evaluate(models, tests, device):
    results = []
    for i, row in enumerate(tests):
        path = ROOT / row['path']
        windows = {'reference_window': row['window'], 'production_rule': detect(path)['box']}
        for protocol, window in windows.items():
            record = {'id': row['id'], 'group': row['leak_group'], 'cohort': row['cohort'],
                      'author': row.get('annotation_author', 'legacy'),
                      'protocol': protocol, 'window': window,
                      'reference_crop': row['crop'], 'path': row['path']}
            for name, model in models.items():
                record[name] = measure(row, predict(model, path, window, 20261022 + i, device))
            results.append(record)
    return results


def build_report(results, models):
    report = {}
    for cohort in ['old', 'new']:
        for protocol in ['reference_window', 'production_rule']:
            selected = [r for r in results if r['cohort'] == cohort and r['protocol'] == protocol]
            if not selected:
                continue
            block = {name: summarize([r[name] for r in selected]) for name in models}
            block['uncertainty'] = {
                'bottom_mae': bootstrap_pairs(selected, 'bottom_mae_px'),
                'top_mae': bootstrap_pairs(selected, 'top_mae_px'),
            }
            report[cohort + '/' + protocol] = block
    # per-annotator breakdown on the new cohort, reference window
    by_author = {}
    for author in sorted({r['author'] for r in results if r['cohort'] == 'new'}):
        selected = [r for r in results if r['cohort'] == 'new'
                    and r['protocol'] == 'reference_window' and r['author'] == author]
        if selected:
            by_author[author] = {name: summarize([r[name] for r in selected]) for name in models}
    report['by_author/reference_window'] = by_author
    return report


def main():
    if not (OUT / 'training_summary.json').exists():
        raise ValueError('Training must finish before evaluation')
    if (OUT / 'comparison.json').exists():
        raise ValueError('Final evaluation exists; do not tune on test outcomes')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parent, _ = ddh.load_checkpoint(ROOT / 'runs_v2/crop.pt', device)
    candidate, _ = ddh.load_checkpoint(OUT / 'crop_candidate.pt', device)
    models = {'parent': parent, 'candidate': candidate}
    rows = ddh.read_rows(OUT / 'split.jsonl')
    tests = [r for r in rows if r['split'] == 'test']
    weights = parameter_change(parent, candidate)
    results = evaluate(models, tests, device)
    report = build_report(results, models)
    output = {'date': '2026-10-22', 'reviewers': ['张林', '成员1', '成员3'],
              'device': device,
              'same_inference_settings': {'samples': 4, 'margin': 0.0, 'seed': 20261022},
              'parameter_change': weights, 'metrics': report, 'predictions': results,
              'candidate_deployed': False,
              'protocol_sha256': hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest()}
    (OUT / 'comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'relative_l2_change': weights['relative_l2_change'],
                      'metrics': {k: {m: {kk: vv for kk, vv in v.items()
                                          if kk in ('n', 'top_mae_px', 'bottom_mae_px', 'vertical_iou', 'pass4')}
                                      for m, v in val.items() if m in models}
                                  for k, val in report.items() if 'uncertainty' not in k}},
                     ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()