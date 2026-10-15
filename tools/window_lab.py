"""Local window experiments. Keep all outputs separate from production labels."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from detect_window import detect


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def grouped_split(rows, pairs, seed=20261013):
    ids = {r['id'] for r in rows}
    parent = {rid: rid for rid in ids}

    def find(rid):
        while parent[rid] != rid:
            rid = parent[rid]
        return rid

    for pair in pairs:
        if pair['a'] in ids and pair['b'] in ids:
            parent[find(pair['b'])] = find(pair['a'])
    groups = sorted({find(rid) for rid in ids})
    random.Random(seed).shuffle(groups)
    a, b = int(len(groups) * .6), int(len(groups) * .8)
    assigned = {g: 'train' if i < a else 'val' if i < b else 'test'
                for i, g in enumerate(groups)}
    return [dict(r, leak_group='NEW_' + find(r['id']), split=assigned[find(r['id'])]) for r in rows]


def prepare(out, corrected_old=False):
    old = read(ROOT / 'data/reports_v2/split.jsonl')
    old = [dict(r, path='data/' + r['path'], dataset='old',
                label_status='legacy_window_reference') for r in old]
    if corrected_old:
        corrected = {r['id']: r for r in read(ROOT / 'data/train_ready_v2.jsonl')}
        old = [dict(r, window=corrected[r['id']]['window'],
                    label_status='corrected_algorithm_reference') for r in old]
    recent = read(ROOT / 'data/分配20261009.jsonl')
    recent = [dict(r, dataset='recent', label_status='human_tool_confirmed')
              for r in recent if r.get('crop_source') == 'human_tool']
    pairs = json.loads((ROOT / 'data/reports/分配20261009_近重复.json').read_text(encoding='utf-8'))['within']
    exact = {}
    for r in recent:
        with Image.open(ROOT / r['path']) as im:
            im = im.convert('RGB')
            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        if digest in exact:
            pairs.append({'a': exact[digest], 'b': r['id']})
        exact[digest] = r['id']
    recent = grouped_split(recent, pairs)
    rows = old + recent
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs in experiment manifest')
    write_rows(out / 'manifest.jsonl', rows)
    write(out / 'protocol.json', {
        'seed': 20261013, 'corrected_old': corrected_old,
        'counts': dict(Counter(r['dataset'] + '/' + r['split'] for r in rows)),
        'limitations': ['Exploratory, not patient independent.',
                        'Old windows include template/algorithm-derived references.',
                        'Recent windows were confirmed with a prefilled tool; not a blind annotation.',
                        'Known near pairs and exact duplicates grouped; undetected related images may remain.',
                        'review_new_v4 excluded because its windows were refreshed by the current rule algorithm.'],
    })
    print(json.dumps(json.loads((out / 'protocol.json').read_text(encoding='utf-8')), ensure_ascii=False))


def metrics(rows):
    valid = [r for r in rows if r.get('box')]
    if not rows:
        return {'n': 0}
    errors = np.array([np.subtract(r['box'], r['window']) for r in valid])
    maximum = np.abs(errors).max(1) if len(valid) else np.array([])
    inward = np.maximum(errors * np.array([1, 1, -1, -1]), 0).max(1) if len(valid) else np.array([])
    return {'n': len(rows), 'detected': len(valid),
            'within4': float((maximum <= 4).sum() / len(rows)),
            'within8': float((maximum <= 8).sum() / len(rows)),
            'within15': float((maximum <= 15).sum() / len(rows)),
            'inward_over30': int((inward > 30).sum()),
            'max_error_p95_detected': float(np.percentile(maximum, 95)) if len(valid) else None,
            'edge_mae_detected': np.abs(errors).mean(0).tolist() if len(valid) else None}


def render(rows, out):
    worst = sorted(rows, key=lambda r: max(abs(a-b) for a, b in zip(r['box'], r['window']))
                   if r.get('box') else 100000, reverse=True)[:12]
    sheet = Image.new('RGB', (1200, 4 * 340), '#151a24')
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(worst):
        im = Image.open(ROOT / r['path']).convert('RGB')
        s = min(390 / im.width, 295 / im.height)
        x, y = i % 3 * 400, i // 3 * 340
        sheet.paste(im.resize((round(im.width*s), round(im.height*s))), (x, y+30))
        for box, color in [(r['window'], '#00ff88'), (r.get('box'), '#ff5555')]:
            if box:
                draw.rectangle([x+box[0]*s, y+30+box[1]*s, x+box[2]*s, y+30+box[3]*s], outline=color, width=2)
        err = max(abs(a-b) for a, b in zip(r['box'], r['window'])) if r.get('box') else None
        draw.text((x+5, y+6), f'{i+1} {r["dataset"]}/{r["split"]} max error={err}px', fill='white')
    sheet.save(out)


def evaluate(out, checkpoint=None):
    rows = read(out / 'manifest.jsonl')
    tag = Path(checkpoint).stem if checkpoint else 'rules'
    model, torch, core = None, None, None
    if checkpoint:
        import torch
        import ddh_v2 as core
        model, _ = core.load_checkpoint(checkpoint, 'cuda' if torch.cuda.is_available() else 'cpu')
        device = next(model.parameters()).device
    results = []
    for i, r in enumerate(rows):
        if checkpoint:
            from torchvision.transforms.functional import to_tensor
            with Image.open(ROOT / r['path']) as im:
                im = im.convert('RGB')
                with torch.no_grad():
                    prediction = model([to_tensor(im).to(device)])[0]
                box, reason = core.select_window(prediction, *im.size)
            d = {'box': box, 'flags': [reason] if reason else []}
        else:
            d = detect(ROOT / r['path'])
        results.append(dict(r, box=d['box'], flags=d['flags']))
        if (i+1) % 100 == 0:
            print(f'{tag}: {i+1}/{len(rows)}', flush=True)
    summary = {ds+'/'+sp: metrics([r for r in results if r['dataset'] == ds and r['split'] == sp])
               for ds in ('old', 'recent') for sp in ('train', 'val', 'calib', 'test')}
    write(out / (tag + '_summary.json'), summary)
    write_rows(out / (tag + '_predictions.jsonl'), results)
    render([r for r in results if r['dataset'] == 'recent'], out / (tag + '_worst.png'))
    print(json.dumps(summary, indent=2), flush=True)


def validation_objective(model, rows, core, torch):
    from torchvision.ops import box_iou
    model.eval()
    per_dataset = {'old': [], 'recent': []}
    with torch.no_grad():
        for r in rows:
            im, target = core.WindowData([r], ROOT)[0]
            o = model([im.cuda()])[0]
            box, _ = core.select_window(o, im.shape[2], im.shape[1])
            score = float(box_iou(torch.tensor([box], dtype=torch.float32), target['boxes'])[0, 0]) if box else 0
            per_dataset[r['dataset']].append(1-score)
    return {k: float(np.mean(v)) for k, v in per_dataset.items()}


def train(out, epochs=6):
    import time
    import torch
    import ddh_v2 as core
    from torch.utils.data import DataLoader
    core.seed_all(20261013)
    torch.set_num_threads(4)
    rows = read(out / 'manifest.jsonl')
    core.validate(rows, ROOT, True)
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    tr = tr + [r for r in tr if r['dataset'] == 'recent'] * 3
    model, meta = core.load_checkpoint(ROOT / 'runs/window.pt', 'cuda')
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    dl = DataLoader(core.WindowData(tr, ROOT, True), batch_size=2, shuffle=True,
                    collate_fn=core.collate_detection)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=1e-4)
    best, history, start = float('inf'), [], time.time()
    for epoch in range(epochs):
        model.train()
        model.backbone.eval()
        loss_sum = 0
        for i, (images, targets) in enumerate(dl):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model([x.cuda() for x in images], [core.moved(t, 'cuda') for t in targets]).values())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach())
            if (i+1) % 100 == 0:
                print(f'epoch {epoch+1}: {i+1}/{len(dl)} batches, elapsed {time.time()-start:.0f}s', flush=True)
        per_dataset = validation_objective(model, va, core, torch)
        objective = float(np.mean(list(per_dataset.values())))
        rec = {'epoch': epoch+1, 'loss': loss_sum/len(dl), 'balanced_val_loss': objective,
               'val_by_dataset': per_dataset,
               'elapsed_seconds': round(time.time()-start, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if objective < best:
            best = objective
            checkpoint = dict(meta, state=model.state_dict(), epoch=epoch+1, seed=20261013,
                              manifest_sha256=hashlib.sha256((out/'manifest.jsonl').read_bytes()).hexdigest(),
                              experiment='frozen backbone; recent samples oversampled 4x; balanced validation')
            torch.save(checkpoint, out / 'window_candidate.pt')
        write(out / 'training_history.json', history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'evaluate', 'train'])
    parser.add_argument('--out', default='data/experiments/window_20261013')
    parser.add_argument('--checkpoint')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--corrected-old', action='store_true')
    args = parser.parse_args()
    out = ROOT / args.out
    if args.action == 'prepare':
        prepare(out, args.corrected_old)
    elif args.action == 'train':
        train(out, args.epochs)
    else:
        evaluate(out, args.checkpoint)


if __name__ == '__main__':
    main()
