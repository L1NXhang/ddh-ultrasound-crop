import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import ddh_v2 as core

MAX_EPOCHS = 30
PATIENCE = 6


def validation(model, loaders, device):
    model.eval()
    result = {}
    with torch.no_grad():
        for cohort, loader in loaders.items():
            total, count = 0., 0
            for batch in loader:
                batch = core.moved(batch, device)
                loss = core.loss_crop(model(batch['image'], batch['roi']), batch,
                                      delta=.04, safe_mode='derived')
                total += float(loss) * len(batch['image'])
                count += len(batch['image'])
            result[cohort] = total / count
    return result, (result['old'] + result['new']) / 2


def loaders(rows):
    train = [r for r in rows if r['split'] == 'train']
    counts = collections.Counter((r['cohort'], r['leak_group']) for r in train)
    groups = {c: len({r['leak_group'] for r in train if r['cohort'] == c}) for c in ['old', 'new']}
    weights = [1 / (counts[r['cohort'], r['leak_group']] * groups[r['cohort']]) for r in train]
    sampler = WeightedRandomSampler(weights, len(train), replacement=True)
    training = DataLoader(core.CropData(train, ROOT, True, 384), batch_size=8,
                          sampler=sampler, num_workers=0)
    val = {c: DataLoader(core.CropData([r for r in rows if r['split'] == 'val' and r['cohort'] == c],
                                       ROOT, False, 384),
                         batch_size=8, shuffle=False, num_workers=0) for c in ['old', 'new']}
    return training, val


def epoch_step(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = core.moved(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = core.loss_crop(model(batch['image'], batch['roi']), batch,
                              delta=.04, safe_mode='derived')
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


def save(model, parent, protocol, epoch):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checkpoint = {'state': state, 'stage': 'crop', 'size': 384,
                  'baseline': parent['baseline'], 'quality_trained': False, 'safe_trained': False,
                  'seed': protocol['seed'], 'epoch': epoch,
                  'parent_sha256': protocol['parent_sha256'],
                  'manifest_sha256': protocol['manifest_sha256'],
                  'reviewers': ['张林', '成员1', '成员3'],
                  'label_semantics': 'final_crop_box', 'crop_policy': 'final_bounds/v1', 'margin': 0.0,
                  'training_method': 'warm_start_merged_members1_2_3_20261022',
                  'schema': 'ddh_crop_candidate/v1'}
    torch.save(checkpoint, OUT / 'crop_candidate.pt')


def fit(model, parent, protocol, train, val, device):
    backbone = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
    heads = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 3e-6}, {'params': heads, 'lr': 3e-5}],
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)
    baseline, baseline_score = validation(model, val, device)
    history, best, bad = [], float('inf'), 0
    (OUT / 'baseline_validation.json').write_text(
        json.dumps({'cohorts': baseline, 'score': baseline_score}), encoding='utf-8')
    print('BASELINE ' + json.dumps(baseline), flush=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.perf_counter()
        loss = epoch_step(model, train, optimizer, device)
        values, score = validation(model, val, device)
        scheduler.step()
        record = {'epoch': epoch, 'train_loss': loss, 'val_old': values['old'],
                  'val_new': values['new'], 'val_score': score,
                  'seconds': round(time.perf_counter() - start, 2)}
        history.append(record)
        if score < best:
            best, bad = score, 0
            save(model, parent, protocol, epoch)
        else:
            bad += 1
        (OUT / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
        if bad >= PATIENCE:
            print('EARLY STOP at epoch %d' % epoch, flush=True)
            break
    return {'baseline_validation': baseline, 'baseline_score': baseline_score,
            'best_score': best, 'epochs_run': len(history),
            'best_epoch': min(history, key=lambda r: r['val_score'])['epoch']}


def main():
    if (OUT / 'crop_candidate.pt').exists():
        raise ValueError('Candidate exists; use a new experiment rather than overwrite')
    torch.set_num_threads(4)
    core.seed_all(20261022)
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((OUT / 'split.jsonl').read_bytes()).hexdigest() == protocol['manifest_sha256'], \
        'split.jsonl changed after protocol freeze'
    assert hashlib.sha256((ROOT / 'runs_v2/crop.pt').read_bytes()).hexdigest() == protocol['parent_sha256'], \
        'production checkpoint changed'
    rows = core.read_rows(OUT / 'split.jsonl')
    core.validate(rows, ROOT, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, parent = core.load_checkpoint(ROOT / 'runs_v2/crop.pt', device)
    train, val = loaders(rows)
    print('DEVICE ' + device, flush=True)
    start = time.perf_counter()
    result = fit(model, parent, protocol, train, val, device)
    assert hashlib.sha256((ROOT / 'runs_v2/crop.pt').read_bytes()).hexdigest() == protocol['parent_sha256']
    result.update(device=device, elapsed_seconds=round(time.perf_counter() - start, 2),
                  production_checkpoint_unchanged=True,
                  candidate_sha256=hashlib.sha256((OUT / 'crop_candidate.pt').read_bytes()).hexdigest())
    (OUT / 'training_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('TRAINING COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()