"""四组同条件对照训练 + 评测入口（ResNet18/50 × 仅L3 / L1+L2+L3 融合）。

严格"同条件"：数据划分、预处理、增强、loss、优化器、调度、早停、选点口径、
batch、epoch、seed 全部来自 ddh_v2.py，本脚本只替换 backbone 与融合方式。

用法：
  python tools/stage_new/train_eval_v3.py --config r18_l3 --epochs 30 --out runs/<dir>/resnet18
  python tools/stage_new/train_eval_v3.py --sanity --epochs 1 --config r50_l3
"""
from __future__ import annotations

import argparse, json, sys, time, hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'tools' / 'stage_new'))

import ddh_v2  # noqa: E402  复用其数据、loss、letterbox、moved、seed、IO
import cropnet_v3 as v3  # noqa: E402


def build_model(cfg, pretrained=True):
    return v3.CropNetV3(cfg, pretrained)


def evaluate_regression(model, rows, root, device, size=384, batch=8):
    """回归口径主指标 + 3 类 softmax 分类口径（与部署同一套 raw_crop 换算）。"""
    ds = ddh_v2.CropData(rows, root, False, size)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    model.eval()
    preds, gts, probs3 = [], [], []
    with torch.no_grad():
        for batch in dl:
            b = ddh_v2.moved(batch, device)
            out = model(b['image'], b['roi'])
            preds.append(out['bounds'].cpu())
            probs3.append(out['head_probs'].cpu() if 'head_probs' in out else out['bounds'].cpu())
            gts.append(b['bounds'].cpu())
    pred = torch.cat(preds).numpy()
    gt = torch.cat(gts).numpy()
    # 像素误差：bounds 是 0..1（相对窗口高度），乘窗口高度还原像素
    err = np.abs(pred - gt)
    for i, r in enumerate(rows):
        y0, y1 = r['window'][1], r['window'][3]
        h = y1 - y0
        top_err = err[i, 0] * h
        bot_err = err[i, 1] * h
        rec = {'id': r['id'], 'split': r['split'], 'group': r['collect_group'],
               # 保存原始坐标（归一化），供案例可视化复现预测线位置
               'pred_bounds': [float(pred[i, 0]), float(pred[i, 1])],
               'gt_bounds': [float(gt[i, 0]), float(gt[i, 1])],
               'window': list(r['window']),
               'top_err_px': float(top_err), 'bot_err_px': float(bot_err),
               'top_err_norm': float(err[i, 0]), 'bot_err_norm': float(err[i, 1]),
               'max_err_norm': float(err[i].max())}
        for tol in (0.02, 0.03, 0.05):
            rec[f'pass_{int(tol * 100)}pct'] = bool(err[i].max() <= tol)
        inter = max(0.0, min(pred[i, 1], gt[i, 1]) - max(pred[i, 0], gt[i, 0]))
        union = (pred[i, 1] - pred[i, 0]) + (gt[i, 1] - gt[i, 0]) - inter
        rec['vert_iou'] = float(inter / max(union, 1e-8))
        yield rec


def summarize(records):
    def med(v):
        return float(np.median(v)) if len(v) else None
    top = [r['top_err_px'] for r in records]
    bot = [r['bot_err_px'] for r in records]
    out = {
        'n': len(records),
        'top_err_px_mean': float(np.mean(top)) if top else None,
        'top_err_px_median': med(top),
        'bot_err_px_mean': float(np.mean(bot)) if bot else None,
        'bot_err_px_median': med(bot),
        'vert_iou_mean': float(np.mean([r['vert_iou'] for r in records])) if records else None,
        'vert_iou_median': med([r['vert_iou'] for r in records]),
    }
    for tol in (2, 3, 5):
        k = f'pass_{tol}pct'
        out[f'pass_{tol}pct'] = float(np.mean([r[k] for r in records])) if records else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(v3.CONFIGS))
    ap.add_argument('--manifest', default='data/reports/split.jsonl')
    ap.add_argument('--root', default='data')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--sanity', action='store_true', help='只跑 --epochs 轮做连通性检查')
    ap.add_argument('--max-train-batches', type=int, default=0, help='sanity 时限制每轮 batch 数')
    args = ap.parse_args()

    ddh_v2.seed_all(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'
    log = open(log_path, 'a', encoding='utf-8')

    def say(obj):
        line = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        print(line, flush=True); log.write(line + '\n'); log.flush()

    rows = ddh_v2.read_rows(args.manifest)
    ddh_v2.validate(rows, args.root, True)
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    if not tr or not va:
        raise ValueError('need train and val rows')
    dev = torch.device(args.device)

    model = build_model(args.config, not args.no_pretrained).to(dev)
    breakdown = v3.param_breakdown(model)
    meta = {'config': args.config, 'backbone': breakdown['backbone_name'], 'fusion': breakdown['fusion'],
            'params': breakdown, 'epochs': args.epochs, 'patience': args.patience, 'batch': args.batch,
            'size': args.size, 'seed': args.seed, 'device': str(dev), 'pretrained': not args.no_pretrained,
            'manifest': args.manifest,
            'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            'n_train': len(tr), 'n_val': len(va),
            'n_test': sum(1 for r in rows if r['split'] == 'test'),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'sanity': args.sanity}
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    say({'event': 'start', **meta})

    ds = ddh_v2.CropData(tr, args.root, True, args.size)
    vds = ddh_v2.CropData(va, args.root, False, args.size)
    counts = Counter(r['leak_group'] for r in tr)
    sampler = WeightedRandomSampler([1 / counts[r['leak_group']] for r in tr], len(tr), replacement=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    vl = DataLoader(vds, batch_size=args.batch, shuffle=False)

    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 3e-5},
                             {'params': head, 'lr': 3e-4}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(args.epochs, 1))

    best, bad, history = float('inf'), 0, []
    t_train0 = time.time()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0; nb = 0
        t0 = time.time()
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            b = ddh_v2.moved(batch, dev)
            o = model(b['image'], b['roi'])
            loss = ddh_v2.loss_crop(o, b, baseline=False, delta=0.04, w_in=2.0, w_out=0.25,
                                    safe_mode='derived')
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            train_loss += float(loss.detach()); nb += 1
            if args.max_train_batches and nb >= args.max_train_batches:
                break
        epoch_sec = time.time() - t0
        model.eval(); vals = []
        with torch.no_grad():
            for batch in vl:
                b = ddh_v2.moved(batch, dev)
                o = model(b['image'], b['roi'])
                vals.append((float(ddh_v2.loss_crop(o, b, baseline=False, delta=0.04, w_in=2.0,
                                                    w_out=0.25, safe_mode='derived')), len(b['image'])))
        score = sum(v * n for v, n in vals) / max(1, sum(n for _, n in vals))
        sched.step()
        rec = {'epoch': epoch + 1, 'train_loss': train_loss / max(1, nb), 'val_objective': score,
               'epoch_sec': round(epoch_sec, 2)}
        history.append(rec); say({'event': 'epoch', **rec})
        if score < best:
            best = score; bad = 0
            torch.save({'state': model.state_dict(), 'stage': 'crop', 'size': args.size,
                        'baseline': False, 'config': args.config,
                        'backbone': breakdown['backbone_name'], 'fusion': breakdown['fusion'],
                        'seed': args.seed, 'epoch': epoch + 1,
                        'manifest_sha256': meta['manifest_sha256']}, out_dir / 'crop.pt')
        else:
            bad += 1
        if bad >= args.patience:
            say({'event': 'early_stop', 'epoch': epoch + 1}); break
    train_sec = time.time() - t_train0
    peak_mem_mb = (torch.cuda.max_memory_allocated(dev) / 1024 ** 2) if dev.type == 'cuda' else None
    (out_dir / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')

    # ---- 评测（test 集固定）----
    st = torch.load(out_dir / 'crop.pt', map_location=dev, weights_only=False)
    model.load_state_dict(st['state']); model.eval()
    test_rows = [r for r in rows if r['split'] == 'test']
    recs = list(evaluate_regression(model, test_rows, args.root, dev, args.size, args.batch))
    with open(out_dir / 'test_per_image.jsonl', 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    val_recs = list(evaluate_regression(model, va, args.root, dev, args.size, args.batch))

    ckpt_bytes = (out_dir / 'crop.pt').stat().st_size
    metrics = {
        'config': args.config, 'backbone': breakdown['backbone_name'], 'fusion': breakdown['fusion'],
        'params_total': breakdown['total'], 'params_backbone': breakdown['backbone'],
        'params_head': breakdown['head'], 'checkpoint_bytes': ckpt_bytes,
        'checkpoint_mb': round(ckpt_bytes / 1024 ** 2, 2),
        'train_time_sec': round(train_sec, 1), 'epochs_run': len(history),
        'best_epoch': int(st['epoch']), 'best_val_objective': float(best),
        'peak_gpu_mem_mb': round(peak_mem_mb, 1) if peak_mem_mb else None,
        'test': summarize(recs), 'val': summarize(val_recs),
        'n_test': len(test_rows),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    say({'event': 'done', **metrics})
    log.close()


if __name__ == '__main__':
    main()
