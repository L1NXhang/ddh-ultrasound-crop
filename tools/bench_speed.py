"""在两阶段真实配置下实测训练速度，估算 1080 张数据的实际耗时。

用法：python tools/bench_speed.py --workdir D:/ddhmodel/bench_out --n 48 --device cuda
合成图片，只为测时间，不产生任何可用模型。
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ddh


def make_data(root, n):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        # 真实截图尺寸 1280x960，成像区大约在中间，模仿实际负载
        arr = np.zeros((960, 1280, 3), np.uint8)
        arr[136:744, 368:912] = rng.integers(0, 255, (608, 544, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / f'{i:04d}.jpg', quality=90)
        s = 'train' if i < n - 12 else 'val' if i < n - 6 else 'calib' if i < n - 3 else 'test'
        rows.append({'id': f'{i:04d}', 'path': f'{i:04d}.jpg', 'window': [368, 136, 912, 744],
                     'crop': [220, 660], 'safe_box': [380, 300, 900, 600], 'quality': 1,
                     'style_group': 'bench', 'source_group': 'bench', 'leak_group': f'G{i:04d}',
                     'split': s})
    ddh.write_rows(root / 'labels.jsonl', rows)
    return rows


def run_stage(stage, root, rows, device, size, batch, epochs=1):
    n_train = sum(r['split'] == 'train' for r in rows)
    args = NS(seed=42, manifest=str(root / 'labels.jsonl'), root=str(root), stage=stage,
              device=device, no_pretrained=True, baseline=False, size=size, batch=batch,
              epochs=epochs, patience=99, output=str(root / f'{stage}.pt'))
    torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
    t0 = time.time()
    ddh.training(args)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == 'cuda' else 0
    return {'stage': stage, '训练图数': n_train, '每轮秒(实测)': round(dt, 1),
            '每图秒': round(dt / n_train, 3),
            '峰值显存GB': round(peak, 2),
            '推算700张每轮分钟': round(dt / n_train * 700 / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    root = Path(args.workdir)
    rows = make_data(root / 'data', args.n)
    out = []
    out.append(run_stage('window', root / 'data', rows, args.device, 640, 2))
    out.append(run_stage('crop', root / 'data', rows, args.device, 384, 8))
    # 推理速度：单图端到端 + MC Dropout 16 次
    wm, wa = ddh.load_checkpoint(root / 'data' / 'window.pt', args.device)
    bm, ba = ddh.load_checkpoint(root / 'data' / 'crop.pt', args.device)
    row = rows[-1]
    t0 = time.time()
    for _ in range(5):
        ddh.pipeline(row, root / 'data', wm, bm, ba, args.device, samples=16)
    infer = (time.time() - t0) / 5
    print(json.dumps({'设备': args.device, 'GPU': torch.cuda.get_device_name(0) if args.device == 'cuda' else '-',
                      '训练': out, '单图推理秒(含MC16次)': round(infer, 3),
                      '推算1080张评估分钟': round(infer * 1080 / 60, 1)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
