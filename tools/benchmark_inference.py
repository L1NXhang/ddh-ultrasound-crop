"""Measure the production prediction path without exporting images or training labels."""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))


def summary(values):
    ordered = sorted(values)
    return {'mean_ms': round(statistics.mean(values)*1000, 2),
            'median_ms': round(statistics.median(values)*1000, 2),
            'p95_ms': round(ordered[min(len(ordered)-1, int(.95*len(ordered)))]*1000, 2),
            'samples': len(values)}


def run(args):
    started = time.perf_counter()
    import torch
    from studio_predictor import ModelRunner
    from detect_window import detect
    imported = time.perf_counter()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    sync = torch.cuda.synchronize if args.device == 'cuda' else lambda: None
    files = sorted(p for p in args.folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
    if not files:
        raise ValueError('No input images')
    count = min(args.count, len(files))
    selected = [files[round(i*(len(files)-1)/max(1,count-1))] for i in range(count)]
    runner = ModelRunner(args.checkpoint, device=args.device)
    runner.load()
    sync()
    loaded = time.perf_counter()
    first = runner.predict(selected[0])
    sync()
    first_done = time.perf_counter()
    if first['crop_box'] is None:
        raise ValueError('First image has no detected window')
    for _ in range(args.warmups):
        runner.predict(selected[0])
    sync()
    full, rule, network = [], [], []
    for repeat in range(args.repeats):
        for path in selected:
            sync()
            t0 = time.perf_counter()
            prediction = runner.predict(path)
            sync()
            full.append(time.perf_counter()-t0)
            if prediction['crop_box'] is None:
                raise ValueError('An image has no window; cannot compare like-for-like')
        print(f'Completed {args.device}: repeat {repeat+1}/{args.repeats}', flush=True)
    for path in selected:
        t0 = time.perf_counter()
        window = detect(path)['box']
        rule.append(time.perf_counter()-t0)
        sync()
        t0 = time.perf_counter()
        runner.predict(path, window)
        sync()
        network.append(time.perf_counter()-t0)
    report = {'schema': 'ddh_inference_benchmark/v1', 'device': args.device, 'torch': torch.__version__,
              'cpu_processor': platform.processor(), 'logical_cpus': os.cpu_count(),
              'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
              'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
              'input_images': count, 'image_folder': str(args.folder.resolve()), 'model': runner.metadata,
              'mc_draws': 4, 'total_network_forwards_per_image': 5,
              'import_ms': round((imported-started)*1000, 2),
              'model_load_ms': round((loaded-imported)*1000, 2),
              'first_prediction_ms': round((first_done-loaded)*1000, 2),
              'first_request_including_import_load_ms': round((first_done-started)*1000, 2),
              'prediction': summary(full), 'window_rules': summary(rule),
              'known_window_prediction': summary(network),
              'estimated_85_prediction_seconds': round(statistics.mean(full)*85, 2),
              'estimated_400_prediction_seconds': round(statistics.mean(full)*400, 2),
              'excludes': ['native UI startup', 'directory inventory', 'PNG export', 'human review time'],
              'outputs_or_training_labels_written': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs_v2/crop.pt')
    parser.add_argument('--device', choices=['cpu','cuda'], required=True)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
