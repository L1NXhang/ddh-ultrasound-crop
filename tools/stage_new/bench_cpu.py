"""E 项补充：CPU 推理时间测量（GPU 被桌面应用抢占时，GPU 计时不可信）。

为什么用 CPU：
  - 本机桌面应用（ChatGPT / Edge / Explorer 等）持续占用 GPU，导致 GPU 计时
    离散度高达 214%，出现"融合模型比基线快 10 倍"这类不可能的结果。
  - 项目本身提供 **CPU 安装包**（README：Windows CPU 安装包，推荐），
    因此 CPU 推理时间是有真实部署意义的指标。
  - CPU 上没有被抢占问题，测量稳定、可复现。

用法：
  .venv/Scripts/python.exe tools/stage_new/bench_cpu.py
"""
from __future__ import annotations

import csv, json, statistics, sys, time
from pathlib import Path

import torch
from PIL import Image

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'tools' / 'stage_new'))

import ddh_v2  # noqa: E402
import cropnet_v3 as v3  # noqa: E402

# 运行目录可通过 argv 指定（默认 corrected 版本）
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (PROJ / 'runs' / '20261103_resnet18_vs_50_corrected')
if not RUN.is_absolute():
    RUN = PROJ / RUN
SIZE = 384
# 样本清单可能继承自旧目录（未标注图与标签无关，可复用同一批 150 张）
GROUPS_JSON_CANDIDATES = [RUN / 'reports' / 'bench_groups.json',
                          PROJ / 'runs' / '20261103_resnet18_vs_50' / 'reports' / 'bench_groups.json'
                          ]
CONFIGS = ['r18_l3', 'r50_l3', 'r18_l123', 'r50_l123', 'r18_l4', 'r18_l1234']


def load_groups():
    for p in GROUPS_JSON_CANDIDATES:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    raise SystemExit('找不到 bench_groups.json，请先运行 bench_inference.py')


def bench_cpu(model, files, warmup=5, repeats=3):
    tensors = []
    for f in files:
        im = Image.open(PROJ / f['path']).convert('RGB')
        x, roi = ddh_v2.letterbox(im, SIZE, False)
        tensors.append((x, roi))
    model.eval()
    # 预热
    with torch.no_grad():
        for x, roi in tensors[:warmup]:
            model(x[None], roi[None])
    all_times = []
    for _ in range(repeats):
        times = []
        with torch.no_grad():
            for x, roi in tensors:
                t0 = time.perf_counter()
                model(x[None], roi[None])
                times.append((time.perf_counter() - t0) * 1000)
        all_times.append(statistics.median(times))
    return min(all_times), all_times


def main():
    groups = load_groups()
    device = torch.device('cpu')
    rows = []
    for cfg in CONFIGS:
        sub = 'resnet18' if cfg.startswith('r18') else 'resnet50'
        ck = RUN / sub / cfg / 'crop.pt'
        if not ck.exists():
            print(f'[skip] {cfg}'); continue
        st = torch.load(ck, map_location=device, weights_only=False)
        model = v3.CropNetV3(st['config'], pretrained=False).to(device)
        model.load_state_dict(st['state']); model.eval()
        pb = v3.param_breakdown(model)
        for g, files in groups.items():
            best, allr = bench_cpu(model, files)
            rows.append({'config': cfg, 'backbone': pb['backbone_name'], 'fusion': pb['fusion'],
                         'group': g, 'n_images': len(files), 'device': 'cpu',
                         'params_total': pb['total'],
                         'checkpoint_mb': round(ck.stat().st_size / 1024 ** 2, 2),
                         'cpu_infer_ms_median': round(best, 2),
                         'cpu_runs': [round(x, 2) for x in allr],
                         'accuracy': None,
                         'accuracy_note': '未标注图片，不计算准确率'})
            print(f'[done] {cfg} {g}: {best:.1f} ms/img')
        del model
    out = RUN / 'reports'
    with open(out / 'bench_cpu_3groups.csv', 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    (out / 'bench_cpu_3groups.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('written bench_cpu_3groups.csv')


if __name__ == '__main__':
    main()
