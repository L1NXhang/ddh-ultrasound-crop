"""E 项：三组样本 × 每组 50 张 —— ResNet18 vs ResNet50 **推理成本**对比。

重要口径（已与负责人确认）：
  这 250 张取自 data/images_分类视图/未使用（未标注）/，**没有任何人工标注**
  （无 window、无 crop）。因此本脚本**只测推理成本**（时间/显存/模型大小/参数量），
  **不计算、也不输出任何准确率类指标** —— 没有标准答案就没有准确率，编一个就是伪造。

三组划分：对未标注目录按文件名排序后用固定步长抽样 150 张，均分三组（A/B/C，各 50 张），
这样"三组"是同一批图上的三次测量，组间差异只反映图片本身（尺寸/内容）的波动。
抽样清单写入 reports/bench_groups.json，可复现。

用法（需要 torch，必须用 .venv 的 python）：
  .venv/Scripts/python.exe tools/stage_new/bench_inference.py
  .venv/Scripts/python.exe tools/stage_new/bench_inference.py --configs r18_l3,r50_l3
"""
from __future__ import annotations

import argparse, csv, json, statistics, sys, time
from pathlib import Path

import torch
from PIL import Image

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'tools' / 'stage_new'))

import ddh_v2  # noqa: E402
import cropnet_v3 as v3  # noqa: E402

UNUSED = PROJ / 'data' / 'images_分类视图' / '未使用（未标注）'
GROUP_SIZE = 50
N_GROUPS = 3
SIZE = 384


def pick_groups():
    files = sorted(p for p in UNUSED.glob('*.jpg'))
    need = GROUP_SIZE * N_GROUPS
    if len(files) < need:
        raise SystemExit(f'需要 {need} 张未标注图，实际只有 {len(files)} 张')
    # 用固定步长取样，避免只取同一批连续帧
    step = max(1, len(files) // need)
    picked = [files[i * step] for i in range(need)]
    return {g: picked[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
            for i, g in enumerate(['A', 'B', 'C'])}


def bench(model, files, device, warmup=10):
    """返回单张前向耗时统计（不含图像解码，含 letterbox 变换）。

    注意：笔记本 GPU 可能被系统进程抢占，导致个别样本出现数十倍离群值。
    因此 warmup 加大到 10，并以 **中位数** 为主指标，均值/标准差仅作参考。
    """
    tensors = []
    for p in files:
        im = Image.open(p).convert('RGB')
        x, roi = ddh_v2.letterbox(im, SIZE, False)
        tensors.append((x, roi))
    model.eval()
    times = []
    with torch.no_grad():
        # 预热：先用前 warmup 张跑一遍，不进统计
        for x, roi in tensors[:warmup]:
            model(x[None].to(device), roi[None].to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        for x, roi in tensors:
            xb = x[None].to(device); rb = roi[None].to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(xb, rb)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default=','.join(v3.CONFIGS))
    ap.add_argument('--ckpt-root', default='runs/20261103_resnet18_vs_50')
    ap.add_argument('--out', default='runs/20261103_resnet18_vs_50/reports')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    groups = pick_groups()
    manifest = {g: [{'id': p.stem, 'path': str(p.relative_to(PROJ))} for p in fs]
                for g, fs in groups.items()}
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'bench_groups.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = []
    for cfg in args.configs.split(','):
        sub = 'resnet18' if cfg.startswith('r18') else 'resnet50'
        ck = Path(args.ckpt_root) / sub / cfg / 'crop.pt'
        if not ck.exists():
            print(f'[skip] {cfg}: 找不到 checkpoint {ck}')
            continue
        state = torch.load(ck, map_location=device, weights_only=False)
        model = v3.CropNetV3(state['config'], pretrained=False).to(device)
        model.load_state_dict(state['state']); model.eval()
        ck_mb = ck.stat().st_size / 1024 ** 2
        pb = v3.param_breakdown(model)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for g, files in groups.items():
            # 重复 3 轮，取「最小的中位数」——桌面应用可能抢占 GPU，
            # 最小值最接近 GPU 独占时的真实性能；同时报告离散度以便读者判断可信度。
            runs = [bench(model, files, device) for _ in range(3)]
            meds = [statistics.median(t) for t in runs]
            times = runs[meds.index(min(meds))]
            rows.append({
                'config': cfg, 'backbone': pb['backbone_name'], 'fusion': pb['fusion'],
                'group': g, 'n_images': len(files), 'device': str(device),
                'params_total': pb['total'], 'checkpoint_mb': round(ck_mb, 2),
                'infer_ms_mean': round(statistics.mean(times), 2),
                'infer_ms_median': round(statistics.median(times), 2),
                'infer_ms_p95': round(sorted(times)[int(len(times) * .95)], 2),
                'infer_ms_std': round(statistics.pstdev(times), 2),
                'median_across_3_runs': [round(m, 2) for m in meds],
                'median_spread_pct': round((max(meds) - min(meds)) / min(meds) * 100, 1),
                # accuracy 字段刻意留空：这些图没有标注，不产生准确率
                'accuracy': None,
                'accuracy_note': '未标注图片，不计算准确率',
            })
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else None
        for r in rows:
            if r['config'] == cfg:
                r['peak_infer_mem_mb'] = round(peak, 1) if peak else None
        del model
        torch.cuda.empty_cache()
        print(f'[done] {cfg}')

    import csv
    if rows:
        with open(outdir / 'bench_inference_3groups.csv', 'w', newline='', encoding='utf-8-sig') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
    (outdir / 'bench_inference_3groups.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=1)[:2000])


if __name__ == '__main__':
    main()
