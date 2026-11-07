#!/usr/bin/env bash
# ============================================================
# DDH 实验：ResNet18 vs ResNet50 同条件对照 + 多尺度融合消融
# 一键复现全部结果（不覆盖任何已有 runs/runs_v2 结果）
#
# 用法：
#   bash scripts/reproduce_all.sh              # 全流程
#   bash scripts/reproduce_all.sh --skip-train # 只重出图/报告（复用已有 checkpoint）
#
# 依赖：
#   训练/评测 -> .venv/Scripts/python.exe (torch 2.5.1+cu121)
#   出图/统计 -> 系统 python（需 matplotlib/pandas/sklearn），默认 D:/PYTHON/python
# ============================================================
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

PY_TORCH="./.venv/Scripts/python.exe"
PY_PLOT="${PY_PLOT:-/d/PYTHON/python}"
RUN="runs/20261103_resnet18_vs_50"
export PYTHONHASHSEED=42

log(){ echo "[$(date '+%F %T')] $*"; }

mkdir -p "$RUN"/{resnet18,resnet50,figures,reports,logs,configs,sanity}

# 固定超参（四组完全相同，唯一变量 = backbone + 融合方式）
COMMON="--manifest data/reports/split.jsonl --root data --epochs 30 --patience 6 --batch 8 --size 384 --seed 42"

if [[ "${1:-}" != "--skip-train" ]]; then
  # ---------- 0) sanity：1 epoch / 20 batch ----------
  log "sanity check r18_l3"
  $PY_TORCH tools/stage_new/train_eval_v3.py --config r18_l3 --sanity --epochs 1 \
      --max-train-batches 20 --out "$RUN/sanity/r18_l3"
  log "sanity check r50_l3"
  $PY_TORCH tools/stage_new/train_eval_v3.py --config r50_l3 --sanity --epochs 1 \
      --max-train-batches 20 --out "$RUN/sanity/r50_l3"

  # ---------- 1) 四组全量训练 ----------
  bash scripts/run_four_models.sh --skip-sanity
fi

# ---------- 2) 结构表 + 图（A 项，不需要 GPU） ----------
log "A: ResNet structure"
$PY_TORCH tools/stage_new/measure_structure.py > /dev/null
$PY_PLOT tools/stage_new/plot_structure.py

# ---------- 3) 汇总表 + 核心柱状图（B/C 项） ----------
log "B/C: summarize"
$PY_PLOT tools/stage_new/summarize_results.py

# ---------- 4) 50 张案例展示（D 项，需要 torch + PIL） ----------
log "D: cases for r18_l3"
$PY_TORCH tools/stage_new/make_cases.py --config r18_l3

# ---------- 5) 三组 × 50 张推理成本（E 项，需要 torch + GPU） ----------
log "E: inference cost (需要 GPU，请等训练结束后再跑)"
$PY_TORCH tools/stage_new/bench_inference.py
$PY_PLOT tools/stage_new/plot_bench.py

log "ALL DONE -> $RUN"
