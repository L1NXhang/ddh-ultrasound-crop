#!/usr/bin/env bash
# DDH 四组同条件对照训练：ResNet18/50 × (仅Layer3 | Layer1+2+3 融合)
# 全程不覆盖任何已有 runs/runs_v2 结果；结果写入 runs/20261103_resnet18_vs_50/
#
# 用法: bash scripts/run_four_models.sh [--skip-sanity] [--only r18_l3,r50_l3]
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
PY="./.venv/Scripts/python.exe"
RUN="runs/20261103_resnet18_vs_50"
mkdir -p "$RUN"/{resnet18,resnet50,figures,reports,logs,configs,sanity}

# 统一超参（来自 ddh_v2.py 默认值，四组完全相同；唯一变量 = backbone + 融合方式）
COMMON="--manifest data/reports/split.jsonl --root data --epochs 30 --patience 6 --batch 8 --size 384 --seed 42"
SEED_TORCH="PYTHONHASHSEED=42"
export PYTHONHASHSEED=42

log(){ echo "[$(date '+%F %T')] $*"; }

# ---------- sanity: 1 epoch, 20 batches ----------
if [[ "${1:-}" != "--skip-sanity" ]]; then
  for cfg in r18_l3 r50_l3; do
    log "sanity $cfg"
    $PY tools/stage_new/train_eval_v3.py --config "$cfg" --sanity --epochs 1 \
        --max-train-batches 20 --out "$RUN/sanity/$cfg" 2>&1 | tail -3
  done
fi

# ---------- 全量训练 ----------
# r18_l3 / r50_l3 先跑（核心对照），再跑两组融合
for cfg in r18_l3 r50_l3 r18_l123 r50_l123; do
  case "$cfg" in
    r18_*) out="$RUN/resnet18/$cfg" ;;
    r50_*) out="$RUN/resnet50/$cfg" ;;
  esac
  if [[ -f "$out/metrics.json" ]]; then
    log "skip $cfg (already done)"; continue
  fi
  log "=== train $cfg -> $out ==="
  $PY tools/stage_new/train_eval_v3.py --config "$cfg" $COMMON --out "$out" 2>&1 | tee -a "$RUN/logs/$cfg.log" | grep -E '"event": "(epoch|done|early_stop)"' || true
done
log "ALL DONE"
