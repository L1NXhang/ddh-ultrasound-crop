#!/usr/bin/env bash
# 四组同条件对照训练 —— 修正版：使用正确的新口径标签
#
# 修正说明（2026-11-03）：
#   首次实验误用 data/reports/split.jsonl（9-21 旧口径，与最终标注仅 1.3% 一致）。
#   正确文件 = data/reports_v2/split.jsonl（9-27，与 train_v2.jsonl 100% 一致，
#   且同样含 train/val/calib/test 划分与 leak_group）。
#   本脚本用正确标签重跑全部四组，输出到 runs/20261103_resnet18_vs_50_corrected/
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
PY="./.venv/Scripts/python.exe"
RUN="runs/20261103_resnet18_vs_50_corrected"
MANIFEST="data/reports_v2/split.jsonl"
mkdir -p "$RUN"/{resnet18,resnet50,figures,reports,logs,configs,sanity}

COMMON="--manifest $MANIFEST --root data --epochs 30 --patience 6 --batch 8 --size 384 --seed 42"
export PYTHONHASHSEED=42

log(){ echo "[$(date '+%F %T')] $*"; }

log "manifest=$MANIFEST  sha256=$(sha256sum "$MANIFEST" | cut -c1-16)"

if [[ "${1:-}" != "--skip-sanity" ]]; then
  log "sanity r18_l3"
  $PY tools/stage_new/train_eval_v3.py --config r18_l3 --sanity --epochs 1 \
      --max-train-batches 20 --out "$RUN/sanity/r18_l3" 2>&1 | tail -2
fi

for cfg in r18_l3 r50_l3 r18_l123 r50_l123; do
  case "$cfg" in
    r18_*) out="$RUN/resnet18/$cfg" ;;
    r50_*) out="$RUN/resnet50/$cfg" ;;
  esac
  if [[ -f "$out/metrics.json" ]]; then log "skip $cfg"; continue; fi
  log "=== train $cfg ==="
  $PY tools/stage_new/train_eval_v3.py --config "$cfg" $COMMON --out "$out" 2>&1 \
    | tee -a "$RUN/logs/$cfg.log" \
    | grep -E '"event": "(done|early_stop)"' || true
done
log "ALL DONE"
