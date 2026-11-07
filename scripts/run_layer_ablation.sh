#!/usr/bin/env bash
# 层级选择验证实验（等主队列四组完成后自动开始）
#
# 验证两个理论预测：
#   1. 仅 Layer4 → 中位误差应 ≥ 16 px（量化下界 = 384/(2×12)）
#   2. L1+L2+L3+L4 → 不优于 L1+L2+L3
#
# 注意：两个配置都必须跑完整个 backbone（要取 L4 必须先过 L1-L3），
# 因此参数量/计算量与基线几乎相同（11.41M vs 11.45M），
# 对照的是"同样的算力下，用哪一层的特征"，而不是算力本身。
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
PY="./.venv/Scripts/python.exe"
RUN="runs/20261103_resnet18_vs_50_corrected"
MANIFEST="data/reports_v2/split.jsonl"
COMMON="--manifest $MANIFEST --root data --epochs 30 --patience 6 --batch 8 --size 384 --seed 42"
export PYTHONHASHSEED=42

log(){ echo "[$(date '+%F %T')] $*"; }

# 等待主队列完成（r50_l123 的 metrics.json 出现）
log "等待主队列完成..."
for i in $(seq 1 240); do
  if [[ -f "$RUN/resnet50/r50_l123/metrics.json" ]]; then
    log "主队列已完成，开始验证实验"; break
  fi
  sleep 30
done

for cfg in r18_l4 r18_l1234; do
  out="$RUN/resnet18/$cfg"
  if [[ -f "$out/metrics.json" ]]; then log "skip $cfg"; continue; fi
  log "=== train $cfg ==="
  $PY tools/stage_new/train_eval_v3.py --config "$cfg" $COMMON --out "$out" 2>&1 \
    | tee -a "$RUN/logs/$cfg.log" \
    | grep -E '"event": "(done|early_stop)"' || true
done
log "验证实验 DONE"
