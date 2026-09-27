#!/usr/bin/env bash
# 把服务器上的标注结果同步回本地（并自动备份）。
# 用法：bash tools/pull_from_server.sh [服务器地址]
set -euo pipefail
HOST="${1:-ubuntu@124.223.0.187}"
DEST=/home/ubuntu/ddh_annotate
cd "$(dirname "$0")/.."

echo "=== 1/3 拉回服务器上的标注 ==="
scp -q "$HOST:$DEST/data/human_A.jsonl" data/ 2>/dev/null || echo "  human_A 还没生成"
scp -q "$HOST:$DEST/data/human_B.jsonl" data/ 2>/dev/null || echo "  human_B 还没生成"
scp -q "$HOST:$DEST/data/human_C.jsonl" data/ 2>/dev/null || echo "  human_C 还没生成"

echo "=== 2/3 本地备份 ==="
mkdir -p D:/ddhmodel/backup
for N in A B C; do
  f="data/human_$N.jsonl"
  [ -f "$f" ] || continue
  cp "$f" "D:/ddhmodel/backup/human_${N}_$(date +%Y%m%d_%H%M).jsonl"
done
ls -1 D:/ddhmodel/backup/ | tail -4

echo "=== 3/3 三人口径对比（同一批图才有意义，这里先看各自的中位数）==="
.venv/Scripts/python.exe tools/merge_labels.py --inputs data/human_A.jsonl data/human_B.jsonl data/human_C.jsonl 2>/dev/null \
  || python3 tools/merge_labels.py --inputs data/human_A.jsonl data/human_B.jsonl data/human_C.jsonl

echo
echo "同步完成。确认无误后合并："
echo "  python tools/merge_labels.py --inputs data/human_A.jsonl data/human_B.jsonl data/human_C.jsonl --apply"
