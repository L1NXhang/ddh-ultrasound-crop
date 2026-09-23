#!/usr/bin/env bash
# 把标注工具部署到服务器，并为 A/B/C 三人各起一个标注服务。
#
# 用法（在项目根目录、Git Bash 里运行）：
#   bash tools/deploy_server.sh <用户名:密码> [服务器地址]
# 例：
#   bash tools/deploy_server.sh ddh:MyPass2026
#   bash tools/deploy_server.sh ddh:MyPass2026 ubuntu@124.223.0.187
#
# 做三件事：打包上传（图片 + 队列 + 工具）→ 远端解包 → 起 3 个服务
set -euo pipefail

AUTH="${1:?用法: bash tools/deploy_server.sh <用户名:密码> [服务器地址]}"
HOST="${2:-ubuntu@124.223.0.187}"
DEST="/home/ubuntu/ddh_annotate"
PORTS=(8765 8766 8767)
NAMES=(A B C)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== 1/4 打包（图片 + 队列 + 工具）==="
TAR=/tmp/ddh_deploy.tgz
FILES=(tools/annotate_app.py tools/webapp data/images data/annotations.todo.jsonl)
for f in data/todo_relabel.jsonl data/todo_relabel_part_A.jsonl data/todo_relabel_part_B.jsonl \
         data/todo_relabel_part_C.jsonl data/todo_window.jsonl; do
  [ -f "$f" ] && FILES+=("$f")
done
tar czf "$TAR" "${FILES[@]}"
ls -lh "$TAR" | awk '{print "  包大小: "$5"  （上传耗时取决于你的上行带宽）"}'

echo "=== 2/4 上传 ==="
scp -q "$TAR" "$HOST:/tmp/"

echo "=== 3/4 远端解包 ==="
ssh "$HOST" "mkdir -p $DEST && tar xzf /tmp/ddh_deploy.tgz -C $DEST && du -sh $DEST/data/images && ls $DEST/data/*.jsonl | head -5"

echo "=== 4/4 启动 3 个标注服务 ==="
for i in 0 1 2; do
  N=${NAMES[$i]}; P=${PORTS[$i]}
  ssh "$HOST" "cd $DEST
    Q=data/todo_relabel_part_${N}.jsonl; [ -f \"\$Q\" ] || Q=data/todo_relabel.jsonl
    pkill -f \"annotate_app.py.*--port ${P}\" 2>/dev/null || true
    sleep 1
    nohup python3 tools/annotate_app.py --manifest \"\$Q\" --root data --reference original \
      --crop-top-offset -70 --top-target 120,200 \
      --out data/human_${N}.jsonl --port ${P} --host 0.0.0.0 --auth '${AUTH}' \
      > /tmp/annotate_${N}.log 2>&1 &
    sleep 2; tail -3 /tmp/annotate_${N}.log"
done

IP="${HOST#*@}"
echo
echo "=== 部署完成：把下面三行发给同学 ==="
for i in 0 1 2; do echo "  同学 ${NAMES[$i]}:  http://${IP}:${PORTS[$i]}    账号密码 ${AUTH}"; done
echo
echo "⚠ 最后一件事：到腾讯云控制台 → 安全组，放行 TCP ${PORTS[*]}（以及 22 已放行）"
echo "  自查命令：curl -I http://${IP}:8765   （返回 401 就说明通了）"
