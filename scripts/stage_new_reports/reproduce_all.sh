#!/usr/bin/env bash
# =============================================================================
#  DDH 裁剪系统 · 第 2 版实验与图表全量复现脚本
#
#  用法：bash scripts/stage_new_reports/reproduce_all.sh
#
#  说明：
#   - 训练/推理脚本必须用 venv 的 python（有 torch，无 matplotlib）
#   - 画图脚本必须用系统 python（有 matplotlib，无 torch）
#   - 两者不能混用，这是本项目的环境约束，见 README
# =============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ"

VENV_PY="$PROJ/.venv/Scripts/python.exe"
SYS_PY="${SYS_PY:-/d/PYTHON/python}"
RUN="runs/20261103_resnet18_vs_50_corrected"
TOOLS="tools/stage_new"

echo "== 项目根目录: $PROJ"
echo "== venv python : $VENV_PY"
echo "== 系统 python : $SYS_PY"
echo

if [[ ! -x "$VENV_PY" ]]; then echo "[FATAL] 找不到 venv python: $VENV_PY" >&2; exit 1; fi
if [[ ! -x "$SYS_PY" ]]; then echo "[FATAL] 找不到系统 python: $SYS_PY" >&2; exit 1; fi

# ---------------------------------------------------------------------------
# 阶段 1：训练 6 组配置（已完成可跳过，加 --skip-train）
# ---------------------------------------------------------------------------
SKIP_TRAIN=0
for a in "$@"; do [[ "$a" == "--skip-train" ]] && SKIP_TRAIN=1; done

if [[ $SKIP_TRAIN -eq 0 ]]; then
  echo "== [1/5] 训练 6 组配置（每组约 10-30 分钟，共约 2 小时）"
  for cfg in r18_l3 r50_l3 r18_l123 r50_l123 r18_l4 r18_l1234; do
    if [[ -f "$RUN/$cfg/metrics.json" ]] || [[ -f "$RUN/resnet18/$cfg/metrics.json" ]]; then
      echo "   [skip] $cfg 已有结果"
      continue
    fi
    echo "   [train] $cfg"
    "$VENV_PY" "$TOOLS/train_eval_v3.py" --config "$cfg" --out "$RUN" \
      --manifest data/reports_v2/split.jsonl
  done
else
  echo "== [1/5] 跳过训练（--skip-train）"
fi

# ---------------------------------------------------------------------------
# 阶段 2：CPU 推理基准（6 组 × 3 组样本 × 50 张）
# ---------------------------------------------------------------------------
echo
echo "== [2/5] CPU 推理基准测试"
"$VENV_PY" "$TOOLS/bench_cpu.py" "$RUN"

# ---------------------------------------------------------------------------
# 阶段 3：汇总指标 + 主图（spin: fig3/fig4/fig5）
# ---------------------------------------------------------------------------
echo
echo "== [3/5] 汇总指标与主图"
"$SYS_PY" "$TOOLS/summarize_results.py" "$RUN"

# ---------------------------------------------------------------------------
# 阶段 4：其余图表
# ---------------------------------------------------------------------------
echo
echo "== [4/5] 生成其余图表"
"$SYS_PY" "$TOOLS/plot_layer_theory.py"
"$SYS_PY" "$TOOLS/plot_doc_figures.py"
"$SYS_PY" "$TOOLS/plot_bench.py"
"$SYS_PY" "$TOOLS/plot_bg_compare.py"
"$SYS_PY" "$TOOLS/plot_manual_vs_auto.py"
"$SYS_PY" "$TOOLS/make_cases.py" --config r18_l3 || echo "   [warn] 案例图生成失败（不影响主流程）"

# ---------------------------------------------------------------------------
# 阶段 5：文档编译
# ---------------------------------------------------------------------------
echo
echo "== [5/5] 编译技术开发说明（12 页）"
DOC="$PROJ/docs/开发文档_第2版"
if [[ -f "$DOC/main.tex" ]]; then
  cp "$RUN"/figures/*.png "$DOC"/ 2>/dev/null || true
  # bibtex.exe 在本机 MiKTeX 上会段错误，改用 bibtex8
  MK="/c/Users/13882/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
  ( cd "$DOC" \
    && xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
    && "$MK/bibtex8.exe" main >/dev/null 2>&1 \
    && xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
    && xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 )
  echo "   -> $DOC/main.pdf"
else
  echo "   [skip] 未找到 $DOC/main.tex"
fi

echo
echo "== 全部完成"
echo "   实验目录 : $RUN"
echo "   图表     : $RUN/figures/"
echo "   文档     : $PROJ/docs/技术开发说明_第2版_20261103.pdf"
