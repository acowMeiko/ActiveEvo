#!/usr/bin/env bash
# ============================================================
# benchmarks/run_all_single.sh — 单向量检索消融实验
#
# 用法:
#   bash benchmarks/run_all_single.sh              # 自动 nohup 后台运行
#   bash benchmarks/run_all_single.sh --foreground # 前台运行
#
# 前置条件:
#   1. 模型已部署 (LLaMA ports 8000-8003, Qwen ports 8004-8007)
#   2. Memory 已通过 curriculum 流程生成了 principles
# ============================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── 自后台化 ────────────────────────────────────────────────
if [[ "${1:-}" != "--foreground" ]]; then
    CONSOLE_LOG="$LOG_DIR/benchmark_single_${TIMESTAMP}.log"
    echo "[INFO] 后台启动单向量消融评测"
    echo "[INFO] LLaMA 日志:  $LOG_DIR/benchmark_single_llama_${TIMESTAMP}.log"
    echo "[INFO] Qwen 日志:   $LOG_DIR/benchmark_single_qwen_${TIMESTAMP}.log"
    echo "[INFO] 控制台日志:  $CONSOLE_LOG"
    nohup bash "$0" --foreground > "$CONSOLE_LOG" 2>&1 &
    echo "[INFO] PID: $!"
    exit 0
fi

export PYTHONUNBUFFERED=1

echo "================================================================"
echo "  单向量消融评测: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  retrieval_method = single (holistic description → KNN)"
echo "================================================================"

# ── 并行启动两个模型 ─────────────────────────────────────────
echo ""
echo "[1/3] 并行启动 LLaMA + Qwen 单向量评测..."

LLAMA_LOG="$LOG_DIR/benchmark_single_llama_${TIMESTAMP}.log"
QWEN_LOG="$LOG_DIR/benchmark_single_qwen_${TIMESTAMP}.log"

python benchmarks/run_suite.py \
  --models llama \
  --mode guided \
  --retrieval-method single \
  --output output/benchmarks-single \
  > "$LLAMA_LOG" 2>&1 &
PID_LLAMA=$!

python benchmarks/run_suite.py \
  --models qwen \
  --mode guided \
  --retrieval-method single \
  --output output/benchmarks-single \
  > "$QWEN_LOG" 2>&1 &
PID_QWEN=$!

echo "  LLaMA PID: $PID_LLAMA  →  $LLAMA_LOG"
echo "  Qwen  PID: $PID_QWEN  →  $QWEN_LOG"

echo ""
echo "[2/3] 等待评测完成..."
wait $PID_LLAMA
echo "  LLaMA 完成: $(date '+%Y-%m-%d %H:%M:%S')"
wait $PID_QWEN
echo "  Qwen  完成: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "[3/3] 生成汇总报告..."

python benchmarks/summary.py \
  --results-dir output/benchmarks-single \
  --format markdown \
  --output output/benchmarks-single/SUMMARY.md

echo ""
echo "================================================================"
echo "  单向量消融评测完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果: output/benchmarks-single/"
echo "================================================================"
