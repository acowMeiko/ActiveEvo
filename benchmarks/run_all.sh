#!/usr/bin/env bash
# ============================================================
# benchmarks/run_all.sh — 运行全部 benchmark 评测 + 汇总
#
# 用法:
#   bash benchmarks/run_all.sh              # 自动 nohup (llama+qwen 各4卡)
#   bash benchmarks/run_all.sh --foreground # 前台运行
#   bash benchmarks/run_all.sh --8gpu       # 8 卡单模型 (默认 llama, 8卡轮询)
#   bash benchmarks/run_all.sh --8gpu qwen  # 8 卡单模型 (qwen)
#
# 前置条件:
#   1. 模型已通过 api_call/run.sh 部署
#   2. Memory 已通过 curriculum 流程生成了 principles
# ============================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── 参数解析 ──────────────────────────────────────────────────
FOREGROUND=false
MODE_8GPU=false
MODEL_8GPU="llama"

for arg in "$@"; do
    case "$arg" in
        --foreground) FOREGROUND=true ;;
        --8gpu)       MODE_8GPU=true ;;
        llama|qwen)   $MODE_8GPU && MODEL_8GPU="$arg" ;;
    esac
done

# ── 自后台化 ──────────────────────────────────────────────────
if [[ "$FOREGROUND" != true ]]; then
    if $MODE_8GPU; then
        TAG="benchmark_8gpu_${MODEL_8GPU}"
    else
        TAG="benchmark_all"
    fi
    CONSOLE_LOG="$LOG_DIR/${TAG}_${TIMESTAMP}.log"
    echo "[INFO] 后台启动 benchmark 评测"
    if $MODE_8GPU; then
        echo "[INFO] 模式: 8-GPU 单模型 (${MODEL_8GPU}_8gpu)"
    else
        echo "[INFO] 模式: 双模型 (llama + qwen)"
        echo "[INFO] LLaMA 日志: $LOG_DIR/benchmark_llama_${TIMESTAMP}.log"
        echo "[INFO] Qwen 日志:  $LOG_DIR/benchmark_qwen_${TIMESTAMP}.log"
    fi
    echo "[INFO] 控制台日志: $CONSOLE_LOG"
    nohup bash "$0" ${MODE_8GPU:+--8gpu $MODEL_8GPU} --foreground > "$CONSOLE_LOG" 2>&1 &
    echo "[INFO] PID: $!"
    exit 0
fi

# ── 以下为实际执行 (--foreground 模式) ──────────────────────
export PYTHONUNBUFFERED=1

# ── 嵌入模型强制 GPU ─────────────────────────────────────────
export MEMUA_EMBEDDING_DEVICE=cuda

echo "================================================================"
if $MODE_8GPU; then
    echo "  Benchmark 评测 (8-GPU 单模型): $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Model: ${MODEL_8GPU}_8gpu (8 卡轮询)"
else
    echo "  Benchmark 评测开始: $(date '+%Y-%m-%d %H:%M:%S')"
fi
echo "================================================================"

# ═══════════════════════════════════════════════════════════════
# 8-GPU 单模型模式: 一个进程, 8 端口轮询
# ═══════════════════════════════════════════════════════════════
if $MODE_8GPU; then
    BM_LOG="$LOG_DIR/benchmark_${MODEL_8GPU}_8gpu_${TIMESTAMP}.log"

    echo ""
    echo "[1/2] 运行 ${MODEL_8GPU}_8gpu benchmark (8 卡轮询, concurrency=128)..."
    echo "  日志: $BM_LOG"

    python benchmarks/run_suite.py \
        --models "${MODEL_8GPU}_8gpu" \
        --mode guided \
        --concurrency 128 \
        --skip deepscaler \
        --output output/benchmarks \
        > "$BM_LOG" 2>&1

    echo "  完成: $(date '+%Y-%m-%d %H:%M:%S')"

    echo ""
    echo "[2/2] 生成汇总..."

    python benchmarks/summary.py \
        --results-dir output/benchmarks \
        --format markdown \
        --output output/benchmarks/SUMMARY.md

    echo ""
    echo "================================================================"
    echo "  Benchmark 评测完成: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  汇总: output/benchmarks/SUMMARY.md"
    echo "================================================================"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════
# 原始双模型模式: llama GPU 0-3 + qwen GPU 4-7, 并行跑
# ═══════════════════════════════════════════════════════════════

# ── Step 1: 并行启动两个模型 ───────────────────────────────────
echo ""
echo "[1/3] 并行启动 LLaMA + Qwen benchmark..."

LLAMA_LOG="$LOG_DIR/benchmark_llama_${TIMESTAMP}.log"
QWEN_LOG="$LOG_DIR/benchmark_qwen_${TIMESTAMP}.log"

python benchmarks/run_suite.py \
  --models llama \
  --mode guided \
  --skip deepscaler \
  --output output/benchmarks \
  > "$LLAMA_LOG" 2>&1 &
PID_LLAMA=$!

python benchmarks/run_suite.py \
  --models qwen \
  --mode guided \
  --skip deepscaler \
  --output output/benchmarks \
  > "$QWEN_LOG" 2>&1 &
PID_QWEN=$!

echo "  LLaMA PID: $PID_LLAMA  →  $LLAMA_LOG"
echo "  Qwen  PID: $PID_QWEN  →  $QWEN_LOG"

# ── Step 2: 等待两个进程结束 ───────────────────────────────────
echo ""
echo "[2/3] 等待评测完成..."
wait $PID_LLAMA
echo "  LLaMA 完成: $(date '+%Y-%m-%d %H:%M:%S')"
wait $PID_QWEN
echo "  Qwen  完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ── Step 3: 生成合并汇总 ──────────────────────────────────────
echo ""
echo "[3/3] 生成汇总报告..."

python benchmarks/summary.py \
  --results-dir output/benchmarks \
  --format markdown \
  --output output/benchmarks/SUMMARY.md

echo ""
echo "================================================================"
echo "  Benchmark 评测全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  汇总: output/benchmarks/SUMMARY.md"
echo "================================================================"
