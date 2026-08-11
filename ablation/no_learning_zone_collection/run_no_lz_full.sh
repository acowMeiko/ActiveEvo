#!/usr/bin/env bash
# ================================================================
# No Learning Zone 消融 — 全自动 (Stage1+2 记忆构建 + Stage3 评测)
#
# 用法:
#   bash ablation/no_learning_zone_collection/run_no_lz_full.sh
#
# 输出:
#   output/no_learning_zone/
#     ├── llama/memory/        # LLaMA 消融记忆
#     ├── qwen/memory/         # Qwen 消融记忆
#     ├── llama/eval/          # LLaMA Stage3 结果 (分 benchmark)
#     └── qwen/eval/           # Qwen Stage3 结果 (分 benchmark)
# ================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── 自后台化 ──────────────────────────────────────────────────
if [[ "${1:-}" != "--foreground" ]]; then
    CONSOLE_LOG="$LOG_DIR/no_lz_full_${TIMESTAMP}.log"
    echo "[INFO] 后台启动 No Learning Zone 消融"
    echo "[INFO] 控制台日志: $CONSOLE_LOG"
    nohup bash "$0" --foreground > "$CONSOLE_LOG" 2>&1 &
    echo "[INFO] PID: $!"
    exit 0
fi

export PYTHONUNBUFFERED=1

# 强制 embedding 模型用 CPU 加载 (GPU 被 vLLM 占满)
export MEMUA_EMBEDDING_DEVICE=cpu
export MEMUA_EMBEDDING_MODEL_PATH="/home/models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

PYTHON="python -u"
ABLATION="$SCRIPT_DIR/run_no_learning_zone_ablation.py"

# ── Benchmark 清单 ────────────────────────────────────────────
declare -A BM_INPUT=(
  ["gsm8k"]="data/test/benchmarks/gsm8k_stage3.json"
  ["math500"]="data/test/benchmarks/math500_stage3.json"
  ["bbh_logic"]="data/test/benchmarks_split/bbh_logic_stage3.json"
  ["bbh_math"]="data/test/benchmarks_split/bbh_math_stage3.json"
  ["svamp"]="data/test/official/svamp/processed/test.json"
  ["general365"]="data/test/official/general365/processed/test.json"
  ["deepscaler"]="data/test/deepscaler.json"
)

OUTPUT_ROOT="$PROJECT_ROOT/output/no_learning_zone"

echo "================================================================"
echo "  No Learning Zone 消融实验"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ═══════════════════════════════════════════════════════════════
# Phase 1: Stage 1+2 — 全量采样 + 记忆入库 (两模型并行)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[Phase 1/2] Stage 1+2: 全量采样 + 记忆入库"

LLAMA_LOG="$LOG_DIR/no_lz_llama_stage12_${TIMESTAMP}.log"
QWEN_LOG="$LOG_DIR/no_lz_qwen_stage12_${TIMESTAMP}.log"

$PYTHON "$ABLATION" \
  --output-dir "$OUTPUT_ROOT/llama" \
  --memory-dir "$OUTPUT_ROOT/llama/memory" \
  --model llama \
  --ports 8000 8001 8002 8003 \
  --stage1-concurrency 128 \
  --stage1-diff-concurrency 96 \
  --stage2-concurrency 64 \
  --skip-stage3 \
  --stage1-sampling-source "$PROJECT_ROOT/output/stage1/01_sampling/llama_unified_sampling_results.json" \
  > "$LLAMA_LOG" 2>&1 &
PID_LLAMA=$!

$PYTHON "$ABLATION" \
  --output-dir "$OUTPUT_ROOT/qwen" \
  --memory-dir "$OUTPUT_ROOT/qwen/memory" \
  --model qwen \
  --ports 8004 8005 8006 8007 \
  --stage1-concurrency 128 \
  --stage1-diff-concurrency 96 \
  --stage2-concurrency 64 \
  --skip-stage3 \
  --stage1-sampling-source "$PROJECT_ROOT/output/stage1/01_sampling/qwen_unified_sampling_results.json" \
  > "$QWEN_LOG" 2>&1 &
PID_QWEN=$!

echo "  LLaMA PID: $PID_LLAMA  →  $LLAMA_LOG"
echo "  Qwen  PID: $PID_QWEN  →  $QWEN_LOG"

echo ""
echo "[Phase 1/2] 等待 Stage 1+2 完成..."
wait $PID_LLAMA
LLAMA_RC=$?
echo "  LLaMA 完成 (exit=$LLAMA_RC): $(date '+%Y-%m-%d %H:%M:%S')"
wait $PID_QWEN
QWEN_RC=$?
echo "  Qwen  完成 (exit=$QWEN_RC): $(date '+%Y-%m-%d %H:%M:%S')"

# ═══════════════════════════════════════════════════════════════
# Phase 2: Stage 3 — 评测所有 benchmark (两模型串行各自 benchmark)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[Phase 2/2] Stage 3: 评测所有 benchmark"

run_eval() {
  local model=$1
  local ports=$2
  local mem_dir=$3
  local out_base=$4

  for bm in gsm8k math500 bbh_logic bbh_math svamp general365 deepscaler; do
    local input="${BM_INPUT[$bm]}"
    echo "  [$model] $bm — $(date '+%H:%M:%S')"
    $PYTHON "$ABLATION" \
      --output-dir "$out_base/$bm" \
      --memory-dir "$mem_dir" \
      --model "$model" \
      --ports $ports \
      --test-input "$input" \
      --levels 4 \
      --skip-stage1 --skip-stage2 \
      --stage3-concurrency 64 \
      2>&1 | sed "s/^/  [$model $bm] /"
    echo "  [$model] $bm 完成: $(date '+%H:%M:%S')"
  done
}

if [[ $LLAMA_RC -eq 0 ]]; then
  echo ""
  echo "--- LLaMA Stage 3 ---"
  LLAMA_EVAL_LOG="$LOG_DIR/no_lz_llama_stage3_${TIMESTAMP}.log"
  run_eval llama "8000 8001 8002 8003" "$OUTPUT_ROOT/llama/memory" "$OUTPUT_ROOT/llama/eval" > "$LLAMA_EVAL_LOG" 2>&1 &
  PID_LLAMA_EVAL=$!
else
  echo "  [SKIP] LLaMA Stage 1+2 失败 (exit=$LLAMA_RC)，跳过 Stage 3"
  PID_LLAMA_EVAL=""
fi

if [[ $QWEN_RC -eq 0 ]]; then
  echo ""
  echo "--- Qwen Stage 3 ---"
  QWEN_EVAL_LOG="$LOG_DIR/no_lz_qwen_stage3_${TIMESTAMP}.log"
  run_eval qwen "8004 8005 8006 8007" "$OUTPUT_ROOT/qwen/memory" "$OUTPUT_ROOT/qwen/eval" > "$QWEN_EVAL_LOG" 2>&1 &
  PID_QWEN_EVAL=$!
else
  echo "  [SKIP] Qwen Stage 1+2 失败 (exit=$QWEN_RC)，跳过 Stage 3"
  PID_QWEN_EVAL=""
fi

# 等待所有 Stage 3 完成
for pid in $PID_LLAMA_EVAL $PID_QWEN_EVAL; do
  [[ -n "$pid" ]] && wait $pid
done

echo ""
echo "================================================================"
echo "  No Learning Zone 消融完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果: $OUTPUT_ROOT/"
echo "================================================================"
