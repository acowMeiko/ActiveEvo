#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ABLATION_DIR="$ROOT_DIR/ablation/single_vector_principle"
cd "$ROOT_DIR"

source "$ROOT_DIR/config.env"

PYTHON_BIN="${PYTHON_BIN:-python}"
PORTS_STR="${PORTS_ENV:-8000 8001 8002 8003}"
MODEL="${MODEL:-$MODEL_NAME}"
CONCURRENCY="${CONCURRENCY:-24}"
TIMEOUT="${TIMEOUT:-600}"
TOP_K="${TOP_K:-5}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.35}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-1024}"
TEST_INPUT="${TEST_INPUT:-$ROOT_DIR/data/test/deepscaler.json}"
SOURCE_MEMORY_ROOT="${SOURCE_MEMORY_ROOT:-/home/czhu/workplace/task/MemUA/output_llama/stage3_rerun_20260423_deepscaler_llama_gpu1_3/memory_views}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

export MEMUA_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export MEMUA_EMBEDDING_MODEL_PATH="$EMBEDDING_MODEL_PATH"
export MEMUA_EMBEDDING_DEVICE="$EMBEDDING_DEVICE"

MEMORY_OUT="$ABLATION_DIR/output/stage2_single_vector_memory"
RUN_OUT="$ABLATION_DIR/output/stage3_deepscaler_single_vector_$RUN_TAG"
LOG_DIR="$ABLATION_DIR/logs"
LOG_FILE="$LOG_DIR/deepscaler_single_vector_$RUN_TAG.log"

IFS=' ' read -r -a PORT_ARRAY <<< "$PORTS_STR"
mkdir -p "$LOG_DIR" "$RUN_OUT"

{
  echo "============================================================"
  echo "Single-vector principle ablation"
  echo "Root:           $ROOT_DIR"
  echo "Ablation dir:   $ABLATION_DIR"
  echo "Source memory:  $SOURCE_MEMORY_ROOT"
  echo "Stage2 memory:  $MEMORY_OUT"
  echo "Stage3 output:  $RUN_OUT"
  echo "Test input:     $TEST_INPUT"
  echo "Model:          $MODEL"
  echo "Ports:          ${PORT_ARRAY[*]}"
  echo "Top-k:          $TOP_K"
  echo "Score threshold:$SCORE_THRESHOLD"
  echo "============================================================"
} | tee "$LOG_FILE"

"$PYTHON_BIN" -u "$ABLATION_DIR/build_single_vector_memory.py" \
  --source-root "$SOURCE_MEMORY_ROOT" \
  --output-dir "$MEMORY_OUT" \
  --levels 1 2 3 4 \
  --embedding-model "$EMBEDDING_MODEL" \
  --device "$EMBEDDING_DEVICE" 2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" -u "$ABLATION_DIR/run_single_vector_stage3.py" \
  --input "$TEST_INPUT" \
  --memory-root "$MEMORY_OUT" \
  --output-dir "$RUN_OUT" \
  --levels 1 2 3 4 \
  --model "$MODEL" \
  --ports "${PORT_ARRAY[@]}" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --top-k "$TOP_K" \
  --score-threshold "$SCORE_THRESHOLD" \
  --answer-max-tokens "$ANSWER_MAX_TOKENS" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-device "$EMBEDDING_DEVICE" 2>&1 | tee -a "$LOG_FILE"

echo "[INFO] Done. Summary: $RUN_OUT/single_vector_rerun_summaries.json" | tee -a "$LOG_FILE"
