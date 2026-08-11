#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/czhu/miniconda3/envs/hcz/bin/python"
DEPLOY_CONFIG="$PROJECT_ROOT/deploy.conf"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/api_call/runtime_config.sh"
load_deployment_runtime "$DEPLOY_CONFIG"
MODEL_NAME="$DEPLOY_SELECTED_MODEL"
GPU_IDS=("${DEPLOY_SELECTED_GPU_IDS[@]}")
PORTS=("${DEPLOY_SELECTED_PORTS[@]}")
SERVER_SEED="$DEPLOY_SELECTED_SERVER_SEED"
TEST_FILE="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/test_subset.json"
MEMORY_DIR="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/v2/memory"
RERANKER_MODEL="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0___6B"
OUTPUT_ROOT="$SCRIPT_DIR/results_seed42"
TEMPERATURES=(0.0 0.2)

export MEMUA_ALLOW_HASH_FALLBACK=0
export MEMUA_FORCE_EMBEDDING=1
export MEMUA_EMBEDDING_DEVICE=cpu
export MEMUA_EMBEDDING_MODEL_PATH="/home/share/models/modelscope/BAAI/bge-m3"

MISSING_GPU_IDS=()
for index in "${!PORTS[@]}"; do
  port="${PORTS[$index]}"
  if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    MISSING_GPU_IDS+=("${GPU_IDS[$index]}")
  fi
done
if [[ ${#MISSING_GPU_IDS[@]} -gt 0 ]]; then
  bash "$PROJECT_ROOT/api_call/run.sh" "$DEPLOY_CONFIG" "$MODEL_NAME" "${MISSING_GPU_IDS[@]}" \
    --daemon --server-seed "$SERVER_SEED"
fi

for attempt in {1..60}; do
  all_ready=true
  for port in "${PORTS[@]}"; do
    if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      all_ready=false
    fi
  done
  if [[ "$all_ready" == true ]]; then
    break
  fi
  if [[ "$attempt" == 60 ]]; then
    echo "Solver endpoints did not become ready within 5 minutes." >&2
    exit 1
  fi
  sleep 5
done

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" tools/prepare_stage3_topk_gate.py \
  --test-file "$TEST_FILE" \
  --memory-dir "$MEMORY_DIR" \
  --reranker-model "$RERANKER_MODEL" \
  --output "$OUTPUT_ROOT" \
  --ports "${PORTS[@]}" \
  --model "$MODEL_NAME" \
  --seed 42 \
  --limit 30 \
  --recall-top-k 10 \
  --gate-top-k 5 \
  --max-selected 1 \
  --confidence-threshold 0.80 \
  --gate-max-tokens 3072 \
  --reranker-device cpu \
  --reranker-batch-size 4

for temperature in "${TEMPERATURES[@]}"; do
  tag="${temperature/./_}"
  "$PYTHON_BIN" tools/evaluate_stage3_gate_adaptation.py \
    --test-file "$TEST_FILE" \
    --gate-records "$OUTPUT_ROOT/gate_records.json" \
    --output "$OUTPUT_ROOT/temperature_${tag}" \
    --ports "${PORTS[@]}" \
    --model "$MODEL_NAME" \
    --seed 42 \
    --limit 30 \
    --temperature "$temperature" \
    --repetition-penalty 1.05
done

"$PYTHON_BIN" tools/summarize_stage3_gate_adaptation.py --root "$OUTPUT_ROOT"
echo "Experiment complete: $OUTPUT_ROOT/experiment_report.md"
echo "Routing and text comparison: $OUTPUT_ROOT/text_comparison.md"
