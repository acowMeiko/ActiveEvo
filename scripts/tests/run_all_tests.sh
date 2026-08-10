#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"
CONFIG_FILE="$ROOT_DIR/config.env"
DEPLOY_CONFIG="$ROOT_DIR/deploy.conf"
if [[ "$DRY_RUN" == "1" ]]; then
  [[ -f "$CONFIG_FILE" ]] || CONFIG_FILE="$ROOT_DIR/config.env.example"
  [[ -f "$DEPLOY_CONFIG" ]] || DEPLOY_CONFIG="$ROOT_DIR/deploy.conf.example"
fi

source "$CONFIG_FILE"
source "$ROOT_DIR/api_call/runtime_config.sh"
load_deployment_runtime "$DEPLOY_CONFIG"

ACTIVE_RUNTIME="$ROOT_DIR/api_call/${LOG_DIR:-logs}/active_deployment.env"
if [[ -f "$ACTIVE_RUNTIME" ]]; then
  source "$ACTIVE_RUNTIME"
  IFS=' ' read -r -a PORT_ARRAY <<< "$ACTIVE_PORTS"
  MODEL_NAME="$ACTIVE_MODEL"
else
  PORT_ARRAY=("${DEPLOY_SELECTED_PORTS[@]}")
  MODEL_NAME="$DEPLOY_SELECTED_MODEL"
fi

export MEMUA_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export MEMUA_EMBEDDING_MODEL_PATH="$EMBEDDING_MODEL_PATH"
export MEMUA_EMBEDDING_DEVICE="$EMBEDDING_DEVICE"
export MEMUA_HF_CACHE_DIR="${MEMUA_HF_CACHE_DIR:-}"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR/stage3" output/stage3/tests
fi

if [[ "$DRY_RUN" != "1" && ! -f "$MEMORY_DIR/principles/principles.json" ]]; then
  echo "[ERROR] Missing memory DB: $ROOT_DIR/$MEMORY_DIR/principles/principles.json" >&2
  echo "[ERROR] Run bash scripts/stage2/run_all_levels.sh first." >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" && ( ! -f data/test/official/svamp/processed/test.json || ! -f data/test/official/general365/processed/test.json ) ]]; then
  "$PYTHON_BIN" scripts/prepare_official_datasets.py
fi

echo "[INFO] ActiveEvo Stage 3 all-test runner"
echo "[INFO] Python:     $PYTHON_BIN"
echo "[INFO] Model:      $MODEL_NAME"
echo "[INFO] Ports:      ${PORT_ARRAY[*]}"
echo "[INFO] Memory:     $ROOT_DIR/$MEMORY_DIR"
echo "[INFO] Output:     $ROOT_DIR/output/stage3/tests"
echo "[INFO] Dry run:    $DRY_RUN"

run_one() {
  local name="$1"
  local input_path="$2"
  local out_dir="$ROOT_DIR/output/stage3/tests/$name"
  local log_file="$ROOT_DIR/$LOG_DIR/stage3/${name}.log"

  if [[ ! -f "$input_path" ]]; then
    echo "[WARN] Skipping $name; missing test file: $input_path" >&2
    return 0
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$out_dir"
  fi
  echo
  echo "[INFO] Running test set: $name"
  cmd=(
    "$PYTHON_BIN" -u scripts/stage3/evaluate_principle_guided_test.py
    --input "$input_path" \
    --output "$out_dir/principle_guided_test_results.json" \
    --summary-output "$out_dir/principle_guided_test_summary.json" \
    --memory-dir "$MEMORY_DIR" \
    --mode guided \
    --concurrency "$EVAL_CONCURRENCY" \
    --model "$MODEL_NAME" \
    --ports "${PORT_ARRAY[@]}" \
    --timeout "$TIMEOUT" \
    --threshold "$THRESHOLD" \
    --top-k "$TOP_K" \
    --embedding-model "$EMBEDDING_MODEL" \
    --answer-max-tokens "$ANSWER_MAX_TOKENS"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

run_one "deepscaler" "data/test/deepscaler.json"
run_one "math500" "data/test/benchmarks/math500_stage3.json"
run_one "gsm8k" "data/test/benchmarks/gsm8k_stage3.json"
run_one "bbh_logic" "data/test/benchmarks_split/bbh_logic_stage3.json"
run_one "bbh_math" "data/test/benchmarks_split/bbh_math_stage3.json"
run_one "svamp" "data/test/official/svamp/processed/test.json"
run_one "general365" "data/test/official/general365/processed/test.json"

echo
echo "[INFO] All available test sets finished."
echo "[INFO] Summaries: $ROOT_DIR/output/stage3/tests/*/principle_guided_test_summary.json"
