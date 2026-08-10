#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/config.env"

IFS=' ' read -r -a PORT_ARRAY <<< "$PORTS"

export MEMUA_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export MEMUA_EMBEDDING_MODEL_PATH="$EMBEDDING_MODEL_PATH"
export MEMUA_EMBEDDING_DEVICE="$EMBEDDING_DEVICE"

DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${TEST_NAME:-}" || -z "${TEST_INPUT:-}" ]]; then
  echo "[ERROR] TEST_NAME and TEST_INPUT must be set by the caller." >&2
  exit 1
fi

if [[ ! -f "$TEST_INPUT" ]]; then
  echo "[ERROR] Missing test file: $ROOT_DIR/$TEST_INPUT" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" && ! -f "$MEMORY_DIR/principles/principles.json" ]]; then
  echo "[ERROR] Missing memory DB: $ROOT_DIR/$MEMORY_DIR/principles/principles.json" >&2
  echo "[ERROR] Run bash scripts/stage2/run_all_levels.sh first." >&2
  exit 1
fi

OUT_DIR="$ROOT_DIR/output/stage3/tests/$TEST_NAME"
LOG_FILE="$ROOT_DIR/$LOG_DIR/stage3_single/${TEST_NAME}.log"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR/stage3_single" "output/stage3/tests/$TEST_NAME"
fi

cmd=(
  "$PYTHON_BIN" -u scripts/stage3/evaluate_principle_guided_test.py
  --input "$TEST_INPUT"
  --output "$OUT_DIR/principle_guided_test_results.json"
  --summary-output "$OUT_DIR/principle_guided_test_summary.json"
  --memory-dir "$MEMORY_DIR"
  --mode guided
  --concurrency "$EVAL_CONCURRENCY"
  --model "$MODEL_NAME"
  --ports "${PORT_ARRAY[@]}"
  --timeout "$TIMEOUT"
  --threshold "$THRESHOLD"
  --top-k "$TOP_K"
  --embedding-model "$EMBEDDING_MODEL"
  --answer-max-tokens "$ANSWER_MAX_TOKENS"
)

echo "[INFO] Test set: $TEST_NAME"
echo "[INFO] Input:    $ROOT_DIR/$TEST_INPUT"
echo "[INFO] Output:   $OUT_DIR"
echo "[INFO] Dry run:  $DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
