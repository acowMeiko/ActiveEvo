#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/ablation/no_learning_zone_collection"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output/deepscaler_no_learning_zone_$RUN_TAG}"
MEMORY_DIR="${MEMORY_DIR:-$OUTPUT_DIR/memory}"
PORTS_ENV="${PORTS_ENV:-8000 8001 8002 8003}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"
"$PYTHON_BIN" -u "$SCRIPT_DIR/run_no_learning_zone_ablation.py" \
  --output-dir "$OUTPUT_DIR" \
  --memory-dir "$MEMORY_DIR" \
  --levels ${LEVELS:-1 2 3 4} \
  --test-input "${TEST_INPUT:-data/test/deepscaler.json}" \
  --ports $PORTS_ENV \
  --stage1-concurrency "${STAGE1_CONCURRENCY:-10}" \
  --stage1-diff-concurrency "${STAGE1_DIFF_CONCURRENCY:-10}" \
  --stage1-batch-size "${STAGE1_BATCH_SIZE:-5}" \
  --stage1-max-trials "${STAGE1_MAX_TRIALS:-15}" \
  --stage2-concurrency "${STAGE2_CONCURRENCY:-8}" \
  --stage3-concurrency "${STAGE3_CONCURRENCY:-24}" \
  --threshold "${THRESHOLD:-0.55}" \
  --top-k "${TOP_K:-5}" \
  --retry-on-wrong "${RETRY_ON_WRONG:-2}"
