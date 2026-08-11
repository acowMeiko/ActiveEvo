#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/czhu/miniconda3/envs/hcz/bin/python"
TEST_FILE="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/test_subset.json"
REFERENCE_GUIDED="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/v2/final_paired/guided/eval_results.json"
OUTPUT_ROOT="$SCRIPT_DIR/results_seed42"
PORTS=(8004 8006 8007)
TEMPERATURES=(0.0 0.2)

all_ready=true
for port in "${PORTS[@]}"; do
  if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    all_ready=false
  fi
done
if [[ "$all_ready" != true ]]; then
  bash "$PROJECT_ROOT/experiments/solver_v1_gpu67/run.sh" deploy
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

"$PYTHON_BIN" tools/prepare_stage3_application_gate.py \
  --test-file "$TEST_FILE" \
  --reference-guided "$REFERENCE_GUIDED" \
  --output "$OUTPUT_ROOT" \
  --ports "${PORTS[@]}" \
  --model llama \
  --seed 42 \
  --limit 30 \
  --confidence-threshold 0.80

for temperature in "${TEMPERATURES[@]}"; do
  tag="${temperature/./_}"
  "$PYTHON_BIN" tools/evaluate_stage3_gate_adaptation.py \
    --test-file "$TEST_FILE" \
    --gate-records "$OUTPUT_ROOT/gate_records.json" \
    --output "$OUTPUT_ROOT/temperature_${tag}" \
    --ports "${PORTS[@]}" \
    --model llama \
    --seed 42 \
    --limit 30 \
    --temperature "$temperature" \
    --repetition-penalty 1.05
done

"$PYTHON_BIN" tools/summarize_stage3_gate_adaptation.py --root "$OUTPUT_ROOT"
echo "Experiment complete: $OUTPUT_ROOT/experiment_report.md"
echo "Text comparison: $OUTPUT_ROOT/text_comparison.md"
