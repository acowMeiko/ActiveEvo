#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/czhu/miniconda3/envs/hcz/bin/python"
TEST_FILE="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/test_subset.json"
REFERENCE_GUIDED="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/v2/final_paired/guided/eval_results.json"
OUTPUT_ROOT="$SCRIPT_DIR/penalty_sweep30_seed42"
PENALTIES=(1.00 1.03 1.05 1.08)
PORTS=(8004 8006 8007)

all_ready=true
for port in "${PORTS[@]}"; do
  if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    all_ready=false
  fi
done

if [[ "$all_ready" != true ]]; then
  echo "Some Solver endpoints are missing; deploying only the missing GPUs."
  bash "$SCRIPT_DIR/run.sh" deploy
fi

for attempt in {1..60}; do
  all_ready=true
  for port in "${PORTS[@]}"; do
    if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      all_ready=false
    fi
  done
  if [[ "$all_ready" == true ]]; then
    echo "All Solver endpoints are ready: ${PORTS[*]}"
    break
  fi
  if [[ "$attempt" == 60 ]]; then
    echo "Solver endpoints did not become ready within 5 minutes." >&2
    bash "$SCRIPT_DIR/run.sh" status >&2
    exit 1
  fi
  if (( attempt % 6 == 0 )); then
    echo "Waiting for Solver endpoints (${attempt}/60 checks)..."
  fi
  sleep 5
done

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

for penalty in "${PENALTIES[@]}"; do
  tag="${penalty/./_}"
  output="$OUTPUT_ROOT/rp_${tag}"
  echo "Starting Solver-v1 Stage 3 sweep: reasoner repetition_penalty=${penalty}"
  "$PYTHON_BIN" tools/evaluate_solver_v1_paired.py \
    --test-file "$TEST_FILE" \
    --reference-guided "$REFERENCE_GUIDED" \
    --output "$output" \
    --ports "${PORTS[@]}" \
    --model llama \
    --seeds 42 \
    --limit 30 \
    --reasoner-repetition-penalty "$penalty" \
    --progress-every 10
done

"$PYTHON_BIN" tools/summarize_solver_v1_penalty_sweep.py --root "$OUTPUT_ROOT"
echo "Sweep complete: $OUTPUT_ROOT/sweep_report.md"
