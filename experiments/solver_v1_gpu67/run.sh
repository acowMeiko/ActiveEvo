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
PORTS=("${DEPLOY_SELECTED_PORTS[@]}")
GPU_IDS=("${DEPLOY_SELECTED_GPU_IDS[@]}")
SERVER_SEED="$DEPLOY_SELECTED_SERVER_SEED"
TEST_FILE="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/test_subset.json"
REFERENCE_GUIDED="$PROJECT_ROOT/experiments/llama_deepscaler_l2_l3_seed42/v2/final_paired/guided/eval_results.json"

MODE="${1:-smoke30}"
case "$MODE" in
  status)
    for port in "${PORTS[@]}"; do
      if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
        echo "[READY] GPU endpoint port ${port}"
      else
        echo "[DOWN]  GPU endpoint port ${port}"
      fi
    done
    exit 0
    ;;
  deploy)
    MISSING_GPU_IDS=()
    for index in "${!PORTS[@]}"; do
      port="${PORTS[$index]}"
      gpu="${GPU_IDS[$index]}"
      if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
        echo "[SKIP] GPU ${gpu} endpoint port ${port} is already ready"
      else
        MISSING_GPU_IDS+=("$gpu")
      fi
    done
    if [[ ${#MISSING_GPU_IDS[@]} -eq 0 ]]; then
      echo "[READY] All Solver endpoints are already running"
      exit 0
    fi
    echo "[DEPLOY] Starting missing GPU endpoints: ${MISSING_GPU_IDS[*]}"
    exec bash "$PROJECT_ROOT/api_call/run.sh" "$DEPLOY_CONFIG" "$MODEL_NAME" "${MISSING_GPU_IDS[@]}" --daemon --server-seed "$SERVER_SEED"
    ;;
  smoke30)
    OUTPUT="$SCRIPT_DIR/smoke30_seed42"
    SEEDS=(42)
    LIMIT_ARGS=(--limit 30)
    ;;
  full100)
    OUTPUT="$SCRIPT_DIR/full100_seed42"
    SEEDS=(42)
    LIMIT_ARGS=(--limit 100)
    ;;
  pilot3seed)
    OUTPUT="$SCRIPT_DIR/full100_seeds11_23_42"
    SEEDS=(11 23 42)
    LIMIT_ARGS=(--limit 100)
    ;;
  *)
    echo "Usage: bash $0 {status|deploy|smoke30|full100|pilot3seed}" >&2
    exit 2
    ;;
esac

for port in "${PORTS[@]}"; do
  if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    echo "[FATAL] Solver endpoint is not ready: port ${port}" >&2
    echo "Run first: bash $0 deploy" >&2
    exit 1
  fi
done

exec "$PYTHON_BIN" "$PROJECT_ROOT/tools/evaluate_solver_v1_paired.py" \
  --test-file "$TEST_FILE" \
  --reference-guided "$REFERENCE_GUIDED" \
  --output "$OUTPUT" \
  --ports "${PORTS[@]}" \
  --model "$MODEL_NAME" \
  --seeds "${SEEDS[@]}" \
  --progress-every 10 \
  "${LIMIT_ARGS[@]}"
