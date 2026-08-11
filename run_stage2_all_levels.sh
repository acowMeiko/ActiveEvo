#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/config.env"
source "$ROOT_DIR/api_call/runtime_config.sh"
load_deployment_runtime "$ROOT_DIR/deploy.conf"

IFS=' ' read -r -a LEVEL_ARRAY <<< "$LEVELS"
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
export MEMUA_STAGE2_DSD_CONCURRENCY="$STAGE2_DSD_CONCURRENCY"
export MEMUA_STAGE2_GENERATION_CONCURRENCY="$STAGE2_GENERATION_CONCURRENCY"
export MEMUA_STAGE2_CANDIDATE_CONCURRENCY="$STAGE2_CANDIDATE_CONCURRENCY"
export MEMUA_STAGE2_VALIDATION_SEED="$STAGE2_VALIDATION_SEED"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

mkdir -p "$LOG_DIR/stage2" "$MEMORY_DIR"

echo "[INFO] MemUA Stage 2"
echo "[INFO] Root:       $ROOT_DIR"
echo "[INFO] Python:     $PYTHON_BIN"
echo "[INFO] Model:      $MODEL_NAME"
echo "[INFO] Levels:     ${LEVEL_ARRAY[*]}"
echo "[INFO] Ports:      ${PORT_ARRAY[*]}"
echo "[INFO] Learning:   $ROOT_DIR/learning_packet"
echo "[INFO] Memory:     $ROOT_DIR/$MEMORY_DIR"
echo "[INFO] Embedding:  $EMBEDDING_MODEL_PATH ($EMBEDDING_DEVICE)"

for level in "${LEVEL_ARRAY[@]}"; do
  dataset="dpo_level${level}"
  packets="$ROOT_DIR/learning_packet/$dataset/04_learning_packets/learning_packets.json"
  log_file="$ROOT_DIR/$LOG_DIR/stage2/${dataset}.log"

  if [[ ! -f "$packets" ]]; then
    echo "[ERROR] Missing learning packets: $packets" >&2
    exit 1
  fi

  echo
  echo "[INFO] Running Stage 2 for $dataset"
  "$PYTHON_BIN" -u pipeline.py \
    --stage 2 \
    --dataset "$dataset" \
    --model "$MODEL_NAME" \
    --memory-dir "$MEMORY_DIR" \
    --stage2-concurrency "$STAGE2_CONCURRENCY" \
    --batch-size "$BATCH_SIZE" \
    --max-trials "$MAX_TRIALS" \
    --timeout "$TIMEOUT" \
    --ports "${PORT_ARRAY[@]}" 2>&1 | tee "$log_file"
done

echo
echo "[INFO] Stage 2 finished."
echo "[INFO] Stage 2 outputs: $ROOT_DIR/memory/dpo_level*/"
echo "[INFO] Live v3 Store:    $ROOT_DIR/$MEMORY_DIR/"
