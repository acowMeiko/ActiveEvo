#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/config.env"

PYTHON_BIN="${PYTHON_BIN:-/home/czhu/miniconda3/envs/hcz/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_benchmarks}"
MEMORY_ROOT="${MEMORY_ROOT:-$SCRIPT_DIR/output/stage2_single_vector_memory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/stage3_benchmarks_single_vector_${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/stage3_benchmarks_single_vector_${RUN_TAG}}"

PORTS_ENV="${PORTS_ENV:-$PORTS}"
read -r -a PORT_ARRAY <<< "$PORTS_ENV"

CONCURRENCY="${CONCURRENCY:-24}"
TIMEOUT="${TIMEOUT:-600}"
TOP_K="${TOP_K:-5}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.35}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-1024}"
RETRY_ON_WRONG="${RETRY_ON_WRONG:-2}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"

export MEMUA_EMBEDDING_MODEL="${MEMUA_EMBEDDING_MODEL:-$EMBEDDING_MODEL}"
export MEMUA_EMBEDDING_MODEL_PATH="${MEMUA_EMBEDDING_MODEL_PATH:-$EMBEDDING_MODEL_PATH}"
export MEMUA_EMBEDDING_DEVICE="${MEMUA_EMBEDDING_DEVICE:-$EMBEDDING_DEVICE}"

if [[ ! -f "$MEMORY_ROOT/cumulative_level4/single_vector_principles.npz" ]]; then
  echo "[INFO] Missing single-vector memory; building it first."
  "$PYTHON_BIN" -u "$SCRIPT_DIR/build_single_vector_memory.py" \
    --source-root "/home/czhu/workplace/task/MemUA/output_llama/stage3_rerun_20260423_deepscaler_llama_gpu1_3/memory_views" \
    --output-root "$MEMORY_ROOT" \
    --embedding-model "$MEMUA_EMBEDDING_MODEL" \
    --embedding-device "$MEMUA_EMBEDDING_DEVICE"
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

declare -a DATASETS=(
  "gsm8k|data/test/benchmarks/gsm8k_stage3.json"
  "math500|data/test/benchmarks/math500_stage3.json"
  "bbh_logic|data/test/benchmarks_split/bbh_logic_stage3.json"
  "bbh_math|data/test/benchmarks_split/bbh_math_stage3.json"
  "svamp|data/test/official/svamp/processed/test.json"
  "general365|data/test/official/general365/processed/test.json"
  "deepscaler|data/test/deepscaler.json"
)

echo "[INFO] Memory root: $MEMORY_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Logs:        $LOG_ROOT"
echo "[INFO] Ports:       ${PORT_ARRAY[*]}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r name input_path <<< "$entry"
  if [[ ! -f "$input_path" ]]; then
    echo "[ERROR] Missing benchmark input: $ROOT_DIR/$input_path" >&2
    exit 1
  fi

  dataset_out="$OUTPUT_ROOT/$name"
  dataset_log="$LOG_ROOT/${name}.log"
  mkdir -p "$dataset_out"

  echo "[INFO] Running benchmark: $name"
  "$PYTHON_BIN" -u "$SCRIPT_DIR/run_single_vector_stage3.py" \
    --input "$input_path" \
    --memory-root "$MEMORY_ROOT" \
    --output-dir "$dataset_out" \
    --levels 4 \
    --model "$MODEL_NAME" \
    --ports "${PORT_ARRAY[@]}" \
    --concurrency "$CONCURRENCY" \
    --timeout "$TIMEOUT" \
    --retry-on-wrong "$RETRY_ON_WRONG" \
    --answer-max-tokens "$ANSWER_MAX_TOKENS" \
    --top-k "$TOP_K" \
    --score-threshold "$SCORE_THRESHOLD" \
    --embedding-model "$MEMUA_EMBEDDING_MODEL" \
    --embedding-device "$MEMUA_EMBEDDING_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --test-scope-key "benchmark_${name}_single_vector" \
    2>&1 | tee "$dataset_log"
done

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for summary_path in sorted(root.glob("*/cumulative_level4_single_vector_summary.json")):
    name = summary_path.parent.name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows[name] = {
        "total": summary.get("total", 0),
        "correct": summary.get("correct", 0),
        "accuracy": summary.get("accuracy", 0.0),
        "retrieval_used_count": summary.get("retrieval_used_count", 0),
        "retrieval_used_ratio": summary.get("retrieval_used_ratio", 0.0),
        "retry_used_count": summary.get("retry_used_count", 0),
        "rescued_by_retry_count": summary.get("rescued_by_retry_count", 0),
    }

(root / "all_benchmark_single_vector_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

echo "[INFO] Done. Combined summary: $OUTPUT_ROOT/all_benchmark_single_vector_summary.json"
