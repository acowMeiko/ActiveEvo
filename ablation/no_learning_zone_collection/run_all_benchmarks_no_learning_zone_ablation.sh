#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/ablation/no_learning_zone_collection"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BUILD_OUTPUT_DIR="${BUILD_OUTPUT_DIR:-$SCRIPT_DIR/output/stage12_no_learning_zone_$RUN_TAG}"
MEMORY_DIR="${MEMORY_DIR:-$BUILD_OUTPUT_DIR/memory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/stage3_benchmarks_no_learning_zone_$RUN_TAG}"
PORTS_ENV="${PORTS_ENV:-8000 8001 8002 8003}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_MEMORY="${BUILD_MEMORY:-1}"

cd "$ROOT_DIR"

if [[ "$BUILD_MEMORY" == "1" ]]; then
  "$PYTHON_BIN" -u "$SCRIPT_DIR/run_no_learning_zone_ablation.py" \
    --output-dir "$BUILD_OUTPUT_DIR" \
    --memory-dir "$MEMORY_DIR" \
    --levels ${LEVELS:-1 2 3 4} \
    --skip-stage3 \
    --ports $PORTS_ENV \
    --stage1-concurrency "${STAGE1_CONCURRENCY:-10}" \
    --stage1-diff-concurrency "${STAGE1_DIFF_CONCURRENCY:-10}" \
    --stage1-batch-size "${STAGE1_BATCH_SIZE:-5}" \
    --stage1-max-trials "${STAGE1_MAX_TRIALS:-15}" \
    --stage2-concurrency "${STAGE2_CONCURRENCY:-8}"
fi

mkdir -p "$OUTPUT_ROOT"

BENCHMARKS=(
  "gsm8k|data/test/benchmarks/gsm8k_stage3.json"
  "math500|data/test/benchmarks/math500_stage3.json"
  "bbh_logic|data/test/benchmarks_split/bbh_logic_stage3.json"
  "bbh_math|data/test/benchmarks_split/bbh_math_stage3.json"
  "svamp|data/test/official/svamp/processed/test.json"
  "general365|data/test/official/general365/processed/test.json"
  "deepscaler|data/test/deepscaler.json"
)

for entry in "${BENCHMARKS[@]}"; do
  name="${entry%%|*}"
  input="${entry#*|}"
  echo "[INFO] Running no-learning-zone benchmark: $name"
  "$PYTHON_BIN" -u "$SCRIPT_DIR/run_no_learning_zone_ablation.py" \
    --output-dir "$OUTPUT_ROOT/$name" \
    --memory-dir "$MEMORY_DIR" \
    --levels "${EVAL_LEVEL:-4}" \
    --test-input "$input" \
    --skip-stage1 \
    --skip-stage2 \
    --ports $PORTS_ENV \
    --stage3-concurrency "${STAGE3_CONCURRENCY:-24}" \
    --threshold "${THRESHOLD:-0.55}" \
    --top-k "${TOP_K:-5}" \
    --retry-on-wrong "${RETRY_ON_WRONG:-2}"
done

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ROOT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {}
for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    candidates = list(bench_dir.glob("stage3/*/01_guided_eval/principle_guided_test_summary.json"))
    if not candidates:
        continue
    summary = json.loads(candidates[0].read_text(encoding="utf-8"))
    rows[bench_dir.name] = {
        "total": summary.get("total"),
        "correct": summary.get("correct"),
        "accuracy": summary.get("accuracy"),
        "runtime_metrics": summary.get("runtime_metrics", {}),
    }
(root / "all_benchmark_no_learning_zone_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
