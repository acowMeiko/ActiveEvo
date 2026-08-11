# Overnight NV Ablation

This experiment builds two isolated Level 1→4 Memory stores from the same
frozen Principle candidate banks, then evaluates Baseline, NV-on, and NV-off on
all 400 DeepScaler questions with independent seeds 42/43/44 at temperature
0.2. Seed 42 is pass@1; any correct rollout is pass@3. A wrong answer never
causes a retry. Infrastructure failures may rerun once on a different endpoint.

## Start

The launcher reads `../../deploy.conf` and, when present,
`../../api_call/logs/active_deployment.env`. It never starts, stops, restarts, or
cleans a model server. If another Stage 2/3 process exists, it waits without
sending a signal.

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
nohup bash experiments/overnight_nv_ablation/run.sh \
  > experiments/overnight_nv_ablation/launcher.log 2>&1 &
```

Run the 4x25 format/management Pilot only. It reuses deterministic DSD
embeddings when their packet hash matches, archives the old live Memory, and
stops before paired Solver validation and Stage 3:

```bash
RUN_ID=format_pilot_100_v2 \
PILOT_ONLY=1 \
PILOT_SIZE_PER_LEVEL=25 \
PILOT_SEED=42 \
WAIT_FOR_EXISTING=1 \
ARCHIVE_OLD_MEMORY=0 \
MEMUA_EMBEDDING_DEVICE=cpu \
SHOW_PROGRESS=1 \
bash experiments/overnight_nv_ablation/run.sh
```

The acceptance report is written to
`runs/format_pilot_100/pilot_report.md`. A failed acceptance check exits with
status 10 after preserving all caches, raw outputs, and candidate mappings.

Preview all commands and preflight checks without launching Python jobs:

```bash
DRY_RUN=1 RUN_ID=dryrun bash experiments/overnight_nv_ablation/run.sh
```

Resume a named run (completed candidates and all completed rollouts, including
wrong answers, are skipped):

```bash
RUN_ID=20260809_230000 bash experiments/overnight_nv_ablation/run.sh
```

Useful controls:

```text
RUN_ID=<name>                 Reuse or name the isolated run directory
WAIT_FOR_EXISTING=1          Wait for pre-existing Stage 2/3 (default)
PILOT_ONLY=1                 Run only the 100-item format/management Pilot
PILOT_SIZE_PER_LEVEL=25      Stable hash sample size for each Level
PILOT_SEED=42                Sampling and validation seed
ARCHIVE_OLD_MEMORY=1         Move the previous live Memory to archive/ first
OVERNIGHT_CONCURRENCY=8      Concurrent question transactions
OVERNIGHT_CANDIDATE_CONCURRENCY=16  Bounded end-to-end Stage-2 candidates
DEPLOY_CONFIG=<path>         Deployment configuration path
DRY_RUN=1                    Print planned commands only
```

## Individual Stage commands

Stage 1 packets are existing frozen inputs for this experiment. A normal Stage
1 command remains:

```bash
python pipeline.py --stage 1 --ports 8100 8101 8102 8103 --model llama
```

Stage 2 now supports reusable preparation and the NV switch:

```bash
# Build/cache only the three DSD routing-facet embeddings
python pipeline.py --stage 2 -d dpo_level1 --backbone llama --ports 8100 8101 8102 8103 \
  --stage2-candidate-concurrency 16 \
  --dsd-cache-dir /path/dsd_level1 --prepare-dsd-only \
  --memory-dir /path/prepare_memory --stage2-output-dir /path/audit_dsd

# Freeze generation + Quality results
python pipeline.py --stage 2 -d dpo_level1 --backbone llama --ports 8100 8101 8102 8103 \
  --dsd-cache-dir /path/dsd_level1 --prepare-candidates-only \
  --candidate-bank /path/candidates_level1.json \
  --memory-dir /path/prepare_memory --stage2-output-dir /path/audit_candidates

# Consume the exact same bank with or without Neighbor Validation
python pipeline.py --stage 2 -d dpo_level1 --backbone llama --ports 8100 8101 8102 8103 \
  --dsd-cache-dir /path/dsd_level1 --candidate-bank /path/candidates_level1.json \
  --neighbor-validation on --memory-dir /path/nv_on/store --stage2-output-dir /path/nv_on/audit

python pipeline.py --stage 2 -d dpo_level1 --backbone llama --ports 8100 8101 8102 8103 \
  --dsd-cache-dir /path/dsd_level1 --candidate-bank /path/candidates_level1.json \
  --neighbor-validation off --memory-dir /path/nv_off/store --stage2-output-dir /path/nv_off/audit
```

Official Stage 3 uses the dedicated evaluator, not the legacy
`retry_on_wrong` evaluator:

```bash
# Base capability: no Memory, same Solver-v1.1 contract later used by Guided.
# Runs all 400 DeepScaler items independently with seeds 42/43/44.
python experiments/overnight_nv_ablation/pass3_eval.py \
  --variant baseline --test-file data/test/deepscaler.json \
  --output-dir experiments/overnight_nv_ablation/runs/base_solver_v1_1_3seed/stage3/baseline \
  --ports 8100 8101 8102 8103 --model llama \
  --concurrency 8 --temperature 0.2

python experiments/overnight_nv_ablation/pass3_eval.py \
  --variant nv_on --test-file data/test/deepscaler.json \
  --memory-dir /path/nv_on/store --output-dir /path/stage3/nv_on \
  --ports 8100 8101 8102 8103 --model llama --temperature 0.2
```

## Manual Queue: Six-Field Card + Stage-2 NV Ablation

This is the current recommended run path. It does not create another launcher
script. Open a foreground tmux session so progress bars stay visible:

```bash
tmux new -s activeevo_nv
```

Paste the full queue below inside tmux. It reuses the existing DSD/BGE cache and
builds a stable 600-item Stage-2 selection with seed 42: Level 1 uses 50 items,
Level 2/3 use 200 items each, and Level 4 uses 150 items. It regenerates one
shared candidate bank, branches exactly those candidates into NV-off and NV-on,
runs 10-item smoke evaluations, then runs both guided variants on all 400
DeepScaler questions. The completed baseline is reused only after
prompt/seed/temperature/completion checks pass.

```bash
# BEGIN ACTIVE_EVO_SIXFIELD_QUEUE
set -euo pipefail

cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code

RUN_ID=sixfield_nv_ablation_$(date +%Y%m%d_%H%M%S)
RUN_ROOT="experiments/overnight_nv_ablation/runs/${RUN_ID}"
QUEUE_LOG="${RUN_ROOT}/queue.log"
DSD_BASE="experiments/overnight_nv_ablation/runs/format_pilot_100_v2/dsd_cache"
SEL_BASE="${RUN_ROOT}/selection"
BASELINE_SRC="experiments/overnight_nv_ablation/runs/base_solver_v1_1_3seed/stage3/baseline"
TEST_FILE="data/test/deepscaler.json"
PORTS="8100 8101 8102 8103"
MODEL="llama"
CONCURRENCY=8
PYTHON_BIN="/home/czhu/miniconda3/envs/hcz/bin/python"

# vLLM already occupies GPUs 0-3. Keep BGE-M3/FAISS embeddings on CPU so the
# Stage-2 process cannot compete with the model servers for GPU 0 memory.
export MEMUA_EMBEDDING_DEVICE=cpu
export MEMUA_FORCE_EMBEDDING=1
export MEMUA_ALLOW_HASH_FALLBACK=0

mkdir -p "${RUN_ROOT}"/{frozen_candidates,stage2,candidate_prepare_memory,memory/nv_off,memory/nv_on,stage3,smoke}
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "Embedding device: ${MEMUA_EMBEDDING_DEVICE}"

echo "[1/10] Code checks"
${PYTHON_BIN} -m compileall template modules experiments/overnight_nv_ablation tests pipeline.py
${PYTHON_BIN} - <<'PY'
import importlib.util
import inspect
import tempfile
from pathlib import Path

test_files = [
    Path("tests/test_principle_schema.py"),
    Path("tests/test_application_gate.py"),
    Path("tests/test_solver_v1.py"),
    Path("tests/test_stage2_v3_contract.py"),
]
for path in test_files:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            params = inspect.signature(fn).parameters
            if not params:
                fn()
            elif list(params) == ["tmp_path"]:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                raise RuntimeError(f"Unsupported test fixture in {path}.{name}: {list(params)}")
            print(f"PASS {path}.{name}")
PY

echo "[2/10] Build 600-item stratified selection and shared candidate bank"
${PYTHON_BIN} experiments/overnight_nv_ablation/pilot_tools.py manifest \
  --run-root "${RUN_ROOT}" --model "${MODEL}" \
  --level-counts 50 200 200 150 --seed 42

for L in 1 2 3 4; do
  ${PYTHON_BIN} pipeline.py --stage 2 -d "dpo_level${L}" --backbone "${MODEL}" --ports ${PORTS} \
    --stage2-candidate-concurrency 16 --stage2-concurrency 8 \
    --dsd-cache-dir "${DSD_BASE}/dpo_level${L}" \
    --stage2-selection-file "${SEL_BASE}/level${L}.json" \
    --prepare-candidates-only \
    --candidate-bank "${RUN_ROOT}/frozen_candidates/level${L}.json" \
    --memory-dir "${RUN_ROOT}/candidate_prepare_memory/level${L}" \
    --stage2-output-dir "${RUN_ROOT}/stage2/candidate_bank/level${L}"
done

echo "[3/10] Build NV-off store from the shared bank"
for L in 1 2 3 4; do
  ${PYTHON_BIN} pipeline.py --stage 2 -d "dpo_level${L}" --backbone "${MODEL}" --ports ${PORTS} \
    --stage2-candidate-concurrency 16 --stage2-concurrency 8 \
    --dsd-cache-dir "${DSD_BASE}/dpo_level${L}" \
    --stage2-selection-file "${SEL_BASE}/level${L}.json" \
    --candidate-bank "${RUN_ROOT}/frozen_candidates/level${L}.json" \
    --neighbor-validation off \
    --memory-dir "${RUN_ROOT}/memory/nv_off/store" \
    --stage2-output-dir "${RUN_ROOT}/memory/nv_off/audit/level${L}"
done

echo "[4/10] Build NV-on store from the same shared bank"
for L in 1 2 3 4; do
  ${PYTHON_BIN} pipeline.py --stage 2 -d "dpo_level${L}" --backbone "${MODEL}" --ports ${PORTS} \
    --stage2-candidate-concurrency 16 --stage2-concurrency 8 \
    --dsd-cache-dir "${DSD_BASE}/dpo_level${L}" \
    --stage2-selection-file "${SEL_BASE}/level${L}.json" \
    --candidate-bank "${RUN_ROOT}/frozen_candidates/level${L}.json" \
    --neighbor-validation on \
    --memory-dir "${RUN_ROOT}/memory/nv_on/store" \
    --stage2-output-dir "${RUN_ROOT}/memory/nv_on/audit/level${L}"
done

echo "[5/10] Verify shared candidate banks and reusable baseline"
${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from modules.execute_module import SOLVER_V1_PROMPT_HASHES

run_root = Path("${RUN_ROOT}")
baseline = Path("${BASELINE_SRC}")
expected_counts = {1: 50, 2: 200, 3: 200, 4: 150}
summary = json.loads((baseline / "eval_summary.json").read_text(encoding="utf-8"))
assert summary["total"] == 400, summary
assert summary["temperature"] == 0.2, summary
assert summary["seeds"] == [42, 43, 44], summary
assert summary["infrastructure_failure_count"] == 0, summary
assert summary["prompt_hashes"]["reasoner_base"] == SOLVER_V1_PROMPT_HASHES["reasoner_base"]
for level in range(1, 5):
    bank = json.loads((run_root / "frozen_candidates" / f"level{level}.json").read_text(encoding="utf-8"))
    assert bank["candidate_count"] == expected_counts[level], (level, bank["candidate_count"])
print("Baseline and shared candidate banks verified.")
PY
ln -sfn "$(pwd)/${BASELINE_SRC}" "${RUN_ROOT}/stage3/baseline"

echo "[6/10] Smoke NV-off, 10 questions x 3 seeds"
${PYTHON_BIN} experiments/overnight_nv_ablation/pass3_eval.py \
  --variant nv_off --test-file "${TEST_FILE}" \
  --memory-dir "${RUN_ROOT}/memory/nv_off/store" \
  --output-dir "${RUN_ROOT}/smoke/nv_off" \
  --ports ${PORTS} --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" --temperature 0.2 --limit 10

echo "[7/10] Smoke NV-on, 10 questions x 3 seeds"
${PYTHON_BIN} experiments/overnight_nv_ablation/pass3_eval.py \
  --variant nv_on --test-file "${TEST_FILE}" \
  --memory-dir "${RUN_ROOT}/memory/nv_on/store" \
  --output-dir "${RUN_ROOT}/smoke/nv_on" \
  --ports ${PORTS} --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" --temperature 0.2 --limit 10

echo "[8/10] Smoke acceptance checks"
${PYTHON_BIN} - <<PY
import json
from pathlib import Path

run_root = Path("${RUN_ROOT}")
summaries = []
for name in ("nv_off", "nv_on"):
    summary = json.loads((run_root / "smoke" / name / "eval_summary.json").read_text(encoding="utf-8"))
    summaries.append(summary)
    assert summary["infrastructure_failure_count"] == 0, summary
    assert summary["solver_decision_missing_count"] <= 1, summary
    assert summary["adapter_attempt_count"] == 0 or summary["adapter_accept_count"] / summary["adapter_attempt_count"] >= 0.95, summary
assert sum(s["solver_use_count"] for s in summaries) >= 1, summaries
assert sum(s["solver_ignore_count"] for s in summaries) >= 1, summaries
print("Smoke passed: parser, adapter, and use/ignore checks are acceptable.")
PY

echo "[9/10] Full NV-off 400 x 3"
${PYTHON_BIN} experiments/overnight_nv_ablation/pass3_eval.py \
  --variant nv_off --test-file "${TEST_FILE}" \
  --memory-dir "${RUN_ROOT}/memory/nv_off/store" \
  --output-dir "${RUN_ROOT}/stage3/nv_off" \
  --ports ${PORTS} --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" --temperature 0.2

echo "[10/10] Full NV-on 400 x 3 and final statistics"
${PYTHON_BIN} experiments/overnight_nv_ablation/pass3_eval.py \
  --variant nv_on --test-file "${TEST_FILE}" \
  --memory-dir "${RUN_ROOT}/memory/nv_on/store" \
  --output-dir "${RUN_ROOT}/stage3/nv_on" \
  --ports ${PORTS} --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" --temperature 0.2

${PYTHON_BIN} experiments/overnight_nv_ablation/analyze.py --run-root "${RUN_ROOT}"

echo "Done. Summary:"
sed -n '1,120p' "${RUN_ROOT}/overnight_summary.md"
# END ACTIVE_EVO_SIXFIELD_QUEUE
```

### Resume the completed 600-item Candidate Bank from run 004704

This queue skips Principle generation and reuses the frozen 50/200/200/150
Candidate Bank. It resumes at NV-off, then runs NV-on, smoke checks, both full
guided evaluations, and the final paired analysis.

```bash
# BEGIN ACTIVE_EVO_RESUME_004704
set -euo pipefail

cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code

RUN_ROOT="experiments/overnight_nv_ablation/runs/sixfield_nv_ablation_20260811_004704"
DSD_BASE="experiments/overnight_nv_ablation/runs/format_pilot_100_v2/dsd_cache"
SEL_BASE="${RUN_ROOT}/selection"
BASELINE_SRC="experiments/overnight_nv_ablation/runs/base_solver_v1_1_3seed/stage3/baseline"
TEST_FILE="data/test/deepscaler.json"
PORTS="8100 8101 8102 8103"
MODEL="llama"
CONCURRENCY=8
PYTHON_BIN="/home/czhu/miniconda3/envs/hcz/bin/python"

export MEMUA_EMBEDDING_DEVICE=cpu
export MEMUA_FORCE_EMBEDDING=1
export MEMUA_ALLOW_HASH_FALLBACK=0

exec > >(tee -a "${RUN_ROOT}/resume_queue.log") 2>&1

echo "[resume 1/8] Preflight endpoints, Candidate Bank, and CPU embedding"
test "${MEMUA_EMBEDDING_DEVICE}" = "cpu"
for P in ${PORTS}; do
  curl --silent --show-error --fail --max-time 5 "http://127.0.0.1:${P}/v1/models" >/dev/null
done
${PYTHON_BIN} - <<PY
import json
from pathlib import Path
root = Path("${RUN_ROOT}")
expected = {1: 50, 2: 200, 3: 200, 4: 150}
for level, count in expected.items():
    path = root / "frozen_candidates" / f"level{level}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == count, (level, payload["candidate_count"], count)
print("Candidate Bank verified:", expected, "total=600")
PY

echo "[resume 2/8] Build NV-off Store from frozen candidates"
for L in 1 2 3 4; do
  ${PYTHON_BIN} pipeline.py --stage 2 -d "dpo_level${L}" --backbone "${MODEL}" --ports ${PORTS} \
    --stage2-candidate-concurrency 16 --stage2-concurrency 8 \
    --dsd-cache-dir "${DSD_BASE}/dpo_level${L}" \
    --stage2-selection-file "${SEL_BASE}/level${L}.json" \
    --candidate-bank "${RUN_ROOT}/frozen_candidates/level${L}.json" \
    --neighbor-validation off \
    --memory-dir "${RUN_ROOT}/memory/nv_off/store" \
    --stage2-output-dir "${RUN_ROOT}/memory/nv_off/audit/level${L}"
done

echo "[resume 3/8] Build NV-on Store from the same frozen candidates"
for L in 1 2 3 4; do
  ${PYTHON_BIN} pipeline.py --stage 2 -d "dpo_level${L}" --backbone "${MODEL}" --ports ${PORTS} \
    --stage2-candidate-concurrency 16 --stage2-concurrency 8 \
    --dsd-cache-dir "${DSD_BASE}/dpo_level${L}" \
    --stage2-selection-file "${SEL_BASE}/level${L}.json" \
    --candidate-bank "${RUN_ROOT}/frozen_candidates/level${L}.json" \
    --neighbor-validation on \
    --memory-dir "${RUN_ROOT}/memory/nv_on/store" \
    --stage2-output-dir "${RUN_ROOT}/memory/nv_on/audit/level${L}"
done

echo "[resume 4/8] Verify reusable baseline and link it"
${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from modules.execute_module import SOLVER_V1_PROMPT_HASHES
baseline = Path("${BASELINE_SRC}")
summary = json.loads((baseline / "eval_summary.json").read_text(encoding="utf-8"))
assert summary["total"] == 400, summary
assert summary["temperature"] == 0.2, summary
assert summary["seeds"] == [42, 43, 44], summary
assert summary["infrastructure_failure_count"] == 0, summary
assert summary["prompt_hashes"]["reasoner_base"] == SOLVER_V1_PROMPT_HASHES["reasoner_base"]
print("Baseline verified.")
PY
mkdir -p "${RUN_ROOT}/stage3" "${RUN_ROOT}/smoke"
ln -sfn "$(pwd)/${BASELINE_SRC}" "${RUN_ROOT}/stage3/baseline"

echo "[resume 5/8] Smoke NV-off and NV-on (10 questions x 3 seeds)"
for V in nv_off nv_on; do
  ${PYTHON_BIN} experiments/overnight_nv_ablation/pass3_eval.py \
    --variant "${V}" --test-file "${TEST_FILE}" \
    --memory-dir "${RUN_ROOT}/memory/${V}/store" \
    --output-dir "${RUN_ROOT}/smoke/${V}" \
    --ports ${PORTS} --model "${MODEL}" \
    --concurrency "${CONCURRENCY}" --temperature 0.2 --limit 10
done

echo "[resume 6/8] Smoke acceptance checks"
${PYTHON_BIN} - <<PY
import json
from pathlib import Path
root = Path("${RUN_ROOT}")
summaries = []
for name in ("nv_off", "nv_on"):
    summary = json.loads((root / "smoke" / name / "eval_summary.json").read_text(encoding="utf-8"))
    summaries.append(summary)
    assert summary["infrastructure_failure_count"] == 0, summary
    assert summary["solver_decision_missing_count"] <= 1, summary
    assert summary["adapter_attempt_count"] == 0 or summary["adapter_accept_count"] / summary["adapter_attempt_count"] >= 0.95, summary
assert sum(s["solver_use_count"] for s in summaries) >= 1, summaries
assert sum(s["solver_ignore_count"] for s in summaries) >= 1, summaries
print("Smoke checks passed.")
PY

echo "[resume 7/8] Full NV-off and NV-on (400 questions x 3 seeds)"
for V in nv_off nv_on; do
  ${PYTHON_BIN} experiments/overnight_nv_ablation/pass3_eval.py \
    --variant "${V}" --test-file "${TEST_FILE}" \
    --memory-dir "${RUN_ROOT}/memory/${V}/store" \
    --output-dir "${RUN_ROOT}/stage3/${V}" \
    --ports ${PORTS} --model "${MODEL}" \
    --concurrency "${CONCURRENCY}" --temperature 0.2
done

echo "[resume 8/8] Final paired analysis"
${PYTHON_BIN} experiments/overnight_nv_ablation/analyze.py --run-root "${RUN_ROOT}"
sed -n '1,160p' "${RUN_ROOT}/overnight_summary.md"
# END ACTIVE_EVO_RESUME_004704
```

Deployment remains controlled only by `deploy.conf`:

```bash
bash api_call/run.sh deploy.conf
bash api_call/test.sh deploy.conf
bash api_call/stop.sh deploy.conf
```

## Outputs

Each run is written below `runs/<RUN_ID>/`. `dsd_cache/` and
`frozen_candidates/` are shared; `memory/nv_on` and `memory/nv_off` are fully
isolated. Live `store/` contains only active/deprecated records, while complete
candidate, Gate, paired-validation, and repair traces stay in `audit/`.

The final files are `overnight_summary.json`, `overnight_summary.md`, and
`overnight.log`. Each Stage 3 directory contains `eval_results.json`,
`eval_summary.json`, `transport_failures.jsonl`, `raw.log`, and a `run.log`
whose first block reports Memory hash, pass@1, pass@3, infrastructure failures,
retrieval use, and runtime.
