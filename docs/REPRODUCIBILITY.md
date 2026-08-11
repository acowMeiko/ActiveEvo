# ActiveEvo Reproducibility Guide

This document describes how to reproduce the paper experiments from the codebase.
It intentionally omits environment installation and dependency troubleshooting.

## 1. Scope

The reproduction pipeline contains four experiment groups:

1. Stage 1 capability sensing and learning-packet construction.
2. Stage 2 lifecycle evolution over difficulty levels.
3. Stage 3 principle-guided evaluation on benchmark test sets.
4. Ablations for learning-zone collection and retrieval representation.

The default Stage 1 code uses the entropy-aware capability signal:

```text
F = correct_count / K
H = mean normalized trajectory entropy
C = 1 - H
S = lambda * F + (1 - lambda) * C
```

Default thresholds:

```text
lambda = 0.7
alpha  = 0.25
beta   = 0.75

panic    if S < alpha
learning if alpha <= S <= beta
master   if S > beta
```

When entropy sensing is enabled, Stage 1 disables early stopping by default so
all tasks use a fixed `K=10` sampling budget.

## 2. Repository Layout

Important paths:

```text
data/
  train/                       # Stage 1 training/sample pool
  test/                        # Stage 3 benchmark inputs

output/
  stage1/01_sampling/          # K-trial capability sensing outputs
  stage1/02_learning_packets/  # mined learning packets by level
  stage3/remediation/          # level-wise rescue/evolution reports
  stage3/tests/                # final benchmark evaluation outputs

memory/
  principles/                  # evolved principle memory
  indices/                     # retrieval indices
  metadata/                    # memory metadata
  snapshots/                   # memory snapshots across rounds

paper/evidence/                # curated evidence files for paper tables
```

Primary code paths:

```text
core/agent_pipeline.py         # full Stage 1/2/3 orchestration
modules/stage1_sampler.py      # K-trial sampling and capability signal
core/api_call.py               # LLM calls and logprob entropy parsing
evaluate_principle_guided_test.py
benchmarks/run_suite.py
```

## 3. Global Configuration

Edit `config.env` or export equivalent variables before running scripts.
The model server must expose OpenAI-compatible `/v1/chat/completions` endpoints.

Recommended Stage 1 capability settings:

```bash
export MEMUA_STAGE1_USE_ENTROPY=1
export MEMUA_STAGE1_LOGPROBS=1
export MEMUA_STAGE1_TOP_LOGPROBS=5
export MEMUA_STAGE1_DISABLE_EARLY_STOP_FOR_ENTROPY=1

export MEMUA_CAPABILITY_LAMBDA=0.7
export MEMUA_CAPABILITY_ALPHA=0.25
export MEMUA_CAPABILITY_BETA=0.75
```

To run the legacy SR-only Stage 1 baseline:

```bash
export MEMUA_STAGE1_USE_ENTROPY=0
```

## 4. Main Experiment: Full Lifecycle Evolution

Run one backbone end to end:

```bash
python -m core.agent_pipeline \
  --levels 1 2 3 4 \
  --backbone qwen \
  --ports 8100 8101 8102 8103 8104 8105 8106 8107 \
  --model qwen \
  --concurrency 64 \
  --diff-concurrency 64 \
  --eval-concurrency 64
```

For Llama:

```bash
python -m core.agent_pipeline \
  --levels 1 2 3 4 \
  --backbone llama \
  --ports 8100 8101 8102 8103 8104 8105 8106 8107 \
  --model llama \
  --concurrency 64 \
  --diff-concurrency 64 \
  --eval-concurrency 64
```

Expected Stage 1 outputs:

```text
output/stage1/01_sampling/{backbone}_unified_sampling_results.json
output/stage1/02_learning_packets/{backbone}/stage1_summary.json
output/stage1/02_learning_packets/{backbone}/level1_learning_packets.json
output/stage1/02_learning_packets/{backbone}/level2_learning_packets.json
output/stage1/02_learning_packets/{backbone}/level3_learning_packets.json
output/stage1/02_learning_packets/{backbone}/level4_learning_packets.json
```

Expected Stage 3/evolution outputs:

```text
output/stage3/remediation/{backbone}_L2_eval_summary.json
output/stage3/remediation/{backbone}_L2_eval_results.json
output/stage3/remediation/{backbone}_L2_rescue.json
output/stage3/remediation/{backbone}_L3_eval_summary.json
output/stage3/remediation/{backbone}_L3_eval_results.json
output/stage3/remediation/{backbone}_L3_rescue.json
```

## 5. Stage 1 Only

Run Stage 1 for one backbone:

```bash
python -m core.agent_pipeline \
  --stage 1 \
  --levels 1 2 3 4 \
  --backbone qwen \
  --ports 8100 8101 8102 8103 8104 8105 8106 8107 \
  --model qwen \
  --concurrency 64 \
  --diff-concurrency 64
```

Run both backbones in parallel:

```bash
python run_stage1_dual.py \
  --sample-concurrency 128 \
  --diff-concurrency 96
```

Resume an interrupted Llama Stage 1 run:

```bash
python run_stage1_llama.py \
  --sample-concurrency 128 \
  --diff-concurrency 96 \
  --resume
```

Inspect capability sensing output:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("output/stage1/01_sampling/qwen_unified_sampling_results.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data["summary"], indent=2, ensure_ascii=False))

sample = data["results"][0]
for key in [
    "correct_count", "success_rate_fixed_k", "avg_entropy",
    "confidence_score", "capability_score", "sr_zone",
    "capability_zone", "zone", "assigned_level"
]:
    print(key, sample.get(key))
PY
```

Required Stage 1 fields for the paper:

```text
trial_records
correct_count
wrong_count
success_rate_fixed_k
avg_entropy
normalized_entropy
confidence_score
capability_score
sr_zone
capability_zone
zone
assigned_level
```

## 6. Stage 2 Only

If Stage 1 packets already exist, run Stage 2 for one level:

```bash
python -m core.agent_pipeline \
  --stage 2 \
  --dataset dpo_level1 \
  --backbone qwen \
  --ports 8100 8101 8102 8103 8104 8105 8106 8107 \
  --model qwen
```

Run all levels through the shell entrypoint:

```bash
bash run_stage2_all_levels.sh
```

Expected memory outputs:

```text
memory/principles/principles.json
memory/indices/
memory/metadata/
memory/snapshots/
```

## 7. Stage 3 Benchmark Evaluation

Run the full benchmark suite after memory is built:

```bash
bash run_all_tests.sh
```

This evaluates the available benchmark files:

```text
data/test/deepscaler.json
data/test/benchmarks/math500_stage3.json
data/test/benchmarks/gsm8k_stage3.json
data/test/benchmarks_split/bbh_logic_stage3.json
data/test/benchmarks_split/bbh_math_stage3.json
data/test/official/svamp/processed/test.json
data/test/official/general365/processed/test.json
```

Expected outputs:

```text
output/stage3/tests/deepscaler/principle_guided_test_summary.json
output/stage3/tests/math500/principle_guided_test_summary.json
output/stage3/tests/gsm8k/principle_guided_test_summary.json
output/stage3/tests/bbh_logic/principle_guided_test_summary.json
output/stage3/tests/bbh_math/principle_guided_test_summary.json
output/stage3/tests/svamp/principle_guided_test_summary.json
output/stage3/tests/general365/principle_guided_test_summary.json
```

Run one benchmark manually:

```bash
python evaluate_principle_guided_test.py \
  --input data/test/deepscaler.json \
  --output output/stage3/tests/deepscaler/principle_guided_test_results.json \
  --summary-output output/stage3/tests/deepscaler/principle_guided_test_summary.json \
  --memory-dir memory \
  --mode guided \
  --concurrency 64 \
  --model qwen \
  --ports 8100 8101 8102 8103 8104 8105 8106 8107 \
  --timeout 600 \
  --threshold 0.55 \
  --top-k 5
```

## 8. Ablations

### 8.1 No Learning Zone Collection

This ablation collects all non-empty zones rather than only the learning zone.

```bash
python ablation/no_learning_zone_collection/run_no_learning_zone_ablation.py \
  --backbone qwen \
  --ports 8100 8101 8102 8103 8104 8105 8106 8107 \
  --model qwen \
  --concurrency 64 \
  --diff-concurrency 64
```

Alternative script:

```bash
bash ablation/no_learning_zone_collection/run_all_benchmarks_no_learning_zone_ablation.sh
```

### 8.2 Single-Vector Principle Retrieval

This ablation replaces the full retrieval representation with a single-vector
principle representation.

```bash
bash ablation/single_vector_principle/run_all_benchmarks_single_vector_ablation.sh
```

## 9. Result Extraction

Summarize Stage 1 capability sensing:

```bash
python - <<'PY'
import json
from pathlib import Path

for backbone in ["qwen", "llama"]:
    path = Path(f"output/stage1/01_sampling/{backbone}_unified_sampling_results.json")
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data.get("summary", {})
    print(backbone)
    print("  total:", s.get("total"))
    print("  master:", s.get("master_zone"))
    print("  learning:", s.get("learning_zone"))
    print("  panic:", s.get("panic_zone"))
    print("  levels:", s.get("levels"))
    print("  entropy_available_trials:", s.get("entropy_available_trials"))
PY
```

Summarize Stage 3 benchmark accuracy:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("output/stage3/tests")
for path in sorted(root.glob("*/principle_guided_test_summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    name = path.parent.name
    accuracy = data.get("accuracy", data.get("acc", data.get("score")))
    correct = data.get("correct", data.get("correct_count"))
    total = data.get("total", data.get("num_samples"))
    print(f"{name}\taccuracy={accuracy}\tcorrect={correct}\ttotal={total}")
PY
```

Summarize level-wise rescue/evolution:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("output/stage3/remediation").glob("*_rescue.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    r = data.get("remediation", {})
    print(
        path.name,
        "rescued=", r.get("rescued_count"),
        "total_panic=", r.get("total_panic"),
        "new_packets=", r.get("new_packets"),
        "levels_updated=", r.get("levels_updated"),
    )
PY
```

## 10. Paper Table Mapping

Use these files to fill the paper tables:

```text
Stage 1 zone/capability table:
  output/stage1/01_sampling/{backbone}_unified_sampling_results.json
  output/stage1/02_learning_packets/{backbone}/stage1_summary.json

Lifecycle evolution table:
  output/stage3/remediation/{backbone}_L*_eval_summary.json
  output/stage3/remediation/{backbone}_L*_rescue.json

Benchmark generalization table:
  output/stage3/tests/*/principle_guided_test_summary.json

Ablation tables:
  ablation/no_learning_zone_collection/
  ablation/single_vector_principle/
```

Minimum values to report:

```text
Stage 1:
  K
  lambda/alpha/beta
  master/learning/panic counts
  capability_zone_counts
  level distribution
  entropy_available_trials

Stage 3:
  dataset
  mode
  top_k
  threshold
  total
  correct
  accuracy

Evolution:
  current_level
  target_level
  total_panic
  rescued_count
  new_packets
  levels_updated
```

## 11. Reproducibility Checklist

Before claiming a reproduced result, verify:

```text
[ ] Stage 1 output contains capability_score and avg_entropy.
[ ] Stage 1 config records use_entropy_signal=true.
[ ] Stage 1 config records disable_early_stop_for_entropy=true.
[ ] K is fixed at 10 for the reported main experiment.
[ ] All reported benchmark summaries come from output/stage3/tests.
[ ] All ablation results are stored separately from main outputs.
[ ] Paper tables cite exact JSON files under output/ or paper/evidence/.
[ ] Logs and summaries preserve backbone, model name, ports, top_k, threshold, and concurrency.
```

## 12. Archiving Evidence

For submission, copy final JSON summaries into `paper/evidence/`:

```bash
mkdir -p paper/evidence/reproduced
cp output/stage1/01_sampling/*_unified_sampling_results.json paper/evidence/reproduced/
cp output/stage1/02_learning_packets/*/stage1_summary.json paper/evidence/reproduced/
cp output/stage3/remediation/*_eval_summary.json paper/evidence/reproduced/ 2>/dev/null || true
cp output/stage3/remediation/*_rescue.json paper/evidence/reproduced/ 2>/dev/null || true
cp output/stage3/tests/*/principle_guided_test_summary.json paper/evidence/reproduced/ 2>/dev/null || true
```

Keep the raw `output/` directory unchanged until all paper numbers have been
cross-checked against the JSON evidence.
