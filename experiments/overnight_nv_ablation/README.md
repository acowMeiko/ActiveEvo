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
cd /path/to/ActiveEvo
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
