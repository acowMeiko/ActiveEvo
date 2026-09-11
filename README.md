# ActiveEvo

ActiveEvo studies experience-driven mathematical reasoning with a capability
signal and structured Memory retrieval.

```text
Stage 1: weak-model rollouts -> capability levels -> LearningBundle
Stage 2: trajectory evidence -> Principle -> quality audit -> Memory Store
Stage 3: SPD retrieval -> runtime gate/adapter -> Solver -> pass@1/pass@3
```

Stage 2 does not use Neighbor Validation. It first measures whether each
Principle improves the source Llama under the frozen post-draft (Forced-C)
protocol, then uses Kimi only as a content-safety audit. A passed record is
stored as `pending_transfer_validation`; it is not searchable until Stage 3
proves transfer to unseen questions.

## Three-stage workflow

The following diagrams describe the code path implemented by the current
branch. They deliberately exclude the removed Neighbor Validation pipeline.

### Stage 1 — Capability signal and learning evidence

```text
Input dataset
Each record must contain:
- question
- answer
- difficulty_level
        |
        v
Dataset validation
validate_stage1_inputs()
Check:
- file and JSON structure
- expected number of questions
- dataset SHA-256
- required fields
- one or more Actor endpoints
        |
        v
Independent weak-model rollouts
Stage1Sampler.sample_all()
For every question:
- run exactly 10 rollouts
- seeds = 42 ... 51
- no early stop after a correct answer
- use the frozen Solver-v1.1 Reasoner -> Finalizer protocol
- save Reasoner / Finalizer / final_answer / endpoint / Reasoner entropy
- infrastructure failure retries the same seed on another endpoint
        |
        v
Final-answer extraction and grading
extract_final_answer()
check_answer_detailed()
The final boxed answer determines is_correct
        |
        v
Capability zoning
Stage1Sampler.finalize_result()

9-10/10 correct
-> Master Zone
-> retained for capability analysis, not used to build Learning Bundles

2-8/10 correct
-> Learning Zone
-> contains both correct and wrong trajectories

0-1/10 correct
-> Panic Zone
-> retained for capability analysis, not used to build Learning Bundles
        |
        v
Accuracy-only Learning levels
Level 1: 8 correct
Level 2: 5-7 correct
Level 3: 3-4 correct
Level 4: 2 correct
        |
        v
One question -> one Learning Bundle
Stage1Miner.run()
build_learning_bundle()
Bundle contains:
- question and expected answer
- all 10 rollout records
- correct_rollout_ids / wrong_rollout_ids
- deterministic correct-wrong trajectory pairing hints
- capability signal and entropy summaries
- sampling configuration and source hash
        |
        v
Within-Level ranking
_rank_bundles()
- random
- high_token_entropy
- low_token_entropy

Accuracy determines the Zone and Level.
Entropy only changes ordering inside the same Level.
        |
        v
Stage 1 artifacts
output/stage1/
- 01_sampling/
- 02_learning_packets/<backbone>/level1-level4_learning_packets.json
- rankings/
- runs/<backbone>/selection_manifests/
- runs/<backbone>/analysis/questions.csv
- runs/<backbone>/figures/
- runs/<backbone>/run_manifest.json
```

### Stage 2 — Source-validated Principle production

```text
Frozen Learning Zone (correct@10 = 2..8)
1258 questions
        |
        v
Exclude the 40 protocol-development questions
        |
        v
Freeze production1200 + reserve18
First batch300 quotas by original correct@10:
96 / 52 / 45 / 32 / 26 / 24 / 25
        |
        v
DeepSeek diagnosis (Thinking, 8192 tokens)
- compare all correct and wrong trajectories
- locate the first consequential error
- preserve successful invariants
- merge equivalent causes; retain at most three independent causes
        |
        v
DeepSeek synthesis (non-Thinking, 4096 tokens)
- emit 0-3 canonical Principle JSON records
- failure_signals must be observable in a completed draft
- action_guidance makes the smallest relevant correction
- verification_step cannot depend on the reference answer
        |
        v
Deterministic format check only
- JSON / schema / required fields
- required_evidence cardinality
- referenced rollout IDs and their correct/wrong identity
No heuristic mathematical rejection
        |
        v
Reuse frozen Stage-1 baseline drafts
Solver-v1.1, temperature=0.7, seeds=42..51
        |
        +---------------- Generic control ----------------+
        |                                                  |
        v                                                  v
Generic draft self-check                           Forced-C Principle check
one shared result/question/seed                    one result/candidate/seed
        |                                                  |
        +---------------- paired effect -------------------+
        |
        v
source_helpful iff:
- delta_vs_generic >= 2
- delta_vs_original >= 0
- gain_vs_generic > loss_vs_generic
        |
        v
Kimi bounded blind content audit (at most 80 helpful candidates)
- at most one strongest candidate per source question
- balanced across the four Stage-1 levels when possible
- source fidelity
- mathematical validity
- instance/answer leakage
- decidable conditions
- executable action and verification
        |
        +-- pass --> pending_transfer_validation
        +-- hard reject --> content_rejected
        +-- repair --> DeepSeek repairs once
                        -> Forced-C pass@10 again
                        -> Kimi re-audit once

Non-helpful records remain available for diagnosis:
- source_harmful
- inconclusive
- protocol_failed
- source_helpful_unreviewed
- audit_inconsistent

Non-helpful candidates do not trigger Kimi diagnostic calls. At most 10 repaired,
still-helpful candidates receive one Kimi re-audit, for a hard total of 90 calls.
All records and repairs are append-only. Nothing is searchable before transfer
validation on unseen questions.
```

### Stage 3 — Applicability retrieval and paired evaluation

```text
Fixed test set
The official DeepScaler evaluation contains 400 questions
        |
        +-------------------------------+
        |                               |
        v                               v
Baseline variant                  Guided variant
No Memory retrieval              Load active Memory Store
        |                               |
        |                               v
        |                         Query semantic decomposition
        |                         PlanModule.extract_semantics()
        |                         Output:
        |                         - core_skill
        |                         - context_constraints
        |                         - task_objective
        |                               |
        |                               v
        |                         Three-facet Memory recall
        |                         MemoryModule.retrieve()
        |                         Recall at most candidate_top_k=12
        |                         Similarity is relevance, not reliability
        |                               |
        |                               v
        |                         Application Gate
        |                         judge_memory_applicability()
        |                         Read current question + routing view
        |                         Reject topic-only or condition-mismatched Memory
        |                         Inspect at most gate_top_k=6 candidates
        |                         Parse failure -> inject nothing
        |                               |
        |                               v
        |                         Current-question Adapter
        |                         adapt_memory_for_question()
        |                         Convert an accepted Principle into:
        |                         - when_to_consider
        |                         - evidence_found in the current problem
        |                         - do_not_use_if
        |                         - failure_to_avoid
        |                         - suggested_action
        |                         - verification
        |                         Adapter reject/failure -> inject nothing
        |                               |
        |                               v
        |                         Conditional Experience Card
        |                         Inject at most one adapted card
        |                         Never inject:
        |                         - source question or source answer
        |                         - Memory ID
        |                         - retrieval score
        |                         - Gate confidence
        |                         - raw Principle
        |                               |
        +---------------+---------------+
                        |
                        v
Frozen Solver V1

Baseline:
question -> Reasoner -> Finalizer

Guided:
question + Conditional Experience Card
-> Solver first decides use or ignore
-> Reasoner
-> Finalizer
        |
        v
Three fixed independent rollouts
- seed 42
- seed 43
- seed 44
- temperature 0.2
- no answer-dependent retry
- infrastructure failure may use one alternative endpoint
        |
        v
Answer matching and metrics

pass@1
-> correctness of seed 42

pass@3
-> at least one of seeds 42/43/44 is correct

Also save:
- retrieval candidates
- Gate judgments
- Adapter raw output and field differences
- exact Solver prompt
- Solver use / ignore decision
- Reasoner and Finalizer outputs
- transport trace
        |
        v
Stage 3 artifacts
output/stage3/<variant>/
- eval_results.json
- eval_summary.json
- prompt_snapshot.json
- transport_failures.jsonl
- raw.log
- run.log
```

### Stage 3 exploratory no-loss Memory experiment

This isolated experiment keeps the 172 source-helpful, content-audited
Principles and adds 92 source-inconclusive Principles that had zero paired loss
against generic self-check, did not regress against the original baseline, and
passed deterministic format checks. The 92 additions remain explicitly marked
as unaudited experimental records; the Stage-2 Store is not modified. The
intervention stays on the existing pre-solve single-card protocol so Memory
composition is the only experimental change.

```bash
experiment_root=output/stage3/production_300_plus_safe_inconclusive_v1
mkdir -p "$experiment_root/guided"
set -o pipefail
export MEMUA_EMBEDDING_MODEL_PATH=/home/share/models/modelscope/BAAI/bge-m3
export MEMUA_EMBEDDING_DEVICE=cpu

python pipeline.py stage3 \
  --variant guided \
  --test-file "$experiment_root/test_from_frozen_baseline.json" \
  --pending-memory-store output/stage2/production_1200_v1/batch_0300/memory/pending_transfer_validation/store.jsonl \
  --generation-records-dir output/stage2/production_1200_v1/batch_0300/raw_candidates/questions \
  --exploratory-inconclusive-root output/stage2/production_1200_v1/batch_0300 \
  --expected-exploratory-count 92 \
  --memory-dir "$experiment_root/memory_snapshot" \
  --output-dir "$experiment_root/guided" \
  --ports 8100 8101 8102 8103 \
  --model llama \
  --concurrency 8 \
  --temperature 0.2 \
  --threshold 0.55 \
  --candidate-top-k 12 \
  --gate-top-k 6 \
  --max-injected 1 \
  --guided-protocol pre_solve \
  --resume \
  2>&1 | tee "$experiment_root/run.log"
```

### Stage 3 retrieval-precision comparison

The isolated retrieval experiment keeps the 172-item Memory snapshot and the
existing three-seed Baseline drafts frozen. It compares six protocols without
regenerating Stage 2 data:

```text
R0 raw question single-vector retrieval
R1 current three-facet soft retrieval
R2 all-three-facets strict retrieval
R3 core recall -> condition veto -> objective reranking
R4 R3 plus per-seed draft-error verification
R5 R3 plus a failure-mode index and per-seed draft verification
```

The first 100 questions form a protocol-calibration split containing all 26
questions where the previous Stage 3 run injected a Memory. The other 300 are
reported separately as held-out evaluation. Four questions that exactly match
a Memory source question are retained for auditing but excluded from the main
transfer metric.

Run all six protocols on both splits:

```bash
mkdir -p output/stage3/retrieval_protocol_compare_v1
set -o pipefail
export MEMUA_EMBEDDING_MODEL_PATH=/home/share/models/modelscope/BAAI/bge-m3
export MEMUA_EMBEDDING_DEVICE=cpu

python pipeline.py stage3 \
  --mode retrieval-compare \
  --retrieval-phase all \
  --retrieval-protocol all \
  --memory-dir output/stage3/production_300_transfer_after_draft_v2/memory_snapshot \
  --baseline-cache-dir output/stage3/production_300_transfer_v1/baseline \
  --current-guided-dir output/stage3/production_300_transfer_after_draft_v2/guided \
  --stage1-packets-root output/stage1/02_learning_packets \
  --output-dir output/stage3/retrieval_protocol_compare_v1 \
  --ports 8100 8101 8102 8103 \
  --model llama \
  --concurrency 8 \
  --resume \
  2>&1 | tee output/stage3/retrieval_protocol_compare_v1/run.log
```

Progress is printed as `阶段:协议 completed/total`. Detailed logs are in
`output/stage3/retrieval_protocol_compare_v1/run.log` and `raw.log`. The main
result is `report.md`; complete per-candidate scores and rejection reasons are
stored below `protocols/` and `effects/`. Re-running the same command resumes
from per-question checkpoints, including normally completed wrong answers.

### Stage 3 three-facet hybrid retrieval

This isolated suite keeps semantic decomposition as the retrieval backbone and
adds source-question deduplication, cross-encoder reranking, BGE-M3 hybrid
retrieval, and soft draft-error reranking. It never overwrites R0-R5 output.

```text
E0 current hierarchical three-facet retrieval
E1 E0 + one candidate per source question
E2 threshold-free three-facet union + source deduplication
E3 E2 + BGE cross-encoder reranking
H1 three-facet BGE-M3 dense+sparse RRF
H2 H1 + BGE-M3 multi-vector RRF
H3 H2 + cross-encoder + 70/30 draft-error soft reranking
```

Run the parser replay and ten-question smoke test first:

```bash
mkdir -p output/stage3/semantic_hybrid_retrieval_v1
export MEMUA_EMBEDDING_MODEL_PATH=/home/share/models/modelscope/BAAI/bge-m3
export MEMUA_EMBEDDING_DEVICE=cpu

python pipeline.py stage3 \
  --mode semantic-hybrid \
  --retrieval-phase smoke \
  --retrieval-protocol all \
  --memory-dir output/stage3/production_300_transfer_after_draft_v2/memory_snapshot \
  --baseline-cache-dir output/stage3/production_300_transfer_v1/baseline \
  --current-guided-dir output/stage3/production_300_transfer_after_draft_v2/guided \
  --stage1-packets-root output/stage1/02_learning_packets \
  --source-comparison-root output/stage3/retrieval_protocol_compare_v1 \
  --output-dir output/stage3/semantic_hybrid_retrieval_v1 \
  --facet-pool-k 30 --candidate-top-k 12 --rerank-top-k 12 --gate-top-k 6 \
  --reranker-model-path models/reranker/bge-reranker-v2-m3 \
  --hybrid-model-path /home/share/models/modelscope/BAAI/bge-m3 \
  --ports 8100 8101 8102 8103 --model llama --concurrency 8 --resume \
  2>&1 | tee output/stage3/semantic_hybrid_retrieval_v1/run.log
```

Progress is in `run.log`; results are in `report.md`, `summary.json`,
`parser_replay/`, `protocols/`, and `effects/` under the output directory.

### Stage 3 strict post-draft transfer

This isolated mode matches the Stage-2 Principle contract: retrieve by joint
agreement across all three semantic facets, rerank to three candidates, and
accept a Principle only when a local checker can quote the corresponding error
from the frozen baseline draft. A treatment revision is compared with an
equal-cost generic draft check. Rejected pairs reuse the baseline rollout
byte-for-byte; the legacy Gate, Adapter, and Solver use/ignore calls are not
used.

```bash
mkdir -p output/stage3/postdraft_strict_transfer_v1
export MEMUA_EMBEDDING_MODEL_PATH=/home/share/models/modelscope/BAAI/bge-m3
export MEMUA_EMBEDDING_DEVICE=cpu

python pipeline.py stage3 \
  --mode postdraft-transfer \
  --postdraft-phase all \
  --memory-arm both \
  --memory-172-dir output/stage3/production_300_transfer_after_draft_v2/memory_snapshot \
  --memory-264-dir output/stage3/production_300_plus_safe_inconclusive_v1/memory_snapshot \
  --baseline-cache-dir output/stage3/production_300_plus_safe_inconclusive_10seed_v1/baseline \
  --split-root output/stage3/retrieval_protocol_compare_v1 \
  --output-dir output/stage3/postdraft_strict_transfer_v1 \
  --reranker-model-path models/reranker/bge-reranker-v2-m3 \
  --candidate-top-k 24 --rerank-top-k 3 --max-injected 1 \
  --ports 8100 8101 8102 8103 --model llama \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --temperature 0.2 --concurrency 8 --timeout 900 --resume \
  2>&1 | tee output/stage3/postdraft_strict_transfer_v1/run.log
```

`all` runs Smoke, calibration, and heldout in order, and stops before a later
phase when the preceding acceptance rule fails. Progress is in `run.log`; the
main outputs are `report.md`, `summary.json`, `effects/`, and `checkpoints/`.

## Project commands

Run all commands from the repository root.

```bash
# Deployment is independent from the experiment controller.
bash api_call/run.sh api_call/actor.conf llama 0 1 2 3 --daemon
bash api_call/test.sh api_call/actor.conf llama
bash api_call/stop.sh api_call/actor.conf

# Stage 2 uses DeepSeek for Principle generation/repair and remote Kimi for
# independent audit, condition cards, and blind structural variants. Secrets
# belong only in the git-ignored stage2.env file.
source stage2.env

# Stage 1 samples the complete external dataset once. Accuracy assigns levels;
# entropy only ranks questions inside each level.
python pipeline.py stage1 \
  --dataset /path/to/deepscaler_merged_6000.json \
  --ports 8100 8101 8102 8103 \
  --resume

# Rebuild the four Stage-1 diagnostic figures without calling a model.
python scripts/stage1/analyze.py --stage1-dir output/stage1

# Stage 2 production is recoverable by phase. Prepare is read-only with
# respect to APIs and freezes the nested 300/1200 manifests.
python pipeline.py stage2 --mode incremental --incremental-phase prepare \
  --target-size 300 --backbone llama

# Run the complete first batch. It automatically completes an isolated real
# two-question Smoke before starting all 300 questions.
mkdir -p output/stage2/production_1200_v1/batch_0300
python pipeline.py stage2 \
  --mode incremental \
  --incremental-phase complete \
  --target-size 300 \
  --backbone llama \
  --ports 8100 8101 8102 8103 \
  --model llama \
  --model-concurrency 8 \
  --validation-concurrency 8 \
  --kimi-max-calls 90 \
  --kimi-initial-audit-limit 80 \
  --kimi-reaudit-limit 10 \
  --kimi-diagnostic-limit 0 \
  --timeout 900 \
  --resume \
  2>&1 | tee output/stage2/production_1200_v1/batch_0300/run.log

# Resume one phase only, or inspect status without calling a model.
python pipeline.py stage2 --mode incremental --incremental-phase generate \
  --target-size 300 --backbone llama --model-concurrency 8 --resume
python pipeline.py stage2 --mode incremental --incremental-phase validate \
  --target-size 300 --ports 8100 8101 8102 8103 \
  --model llama --validation-concurrency 8 --resume
python pipeline.py stage2 --mode incremental --incremental-phase audit-store \
  --target-size 300 --ports 8100 8101 8102 8103 \
  --model llama --model-concurrency 8 --validation-concurrency 8 \
  --kimi-max-calls 90 --kimi-initial-audit-limit 80 \
  --kimi-reaudit-limit 10 --kimi-diagnostic-limit 0 --resume

# Expand to 1200 without regenerating the hash-compatible first 300. This
# phase calls DeepSeek only; it never calls Kimi or the local Llama endpoints.
mkdir -p output/stage2/production_1200_v1/batch_1200
python pipeline.py stage2 --mode incremental --incremental-phase generate \
  --target-size 1200 --backbone llama --model-concurrency 8 --resume \
  2>&1 | tee output/stage2/production_1200_v1/batch_1200/run_generate.log
python pipeline.py stage2 --mode incremental --incremental-phase status \
  --target-size 300

# DeepSeek异常题采用本地Qwen隔离恢复。真实产物复核后的恢复规模为626题：
# 573题DeepSeek完整结果和1题合法no_reliable_principle保持只读。
# 必须先跑8题Smoke，再累计跑到30题；全量入口受monitor_0030.json硬门槛保护。
export ACTIVEEVO_QWEN_GENERATOR_URLS="http://127.0.0.1:8200/v1/chat/completions,http://127.0.0.1:8201/v1/chat/completions,http://127.0.0.1:8202/v1/chat/completions,http://127.0.0.1:8203/v1/chat/completions"

python pipeline.py stage2 --mode incremental \
  --incremental-phase recover-generate --target-size 1200 \
  --generator qwen --recovery-limit 8 --model-concurrency 8 --resume \
  2>&1 | tee output/stage2/production_1200_v1/batch_1200/qwen_recovery/run_smoke8.log

python pipeline.py stage2 --mode incremental \
  --incremental-phase recover-generate --target-size 1200 \
  --generator qwen --recovery-limit 30 --model-concurrency 8 --resume \
  2>&1 | tee output/stage2/production_1200_v1/batch_1200/qwen_recovery/run_monitor30.log

# 只有monitor_0030.json中的passed=true时，下列命令才会运行全部恢复题。
python pipeline.py stage2 --mode incremental \
  --incremental-phase recover-generate --target-size 1200 \
  --generator qwen --recovery-limit 0 --model-concurrency 8 --resume \
  2>&1 | tee output/stage2/production_1200_v1/batch_1200/qwen_recovery/run_all.log

# 1200题全部达到合法终态后，Kimi只分层盲审80条用于估计质量；不做逐条准入。
python pipeline.py stage2 --mode incremental \
  --incremental-phase sample-audit --target-size 1200 \
  --kimi-audit-sample-size 80 --kimi-max-calls 90 \
  --model-concurrency 8 --resume \
  2>&1 | tee output/stage2/production_1200_v1/batch_1200/quality_sample/run.log

# Qwen恢复日志/质量门槛/混合Bank/Kimi报告：
# output/stage2/production_1200_v1/batch_1200/qwen_recovery/
# output/stage2/production_1200_v1/batch_1200/raw_candidates/mixed_candidate_bank.json
# output/stage2/production_1200_v1/batch_1200/raw_candidates/format_valid_mixed_bank.json
# output/stage2/production_1200_v1/batch_1200/quality_sample/audit_report.md

# Progress/log/result locations:
# output/stage2/production_1200_v1/batch_0300/run.log
# output/stage2/production_1200_v1/batch_0300/report.md
# output/stage2/production_1200_v1/batch_0300/memory/

# Isolated 40-source + 40-variant Principle protocol experiment. None of these
# phases writes to memory/. Run the provider probe and historical dev8 Smoke
# before paying for the full experiment.
python pipeline.py stage2 --mode principle-test --test-phase prepare
python pipeline.py stage2 --mode principle-test --test-phase probe
python pipeline.py stage2 --mode principle-test --test-phase smoke \
  --ports 8100 8101 8102 8103 --model llama --validation-concurrency 4

# Fast eight-question role comparison. This reuses the frozen DeepSeek
# smoke8 Candidate Bank, generates only the matching Kimi branch, performs
# one-pass blind cross-audit, and evaluates seeds 42/43/44 on Llama. It never
# writes to memory/ and resumes at the completed model-call boundary.
python pipeline.py stage2 \
  --mode principle-test \
  --test-phase role-compare \
  --test-root output/stage2/principle_test_80_v1/smoke8 \
  --ports 8101 8103 \
  --model llama \
  --model-concurrency 8 \
  --validation-concurrency 4 \
  --timeout 900 \
  --resume
# Results:
# output/stage2/principle_test_80_v1/smoke8/role_compare_v1/report.md

# Full experiment. The variants phase stops after producing the human review.
python pipeline.py stage2 --mode principle-test --test-phase variants --model-concurrency 2
# Manually approve all 40 rows in:
# output/stage2/principle_test_80_v1/variants/variant_approvals.json
python pipeline.py stage2 --mode principle-test --test-phase generate --model-concurrency 2
python pipeline.py stage2 --mode principle-test --test-phase audit --model-concurrency 2
python pipeline.py stage2 --mode principle-test --test-phase source \
  --ports 8100 8101 8102 8103 --model llama --validation-concurrency 4
python pipeline.py stage2 --mode principle-test --test-phase transfer \
  --ports 8100 8101 8102 8103 --model llama --validation-concurrency 4
python pipeline.py stage2 --mode principle-test --test-phase robustness \
  --ports 8100 8101 8102 8103 --model llama --validation-concurrency 4
python pipeline.py stage2 --mode principle-test --test-phase report
python pipeline.py stage2 --mode principle-test --test-phase status

# Stage 3 baseline and guided pass@3 evaluation.
python pipeline.py stage3 --variant baseline
python pipeline.py stage3 --variant guided --memory-dir memory/store

# Stage 3 retrieval protocol comparison: see the full command above.
python pipeline.py stage3 --mode retrieval-compare --help

# Static checks and unit contracts.
bash scripts/tests/run_all_tests.sh
```

## Model roles on a new machine

The repository does not contain weights. Model mount points are reserved under
`models/actor`, `models/generator`, `models/auditor`, and `models/embedding`.
The default Stage 2 generator is the DeepSeek API. The local Qwen generator is
used only to recover DeepSeek transport/protocol failures, with explicit
generator provenance in the mixed Candidate Bank. Moonshot Kimi is an
independent remote auditor; in production it reviews only a frozen stratified
quality-estimation sample. Llama remains the local actor whose behavior is
measured.

## Directory guide

| Path | Responsibility |
|---|---|
| `pipeline.py` | Only project-level Python controller |
| `core/config.py` | Central paths, model roles, and Stage defaults |
| `modules/stage2/` | Inspectable Memory Factory functions |
| `modules/` | Stage 1 sampling, Memory, retrieval and Solver modules |
| `scripts/` | Optional stage-specific utilities |
| `api_call/` | Model deployment, health checks and shutdown |
| `memory/store/` | Versioned searchable Memory and Level snapshots |
| `output/` | Generated run artifacts; not source code |
| `tests/` | Unit and contract tests |

The precise Stage 2 data flow and function map is documented in
[docs/STAGE2_MEMORY_FACTORY.md](docs/STAGE2_MEMORY_FACTORY.md).
The persistent 40-to-100-to-600 freezing workflow is documented in
[docs/STAGE2_INCREMENTAL_FREEZING.md](docs/STAGE2_INCREMENTAL_FREEZING.md).
