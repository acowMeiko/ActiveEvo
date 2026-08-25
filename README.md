# ActiveEvo

ActiveEvo studies experience-driven mathematical reasoning with a capability
signal and structured Memory retrieval.

```text
Stage 1: weak-model rollouts -> capability levels -> LearningBundle
Stage 2: trajectory evidence -> Principle -> quality audit -> Memory Store
Stage 3: SPD retrieval -> runtime gate/adapter -> Solver -> pass@1/pass@3
```

Stage 2 does not use Neighbor Validation. A Principle becomes searchable only
after schema, leakage, source-fidelity and actionability checks. Its downstream
effect is measured later by Stage 3 paired evaluation.

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

### Stage 2 — Principle generation and Memory Factory

```text
Stage 1 Learning Bundles
One Bundle contains one question with correct and wrong trajectories
        |
        v
Canonical input construction
build_learning_bundles()
validate_learning_bundle()
Required evidence:
- question and expected answer
- complete rollout records
- at least one correct rollout
- at least one wrong rollout
- valid trajectory references
        |
        v
Principle Generation
generate_principles()
Input:
- source question
- all correct and wrong rollouts
- deterministic trajectory-pairing hints

The Generator first infers:
- stable success consensus
- first causal divergence of wrong trajectories
- 1-3 failure families with similar root causes

The Generator outputs:
- routing semantics
  - core_skill
  - context_constraints
  - task_objective
- one candidate Principle per failure family
        |
        v
Canonical Principle Schema Check
validate_principle_content(require_extended=True)
Required fields:
- introspection
- trigger_condition
- required_evidence
- exclusion_condition
- failure_mode
- action_guidance
- verification_step
- generalization_level
- scope

Invalid schema or invalid rollout references
-> rejected
        |
        v
Deterministic leakage check
deterministic_leakage_check()
Reject source-specific:
- answer payload
- copied significant numbers
        |
        v
Independent Quality Audit
audit_principle()
Judge every dimension separately:
- source_fidelity
- instance_leakage, including source-only entities or conclusions
- trigger_decidability
- evidence_grounding
- exclusion_decidability
- action_executability
- verification_executability
- generalization_support
        |
        v
Quality decision

pass
-> quality_pass

repair
-> repair_principle()
-> at most one directed repair
-> run the complete audit again

reject / invalid repaired result
-> rejected and retained in audit artifacts
        |
        v
Derive four Memory views
build_memory_views()

routing_view
-> three-dimensional indexing and applicability judgment

solver_view
-> lesson / action guidance / verification only

validation_view
-> quality and later validation evidence

audit_view
-> source, schema, generation and traceability records
        |
        v
Reconcile with the cumulative Memory Store
reconcile_candidate()
Retrieve only near-equivalent existing memories, then choose:
- add
- covered
- update
- merge
- reject

update / merge
-> audit the resulting Principle again
-> failed re-audit becomes reject
        |
        v
Versioned Memory commit
- append or replace version
- keep historical versions
- index only active memories
- save three facet indices
- create the Level snapshot
        |
        v
Stage 2 artifacts
output/stage2/<level>/
- 00_learning_bundles/
- 04_generated_principles/
- 06_quality_audits/
- 08_memory_reconciliation/
- checkpoint.json
- summary.json

memory/store/
- active Principle records
- version history
- core_skill / context_constraints / task_objective indices
- Level snapshots
```

Stage 2 is a source-evidence and content-quality gate. It does not use Neighbor
Validation and it does not claim that a quality-approved Principle improves
accuracy. Positive and negative effects are measured by Stage 3.

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

# Stage 2 cumulatively builds the same Store by curriculum level.
python pipeline.py stage2 --level level1 --memory-dir memory/store
python pipeline.py stage2 --level level2 --memory-dir memory/store
python pipeline.py stage2 --level level3 --memory-dir memory/store
python pipeline.py stage2 --level level4 --memory-dir memory/store

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

# After the 40-question protocol is accepted, freeze it once. Later targets
# reuse all hash-compatible completed questions and call the APIs only for the
# new delta: 40 -> 100 (+60) -> 600 (+500) -> all.
python pipeline.py stage2 --mode incremental --incremental-phase freeze \
  --target-size 40 \
  --incremental-target-root output/stage2/principle_test_80_v1
python pipeline.py stage2 --mode incremental --incremental-phase prepare --target-size 100
python pipeline.py stage2 --mode incremental --incremental-phase run \
  --target-size 100 --model-concurrency 2
# Review target_0100 and fill its hash-bound quality_decision.json, then:
python pipeline.py stage2 --mode incremental --incremental-phase freeze \
  --target-size 100 \
  --quality-decision output/stage2/incremental/target_0100/quality_decision.json
python pipeline.py stage2 --mode incremental --incremental-phase status

# Stage 3 baseline and guided pass@3 evaluation.
python pipeline.py stage3 --variant baseline
python pipeline.py stage3 --variant guided --memory-dir memory/store

# Static checks and unit contracts.
bash scripts/tests/run_all_tests.sh
```

## Model roles on a new machine

The repository does not contain weights. Model mount points are reserved under
`models/actor`, `models/generator`, `models/auditor`, and `models/embedding`.
The default Stage 2 generator is the DeepSeek API. The isolated Principle
calibration uses Moonshot Kimi as an independent remote auditor and variant
generator. Llama remains the local actor whose behavior is measured. Historical
Qwen comparison code is not part of this 40+40 protocol.

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
