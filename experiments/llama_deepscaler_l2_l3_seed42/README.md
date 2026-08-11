# Llama DeepScaler Level 2/3 Memory-Effect Audit

This folder is an independent 100-item audit. It does not run the four curriculum rounds and does not modify the pipeline's iterative cumulative-success rule.

## Fixed protocol

- Model: `/home/share/models/modelscope/llama/Meta-Llama-3.1-8B-Instruct`
- GPUs/endpoints: GPU 2 → port 8002; GPU 3 → port 8003
- Server and request seed: 42
- Data: 50 Level-2 + 50 Level-3 DeepScaler items, stratified without replacement
- Subset SHA-256: `daa020eba205b14c4cc251163295299f74ad26520eb56191f920b20ab18391ad`
- Solver: one complete inference per problem, temperature 0, no retry, 2048 output-token cap
- Retrieval: real BGE-M3 embeddings, DSD threshold 0.55, up to 10 candidates
- Applicability: Llama judges trigger, scope, and failure-mode; confidence ≥ 0.70; up to 3 memories injected
- Injection: only `When / Do / Scope / Avoid`; introspection is not sent to the solver

## Main result

- Base: 16/100
- Guided: 15/100
- Paired delta: −1 percentage point
- Improved/degraded: 6/7
- Exact McNemar p: 1.0
- Paired-bootstrap 95% CI: [−8, +6] percentage points
- Principle invoked: 27/100 samples
- Injected: 32 memories (30 unique); 22 samples received one and 5 received two

The evaluated store is legacy: all 32 injected records retained only action text, so trigger/scope/failure fields were explicitly derived and marked `legacy_derived`. This run validates the new code path and exposes the need to rebuild a canonical five-field store; it is not suitable as the final paper efficacy result.

## Counterfactual replay

The 27 invoked samples were replayed with one active pair per endpoint: the same item first ran base and then guided on the same endpoint and seed.

- Base: 6/27
- Guided: 5/27
- Improved/degraded: 3/4
- Final answer changed: 22/27
- Exact McNemar p: 1.0

Thus the principle prompt materially changes model outputs, but the current legacy memory is not reliably beneficial.

## Artifacts

- `test_subset.json`: exact sampled questions and source indices
- `full/base/`: base results and summary
- `full/guided/`: guided results, full routing traces, injected text, and per-item checkpoints
- `analysis/paired_summary.json`: aggregate paired statistics
- `analysis/paired_cases.csv`: compact per-item comparison
- `analysis/paired_cases.json`: complete answers, routing judgments, and injected memories
- `analysis/paired_report.md`: readable multi-case analysis
- `analysis/injected_counterfactual_replay.json`: strict serial replay for invoked samples

Each guided item has `ready_to_infer`, `inference_completed`, and `result_committed` checkpoints under `full/guided/checkpoints/item_*`.
