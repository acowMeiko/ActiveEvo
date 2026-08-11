# Stage 3 Math500 Results

This table records `MemUA` Stage 3 guided-evaluation results on `data/test/benchmarks/math500_stage3.json`.

| Run Date | Dataset | Test Set | Model | Ports | Memory Source | Total | Correct | Accuracy | Accuracy@3 | Retrieval Used | Retrieval Ratio | Retry Used | Retry Ratio | Rescued by Retry | Status | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-04-21 | `dpo_level2` | `math500_stage3.json` | `base` | `8000,8001,8002,8003` | `memory/principles/principles.json` | 500 | 363 | 0.726 | 0.726 | 375 | 0.750 | 162 | 0.324 | 25 | `done` | `evaluate_principle_guided_test.py --input data/test/benchmarks/math500_stage3.json --memory-dir memory --answer-max-tokens 1024` |
