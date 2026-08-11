# ActiveEvo memory-routing v2 experiment

## Authoritative result

Use only `final_paired/` for analysis:

- `final_paired/base/`: final base outputs and summary.
- `final_paired/guided/`: final guided outputs, complete Top-10 traces, and injected solver text.
- `final_paired/analysis/paired_report.md`: concise paired report.
- `final_paired/analysis/paired_summary.json`: machine-readable statistics.
- `final_paired/analysis/paired_cases.csv`: one row per sample for manual case analysis.
- `final_paired/analysis/paired_cases.json`: the same audit with untruncated text fields.

The older `smoke/`, `full/`, `final/`, and `paired/` directories are prompt/matcher engineering audits. They are intentionally retained for provenance but must not be reported as final results.

## Frozen protocol

- Input: the unchanged `../test_subset.json`, SHA256 `daa020eba205b14c4cc251163295299f74ad26520eb56191f920b20ab18391ad`.
- Composition: 50 DeepScaler Level-2 + 50 Level-3 questions.
- Solver: local Meta-Llama-3.1-8B-Instruct.
- Request and server seed: 42; temperature 0; stable item-to-endpoint mapping.
- Solver calls: exactly one per item; no answer-dependent retry.
- Recall: BGE-M3, threshold-free three-facet candidate union, Top-10.
- Applicability reranker: Qwen3-Reranker-0.6B, absolute threshold 0.50.
- Injection: final Top-0/1.
- Solver-visible memory: `Lesson learned` and `Recommended action` only.
- Physical GPUs: GPU 3 (A6000) and GPU 4 (RTX 3090). GPU 2 was occupied by another TP job and was not interrupted.

## Canonical memory

`memory/` is a new independent v2 store derived read-only from the legacy store and original Stage-2 artifacts. It contains 1,245 canonical five-field records. The 250 legacy records that could not be mapped back reliably are excluded and listed in `memory/metadata/migration_report.json`.

## Known data issue

Test item 61 supplies `5057/9990` for `0.3\overline{206}`. Direct algebra yields `3203/9990`. The supplied label was not changed in the headline paired evaluation; the issue is recorded in the final report.
