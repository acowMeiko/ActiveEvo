# Reasoning-first prompt audit

This directory tests the solver format in which a concise derivation and one
sanity check precede the unique final boxed answer.

## Controlled setup

- Same fixed 100 DeepScaler Level-2/3 items and seed 42.
- Meta-Llama-3.1-8B-Instruct on GPU 6/7, one call per condition and no retry.
- Completion budget increased from 2048 to 4096 tokens.
- Guided evaluation replays the exact Top-10 traces, selected memory IDs, and
  25 solver-visible memory strings from `../final_paired/guided/eval_results.json`.
- The failed CPU-reranker attempt is retained under `final_paired/guided/` as
  an execution audit and is not a result. Use `final_paired/guided_replay/`.

## Result

- Base: 7/100; explicit final answer: 92/100; 9 outputs reached 4096 tokens.
- Guided replay: 10/100; explicit final answer: 89/100; 11 outputs reached
  4096 tokens.
- Paired difference: +3 points; McNemar p=0.4531; paired bootstrap 95% CI
  [-2%, +8%].
- Injected subgroup: Base 0/25, Guided 2/25, with two answer-level rescues and
  no answer-level harms.
- Manual audit: idx15 is a genuine reasoning correction; idx79 has the right
  final value but an invalid derivation, so it is not a sound reasoning rescue.

See `final_paired/analysis/paired_report.md` for the full paired summary.
