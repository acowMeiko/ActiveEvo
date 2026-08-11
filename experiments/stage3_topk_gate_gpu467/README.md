# Stage 3 Top-K Gate experiment

This is a 30-item protocol screen on GPUs 4/6/7. It does not modify the
iterative cumulative-success implementation and does not perform dataset/store
deduplication.

Pipeline:

1. BGE-M3 threshold-free three-facet Top-10 recall.
2. Qwen3-Reranker-0.6B relevance ranking; retain Top-5 without treating its
   score as memory reliability.
3. One batched Llama Application Gate call over the five candidates; zero
   candidates may be accepted.
4. Select at most one accepted candidate and adapt it to the current problem.
5. Run Base once for every item. Run Raw and Adapted only on eligible items at
   Reasoner temperatures 0.0 and 0.2. Repetition penalty is fixed at 1.05 and
   Finalizer temperature remains 0.0.

Run from the project root:

```bash
bash experiments/stage3_topk_gate_gpu467/run.sh
```

Outputs under `results_seed42/` include complete Top-10 retrieval, reranking,
all Gate judgments, selected raw/adapted texts, Solver outputs, and the combined
report.

For the independent 100-pair DeepSeek audit of Application Gate precision and
recall, follow [`ANNOTATION.md`](ANNOTATION.md).
