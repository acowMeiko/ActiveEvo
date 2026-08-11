# DeepSeek blind Gate annotation (100 pairs)

This audit labels 100 question–Memory candidate pairs independently. DeepSeek
does not receive the Llama Gate decision, retrieval score, reranker score, or
sampling stratum. The saved task artifact retains those hidden fields locally
so the evaluation step can compute Gate precision and recall.

The 100 pairs are a deterministic diagnostic sample (seed 42): all available
Llama-accepted pairs are prioritized, then high-rank rejects, then remaining
rejects. It is suitable for Gate error analysis, but its class prevalence must
not be presented as an estimate of the full candidate distribution.

After the current Top-K experiment creates `results_seed42/gate_records.json`,
freeze the annotation tasks:

For the initial balanced 10-pair smoke test (five accepted and five rejected
when enough candidates exist):

```bash
python tools/deepseek_gate_annotation.py prepare \
  --gate-records experiments/stage3_topk_gate_gpu467/results_seed42/gate_records.json \
  --output experiments/stage3_topk_gate_gpu467/results_seed42/deepseek_annotation_smoke10 \
  --sample-size 10 \
  --accepted-cap 5 \
  --seed 42
```

After checking those labels, freeze the full 100-pair audit without an
accepted cap:

```bash
python tools/deepseek_gate_annotation.py prepare \
  --gate-records experiments/stage3_topk_gate_gpu467/results_seed42/gate_records.json \
  --output experiments/stage3_topk_gate_gpu467/results_seed42/deepseek_annotation \
  --sample-size 100 \
  --seed 42
```

Set the API key only in the shell environment, then run the resumable labeler.
Replace the API base and model with the values supplied by the provider:

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'
python tools/deepseek_gate_annotation.py label \
  --tasks experiments/stage3_topk_gate_gpu467/results_seed42/deepseek_annotation/annotation_tasks.json \
  --labels experiments/stage3_topk_gate_gpu467/results_seed42/deepseek_annotation/deepseek_labels.jsonl \
  --api-base YOUR_OPENAI_COMPATIBLE_API_BASE \
  --model YOUR_DEEPSEEK_MODEL \
  --concurrency 4
```

The JSONL file is append-only. Re-running the same command skips successful
pairs and retries failed or schema-invalid pairs.

Compute the Gate audit:

```bash
python tools/deepseek_gate_annotation.py evaluate \
  --tasks experiments/stage3_topk_gate_gpu467/results_seed42/deepseek_annotation/annotation_tasks.json \
  --labels experiments/stage3_topk_gate_gpu467/results_seed42/deepseek_annotation/deepseek_labels.jsonl \
  --output experiments/stage3_topk_gate_gpu467/results_seed42/deepseek_annotation
```

Use `annotation_adjudication.csv` to review model disagreements first. The
DeepSeek output is a silver label; any Gate metric reported in the paper should
be confirmed by human adjudication.
