# Single-Vector Principle Ablation

This ablation removes MemUA's three-facet retrieval and indexes only the principle text with one embedding per memory item.

Fairness choices:
- Stage 2 uses the same cumulative principle views as the main Deepscaler rerun.
- Stage 3 uses the same `GUIDED_ANSWER_PROMPT`, retry prompt, answer matcher, retry budget, and curriculum carry-forward style as the main evaluation.
- The only intended retrieval difference is single-vector question-to-principle cosine retrieval.
- All outputs stay under this ablation directory.

Run:

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
PORTS_ENV="8000 8001 8002 8003" bash ablation/single_vector_principle/run_deepscaler_single_vector_ablation.sh
```

Outputs:

```text
ablation/single_vector_principle/output/stage2_single_vector_memory/
ablation/single_vector_principle/output/stage3_deepscaler_single_vector_*/
ablation/single_vector_principle/logs/
```
