# ActiveEvo workflow

1. Configure and deploy the actor and auditor endpoints through `api_call/`.
2. Run Stage 1 once with an explicit dataset path. It preserves the paper's
   Master/Learning/Panic zones; accuracy divides Learning into level1-level4,
   and entropy only provides within-level rankings.

```bash
python pipeline.py stage1 \
  --dataset /path/to/deepscaler_merged_6000.json \
  --ports 8100 8101 8102 8103 \
  --resume
```
3. Run Stage 2 sequentially for `level1` through `level4` using one Store.
4. Inspect each Level's generated Principles, audits, reconciliation decisions,
   Store commit and snapshot before continuing.
5. Freeze the final Store and run identical Stage 3 baseline/guided evaluations.

Stage 1 and Stage 2 may run on another machine. Copy the repository, Stage-1
input data, `output/stage1`, and the selected `memory/store` snapshot. No code
path depends on the old machine's absolute model directories.
