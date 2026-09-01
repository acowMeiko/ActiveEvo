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
3. Freeze the 1200-question Stage 2 production manifest and run the nested
   300-question batch first. The production controller generates Principles,
   applies the deterministic format check, runs Forced-C source validation,
   and then spends the bounded Kimi audit budget.
4. Inspect `batch_0300/report.md` and both Memory branches. Only
   `pending_transfer_validation` has passed source-effect and content checks;
   it is still not a claim of cross-question generalization.
5. After the 300-question batch is accepted, resume the same frozen manifest
   at 1200 questions. Stage 3 then performs transfer validation and retrieval
   evaluation on unseen questions.

Stage 1 and Stage 2 may run on another machine. Copy the repository, Stage-1
input data, `output/stage1`, and the selected `memory/store` snapshot. No code
path depends on the old machine's absolute model directories.
