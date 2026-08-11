# Stage 3 structural gate and adaptation experiment

This experiment uses GPUs 4/6/7 (ports 8004/8006/8007) and a fixed 30-item,
seed-42 replay. It performs these operations in order:

1. Compare each frozen selected memory's source problem and structure with the
   current problem using a fail-closed structural Application Gate.
2. Adapt only accepted memories and freeze both raw and adapted solver texts.
3. Run Base, accepted Raw, and accepted Adapted conditions at Reasoner
   temperatures 0.0 and 0.2 with repetition penalty 1.05.
4. Produce a report and exact raw/adapted text diffs.

Run from the project root:

```bash
bash experiments/stage3_gate_adaptation_gpu467/run.sh
```

Outputs are under `results_seed42/`:

- `gate_records.json`: full source/current gate judgments and adaptations.
- `text_comparison.md`: source text, adapted text, and unified diffs.
- `temperature_0_0/` and `temperature_0_2/`: full Solver outputs.
- `experiment_report.md`: combined protocol metrics.
- `experiment_summary.json`: machine-readable combined summary.

The Stage 2 schema/generation code is updated to emit the extended fields, but
this command does not regenerate or mutate the existing Memory store.
