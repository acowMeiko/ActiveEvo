# Solver-v1 GPU 4/6/7 reliability test

This experiment freezes a two-call solver protocol:

- Reasoner: maximum 3072 completion tokens, temperature 0.1.
- Finalizer: maximum 128 completion tokens, temperature 0.
- No answer-dependent retry.
- Base and Guided are paired on the same endpoint and request seed.
- One complete pair is active per endpoint; pair order is counterbalanced.
- Retrieval and injected memory are replayed from the prior frozen 100-item run.
- GPU 4/6/7 serve ports 8004/8006/8007; each endpoint runs one complete pair at a time.

## Run

From the project root:

```bash
bash experiments/solver_v1_gpu67/run.sh status
bash experiments/solver_v1_gpu67/run.sh deploy
bash experiments/solver_v1_gpu67/run.sh status
bash experiments/solver_v1_gpu67/run.sh smoke30
bash experiments/solver_v1_gpu67/run.sh full100
```

If both endpoints are already `READY`, skip `deploy`.  After a new deployment,
wait until both endpoints are ready before starting `smoke30`.

Only after the 30-item smoke test passes the formatting and loop checks should
the 100-item command be run.  The optional three-seed development pilot is:

```bash
bash experiments/solver_v1_gpu67/run.sh pilot3seed
```

To test Reasoner-only repetition penalties on the same frozen 30-item Stage 3
set, run all four settings (`1.00`, `1.03`, `1.05`, and `1.08`) with one command:

```bash
bash experiments/solver_v1_gpu67/run_penalty_sweep30.sh
```

The sweep command starts only missing GPU endpoints, waits up to five minutes
for model loading, and then runs all four settings. Existing healthy endpoints
are left untouched.

The Finalizer remains fixed at repetition penalty `1.00`. The generated
`penalty_sweep30_seed42/sweep_report.md` recommends a preliminary setting based
on hard loops, token-limit hits, Reasoner format validity, and output length.
Accuracy is reported but is not used to tune the penalty on this 30-item set.

## Outputs

Each run writes:

- `paired_results.json`: complete Base/Guided reasoning, final answers, usage,
  finish reasons, token-limit flags, repetition diagnostics, and pair outcome.
- `summary.json`: accuracy, rescues, harms, net uplift, per-seed results, and
  historical uplift grouped by selected memory ID.
- `prompt_snapshot.json`: the exact prompt templates and hashes.

The penalty sweep additionally writes `sweep_summary.json` and
`sweep_report.md` above the four per-setting result directories.

The runner prints progress only every 10 completed pairs.  Inspect `summary.json`
instead of repeatedly loading the large result file while a run is active.
