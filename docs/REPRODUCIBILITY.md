# Reproducibility

Copy `deploy.conf.example` to `deploy.conf` and `config.env.example` to
`config.env`. Record the model revision, dataset hash, prompt hashes, seed,
temperature, token limits, and Memory snapshot hash for each experiment.

The official Stage 3 evaluator runs independent seeds 42, 43, and 44. Seed 42
defines pass@1; pass@3 is true when any of the three independent rollouts is
correct. An incorrect answer never triggers a retry. Transport failures may be
retried on a different endpoint and are saved separately.

Generated results belong under ignored `runs/`, `output/`, or `results/`
directories. Do not commit model weights, datasets, Memory stores, or local
deployment manifests.

