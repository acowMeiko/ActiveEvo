# ActiveEvo

ActiveEvo is an experimental framework for building, validating, versioning,
retrieving, and applying reusable reasoning experience for language models.

This repository contains code only. Datasets, generated Memory stores, model
weights, experiment outputs, paper files, and review material are intentionally
excluded.

## Pipeline

```text
Stage 1: sample trajectories and build learning packets
    -> Stage 2: generate, validate, and version Principles as Memory
    -> Stage 3: retrieve Memory, gate applicability, and evaluate the solver
```

The main entry point is `pipeline.py`. Helper scripts live under `scripts/`
so the repository root stays small and readable.

## Setup

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp deploy.conf.example deploy.conf
cp config.env.example config.env
```

Edit `deploy.conf` and `config.env` for your local model paths, GPUs, ports,
embedding model, concurrency, and token limits. Both local config files are
ignored by Git.

## Model Deployment

```bash
bash api_call/run.sh deploy.conf
bash api_call/test.sh deploy.conf llama
bash api_call/stop.sh deploy.conf
```

The default port mapping is `port = DEPLOY_PORT_BASE + GPU_ID`.

## Common Commands

Stage 1:

```bash
python pipeline.py --stage 1 --dataset dpo_level1 --ports 8100 8101 8102 8103
python scripts/stage1/run_stage1_llama.py --resume
python scripts/stage1/build_learning_packets.py --dataset dpo_level1 --ports 8100 8101
```

Stage 2:

```bash
python pipeline.py --stage 2 --dataset dpo_level1 \
  --memory-dir memory/v3 --ports 8100 8101 8102 8103

bash scripts/stage2/run_all_levels.sh
```

Stage 3:

```bash
python pipeline.py --stage 3 --dataset dpo_level1 \
  --test-input data/test/deepscaler.json \
  --memory-dir memory/v3 --ports 8100 8101 8102 8103

python scripts/stage3/evaluate_principle_guided_test.py --help
```

Tests:

```bash
python tests/run_stage2_v3_contracts.py
DRY_RUN=1 bash scripts/tests/run_all_tests.sh
```

## Repository Layout

| Path | Responsibility |
|---|---|
| `pipeline.py` | Main Stage 1/2/3 command-line entry point |
| `core/` | Pipeline orchestration, API routing, configuration, and checkpoints |
| `modules/` | Stage logic, Memory, DSD, gating, validation, retrieval, and reranking |
| `template/` | Frozen prompt templates |
| `utils/` | Shared path, IO, embedding, and evaluation helpers |
| `api_call/` | Local vLLM deployment, health check, and shutdown scripts |
| `scripts/` | Organized helper entry points for stages, deploy, and tests |
| `experiments/` | Reproducible experiment controllers and statistical analysis |
| `ablation/` | Ablation implementations |
| `benchmarks/` | Dataset evaluators, answer matching, and summaries |
| `tests/` | Contract and unit tests |
| `tools/` | Analysis and maintenance utilities |
| `docs/` | Architecture and reproducibility documentation |

## Generated Files

Do not commit datasets, learning packets, Memory stores, checkpoints, vector
indices, experiment outputs, model weights, local configs, or paper material.
These are ignored by `.gitignore`.

See `docs/PROJECT_STRUCTURE.md` and `docs/REPRODUCIBILITY.md` for implementation
and reproducibility notes.
