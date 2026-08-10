# ActiveEvo

ActiveEvo is an experimental framework for building, validating, versioning,
retrieving, and applying reusable reasoning experience for language models.
The repository contains code only: datasets, generated Memory stores, model
weights, experiment outputs, paper files, and review material are excluded.

## Pipeline

```text
Stage 1: sample trajectories and identify informative learning cases
    ↓
Stage 2: generate Principles, validate them, and commit versioned Memory
    ↓
Stage 3: retrieve candidate Memory, apply the gate, and evaluate the solver
```

Stage 2 stores four derived views of each canonical Memory record:

- `routing_view`: structured fields used by retrieval and applicability checks.
- `solver_view`: executable guidance that may enter the solver prompt.
- `validation_view`: paired validation evidence and historical outcomes.
- `audit_view`: provenance, repairs, configuration, and version history.

## Repository layout

| Path | Responsibility |
|---|---|
| `core/` | Pipeline orchestration, API routing, configuration, and checkpoints |
| `modules/` | Stage implementations, Memory, DSD, gating, validation, and reranking |
| `template/` | Frozen prompt templates |
| `api_call/` | Local vLLM deployment, health check, and shutdown scripts |
| `benchmarks/` | Dataset evaluators, answer matching, and summaries |
| `experiments/` | Reproducible experiment controllers and statistical analysis |
| `ablation/` | Ablation implementations |
| `tests/` | Contract and unit tests |
| `tools/` | Analysis and maintenance utilities |
| `docs/` | Architecture and reproducibility documentation |

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp deploy.conf.example deploy.conf
cp config.env.example config.env
```

Edit the two local configuration files before running a model. They are ignored
by Git so that model paths and machine-specific settings cannot be committed.

## Model deployment

```bash
bash api_call/run.sh deploy.conf
bash api_call/test.sh deploy.conf
bash api_call/stop.sh deploy.conf
```

The default mapping is `port = DEPLOY_PORT_BASE + GPU_ID`.

## Running the stages

```bash
# Stage 1
python pipeline.py --stage 1 --dataset dpo_level1 \
  --model llama --ports 8100 8101 8102 8103

# Stage 2
python pipeline.py --stage 2 --dataset dpo_level1 \
  --model llama --ports 8100 8101 8102 8103 \
  --memory-dir memory/v3

# Stage 3
python pipeline.py --stage 3 --dataset dpo_level1 \
  --test-input data/test/deepscaler.json \
  --memory-dir memory/v3 --model llama \
  --ports 8100 8101 8102 8103
```

The official three-seed Base/NV evaluation protocol is under
`experiments/overnight_nv_ablation/`.

## Tests

```bash
python tests/run_stage2_v3_contracts.py
DRY_RUN=1 bash run_all_tests.sh
```

## Generated files

The following are deliberately local and ignored by Git:

- datasets and learning packets;
- Memory stores, snapshots, checkpoints, and vector indices;
- experiment outputs, logs, and error analysis;
- model weights and embedding caches;
- paper sources, reviews, and submission material.

See `docs/PROJECT_MAP.md` and `docs/REPRODUCIBILITY.md` for implementation-level
details.
