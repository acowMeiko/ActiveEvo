# ActiveEvo

ActiveEvo studies experience-driven mathematical reasoning with a capability
signal and structured Memory retrieval.

```text
Stage 1: weak-model rollouts -> capability levels -> LearningBundle
Stage 2: trajectory evidence -> Principle -> quality audit -> Memory Store
Stage 3: SPD retrieval -> runtime gate/adapter -> Solver -> pass@1/pass@3
```

Stage 2 does not use Neighbor Validation. A Principle becomes searchable only
after schema, leakage, source-fidelity and actionability checks. Its downstream
effect is measured later by Stage 3 paired evaluation.

## Project commands

Run all commands from the repository root.

```bash
# Deployment is independent from the experiment controller.
bash api_call/run.sh api_call/actor.conf llama 0 1 2 3 --daemon
bash api_call/test.sh api_call/actor.conf llama
bash api_call/stop.sh api_call/actor.conf

# Stage-2 auditor is a separate service (example maps GPU 4 to port 8200).
bash api_call/run.sh api_call/auditor.conf qwen 4 --daemon
export ACTIVEEVO_GENERATOR_API_KEY='your-deepseek-key'
export ACTIVEEVO_AUDITOR_URL='http://localhost:8200/v1/chat/completions'

# Stage 1 samples the complete external dataset once. Accuracy assigns levels;
# entropy only ranks questions inside each level.
python pipeline.py stage1 \
  --dataset /path/to/deepscaler_merged_6000.json \
  --ports 8100 8101 8102 8103 \
  --resume

# Rebuild the four Stage-1 diagnostic figures without calling a model.
python scripts/stage1/analyze.py --stage1-dir output/stage1

# Stage 2 cumulatively builds the same Store by curriculum level.
python pipeline.py stage2 --level level1 --memory-dir memory/store
python pipeline.py stage2 --level level2 --memory-dir memory/store
python pipeline.py stage2 --level level3 --memory-dir memory/store
python pipeline.py stage2 --level level4 --memory-dir memory/store

# Stage 3 baseline and guided pass@3 evaluation.
python pipeline.py stage3 --variant baseline
python pipeline.py stage3 --variant guided --memory-dir memory/store

# Static checks and unit contracts.
bash scripts/tests/run_all_tests.sh
```

## Model roles on a new machine

The repository does not contain weights. Model mount points are reserved under
`models/actor`, `models/generator`, `models/auditor`, and `models/embedding`.
The default Stage 2 generator is DeepSeek V4 API; the default independent local
auditor is Qwen3.5-27B-FP8. See [models/README.md](models/README.md).
Install its serving environment separately with
`api_call/requirements-auditor.txt`; Qwen3.5 requires a current vLLM build.

## Directory guide

| Path | Responsibility |
|---|---|
| `pipeline.py` | Only project-level Python controller |
| `core/config.py` | Central paths, model roles, and Stage defaults |
| `modules/stage2/` | Inspectable Memory Factory functions |
| `modules/` | Stage 1 sampling, Memory, retrieval and Solver modules |
| `scripts/` | Optional stage-specific utilities |
| `api_call/` | Model deployment, health checks and shutdown |
| `memory/store/` | Versioned searchable Memory and Level snapshots |
| `output/` | Generated run artifacts; not source code |
| `tests/` | Unit and contract tests |

The precise Stage 2 data flow and function map is documented in
[docs/STAGE2_MEMORY_FACTORY.md](docs/STAGE2_MEMORY_FACTORY.md).
