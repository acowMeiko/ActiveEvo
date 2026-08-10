# Project structure

The public-facing entry point is `pipeline.py`, which delegates orchestration
to `core/agent_pipeline.py`.

```text
pipeline.py
├── core/
│   ├── agent_pipeline.py      # Stage orchestration
│   ├── api_call.py            # OpenAI-compatible HTTP client
│   ├── config.py              # Runtime configuration
│   ├── endpoint_pool.py       # Endpoint routing
│   └── thread_checkpoint.py   # Recoverable working state
├── modules/
│   ├── stage1_sampler.py      # Trajectory sampling
│   ├── stage1_miner.py        # Positive/negative trajectory analysis
│   ├── dsd_module.py          # Three-facet problem representation
│   ├── plan_module.py         # Principle generation and gates
│   ├── principle_schema.py    # Canonical Memory record and four views
│   ├── stage2_v3.py           # Validation and versioned Store commit
│   ├── memory.py              # Persistent Memory and indices
│   ├── memory_module.py       # Retrieval interface
│   └── execute_module.py      # Frozen two-stage solver
├── template/                  # Prompt contracts
├── benchmarks/                # Evaluation and answer matching
├── experiments/               # Reproducible experiment protocols
├── ablation/                  # Method ablations
├── tests/                     # Contract and unit tests
└── tools/                     # Analysis and maintenance utilities
```

## Runtime data flow

```text
dataset
  -> Stage 1 learning packets
  -> Stage 2 Principle candidates
  -> schema and quality checks
  -> optional neighbor validation
  -> versioned Memory Store
  -> Stage 3 retrieval and application gate
  -> Base/Guided paired evaluation
```

All generated nodes in this flow are ignored by Git.

