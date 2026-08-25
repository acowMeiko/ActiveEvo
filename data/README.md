# Data

The Stage 1 capability-zoning dataset is committed at:

```text
data/deepscaler_merged_6000.json
```

It contains 6,000 records. Every record has `question`, `answer`, and
`difficulty_level` keys. Source row 5743 has an empty `answer`; Stage 1 retains
it as `invalid_input` outside Master/Learning/Panic instead of silently changing
the frozen source. The frozen file SHA-256 is:

```text
9d8b634043cb008076cbf60b7c8a01fddaeb2a83f9d3a8de564b05dd176eb225
```

Verify it after cloning or copying:

```bash
sha256sum data/deepscaler_merged_6000.json
```

Generated rollouts, Learning Bundles, checkpoints and evaluation outputs stay
under `output/` and are intentionally excluded from Git.
