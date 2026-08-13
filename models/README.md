# Model mount points

Model weights are not part of the repository. On a new machine, download or
mount models under these role directories and configure their endpoints in
`core/config.py` or the documented environment variables.

- `actor/`: weak model used by Stage 1 and Stage 3.
- `generator/`: optional local Principle generator; the default is DeepSeek V4 API.
- `auditor/`: default local quality auditor, Qwen3.5-27B-FP8 on a 48GB GPU.
- `embedding/`: default routing encoder, BGE-M3.

Keep each role directory name stable. Replacing a model should not require a
change to Stage 1/2/3 business logic.

Example downloads on the new machine:

```bash
huggingface-cli download Qwen/Qwen3.5-27B-FP8 --local-dir models/auditor
huggingface-cli download BAAI/bge-m3 --local-dir models/embedding
# Download the chosen weak actor in the same way into models/actor.
```
