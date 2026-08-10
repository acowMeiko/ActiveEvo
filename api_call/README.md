# Local model deployment

The scripts in this directory manage one OpenAI-compatible vLLM endpoint per
selected GPU. Copy `deploy.conf.example` to `deploy.conf` in the project root
and configure local model paths there.

```bash
bash api_call/run.sh deploy.conf
bash api_call/test.sh deploy.conf
bash api_call/stop.sh deploy.conf
```

The selected deployment is recorded under the ignored `api_call/logs/`
directory. Health checks and shutdown use that runtime manifest, so command-line
GPU overrides remain consistent across all three operations.

By default, each endpoint uses `DEPLOY_PORT_BASE + GPU_ID`.

