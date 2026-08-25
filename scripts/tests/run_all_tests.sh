#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m compileall -q pipeline.py core modules utils scripts tests tools ablation

if python -c 'import pytest' >/dev/null 2>&1; then
  python -m pytest -q tests
else
  python - <<'PY'
import importlib
import inspect

modules = (
    "tests.test_application_gate", "tests.test_matcher", "tests.test_principle_schema",
    "tests.test_reranker_routing", "tests.test_solver_v1", "tests.test_stage1_signal",
    "tests.test_stage2_factory", "tests.test_stage3_experiment_metrics",
    "tests.test_stage2_source_validation",
    "tests.test_stage2_principle_test", "tests.test_stage2_model_client",
    "tests.test_stage2_incremental_registry",
)
count = 0
for module_name in modules:
    module = importlib.import_module(module_name)
    for name, function in sorted(inspect.getmembers(module, inspect.isfunction)):
        if name.startswith("test_") and not inspect.signature(function).parameters:
            function()
            count += 1
print(f"pytest is unavailable; dependency-light contracts passed ({count}/{count}).")
PY
fi
