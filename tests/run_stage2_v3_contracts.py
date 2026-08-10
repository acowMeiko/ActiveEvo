#!/usr/bin/env python3
"""Run the Stage-2 v3 contract tests without requiring pytest."""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEST_ROOT))

import test_application_gate as application_gate
import test_principle_schema as principle_schema
import test_solver_v1 as solver_v1
import test_stage2_v3_contract as stage2_v3


def _run_zero_arg_tests(module) -> int:
    count = 0
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        function = getattr(module, name)
        if function.__code__.co_argcount == 0:
            function()
            count += 1
    return count


def main() -> None:
    count = 0
    for module in (principle_schema, application_gate, solver_v1, stage2_v3):
        count += _run_zero_arg_tests(module)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        stage2_v3.test_checkpoint_resume_and_parent(root / "checkpoint")
        stage2_v3.test_memory_replacement_keeps_version_history(root / "memory")
        count += 2
    print(f"Stage-2 v3 contracts: {count} passed")


if __name__ == "__main__":
    main()
