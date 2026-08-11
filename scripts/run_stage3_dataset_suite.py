#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/home/czhu/miniconda3/envs/hcz/bin/python")
MD_PATH = ROOT / "docs" / "experiments" / "STAGE3_DATASET_SUITE_RESULTS.md"
OFFICIAL_PREP_SCRIPT = ROOT / "scripts" / "prepare_official_datasets.py"
EVAL_SCRIPT = ROOT / "evaluate_principle_guided_test.py"
MATH500_SUMMARY = ROOT / "output" / "stage3" / "dpo_level2" / "01_guided_eval" / "principle_guided_test_summary.json"
SUITE_OUTPUT_ROOT = ROOT / "output" / "stage3_suite" / "dpo_level2"
LOG_ROOT = ROOT / "logs" / "stage3_suite"


DATASETS: List[Dict[str, object]] = [
    {
        "family": "math500",
        "subset": "full",
        "test_file": "data/test/benchmarks/math500_stage3.json",
        "summary_path": MATH500_SUMMARY,
        "results_path": ROOT / "output" / "stage3" / "dpo_level2" / "01_guided_eval" / "principle_guided_test_results.json",
        "rerun": False,
    },
    {
        "family": "deepscaler",
        "subset": "full",
        "test_file": "data/test/deepscaler.json",
        "summary_path": SUITE_OUTPUT_ROOT / "deepscaler" / "principle_guided_test_summary.json",
        "results_path": SUITE_OUTPUT_ROOT / "deepscaler" / "principle_guided_test_results.json",
        "rerun": True,
    },
    {
        "family": "bbh",
        "subset": "logic",
        "test_file": "data/test/benchmarks_split/bbh_logic_stage3.json",
        "summary_path": SUITE_OUTPUT_ROOT / "bbh_logic" / "principle_guided_test_summary.json",
        "results_path": SUITE_OUTPUT_ROOT / "bbh_logic" / "principle_guided_test_results.json",
        "rerun": True,
    },
    {
        "family": "bbh",
        "subset": "math",
        "test_file": "data/test/benchmarks_split/bbh_math_stage3.json",
        "summary_path": SUITE_OUTPUT_ROOT / "bbh_math" / "principle_guided_test_summary.json",
        "results_path": SUITE_OUTPUT_ROOT / "bbh_math" / "principle_guided_test_results.json",
        "rerun": True,
    },
    {
        "family": "svamp",
        "subset": "official_test",
        "test_file": "data/test/official/svamp/processed/test.json",
        "summary_path": SUITE_OUTPUT_ROOT / "svamp" / "principle_guided_test_summary.json",
        "results_path": SUITE_OUTPUT_ROOT / "svamp" / "principle_guided_test_results.json",
        "rerun": True,
    },
    {
        "family": "general365",
        "subset": "official_test",
        "test_file": "data/test/official/general365/processed/test.json",
        "summary_path": SUITE_OUTPUT_ROOT / "general365" / "principle_guided_test_summary.json",
        "results_path": SUITE_OUTPUT_ROOT / "general365" / "principle_guided_test_results.json",
        "rerun": True,
    },
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_summary(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_num(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _summary_get(summary: Dict[str, object], *keys: str, default: object = "-") -> object:
    current: object = summary
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _render_row(entry: Dict[str, object], summary: Dict[str, object] | None = None, status: str | None = None) -> str:
    family = str(entry["family"])
    subset = str(entry["subset"])
    test_file = str(entry["test_file"])
    current_status = status or ("done" if summary else "pending")
    if summary:
        total = _format_num(_summary_get(summary, "performance", "scope_total"))
        correct = _format_num(_summary_get(summary, "performance", "correct"))
        accuracy = _format_num(_summary_get(summary, "performance", "accuracy"))
        accuracy_at_3 = "-"
        first_attempt_acc = "-"
        retrieval_used = _format_num(_summary_get(summary, "retrieval", "retrieved_samples"))
        retrieval_ratio = _format_num(_summary_get(summary, "retrieval", "retrieval_used_ratio"))
        retry_used = "-"
        retry_ratio = "-"
        rescued = "-"
    else:
        total = correct = accuracy = accuracy_at_3 = first_attempt_acc = "-"
        retrieval_used = retrieval_ratio = retry_used = retry_ratio = rescued = "-"
    notes = "`memory/principles/principles.json`, `ports=8000,8001,8002,8003`, `answer_max_tokens=1024`"
    return (
        f"| 2026-04-21 | `{family}` | `{subset}` | `{test_file}` | "
        f"{total} | {correct} | {accuracy} | {accuracy_at_3} | {first_attempt_acc} | "
        f"{retrieval_used} | {retrieval_ratio} | {retry_used} | {retry_ratio} | {rescued} | "
        f"`{current_status}` | {notes} |"
    )


def update_markdown(rows: Dict[str, str]) -> None:
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    new_lines: List[str] = []
    for line in lines:
        replaced = False
        for key, row in rows.items():
            family, subset = key.split("::", 1)
            prefix = f"| 2026-04-21 | `{family}` | `{subset}` |"
            if line.startswith(prefix):
                new_lines.append(row)
                replaced = True
                break
        if not replaced:
            new_lines.append(line)
    MD_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def ensure_official_datasets() -> None:
    svamp_test = ROOT / "data" / "test" / "official" / "svamp" / "processed" / "test.json"
    general_test = ROOT / "data" / "test" / "official" / "general365" / "processed" / "test.json"
    if svamp_test.exists() and general_test.exists():
        return
    subprocess.run([str(PYTHON_BIN), str(OFFICIAL_PREP_SCRIPT)], cwd=str(ROOT), check=True)


def validate_inputs() -> None:
    required = [
        ROOT / "memory" / "principles" / "principles.json",
        EVAL_SCRIPT,
        MD_PATH,
        PYTHON_BIN,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")
    for entry in DATASETS:
        path = ROOT / str(entry["test_file"])
        if not path.exists():
            raise FileNotFoundError(f"Missing test file: {path}")


def run_one(entry: Dict[str, object]) -> None:
    family = str(entry["family"])
    subset = str(entry["subset"])
    family_key = family if family != "bbh" else f"{family}_{subset}"
    summary_path = Path(entry["summary_path"])
    results_path = Path(entry["results_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{family_key}.log"
    test_path = ROOT / str(entry["test_file"])

    cmd = [
        str(PYTHON_BIN),
        "-u",
        str(EVAL_SCRIPT),
        "--input",
        str(test_path),
        "--output",
        str(results_path),
        "--summary-output",
        str(summary_path),
        "--memory-dir",
        str(ROOT / "memory"),
        "--mode",
        "guided",
        "--concurrency",
        "20",
        "--model",
        "base",
        "--ports",
        "8000",
        "8001",
        "8002",
        "8003",
        "--timeout",
        "600",
        "--threshold",
        "0.55",
        "--top-k",
        "5",
        "--answer-max-tokens",
        "1024",
    ]

    rows = {f"{family}::{subset}": _render_row(entry, None, status="running")}
    update_markdown(rows)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=log_file, stderr=subprocess.STDOUT)

    summary = _load_summary(summary_path)
    update_markdown({f"{family}::{subset}": _render_row(entry, summary, status="done")})


def seed_existing_results() -> None:
    rows: Dict[str, str] = {}
    for entry in DATASETS:
        key = f"{entry['family']}::{entry['subset']}"
        summary_path = Path(entry["summary_path"])
        if summary_path.exists():
            rows[key] = _render_row(entry, _load_summary(summary_path), status="done")
        else:
            rows[key] = _render_row(entry, None, status="pending")
    update_markdown(rows)


def main() -> None:
    ensure_official_datasets()
    validate_inputs()
    seed_existing_results()

    for entry in DATASETS:
        if not bool(entry["rerun"]):
            continue
        summary_path = Path(entry["summary_path"])
        if summary_path.exists():
            summary = _load_summary(summary_path)
            if int(_summary_get(summary, "performance", "scope_total", default=0) or 0) > 0:
                update_markdown({f"{entry['family']}::{entry['subset']}": _render_row(entry, summary, status="done")})
                continue
        run_one(entry)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
