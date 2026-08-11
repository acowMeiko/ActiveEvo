#!/usr/bin/env python
import json
import time
from pathlib import Path


SUMMARY_PATH = Path("/home/czhu/workplace/task/MemUA/output/stage3/dpo_level2/01_guided_eval/principle_guided_test_summary.json")
MD_PATH = Path("/home/czhu/workplace/task/MemUA/docs/experiments/STAGE3_MATH500_RESULTS.md")


def _summary_get(summary: dict, *keys: str, default="-"):
    current = summary
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def render_row(summary: dict) -> str:
    return (
        f"| 2026-04-21 | `dpo_level2` | `math500_stage3.json` | `base` | "
        f"`8000,8001,8002,8003` | `memory/principles/principles.json` | "
        f"{_summary_get(summary, 'performance', 'scope_total')} | {_summary_get(summary, 'performance', 'correct')} | "
        f"{_summary_get(summary, 'performance', 'accuracy')} | - | "
        f"{_summary_get(summary, 'retrieval', 'retrieved_samples')} | {_summary_get(summary, 'retrieval', 'retrieval_used_ratio')} | "
        f"- | - | "
        f"- | `done` | "
        f"`evaluate_principle_guided_test.py --input data/test/benchmarks/math500_stage3.json --memory-dir memory --answer-max-tokens 1024` |"
    )


def update_md(summary: dict) -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines = []
    replaced = False
    for line in lines:
        if line.startswith("| 2026-04-21 | `dpo_level2` | `math500_stage3.json` |"):
            new_lines.append(render_row(summary))
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(render_row(summary))
    MD_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> None:
    for _ in range(12 * 60):  # up to 12 hours, poll every 10s
        if SUMMARY_PATH.exists():
            try:
                summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            except Exception:
                time.sleep(10)
                continue
            if isinstance(summary, dict) and int(_summary_get(summary, "performance", "scope_total", default=0) or 0) == 500:
                update_md(summary)
                print("Filled markdown table from completed summary.")
                return
        time.sleep(10)
    raise SystemExit("Timed out waiting for stage3 summary.")


if __name__ == "__main__":
    main()
