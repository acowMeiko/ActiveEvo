#!/usr/bin/env python3
"""Combine structural-gate and temperature/adaptation experiment summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    gate = load_object(args.root / "gate_summary.json")
    runs = []
    for path in sorted(args.root.glob("temperature_*/summary.json")):
        runs.append(load_object(path))
    if not runs:
        raise SystemExit(f"No temperature_*/summary.json under {args.root}")
    runs.sort(key=lambda value: float(value["reasoner_temperature"]))
    payload = {
        "stage": 3,
        "experiment": "structural_gate_raw_vs_adapted_temperature_sweep",
        "gate": gate,
        "temperature_runs": runs,
        "text_comparison": str(args.root / "text_comparison.md"),
    }
    lines = [
        "# Stage 3 gate, adaptation, and temperature experiment",
        "",
        "## Structural gate",
        "",
        f"- Candidate pairs judged: {gate['candidate_count']}",
        f"- Gate accepted: {gate['gate_accepted_count']}",
        f"- Gate rejected: {gate['gate_rejected_count']}",
        f"- Raw/adapted eligible: {gate['experiment_eligible_count']}",
        f"- Gate calls format-parsed: {gate.get('gate_format_parsed_count', 0)}/{gate.get('gate_item_count', gate.get('candidate_count', 0))}",
        f"- Gate calls strictly compliant: {gate.get('gate_protocol_compliant_count', 0)}/{gate.get('gate_item_count', gate.get('candidate_count', 0))}",
        f"- Accepted indices: {gate['accepted_indices']}",
        f"- Rejected indices: {gate['rejected_indices']}",
        "",
        "Exact source, raw, adapted, and unified-diff text is in `text_comparison.md`.",
        "",
        "## Solver results",
        "",
        "| temperature | condition | injected | correct | hard loops | caps | reasoner format | finalizer format | avg tokens |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        temperature = float(run["reasoner_temperature"])
        for condition in ("base", "raw", "adapted"):
            metric = run["conditions"][condition]
            n = int(metric["count"])
            lines.append(
                f"| {temperature:.1f} | {condition} | {metric['memory_injected']} | "
                f"{metric['correct']}/{n} | {metric['hard_loops']}/{n} | "
                f"{metric['reasoner_token_limit']}/{n} | {metric['reasoner_format_valid']}/{n} | "
                f"{metric['finalizer_format_valid']}/{n} | {metric['average_reasoner_tokens']:.1f} |"
            )
        pair = run["paired_raw_to_adapted"]
        lines.extend([
            "",
            f"At temperature {temperature:.1f}, Raw→Adapted on {pair['eligible_count']} eligible cases: "
            f"rescues={pair['rescues']}, harms={pair['harms']}, changed final answers={pair['changed_final_answers']}.",
            "",
        ])
    lines.extend([
        "## Interpretation guardrails",
        "",
        "- Gate rejection and text adaptation are frozen before Solver evaluation.",
        "- Raw and Adapted inject memory on exactly the same eligible items.",
        "- Base is run once per item; Raw and Adapted are run only for eligible items.",
        "- Rejected/no-candidate items trigger no duplicate Raw/Adapted Solver calls.",
        "- Thirty items are a protocol screen, not sufficient for a paper-level accuracy claim.",
        "",
    ])
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "experiment_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.root / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report={args.root / 'experiment_report.md'}", flush=True)


if __name__ == "__main__":
    main()
