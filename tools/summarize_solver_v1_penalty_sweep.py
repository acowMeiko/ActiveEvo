#!/usr/bin/env python3
"""Summarize a Solver-v1 Reasoner repetition-penalty sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _safe_rate(value: float, denominator: int) -> float:
    return value / denominator if denominator else 0.0


def _row(summary_path: Path) -> dict[str, Any]:
    summary = _load(summary_path)
    metrics = summary["metrics"]
    pair_count = int(metrics["pair_count"])
    condition_count = 2 * pair_count
    hard_loops = int(metrics["base_hard_loops"] + metrics["guided_hard_loops"])
    token_limits = int(
        metrics["base_reasoner_token_limit"] + metrics["guided_reasoner_token_limit"]
    )
    reasoner_formats = int(
        metrics["base_reasoner_format_valid"] + metrics["guided_reasoner_format_valid"]
    )
    finalizer_formats = int(
        metrics["base_finalizer_format_valid"] + metrics["guided_finalizer_format_valid"]
    )
    average_reasoner_tokens = (
        float(metrics["base_reasoner_avg_tokens"])
        + float(metrics["guided_reasoner_avg_tokens"])
    ) / 2
    return {
        "summary_path": str(summary_path),
        "penalty": float(summary["reasoner_repetition_penalty"]),
        "pair_count": pair_count,
        "condition_count": condition_count,
        "hard_loop_count": hard_loops,
        "hard_loop_rate": _safe_rate(hard_loops, condition_count),
        "token_limit_count": token_limits,
        "token_limit_rate": _safe_rate(token_limits, condition_count),
        "reasoner_format_valid_count": reasoner_formats,
        "reasoner_format_valid_rate": _safe_rate(reasoner_formats, condition_count),
        "finalizer_format_valid_count": finalizer_formats,
        "finalizer_format_valid_rate": _safe_rate(finalizer_formats, condition_count),
        "average_reasoner_tokens": average_reasoner_tokens,
        # Accuracy is diagnostic only in this small protocol-reliability sweep.
        "base_accuracy": float(metrics["base_accuracy"]),
        "guided_accuracy": float(metrics["guided_accuracy"]),
        "rescues": int(metrics["rescues"]),
        "harms": int(metrics["harms"]),
        "injected_rescues": int(metrics["injected_rescues"]),
        "injected_harms": int(metrics["injected_harms"]),
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["hard_loop_count"]),
        float(row["token_limit_count"]),
        -float(row["reasoner_format_valid_rate"]),
        float(row["average_reasoner_tokens"]),
        float(row["penalty"]),
    )


def _markdown(rows: list[dict[str, Any]], best: dict[str, Any]) -> str:
    lines = [
        "# Solver-v1 repetition-penalty sweep",
        "",
        "This is a Stage 3 protocol-reliability sweep. Retrieval and injected memory are replayed and fixed.",
        "The preliminary recommendation is selected lexicographically by: hard loops, token-limit hits, Reasoner format validity, average Reasoner tokens, then the lower penalty.",
        "Accuracy, rescues, and harms are reported only as diagnostics and are not used for selection on 30 items.",
        "",
        f"Preliminary recommendation: **repetition_penalty={best['penalty']:.2f}**",
        "",
        "| penalty | hard loops | caps | reasoner format | avg reasoner tokens | Base acc. | Guided acc. | rescues | harms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {penalty:.2f} | {hard_loop_count}/{condition_count} | "
            "{token_limit_count}/{condition_count} | {reasoner_format_valid_rate:.1%} | "
            "{average_reasoner_tokens:.1f} | {base_accuracy:.1%} | {guided_accuracy:.1%} | "
            "{rescues} | {harms} |".format(**row)
        )
    lines.extend([
        "",
        "Before freezing this value, manually inspect every hard-loop case and every penalty-induced loss of a previously correct answer.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.root.glob("rp_*/summary.json"))
    if not paths:
        raise SystemExit(f"No rp_*/summary.json files found under {args.root}")
    rows = sorted((_row(path) for path in paths), key=lambda row: row["penalty"])
    best = min(rows, key=_rank_key)
    payload = {
        "experiment": "solver_v1_reasoner_repetition_penalty_sweep",
        "stage": 3,
        "accuracy_used_for_selection": False,
        "selection_order": [
            "minimize_hard_loop_count",
            "minimize_token_limit_count",
            "maximize_reasoner_format_valid_rate",
            "minimize_average_reasoner_tokens",
            "prefer_lower_penalty_on_ties",
        ],
        "preliminary_recommendation": best["penalty"],
        "results": rows,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "sweep_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.root / "sweep_report.md").write_text(_markdown(rows, best), encoding="utf-8")
    print(
        f"recommended_rp={best['penalty']:.2f} "
        f"hard_loops={best['hard_loop_count']} caps={best['token_limit_count']} "
        f"report={args.root / 'sweep_report.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
