#!/usr/bin/env python3
"""Create paired statistical report for the three overnight variants."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_mcnemar(gain: int, loss: int) -> float:
    n = gain + loss
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(gain, loss) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def compare(base: List[Dict[str, Any]], other: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    by_idx = {int(item["idx"]): bool(item.get(field)) for item in base}
    pairs = [(by_idx[int(item["idx"])], bool(item.get(field))) for item in other if int(item["idx"]) in by_idx]
    gain = sum((not left) and right for left, right in pairs)
    loss = sum(left and (not right) for left, right in pairs)
    deltas = [int(right) - int(left) for left, right in pairs]
    rng = random.Random(42)
    samples = []
    for _ in range(10000):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) if deltas else 0.0)
    samples.sort()
    return {
        "n": len(pairs), "gain": gain, "loss": loss, "net_gain": gain - loss,
        "accuracy_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "mcnemar_exact_p": exact_mcnemar(gain, loss),
        "paired_bootstrap_95_ci": [samples[249], samples[9749]] if samples else [0.0, 0.0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    variants = ("baseline", "nv_on", "nv_off")
    summaries = {name: load(root / "stage3" / name / "eval_summary.json") for name in variants}
    results = {name: load(root / "stage3" / name / "eval_results.json") for name in variants}
    comparisons = {}
    for name in ("nv_on", "nv_off"):
        comparisons[name] = {
            "vs_baseline_pass1": compare(results["baseline"], results[name], "pass1_correct"),
            "vs_baseline_pass3": compare(results["baseline"], results[name], "pass3_correct"),
        }
    comparisons["nv_on_vs_nv_off"] = {
        "pass1": compare(results["nv_off"], results["nv_on"], "pass1_correct"),
        "pass3": compare(results["nv_off"], results["nv_on"], "pass3_correct"),
    }
    report = {"protocol": {"seeds": [42, 43, 44], "temperature": 0.2, "answer_dependent_retry": False},
              "variants": summaries, "paired_comparisons": comparisons}
    (root / "overnight_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Overnight NV Ablation",
        "",
        "| Variant | Active Memory | Pass@1 | Pass@3 | Infra failures | Retrieval used | Gate selected | Adapter accepted | Solver use/ignore/missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variants:
        s = summaries[name]
        total = int(s.get("total", 400))
        lines.append(
            f"| {name} | {s['active_memory_count']} | "
            f"{s['pass1_correct']}/{total} ({s['pass1_accuracy']:.3%}) | "
            f"{s['pass3_correct']}/{total} ({s['pass3_accuracy']:.3%}) | "
            f"{s['infrastructure_failure_count']} | {s['retrieval_used_count']} | "
            f"{s.get('application_gate_selected_count', 0)} | {s.get('adapter_accept_count', 0)} | "
            f"{s.get('solver_use_count', 0)}/{s.get('solver_ignore_count', 0)}/{s.get('solver_decision_missing_count', 0)} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, blocks in comparisons.items():
        lines.append(f"### {name}")
        lines.append("")
        for metric, value in blocks.items():
            ci = value["paired_bootstrap_95_ci"]
            lines.append(f"- {metric}: gain={value['gain']}, loss={value['loss']}, net={value['net_gain']}, delta={value['accuracy_delta']:.3%}, McNemar p={value['mcnemar_exact_p']:.6g}, bootstrap 95% CI=[{ci[0]:.3%}, {ci[1]:.3%}].")
        lines.append("")
    markdown = "\n".join(lines) + "\n"
    (root / "overnight_summary.md").write_text(markdown, encoding="utf-8")
    headers = []
    for name in variants:
        headers.append((root / "stage3" / name / "run.log").read_text(encoding="utf-8"))
    (root / "overnight.log").write_text(markdown + "\n" + "\n".join(headers), encoding="utf-8")


if __name__ == "__main__":
    main()
