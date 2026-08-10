"""Compare paired base/guided results and summarize memory routing effects."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.execute_module import extract_final_answer


def correctness_transition(base: bool, guided: bool) -> str:
    if not base and guided:
        return "improved"
    if base and not guided:
        return "degraded"
    return "both_correct" if base else "both_wrong"


def normalized_answer(text: str) -> str:
    answer = extract_final_answer(text or "") or ""
    return re.sub(r"\s+", "", answer).lower()


def mcnemar_exact_p(improved: int, degraded: int) -> float:
    discordant = improved + degraded
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(improved, degraded) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_delta(rows: list[Dict[str, Any]], seed: int, samples: int = 10000) -> Dict[str, float]:
    rng = random.Random(seed)
    n = len(rows)
    deltas = []
    for _ in range(samples):
        chosen = [rows[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(int(row["guided_correct"]) - int(row["base_correct"]) for row in chosen) / n)
    deltas.sort()
    return {
        "mean": sum(deltas) / len(deltas),
        "ci95_low": deltas[int(0.025 * len(deltas))],
        "ci95_high": deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))],
        "samples": samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--guided", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-file", help="Original indexed test subset, used for level metadata")
    parser.add_argument("--counterfactual-replay", help="Serial base→guided replay for invoked samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_items = {int(row["idx"]): row for row in json.loads(Path(args.base).read_text(encoding="utf-8"))}
    guided_items = {int(row["idx"]): row for row in json.loads(Path(args.guided).read_text(encoding="utf-8"))}
    test_items = json.loads(Path(args.test_file).read_text(encoding="utf-8")) if args.test_file else []
    if set(base_items) != set(guided_items):
        raise ValueError("Base/guided item indices do not match")

    rows = []
    selected_frequency: Counter[str] = Counter()
    schema_sources: Counter[str] = Counter()
    candidate_total = 0
    gate_parse_failures = 0
    for idx in sorted(base_items):
        base = base_items[idx]
        guided = guided_items[idx]
        trace = guided.get("retrieval_trace", {}) or {}
        selected_ids = trace.get("selected_ids", []) or []
        for memory_id in selected_ids:
            selected_frequency[memory_id] += 1
        for injected in trace.get("injected_memory", []) or []:
            schema_sources[str(injected.get("schema_source", "unknown"))] += 1
        candidate_total += len(trace.get("candidate_ids", []) or [])
        gate = trace.get("applicability")
        if gate and gate.get("status") != "success":
            gate_parse_failures += 1
        source_item = test_items[idx] if idx < len(test_items) else guided.get("source_item", {})
        level = int((source_item.get("_sampling", {}) or {}).get("stratified_level", 0) or 0)
        base_final = normalized_answer(base.get("model_answer", ""))
        guided_final = normalized_answer(guided.get("model_answer", ""))
        rows.append({
            "idx": idx,
            "level": level,
            "question": guided.get("question", ""),
            "base_correct": bool(base.get("is_correct")),
            "guided_correct": bool(guided.get("is_correct")),
            "transition": correctness_transition(bool(base.get("is_correct")), bool(guided.get("is_correct"))),
            "principle_invoked": bool(selected_ids),
            "selected_count": len(selected_ids),
            "selected_ids": selected_ids,
            "full_response_changed": (base.get("model_answer") or "").strip() != (guided.get("model_answer") or "").strip(),
            "final_answer_changed": base_final != guided_final,
            "base_final_answer": base_final,
            "guided_final_answer": guided_final,
            "base_answer": base.get("model_answer", ""),
            "guided_answer": guided.get("model_answer", ""),
            "injected_principle_text": guided.get("injected_principle_text", ""),
            "retrieval_trace": trace,
        })

    transitions = Counter(row["transition"] for row in rows)
    invoked = [row for row in rows if row["principle_invoked"]]
    not_invoked = [row for row in rows if not row["principle_invoked"]]
    by_level = {}
    for level in sorted({row["level"] for row in rows}):
        subset = [row for row in rows if row["level"] == level]
        by_level[f"level_{level}"] = {
            "n": len(subset),
            "base_correct": sum(row["base_correct"] for row in subset),
            "guided_correct": sum(row["guided_correct"] for row in subset),
            "invoked": sum(row["principle_invoked"] for row in subset),
            "transitions": dict(Counter(row["transition"] for row in subset)),
        }

    summary = {
        "n": len(rows),
        "seed": args.seed,
        "base_accuracy": sum(row["base_correct"] for row in rows) / len(rows),
        "guided_accuracy": sum(row["guided_correct"] for row in rows) / len(rows),
        "accuracy_delta": (sum(row["guided_correct"] for row in rows) - sum(row["base_correct"] for row in rows)) / len(rows),
        "transitions": dict(transitions),
        "mcnemar_exact_p": mcnemar_exact_p(transitions["improved"], transitions["degraded"]),
        "paired_bootstrap": bootstrap_delta(rows, args.seed),
        "routing": {
            "samples_with_principle": len(invoked),
            "sample_invocation_rate": len(invoked) / len(rows),
            "candidate_total": candidate_total,
            "selected_total": sum(row["selected_count"] for row in rows),
            "selected_count_distribution": dict(Counter(row["selected_count"] for row in rows)),
            "unique_selected_memories": len(selected_frequency),
            "selected_frequency": dict(selected_frequency.most_common()),
            "schema_sources": dict(schema_sources),
            "gate_parse_failures": gate_parse_failures,
        },
        "answer_change": {
            "all_full_response_changed": sum(row["full_response_changed"] for row in rows),
            "all_final_answer_changed": sum(row["final_answer_changed"] for row in rows),
            "invoked_full_response_changed": sum(row["full_response_changed"] for row in invoked),
            "invoked_final_answer_changed": sum(row["final_answer_changed"] for row in invoked),
            "not_invoked_full_response_changed": sum(row["full_response_changed"] for row in not_invoked),
            "not_invoked_final_answer_changed": sum(row["final_answer_changed"] for row in not_invoked),
        },
        "invoked_transitions": dict(Counter(row["transition"] for row in invoked)),
        "not_invoked_transitions": dict(Counter(row["transition"] for row in not_invoked)),
        "by_level": by_level,
    }

    replay_rows = []
    if args.counterfactual_replay:
        replay_rows = json.loads(Path(args.counterfactual_replay).read_text(encoding="utf-8"))
        replay_transitions = Counter(
            correctness_transition(bool(row.get("base_correct")), bool(row.get("guided_correct")))
            for row in replay_rows
        )
        replay_final_changed = sum(
            re.sub(r"\s+", "", str(row.get("base_final_answer") or "")).lower()
            != re.sub(r"\s+", "", str(row.get("guided_final_answer") or "")).lower()
            for row in replay_rows
        )
        summary["serial_invoked_counterfactual"] = {
            "n": len(replay_rows),
            "base_correct": sum(bool(row.get("base_correct")) for row in replay_rows),
            "guided_correct": sum(bool(row.get("guided_correct")) for row in replay_rows),
            "transitions": dict(replay_transitions),
            "mcnemar_exact_p": mcnemar_exact_p(replay_transitions["improved"], replay_transitions["degraded"]),
            "full_response_changed": sum(bool(row.get("full_response_changed")) for row in replay_rows),
            "final_answer_changed": replay_final_changed,
            "protocol": "one active pair per endpoint; same item, endpoint, seed; base immediately followed by guided",
        }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "paired_cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "paired_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "idx", "level", "base_correct", "guided_correct", "transition", "principle_invoked",
            "selected_count", "selected_ids", "full_response_changed", "final_answer_changed",
            "base_final_answer", "guided_final_answer", "question",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            compact = {key: row[key] for key in fields}
            compact["selected_ids"] = "|".join(row["selected_ids"])
            writer.writerow(compact)

    examples = [row for row in rows if row["principle_invoked"] and row["transition"] in {"improved", "degraded"}]
    examples += [row for row in rows if row["principle_invoked"] and row["transition"] == "both_wrong"][:2]
    report = [
        "# Paired DeepScaler Memory-Effect Report",
        "",
        f"- N: {summary['n']} (Level 2/3 stratified)",
        f"- Base: {summary['base_accuracy']:.1%}",
        f"- Guided: {summary['guided_accuracy']:.1%}",
        f"- Delta: {summary['accuracy_delta']:+.1%}",
        f"- Improved / degraded: {transitions['improved']} / {transitions['degraded']}",
        f"- Exact McNemar p: {summary['mcnemar_exact_p']:.4f}",
        f"- Principle invoked: {len(invoked)}/{len(rows)} ({len(invoked)/len(rows):.1%})",
        f"- Injected memories: {summary['routing']['selected_total']} ({summary['routing']['unique_selected_memories']} unique)",
        f"- Injected schema sources: {dict(schema_sources)}",
        "",
        "## Interpretation",
        "",
        "This is a code-path and routing audit, not a final paper result. The evaluated legacy store retained only action text; routing fields were explicitly derived for compatibility. A canonical five-field store must be rebuilt before claiming the applicability-gate result.",
        "",
        "## Selected cases",
        "",
    ]
    if replay_rows:
        replay_summary = summary["serial_invoked_counterfactual"]
        report[10:10] = [
            f"- Serial invoked replay: base {replay_summary['base_correct']}/{replay_summary['n']}, guided {replay_summary['guided_correct']}/{replay_summary['n']}",
            f"- Serial replay improved / degraded: {replay_summary['transitions'].get('improved', 0)} / {replay_summary['transitions'].get('degraded', 0)}",
            f"- Serial replay final answer changed: {replay_summary['final_answer_changed']}/{replay_summary['n']}",
        ]
    for row in examples[:8]:
        report.extend([
            f"### Item {row['idx']} · Level {row['level']} · {row['transition']}",
            "",
            f"Question: {row['question']}",
            "",
            f"Selected: {', '.join(row['selected_ids'])}",
            "",
            f"Injected memory:\n```text\n{row['injected_principle_text']}\n```",
            "",
            f"Base final: `{row['base_final_answer']}`; guided final: `{row['guided_final_answer']}`.",
            "",
        ])
    (output / "paired_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
