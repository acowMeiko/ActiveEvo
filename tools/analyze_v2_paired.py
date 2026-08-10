#!/usr/bin/env python3
"""Generate paired statistics and human-readable injection audit for v2."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.matcher import extract_final_answer


def exact_mcnemar_p(base_only: int, guided_only: int) -> float:
    discordant = base_only + guided_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(base_only, guided_only) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def percentile(value: np.ndarray, q: float) -> float:
    return float(np.percentile(value, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--guided", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    guided = json.loads(Path(args.guided).read_text(encoding="utf-8"))
    tests = json.loads(Path(args.test_file).read_text(encoding="utf-8"))
    base_by_idx = {row["idx"]: row for row in base}
    guided_by_idx = {row["idx"]: row for row in guided}
    indices = sorted(set(base_by_idx) & set(guided_by_idx))
    if indices != list(range(len(tests))):
        raise ValueError("Base/guided results do not cover the same complete test subset")

    base_correct = np.array([bool(base_by_idx[i].get("is_correct")) for i in indices], dtype=np.int8)
    guided_correct = np.array([bool(guided_by_idx[i].get("is_correct")) for i in indices], dtype=np.int8)
    differences = guided_correct - base_correct
    both_correct = int(np.sum((base_correct == 1) & (guided_correct == 1)))
    base_only = int(np.sum((base_correct == 1) & (guided_correct == 0)))
    guided_only = int(np.sum((base_correct == 0) & (guided_correct == 1)))
    both_wrong = int(np.sum((base_correct == 0) & (guided_correct == 0)))

    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.integers(0, len(indices), size=(args.bootstrap_samples, len(indices)))
    bootstrap_differences = differences[sampled_indices].mean(axis=1)

    selected = [guided_by_idx[i] for i in indices if guided_by_idx[i].get("retrieval_used")]
    selected_indices = {row["idx"] for row in selected}
    candidate_counts = [len(guided_by_idx[i].get("retrieval_trace", {}).get("reranked_candidates", [])) for i in indices]
    top_scores = []
    all_candidate_scores = []
    schema_sources = set()
    forbidden_exposures = []
    for i in indices:
        row = guided_by_idx[i]
        trace = row.get("retrieval_trace", {})
        ranked = trace.get("reranked_candidates", [])
        if ranked:
            top_scores.append(float(ranked[0]["rerank_score"]))
            all_candidate_scores.extend(float(entry["rerank_score"]) for entry in ranked)
        schema_sources.update(trace.get("candidate_schema_sources", {}).values())
        injected = row.get("injected_principle_text", "")
        forbidden = [token for token in ("retrieval_score", "rerank_score", "Trigger:", "Scope:", "Failure", "memory_id") if token in injected]
        if forbidden:
            forbidden_exposures.append({"idx": i, "tokens": forbidden})

    cases = []
    for i in indices:
        b = base_by_idx[i]
        g = guided_by_idx[i]
        trace = g.get("retrieval_trace", {})
        ranked = trace.get("reranked_candidates", [])
        injected_content = (trace.get("injected_memory") or [{}])[0].get("content", {}) if g.get("retrieval_used") else {}
        cases.append({
            "idx": i,
            "difficulty_level": tests[i].get("difficulty_level"),
            "question": tests[i].get("question"),
            "gold": tests[i].get("answer"),
            "base_final_answer": extract_final_answer(b.get("model_answer", "")),
            "guided_final_answer": extract_final_answer(g.get("model_answer", "")),
            "base_correct": bool(b.get("is_correct")),
            "guided_correct": bool(g.get("is_correct")),
            "outcome": "guided_rescue" if (not b.get("is_correct") and g.get("is_correct")) else "guided_harm" if (b.get("is_correct") and not g.get("is_correct")) else "both_correct" if b.get("is_correct") else "both_wrong",
            "memory_injected": bool(g.get("retrieval_used")),
            "selected_memory_ids": trace.get("selected_ids", []),
            "top_rerank_score": float(ranked[0]["rerank_score"]) if ranked else None,
            "candidate_count": len(ranked),
            "lesson": injected_content.get("introspection", ""),
            "action_guidance": injected_content.get("action_guidance", ""),
            "injected_text": g.get("injected_principle_text", ""),
            "base_solver_url": b.get("solver_url"),
            "guided_solver_url": g.get("solver_url"),
        })

    by_level = {}
    for level in sorted({str(item.get("difficulty_level")) for item in tests}):
        subset = [case for case in cases if str(case["difficulty_level"]) == level]
        by_level[level] = {
            "count": len(subset),
            "base_correct": sum(case["base_correct"] for case in subset),
            "guided_correct": sum(case["guided_correct"] for case in subset),
            "injected": sum(case["memory_injected"] for case in subset),
        }

    injected_cases = [case for case in cases if case["memory_injected"]]
    non_injected_cases = [case for case in cases if not case["memory_injected"]]

    def subgroup(rows):
        return {
            "count": len(rows),
            "base_correct": sum(case["base_correct"] for case in rows),
            "guided_correct": sum(case["guided_correct"] for case in rows),
            "guided_rescues": sum(case["outcome"] == "guided_rescue" for case in rows),
            "guided_harms": sum(case["outcome"] == "guided_harm" for case in rows),
        }

    summary = {
        "experiment": {
            "n": len(indices),
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "single_solver_call": all(base_by_idx[i].get("attempt_count") == 1 and guided_by_idx[i].get("attempt_count") == 1 for i in indices),
            "base_final_answer_count": sum(extract_final_answer(base_by_idx[i].get("model_answer", "")) is not None for i in indices),
            "guided_final_answer_count": sum(extract_final_answer(guided_by_idx[i].get("model_answer", "")) is not None for i in indices),
            "stable_endpoint_mapping": all(base_by_idx[i].get("solver_url") == guided_by_idx[i].get("solver_url") for i in indices),
        },
        "paired_performance": {
            "base_correct": int(base_correct.sum()),
            "guided_correct": int(guided_correct.sum()),
            "base_accuracy": float(base_correct.mean()),
            "guided_accuracy": float(guided_correct.mean()),
            "absolute_difference": float(differences.mean()),
            "both_correct": both_correct,
            "base_only_correct": base_only,
            "guided_only_correct": guided_only,
            "both_wrong": both_wrong,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(base_only, guided_only),
            "paired_bootstrap_95_ci": [percentile(bootstrap_differences, 2.5), percentile(bootstrap_differences, 97.5)],
        },
        "retrieval_and_injection": {
            "candidate_top_k": 10,
            "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts),
            "injected_samples": len(selected),
            "injection_rate": len(selected) / len(indices),
            "max_memories_per_solver": max((row.get("retrieval_count", 0) for row in guided), default=0),
            "top_rerank_score_median": float(np.median(top_scores)),
            "top_rerank_score_min": min(top_scores),
            "top_rerank_score_max": max(top_scores),
            "all_candidate_score_median": float(np.median(all_candidate_scores)),
            "schema_sources": sorted(schema_sources),
            "forbidden_solver_field_exposures": forbidden_exposures,
            "injected_subgroup": subgroup(injected_cases),
            "non_injected_subgroup": subgroup(non_injected_cases),
            "non_injected_final_answer_disagreements": sum(
                case["base_final_answer"] != case["guided_final_answer"] for case in non_injected_cases
            ),
        },
        "by_level": by_level,
        "known_gold_issue": {
            "idx": 61,
            "question": tests[61]["question"],
            "provided_gold": tests[61]["answer"],
            "audited_value": "\\frac{3203}{9990}",
            "note": "For 0.3\\overline{206}, 10x=3.206206... and 10000x=3206.206206..., hence 9990x=3203. The provided gold was left unchanged for the paired headline metric.",
        },
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "paired_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "paired_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    perf = summary["paired_performance"]
    retrieval = summary["retrieval_and_injection"]
    injected_stats = retrieval["injected_subgroup"]
    non_injected_stats = retrieval["non_injected_subgroup"]
    report = f"""# ActiveEvo v2 paired 100-sample report

## Frozen protocol

- Fixed 100-item DeepScaler Level-2/3 subset, seed 42.
- One solver call per item, no answer-dependent retry.
- Stable item-to-endpoint mapping; base and guided use the same endpoint for every item.
- Guided recall: threshold-free BGE-M3 three-facet union, exactly 10 candidates.
- Applicability: Qwen3-Reranker-0.6B, absolute threshold 0.50, final top-0/1.
- Solver view: `Lesson learned` + `Recommended action` only; no IDs, scores, trigger, scope, failure mode, or reliability metadata.

## Paired outcome

| Metric | Base | Guided |
|---|---:|---:|
| Correct | {perf['base_correct']}/100 | {perf['guided_correct']}/100 |
| Accuracy | {perf['base_accuracy']:.1%} | {perf['guided_accuracy']:.1%} |
| Explicit final answers | {summary['experiment']['base_final_answer_count']}/100 | {summary['experiment']['guided_final_answer_count']}/100 |

- Difference (guided - base): {perf['absolute_difference']:+.1%}.
- Paired cells: both correct={both_correct}, base-only={base_only}, guided-only={guided_only}, both wrong={both_wrong}.
- Exact McNemar two-sided p={perf['mcnemar_exact_two_sided_p']:.4f}.
- Paired bootstrap 95% CI for accuracy difference: [{perf['paired_bootstrap_95_ci'][0]:+.1%}, {perf['paired_bootstrap_95_ci'][1]:+.1%}].

## Routing and injection

- Every item produced {retrieval['candidate_count_min']}–{retrieval['candidate_count_max']} reranked candidates.
- Memory was injected for {retrieval['injected_samples']}/100 items ({retrieval['injection_rate']:.1%}); at most {retrieval['max_memories_per_solver']} memory per solver.
- Median top rerank score: {retrieval['top_rerank_score_median']:.4f}; all-candidate median: {retrieval['all_candidate_score_median']:.4f}.
- Canonical schema sources: {', '.join(retrieval['schema_sources'])}.
- Forbidden routing/meta fields exposed to solver: {len(retrieval['forbidden_solver_field_exposures'])} cases.
- Injected subgroup: base {injected_stats['base_correct']}/{injected_stats['count']}, guided {injected_stats['guided_correct']}/{injected_stats['count']}, rescues={injected_stats['guided_rescues']}, harms={injected_stats['guided_harms']}.
- Non-injected subgroup: base {non_injected_stats['base_correct']}/{non_injected_stats['count']}, guided {non_injected_stats['guided_correct']}/{non_injected_stats['count']}, rescues={non_injected_stats['guided_rescues']}, harms={non_injected_stats['guided_harms']}.
- Non-injected final-answer disagreements: {retrieval['non_injected_final_answer_disagreements']}/{non_injected_stats['count']}. These receive the same direct prompt and expose residual batched-inference nondeterminism despite a fixed request seed.

## Interpretation

Overall accuracy is tied. Within the actually injected subgroup, guided has {injected_stats['guided_rescues']} rescue(s) and {injected_stats['guided_harms']} harm(s); the remaining discordant outcome occurs without injection and therefore cannot be attributed to memory. The reranker estimates applicability, not truth or historical reliability. This run supports keeping applicability and reliability as separate signals, revalidating stored lessons with the repaired matcher, and running multiple seeds before claiming that accumulated principles are beneficial.

## Dataset audit note

Item 61 (`0.3\\overline{{206}}`) has provided gold `{tests[61]['answer']}`. Algebra gives `3203/9990`; the headline paired metric retains the supplied gold so both conditions are evaluated identically.
"""
    (output / "paired_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
