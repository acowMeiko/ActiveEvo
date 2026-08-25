#!/usr/bin/env python3
"""Merge completed Stage-2 protocol rounds into one self-contained JSON file."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(probability * len(values))))
    return values[index]


def merge_rounds(round_roots: list[Path]) -> dict[str, Any]:
    questions: dict[str, dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    all_effects: list[dict[str, Any]] = []
    audit_decisions: Counter[str] = Counter()
    round_metadata: list[dict[str, Any]] = []

    for round_index, raw_root in enumerate(round_roots, 1):
        root = raw_root.resolve()
        manifest_path = root / "manifests" / "source40.json"
        bundle_path = root / "source_bundles" / "source40_bundles.json"
        bank_path = root / "raw_candidates" / "candidate_bank.json"
        audit_path = root / "production_audit" / "all_records.json"
        rollout_path = root / "llama_validation" / "rollouts.json"
        effects_path = root / "llama_validation" / "candidate_effects.json"
        summary_path = root / "summary.json"
        required = (
            manifest_path, bundle_path, bank_path, audit_path,
            rollout_path, effects_path, summary_path,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Incomplete protocol round {root}: {missing}")

        manifest = load_json(manifest_path)
        bundles = load_json(bundle_path)
        bank = load_json(bank_path)
        audits = load_json(audit_path)
        rollouts = load_json(rollout_path)
        effects = load_json(effects_path)
        summary = load_json(summary_path)
        round_id = f"round_{round_index}"

        bundle_by_qid = {str(item["question_id"]): item for item in bundles}
        candidate_by_id = {
            str(item["candidate_id"]): item for item in bank.get("candidates", [])
        }
        audit_by_id = {
            str(item["candidate_id"]): item for item in audits
        }
        effect_by_id = {
            str(item["candidate_id"]): item for item in effects
        }
        baseline_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        guided_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rollout in rollouts:
            if rollout.get("status") != "complete":
                raise RuntimeError(
                    f"Non-complete rollout in {root}: {rollout.get('record_id')}"
                )
            if rollout.get("arm") == "baseline":
                baseline_by_qid[str(rollout["question_id"])].append(rollout)
            elif rollout.get("arm") == "guided":
                guided_by_candidate[str(rollout["candidate_id"])].append(rollout)

        manifest_qids = [str(value) for value in manifest["source_question_ids"]]
        if len(manifest_qids) != 20 or len(set(manifest_qids)) != 20:
            raise RuntimeError(f"Expected 20 unique questions in {root}")
        overlap = set(manifest_qids) & set(questions)
        if overlap:
            raise RuntimeError(f"Question overlap across rounds: {sorted(overlap)}")

        for qid in manifest_qids:
            baseline = sorted(baseline_by_qid[qid], key=lambda item: item["seed"])
            if len(baseline) != 10:
                raise RuntimeError(f"Question {qid} has {len(baseline)} baseline rollouts")
            questions[qid] = {
                "question_id": qid,
                "round_id": round_id,
                "bundle": bundle_by_qid[qid],
                "baseline_rollouts": baseline,
                "candidates": [],
            }

        for candidate_id, candidate in candidate_by_id.items():
            if candidate_id in candidate_ids:
                raise RuntimeError(f"Duplicate candidate ID: {candidate_id}")
            candidate_ids.add(candidate_id)
            audit = audit_by_id[candidate_id]
            effect = effect_by_id[candidate_id]
            guided = sorted(
                guided_by_candidate.get(candidate_id, []),
                key=lambda item: item["seed"],
            )
            decision = str(audit.get("final_decision", "unknown"))
            audit_decisions[decision] += 1
            expected_guided = 10 if decision == "pass" else 0
            if len(guided) != expected_guided:
                raise RuntimeError(
                    f"Candidate {candidate_id} has {len(guided)} guided rollouts; "
                    f"expected {expected_guided} for decision={decision}"
                )
            qid = str(candidate["question_id"])
            questions[qid]["candidates"].append({
                "candidate_id": candidate_id,
                "raw_candidate": candidate,
                "audit_record": audit,
                "effect": effect,
                "guided_rollouts": guided,
            })
            if effect.get("evaluated"):
                all_effects.append(effect)

        round_metadata.append({
            "round_id": round_id,
            "root": str(root),
            "manifest_hash": manifest["manifest_hash"],
            "candidate_bank_hash": summary["generation"]["candidate_bank_hash"],
            "question_count": len(manifest_qids),
            "candidate_count": len(candidate_by_id),
            "summary": summary,
            "artifact_hashes": {
                "manifest": file_hash(manifest_path),
                "candidate_bank": file_hash(bank_path),
                "audits": file_hash(audit_path),
                "rollouts": file_hash(rollout_path),
                "effects": file_hash(effects_path),
            },
        })

    per_question_effects: list[dict[str, Any]] = []
    effect_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for effect in all_effects:
        effect_by_question[str(effect["question_id"])].append(effect)
    for qid, question in questions.items():
        effects = effect_by_question.get(qid, [])
        deltas = [int(item["delta_correct"]) for item in effects]
        per_question_effects.append({
            "question_id": qid,
            "round_id": question["round_id"],
            "level": question["bundle"]["level"],
            "evaluated_candidate_count": len(effects),
            "mean_delta_correct": sum(deltas) / len(deltas) if deltas else None,
            "best_delta_correct": max(deltas) if deltas else None,
            "worst_delta_correct": min(deltas) if deltas else None,
        })

    question_scores = [
        item["mean_delta_correct"] / 10.0
        for item in per_question_effects
        if item["mean_delta_correct"] is not None
    ]
    rng = random.Random(42)
    bootstrap = []
    for _ in range(10_000):
        sample = [question_scores[rng.randrange(len(question_scores))] for _ in question_scores]
        bootstrap.append(sum(sample) / len(sample))
    bootstrap.sort()

    baseline_correct = sum(int(item["baseline_correct_at_10"]) for item in all_effects)
    guided_correct = sum(int(item["guided_correct_at_10"]) for item in all_effects)
    gain = sum(int(item["paired"]["gain"]) for item in all_effects)
    loss = sum(int(item["paired"]["loss"]) for item in all_effects)
    summary = {
        "question_count": len(questions),
        "candidate_count": len(candidate_ids),
        "audit_decisions": dict(audit_decisions),
        "evaluated_candidate_count": len(all_effects),
        "paired_rollout_count": len(all_effects) * 10,
        "baseline_correct": baseline_correct,
        "guided_correct": guided_correct,
        "baseline_accuracy": baseline_correct / (len(all_effects) * 10),
        "guided_accuracy": guided_correct / (len(all_effects) * 10),
        "accuracy_delta": (guided_correct - baseline_correct) / (len(all_effects) * 10),
        "gain": gain,
        "loss": loss,
        "net_gain": gain - loss,
        "helpful_candidate_count": sum(
            int(item["delta_correct"]) >= 2 for item in all_effects
        ),
        "harmful_candidate_count": sum(
            int(item["delta_correct"]) <= -2 for item in all_effects
        ),
        "question_level": {
            "evaluated_question_count": len(question_scores),
            "no_accepted_candidate_count": len(questions) - len(question_scores),
            "positive_mean_effect_count": sum(value > 0 for value in question_scores),
            "zero_mean_effect_count": sum(value == 0 for value in question_scores),
            "negative_mean_effect_count": sum(value < 0 for value in question_scores),
            "mean_accuracy_delta": sum(question_scores) / len(question_scores),
            "paired_bootstrap_95_ci": [
                percentile(bootstrap, 0.025), percentile(bootstrap, 0.975),
            ],
            "bootstrap_seed": 42,
            "bootstrap_samples": 10_000,
        },
        "delta_distribution": dict(sorted(Counter(
            int(item["delta_correct"]) for item in all_effects
        ).items())),
    }
    return {
        "schema_version": "stage2_protocol_combined40.1",
        "description": (
            "Self-contained merge of two frozen 20-question protocol rounds. "
            "Candidate-level pairs are not independent because a question may "
            "have multiple candidate Principles."
        ),
        "rounds": round_metadata,
        "summary": summary,
        "per_question_effects": per_question_effects,
        "questions": [questions[qid] for qid in sorted(questions)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = merge_rounds(args.round_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
