"""Deterministic Stage 1 evidence bundling and within-level rankings."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _split_steps(path: str) -> List[str]:
    text = str(path or "").strip()
    if not text:
        return []
    parts = re.split(r"\n(?:Step\s+)?\d+[.)]\s*|\n{2,}", text, flags=re.IGNORECASE)
    steps = [part.strip() for part in parts if len(part.strip()) >= 10]
    return steps or [text]


def divergence_score(correct_path: str, wrong_path: str) -> Tuple[float, int, str]:
    correct = _split_steps(correct_path)
    wrong = _split_steps(wrong_path)
    if not correct or not wrong:
        return 0.0, 0, ""
    matched = 0
    for left, right in zip(correct, wrong):
        if SequenceMatcher(None, left.lower(), right.lower()).ratio() >= 0.80:
            matched += 1
        else:
            break
    return matched / len(correct), matched, correct[matched] if matched < len(correct) else ""


def mine_pairs(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = [item for item in sample.get("trial_records", []) if isinstance(item, dict)]
    correct = [item for item in records if item.get("is_correct") is True and item.get("response")]
    wrong = [item for item in records if item.get("is_correct") is False and item.get("response")]
    if not records:
        correct = [{"rollout_id": f"correct_{i}", "response": value} for i, value in enumerate(sample.get("correct_trajectories", []))]
        wrong = [{"rollout_id": f"wrong_{i}", "response": value} for i, value in enumerate(sample.get("wrong_trajectories", []))]
    pairs: List[Dict[str, Any]] = []
    for rejected in wrong:
        candidates = []
        for chosen in correct:
            score, index, text = divergence_score(chosen["response"], rejected["response"])
            candidates.append((score, chosen, index, text))
        if not candidates:
            continue
        score, chosen, index, text = max(candidates, key=lambda item: item[0])
        pairs.append({
            "wrong_rollout_id": rejected.get("rollout_id"),
            "closest_correct_rollout_id": chosen.get("rollout_id"),
            "divergence_score": score,
            "divergence_step": index,
            "divergence_step_text": text,
            # compatibility labels for older contract tests
            "anchor": chosen["response"],
            "wrong_path": rejected["response"],
            "similarity": score,
            "divergence_step_idx": index,
        })
    return sorted(pairs, key=lambda item: str(item.get("wrong_rollout_id")))


def _stable_random_key(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{question_id}:{seed}".encode()).hexdigest()


def _rank_bundles(bundles: Sequence[Dict[str, Any]], experiment_seed: int) -> Dict[str, List[str]]:
    def entropy(bundle: Dict[str, Any]) -> float:
        value = bundle["capability_signal"]["entropy"].get("h_token")
        return float(value) if isinstance(value, (int, float)) else float("nan")

    random_order = sorted(bundles, key=lambda item: _stable_random_key(item["question_id"], experiment_seed))
    high = sorted(bundles, key=lambda item: (math.isnan(entropy(item)), -entropy(item) if not math.isnan(entropy(item)) else 0, item["question_id"]))
    low = sorted(bundles, key=lambda item: (math.isnan(entropy(item)), entropy(item) if not math.isnan(entropy(item)) else 0, item["question_id"]))
    return {
        "random": [item["question_id"] for item in random_order],
        "high_token_entropy": [item["question_id"] for item in high],
        "low_token_entropy": [item["question_id"] for item in low],
    }


def build_learning_bundle(sample: Dict[str, Any], dataset_sha256: str = "") -> Dict[str, Any]:
    question_id = str(sample.get("question_id") or sample.get("idx"))
    records = [dict(item) for item in sample.get("trial_records", []) if isinstance(item, dict)]
    bundle = {
        "schema_version": "stage1.1",
        "bundle_id": "bundle_" + hashlib.sha256(question_id.encode()).hexdigest()[:16],
        "question_id": question_id,
        "question_text": sample.get("question", ""),
        "expected_answer": sample.get("expected_answer", ""),
        "level": f"level{sample.get('assigned_level')}",
        "original_difficulty": sample.get("orig_level", ""),
        "dataset_sha256": dataset_sha256,
        "sampling_config": dict(sample.get("sampling_config") or {}),
        "capability_signal": dict(sample.get("capability_signal") or {}),
        "rollouts": records,
        "correct_rollout_ids": [item["rollout_id"] for item in records if item.get("is_correct") is True],
        "wrong_rollout_ids": [item["rollout_id"] for item in records if item.get("is_correct") is False],
        "trajectory_pairs": mine_pairs(sample),
        "within_level_rankings": {},
        "selection": {"selected": False, "policy": "", "reason": ""},
        "evidence_complete": (
            len(records) == int(sample.get("planned_trials") or 10)
            and all(item.get("transport_status") == "success" for item in records)
        ),
    }
    payload = json.dumps(bundle, ensure_ascii=False, sort_keys=True, default=str)
    bundle["source_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    return bundle


def _proportional_quotas(level_counts: Dict[str, int], budget: int) -> Dict[str, int]:
    total = sum(level_counts.values())
    if total <= 0:
        return {level: 0 for level in level_counts}
    target = min(int(budget), total)
    raw = {level: target * count / total for level, count in level_counts.items()}
    quotas = {level: min(level_counts[level], int(math.floor(value))) for level, value in raw.items()}
    remaining = target - sum(quotas.values())
    for level in sorted(level_counts, key=lambda key: (-(raw[key] - math.floor(raw[key])), key)):
        if remaining <= 0:
            break
        if quotas[level] < level_counts[level]:
            quotas[level] += 1
            remaining -= 1
    return quotas


class Stage1Miner:
    """Build one bundle per question; no LLM calls occur in this step."""

    def __init__(self, *args: Any, experiment_seed: int = 42, dataset_sha256: str = "", **kwargs: Any):
        del args, kwargs
        self.experiment_seed = int(experiment_seed)
        self.dataset_sha256 = dataset_sha256

    async def run(self, level_data: Dict[str, List[Dict[str, Any]]], concurrency: int = 0) -> Dict[str, Any]:
        del concurrency
        output = {f"level{level}": [] for level in range(1, 5)}
        rankings: Dict[str, Dict[str, List[str]]] = {}
        for level in range(1, 5):
            bundles = [build_learning_bundle(item, self.dataset_sha256) for item in level_data.get(str(level), [])]
            level_rankings = _rank_bundles(bundles, self.experiment_seed)
            positions = {
                policy: {question_id: index + 1 for index, question_id in enumerate(ids)}
                for policy, ids in level_rankings.items()
            }
            for bundle in bundles:
                bundle["within_level_rankings"] = {
                    policy: positions[policy][bundle["question_id"]] for policy in positions
                }
            output[f"level{level}"] = sorted(bundles, key=lambda item: item["question_id"])
            rankings[f"level{level}"] = level_rankings
        return {
            "packets": output,
            "rankings": rankings,
            "stats": {
                "total_questions": sum(len(value) for value in output.values()),
                "total_pairs_found": sum(len(item["trajectory_pairs"]) for value in output.values() for item in value),
                "llm_diff_calls": 0,
            },
        }

    @staticmethod
    def save(packets: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for level, items in packets.items():
            path = output_dir / f"{level}_learning_packets.json"
            path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
            paths[level] = path
        return paths


def save_rankings_and_manifests(
    rankings: Dict[str, Dict[str, List[str]]],
    output_root: Path,
    budgets: Sequence[int] = (100, 200, 400, 600),
) -> None:
    ranking_root = output_root / "rankings"
    manifest_root = output_root / "selection_manifests"
    ranking_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    counts = {level: len(next(iter(policies.values()), [])) for level, policies in rankings.items()}
    for policy in ("random", "high_token_entropy", "low_token_entropy"):
        value = {level: policies.get(policy, []) for level, policies in rankings.items()}
        (ranking_root / f"{policy}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    for budget in budgets:
        quotas = _proportional_quotas(counts, budget)
        for policy in ("random", "high_token_entropy", "low_token_entropy"):
            by_level = {level: rankings[level][policy][:quotas[level]] for level in rankings}
            payload = {
                "budget": budget,
                "actual_count": sum(len(ids) for ids in by_level.values()),
                "policy": policy,
                "level_quotas": quotas,
                "question_ids_by_level": by_level,
                "question_ids": [question_id for level in sorted(by_level) for question_id in by_level[level]],
            }
            (manifest_root / f"budget_{budget}_{policy}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )


def write_analysis_csv(packets: Dict[str, List[Dict[str, Any]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id", "level", "success_rate", "h_token", "h_answer", "h_gap",
        "correct_entropy_mean", "wrong_entropy_mean", "entropy_coverage",
        "random_rank", "high_entropy_rank", "low_entropy_rank",
        "quality_pass", "memory_operation", "retrieved", "stage3_outcome",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for level in sorted(packets):
            for bundle in packets[level]:
                signal = bundle["capability_signal"]
                entropy = signal["entropy"]
                ranks = bundle["within_level_rankings"]
                writer.writerow({
                    "question_id": bundle["question_id"], "level": level,
                    "success_rate": signal["success_rate"], "h_token": entropy.get("h_token"),
                    "h_answer": entropy.get("h_answer"), "h_gap": entropy.get("h_gap"),
                    "correct_entropy_mean": entropy.get("correct_mean"),
                    "wrong_entropy_mean": entropy.get("wrong_mean"),
                    "entropy_coverage": entropy.get("coverage"),
                    "random_rank": ranks.get("random"),
                    "high_entropy_rank": ranks.get("high_token_entropy"),
                    "low_entropy_rank": ranks.get("low_token_entropy"),
                    "quality_pass": "", "memory_operation": "", "retrieved": "", "stage3_outcome": "",
                })
