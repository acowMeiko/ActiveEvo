"""Pure Stage-2 validation helpers.

This module contains no model calls.  Keeping the admission and split logic
pure makes the causal-memory contract independently testable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from modules.principle_schema import PRINCIPLE_FIELDS


PAIR_OUTCOMES = ("gain", "loss", "both_correct", "both_wrong", "infrastructure_failed")


def classify_pair(baseline_correct: bool, injected_correct: bool) -> str:
    if injected_correct and not baseline_correct:
        return "gain"
    if baseline_correct and not injected_correct:
        return "loss"
    if baseline_correct and injected_correct:
        return "both_correct"
    return "both_wrong"


def summarize_pairs(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {name: 0 for name in PAIR_OUTCOMES}
    for result in results:
        outcome = str(result.get("outcome", ""))
        if outcome in counts:
            counts[outcome] += 1
    total = sum(counts[name] for name in PAIR_OUTCOMES if name != "infrastructure_failed")
    discordant = counts["gain"] + counts["loss"]
    net_gain = counts["gain"] - counts["loss"]
    return {
        **{f"{name}_count": value for name, value in counts.items()},
        "pair_count": total,
        "attempted_pair_count": total + counts["infrastructure_failed"],
        "discordant_count": discordant,
        "net_gain": net_gain,
        "historical_gain_rate": counts["gain"] / total if total else 0.0,
        "harm_rate": counts["loss"] / total if total else 0.0,
        "baseline_accuracy": (counts["loss"] + counts["both_correct"]) / total if total else 0.0,
        "injected_accuracy": (counts["gain"] + counts["both_correct"]) / total if total else 0.0,
    }


def stable_dev_holdout_split(
    neighbors: Sequence[Dict[str, Any]],
    *,
    candidate_id: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically split 4-6 accepted neighbors without data leakage."""
    ranked = sorted(
        neighbors,
        key=lambda item: hashlib.sha256(
            f"{candidate_id}{item.get('question_id', '')}{seed}".encode("utf-8")
        ).hexdigest(),
    )
    dev_size = len(ranked) // 2
    return ranked[:dev_size], ranked[dev_size:]


def admission_decision(
    *,
    schema_valid: bool,
    source_fidelity_pass: bool,
    answer_leakage_detected: bool,
    accepted_neighbors: int,
    holdout_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    reasons: List[str] = []
    if not schema_valid:
        reasons.append("schema_invalid")
    if not source_fidelity_pass:
        reasons.append("source_fidelity_failed")
    if answer_leakage_detected:
        reasons.append("answer_leakage_detected")
    if accepted_neighbors < 4:
        reasons.append("insufficient_validation_evidence")
    if int(holdout_summary.get("infrastructure_failed_count", 0)) > 0:
        reasons.append("infrastructure_incomplete")
    if int(holdout_summary.get("gain_count", 0)) < 1:
        reasons.append("no_holdout_gain")
    if int(holdout_summary.get("loss_count", 0)) > 0:
        reasons.append("holdout_harm_detected")
    if int(holdout_summary.get("net_gain", 0)) <= 0:
        reasons.append("non_positive_holdout_net_gain")

    hard_reject = any(
        reason in reasons
        for reason in ("schema_invalid", "source_fidelity_failed", "answer_leakage_detected", "holdout_harm_detected")
    )
    status = "active" if not reasons else ("rejected" if hard_reject else "provisional")
    return {"status": status, "qualified_for_memory": status == "active", "reasons": reasons}


def deterministic_leakage_check(
    content: Mapping[str, Any],
    *,
    source_question: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """Detect obvious source-instance constants copied into principle fields.

    Single digits are ignored because mathematical roles such as first/second
    steps and universal constants otherwise cause excessive false positives.
    The LLM quality audit performs the semantic leakage check.
    """
    principle_text = " ".join(str(content.get(field, "")) for field in PRINCIPLE_FIELDS)
    source_numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?", source_question or ""))
    significant = {value for value in source_numbers if len(re.sub(r"\D", "", value)) >= 2}
    copied = sorted(value for value in significant if value and value in principle_text)
    expected = re.sub(r"\s+", "", expected_answer or "")
    answer_copied = bool(len(expected) >= 2 and expected in re.sub(r"\s+", "", principle_text))
    return {
        "detected": bool(copied or answer_copied),
        "copied_source_constants": copied,
        "expected_answer_copied": answer_copied,
    }


def render_solver_view(view: Mapping[str, Any], index: int = 1) -> str:
    """Render only executable experience fields for Frozen Solver V1."""
    return "\n".join(
        [
            f"[Past Experience {index}]",
            f"- Lesson learned: {str(view.get('lesson', '')).strip()}",
            f"- Recommended action: {str(view.get('action_guidance', '')).strip()}",
            f"- Verification: {str(view.get('verification_step', '')).strip()}",
        ]
    )


def atomic_save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


class Stage2Checkpoint:
    """Dataset-level resumable checkpoint stored outside long-term memory."""

    def __init__(self, root: Path, run_id: str):
        self.root = Path(root) / "checkpoints" / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest = self.root / "latest.json"

    def load(self) -> Dict[str, Any]:
        if not self.latest.exists():
            return {}
        with self.latest.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}

    def save(self, state: Mapping[str, Any]) -> Path:
        previous = self.load()
        payload = {
            **dict(state),
            "checkpoint_id": f"{state.get('run_id', 'stage2')}:{state.get('current_step', 'unknown')}",
            "previous_checkpoint_id": previous.get("checkpoint_id"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_save_json(self.latest, payload)
        return self.latest
