"""Stable Stage-1 -> Stage-2 data contracts and normalization."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence


def _stable_id(prefix: str, *parts: str) -> str:
    payload = ":".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _question_id(packet: Dict[str, Any], index: int) -> str:
    return str(packet.get("question_id") or packet.get("idx") or index)


def _trajectory_records(packet: Dict[str, Any], question_id: str) -> List[Dict[str, Any]]:
    records = packet.get("rollouts") or packet.get("trial_records")
    normalized: List[Dict[str, Any]] = []
    if isinstance(records, list) and records:
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            response = str(record.get("response") or record.get("answer") or "").strip()
            if not response:
                continue
            correct = record.get("correct") if isinstance(record.get("correct"), bool) else bool(record.get("is_correct"))
            normalized.append({
                "rollout_id": str(record.get("rollout_id") or _stable_id("rollout", question_id, str(index), response)),
                "response": response,
                "correct": correct,
                "seed": record.get("seed"),
                "model": str(record.get("model", "")),
                "prompt_version": str(record.get("prompt_version", "")),
                "prompt_hash": str(record.get("prompt_hash", "")),
                "final_answer": record.get("final_answer"),
                "endpoint": str(record.get("endpoint", "")),
                "transport_status": str(record.get("transport_status", "")),
                "entropy": dict(record.get("entropy") or {}),
            })
        return normalized

    # Compatibility with old Stage-1 packets. This path is explicitly marked
    # incomplete so an audit never mistakes a single pair for full evidence.
    pairs = (
        (packet.get("chosen_trajectory") or packet.get("chosen_path"), True),
        (packet.get("hard_negative_trajectory") or packet.get("rejected_path"), False),
    )
    for index, (response, correct) in enumerate(pairs):
        text = str(response or "").strip()
        if text:
            normalized.append({
                "rollout_id": _stable_id("legacy_rollout", question_id, str(index), text),
                "response": text,
                "correct": correct,
                "seed": None,
                "model": "",
                "prompt_version": "legacy_packet",
            })
    return normalized


def build_learning_bundles(packets: Sequence[Dict[str, Any]], level: str) -> List[Dict[str, Any]]:
    """Group every packet for one question into one evidence-complete bundle."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for index, packet in enumerate(packets):
        if isinstance(packet, dict):
            grouped.setdefault(_question_id(packet, index), []).append(packet)

    bundles: List[Dict[str, Any]] = []
    for question_id, rows in sorted(grouped.items()):
        first = rows[0]
        rollout_by_id: Dict[str, Dict[str, Any]] = {}
        differences: List[Dict[str, Any]] = []
        evidence_complete = any(bool(row.get("evidence_complete")) for row in rows)
        trajectory_pairs: List[Dict[str, Any]] = []
        for row in rows:
            for rollout in _trajectory_records(row, question_id):
                rollout_by_id[rollout["rollout_id"]] = rollout
            diff = row.get("contrast") or row.get("diff") or {}
            values = diff.get("differences", []) if isinstance(diff, dict) else []
            if isinstance(values, list):
                differences.extend(item for item in values if isinstance(item, dict))
            values = row.get("trajectory_pairs", [])
            if isinstance(values, list):
                trajectory_pairs.extend(item for item in values if isinstance(item, dict))

        rollouts = list(rollout_by_id.values())
        correct = [item for item in rollouts if item["correct"]]
        wrong = [item for item in rollouts if not item["correct"]]
        if not correct or not wrong:
            continue
        source_dsd = first.get("source_dsd") or first.get("problem_semantics") or {}
        bundle = {
            "schema_version": str(first.get("schema_version") or "legacy"),
            "bundle_id": str(first.get("bundle_id") or _stable_id("bundle", level, question_id)),
            "question_id": question_id,
            "question_text": str(first.get("question_text") or first.get("question") or "").strip(),
            "expected_answer": str(first.get("expected_answer") or first.get("answer") or "").strip(),
            "level": level,
            "capability_signal": first.get("capability_signal") or {"success_rate": first.get("success_rate")},
            "sampling_config": dict(first.get("sampling_config") or {}),
            "dataset_sha256": str(first.get("dataset_sha256", "")),
            "original_difficulty": first.get("original_difficulty", first.get("orig_level", "")),
            "within_level_rankings": dict(first.get("within_level_rankings") or {}),
            "selection": dict(first.get("selection") or {}),
            "source_dsd": source_dsd if isinstance(source_dsd, dict) else {},
            "rollouts": rollouts,
            "correct_rollout_ids": [item["rollout_id"] for item in correct],
            "wrong_rollout_ids": [item["rollout_id"] for item in wrong],
            "stage1_differences": differences,
            "trajectory_pairs": trajectory_pairs,
            "evidence_complete": evidence_complete,
            "evidence_warning": "" if evidence_complete else "partial_trajectory_evidence",
        }
        bundle["source_hash"] = str(first.get("source_hash") or hashlib.sha256(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest())
        bundles.append(bundle)
    return bundles


def validate_learning_bundle(bundle: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in ("bundle_id", "question_id", "question_text", "expected_answer", "level"):
        if not str(bundle.get(field, "")).strip():
            errors.append(f"missing_{field}")
    rollouts = bundle.get("rollouts")
    if not isinstance(rollouts, list):
        errors.append("rollouts_must_be_list")
        return errors
    if not any(item.get("correct") is True for item in rollouts if isinstance(item, dict)):
        errors.append("missing_correct_rollout")
    if not any(item.get("correct") is False for item in rollouts if isinstance(item, dict)):
        errors.append("missing_wrong_rollout")
    return errors
