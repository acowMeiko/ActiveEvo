"""Build an isolated Stage-3 retrieval snapshot from transfer-pending memories.

The Stage-2 production store is append-only and keeps source-helpful memories as
``pending_transfer_validation``.  Stage 3 needs to retrieve those records in
order to perform that transfer validation, but the normal Memory reader only
indexes ``active`` records.  This module creates an evaluation-only copy whose
status override is explicit and whose source store is never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.io_utils import atomic_save_json


ROUTING_FIELDS = ("core_skill", "context_constraints", "task_objective")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def _load_routing_by_question(generation_dir: Path) -> Dict[str, Dict[str, str]]:
    routing: Dict[str, Dict[str, str]] = {}
    for path in sorted(generation_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        question_id = str(value.get("question_id", path.stem))
        semantics = (value.get("synthesis") or {}).get("routing_semantics") or {}
        normalized = {
            field: str(semantics.get(field, "")).strip()
            for field in ROUTING_FIELDS
        }
        if all(normalized.values()):
            routing[question_id] = normalized
    return routing


def select_safe_inconclusive(batch_root: str | Path) -> List[Dict[str, Any]]:
    """Select source-inconclusive candidates with no observed paired harm.

    This is deliberately stricter than ``effect_label == inconclusive``:
    the candidate must have zero losses against the generic self-check, must
    not regress against the original baseline, and must pass the deterministic
    Stage-2 format/reference contract.  Content audit is intentionally not
    implied by this experimental selection.
    """
    root = Path(batch_root).resolve()
    bank_path = root / "raw_candidates" / "candidate_bank.json"
    effects_path = root / "local_validation" / "candidate_effects_raw.json"
    format_dir = root / "format_checks"
    if not bank_path.is_file():
        raise FileNotFoundError(f"Missing candidate bank: {bank_path}")
    if not effects_path.is_file():
        raise FileNotFoundError(f"Missing source effects: {effects_path}")
    if not format_dir.is_dir():
        raise FileNotFoundError(f"Missing format checks: {format_dir}")

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    candidates = {
        str(row.get("candidate_id", "")): row
        for row in bank.get("candidates", [])
        if isinstance(row, dict) and row.get("candidate_id")
    }
    effects = json.loads(effects_path.read_text(encoding="utf-8"))
    selected: List[Dict[str, Any]] = []
    for effect in effects:
        if not isinstance(effect, dict) or effect.get("effect_label") != "inconclusive":
            continue
        candidate_id = str(effect.get("candidate_id", ""))
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Effect references unknown candidate: {candidate_id}")
        paired = effect.get("paired_vs_generic") or {}
        if int(paired.get("loss", 0)) != 0:
            continue
        if int(effect.get("delta_vs_original", 0)) < 0:
            continue
        format_path = format_dir / f"{candidate_id}.json"
        if not format_path.is_file():
            raise FileNotFoundError(f"Missing format check: {format_path}")
        format_check = json.loads(format_path.read_text(encoding="utf-8"))
        if bool(format_check.get("hard_invalid")):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            raise ValueError(f"Candidate {candidate_id} has no canonical content")
        content_hash = _json_hash(content)
        selected.append({
            "memory_id": candidate_id,
            "version_id": f"{candidate_id}_raw_{content_hash[:8]}",
            "parent_version_id": "",
            "operation": "add",
            "status": "source_inconclusive_no_loss",
            "question_id": str(candidate.get("question_id", "")),
            "level": candidate.get("level", ""),
            "content": content,
            "content_hash": content_hash,
            "source_effect": effect,
            "audit_result": {},
            "content_audit_status": "not_run",
            "intervention_protocol": "forced_c_v1",
            "schema_version": "experimental_safe_inconclusive_memory.1",
        })
    selected.sort(key=lambda row: str(row["memory_id"]))
    return selected


def prepare_transfer_snapshot(
    *,
    pending_store: str | Path,
    generation_dir: str | Path,
    snapshot_dir: str | Path,
    exploratory_batch_root: str | Path | None = None,
    expected_exploratory_count: int | None = None,
) -> Dict[str, Any]:
    """Create a hash-bound, evaluation-only active snapshot for Stage 3."""
    pending_path = Path(pending_store).resolve()
    generation_path = Path(generation_dir).resolve()
    destination = Path(snapshot_dir).resolve()
    if not pending_path.is_file():
        raise FileNotFoundError(f"Missing pending Memory store: {pending_path}")
    if not generation_path.is_dir():
        raise FileNotFoundError(f"Missing Stage-2 generation records: {generation_path}")

    validated_records = _load_jsonl(pending_path)
    exploratory_records = (
        select_safe_inconclusive(exploratory_batch_root)
        if exploratory_batch_root else []
    )
    if expected_exploratory_count is not None and len(exploratory_records) != int(expected_exploratory_count):
        raise ValueError(
            "Safe inconclusive candidate count mismatch: "
            f"expected {expected_exploratory_count}, found {len(exploratory_records)}"
        )
    records = validated_records + exploratory_records
    routing_by_question = _load_routing_by_question(generation_path)
    prepared: List[Dict[str, Any]] = []
    missing_routing: List[str] = []
    seen_ids: set[str] = set()
    for source in records:
        memory_id = str(source.get("memory_id", "")).strip()
        question_id = str(source.get("question_id", "")).strip()
        if not memory_id or memory_id in seen_ids:
            raise ValueError(f"Missing or duplicate memory_id: {memory_id!r}")
        source_status = str(source.get("status", ""))
        if source_status not in {"pending_transfer_validation", "source_inconclusive_no_loss"}:
            raise ValueError(
                f"Memory {memory_id} has unexpected source status {source_status!r}"
            )
        semantics = routing_by_question.get(question_id)
        if not semantics:
            missing_routing.append(memory_id)
            continue
        item = dict(source)
        item.update(semantics)
        item["source_status"] = source_status
        item["evidence_tier"] = (
            "source_helpful_content_audited"
            if source_status == "pending_transfer_validation"
            else "source_inconclusive_zero_observed_loss_unaudited"
        )
        item["status"] = "active"
        item["evaluation_only_status_override"] = True
        prepared.append(item)
        seen_ids.add(memory_id)

    if missing_routing:
        preview = ", ".join(missing_routing[:10])
        raise ValueError(
            f"Missing three-facet routing semantics for {len(missing_routing)} memories: {preview}"
        )
    if len(prepared) != len(records):
        raise ValueError(f"Prepared {len(prepared)} of {len(records)} Memory records")

    principles_path = destination / "principles" / "principles.json"
    principles_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_json(principles_path, prepared)
    manifest = {
        "schema_version": (
            "stage3_transfer_snapshot.2" if exploratory_records
            else "stage3_transfer_snapshot.1"
        ),
        "evaluation_only": True,
        "source_status": (
            "mixed" if exploratory_records else "pending_transfer_validation"
        ),
        "source_statuses": [
            "pending_transfer_validation",
            *(["source_inconclusive_no_loss"] if exploratory_records else []),
        ],
        "snapshot_status": "active",
        "source_store": str(pending_path),
        "source_store_sha256": _sha256_file(pending_path),
        "generation_records_dir": str(generation_path),
        "memory_count": len(prepared),
        "validated_memory_count": len(validated_records),
        "exploratory_memory_count": len(exploratory_records),
        "exploratory_batch_root": (
            str(Path(exploratory_batch_root).resolve()) if exploratory_batch_root else ""
        ),
        "exploratory_selection_rule": (
            "effect_label=inconclusive AND paired_vs_generic.loss=0 "
            "AND delta_vs_original>=0 AND format_valid"
            if exploratory_records else ""
        ),
        "exploratory_ids_hash": _json_hash(
            [row["memory_id"] for row in exploratory_records]
        ),
        "routing_fields": list(ROUTING_FIELDS),
        "principles_sha256": _sha256_file(principles_path),
    }
    metadata_path = destination / "metadata" / "evaluation_snapshot.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_json(metadata_path, manifest)
    return manifest
