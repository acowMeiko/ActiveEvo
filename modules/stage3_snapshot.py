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


def prepare_transfer_snapshot(
    *,
    pending_store: str | Path,
    generation_dir: str | Path,
    snapshot_dir: str | Path,
) -> Dict[str, Any]:
    """Create a hash-bound, evaluation-only active snapshot for Stage 3."""
    pending_path = Path(pending_store).resolve()
    generation_path = Path(generation_dir).resolve()
    destination = Path(snapshot_dir).resolve()
    if not pending_path.is_file():
        raise FileNotFoundError(f"Missing pending Memory store: {pending_path}")
    if not generation_path.is_dir():
        raise FileNotFoundError(f"Missing Stage-2 generation records: {generation_path}")

    records = _load_jsonl(pending_path)
    routing_by_question = _load_routing_by_question(generation_path)
    prepared: List[Dict[str, Any]] = []
    missing_routing: List[str] = []
    seen_ids: set[str] = set()
    for source in records:
        memory_id = str(source.get("memory_id", "")).strip()
        question_id = str(source.get("question_id", "")).strip()
        if not memory_id or memory_id in seen_ids:
            raise ValueError(f"Missing or duplicate memory_id: {memory_id!r}")
        if source.get("status") != "pending_transfer_validation":
            raise ValueError(
                f"Memory {memory_id} has unexpected source status {source.get('status')!r}"
            )
        semantics = routing_by_question.get(question_id)
        if not semantics:
            missing_routing.append(memory_id)
            continue
        item = dict(source)
        item.update(semantics)
        item["source_status"] = source["status"]
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
        "schema_version": "stage3_transfer_snapshot.1",
        "evaluation_only": True,
        "source_status": "pending_transfer_validation",
        "snapshot_status": "active",
        "source_store": str(pending_path),
        "source_store_sha256": _sha256_file(pending_path),
        "generation_records_dir": str(generation_path),
        "memory_count": len(prepared),
        "routing_fields": list(ROUTING_FIELDS),
        "principles_sha256": _sha256_file(principles_path),
    }
    metadata_path = destination / "metadata" / "evaluation_snapshot.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_json(metadata_path, manifest)
    return manifest

