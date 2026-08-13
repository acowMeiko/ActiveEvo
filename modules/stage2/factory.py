"""Stage 2 orchestration: Stage-1 evidence -> versioned Memory Store."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from modules.principle_schema import SCHEMA_VERSION, build_memory_views
from modules.stage2.contracts import build_learning_bundles, validate_learning_bundle
from modules.stage2.model_client import Stage2ModelClient
from modules.stage2.principle_generation import generate_principles, repair_principle
from modules.stage2.quality_audit import audit_principle
from modules.stage2.reconciliation import reconcile_candidate
from utils.io_utils import load_json
from utils.output_paths import stage1_level_packets_file


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _candidate_id(bundle_id: str, family_id: str) -> str:
    return "candidate_" + hashlib.sha256(f"{bundle_id}:{family_id}".encode()).hexdigest()[:16]


def _memory_record(candidate: Dict[str, Any], bundle: Dict[str, Any], round_id: int) -> Dict[str, Any]:
    views = build_memory_views(
        candidate["content"],
        source_dsd=candidate.get("source_dsd", {}),
        validation={"quality_audit": candidate.get("quality_audit", {})},
        audit={
            "bundle_id": bundle["bundle_id"],
            "source_hash": bundle["source_hash"],
            "evidence_complete": bundle["evidence_complete"],
            "supporting_wrong_rollout_ids": candidate["supporting_wrong_rollout_ids"],
            "contrast_correct_rollout_ids": candidate["contrast_correct_rollout_ids"],
            "failure_family_id": candidate["family_id"],
            "root_cause": candidate["root_cause"],
        },
    )
    return {
        **candidate["content"],
        **views,
        "content": candidate["content"],
        "source_dsd": candidate.get("source_dsd", {}),
        "principle": candidate["content"]["action_guidance"],
        "principle_text": candidate["content"]["action_guidance"],
        "question": bundle["question_text"],
        "source_question_id": bundle["question_id"],
        "source_round": round_id,
        "source_dataset": bundle["level"],
        "status": "active",
        "schema_version": SCHEMA_VERSION,
        "quality_metrics": {"source_fidelity_pass": True},
    }


async def build_level_memory(
    pipeline: Any,
    *,
    level: str,
    round_id: int,
    backbone: str = "",
    model_concurrency: int = 8,
    max_records: int = 0,
    run_tag: str = "",
    prepare_candidates_only: bool = False,
    candidate_bank: str = "",
    selection_file: str = "",
    artifact_dir: str = "",
) -> Dict[str, Any]:
    """Build one curriculum level without neighbor-task validation."""
    started = time.monotonic()
    packet_path = stage1_level_packets_file(pipeline.config.paths.output_dir, round_id, backbone)
    packets = load_json(str(packet_path)) if packet_path.exists() else []
    if not isinstance(packets, list) or not packets:
        return {"status": "empty_stage1", "level": level, "packet_path": str(packet_path)}
    bundles = build_learning_bundles(packets, level)
    if selection_file:
        selected = load_json(selection_file)
        selected_ids = set(selected.get("question_ids", []) if isinstance(selected, dict) else selected)
        bundles = [item for item in bundles if item["question_id"] in selected_ids]
    if max_records:
        bundles = bundles[: int(max_records)]

    artifact_root = Path(artifact_dir) if artifact_dir else (
        pipeline.config.paths.output_dir / "stage2" / level
    )
    if run_tag:
        artifact_root = artifact_root / "runs" / run_tag
    _save(artifact_root / "00_learning_bundles" / "bundles.json", bundles)
    invalid_bundles = [
        {"bundle_id": item["bundle_id"], "errors": validate_learning_bundle(item)}
        for item in bundles if validate_learning_bundle(item)
    ]
    _save(artifact_root / "00_learning_bundles" / "invalid.json", invalid_bundles)

    run_hash = hashlib.sha256(json.dumps({
        "level": level,
        "bundle_hashes": [item["source_hash"] for item in bundles],
        "generator": pipeline.config.stage2.generator_model,
        "auditor": pipeline.config.stage2.auditor_model,
        "schema": SCHEMA_VERSION,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint_path = artifact_root / "checkpoint.json"
    candidate_path = artifact_root / "04_generated_principles" / "candidates.json"
    checkpoint = load_json(str(checkpoint_path)) if checkpoint_path.exists() else {}
    if not isinstance(checkpoint, dict) or checkpoint.get("run_hash") != run_hash:
        checkpoint = {"run_hash": run_hash, "completed_question_ids": [], "failed_question_ids": []}
    summary_path = artifact_root / "summary.json"
    if checkpoint.get("status") == "complete" and summary_path.exists():
        summary = load_json(str(summary_path))
        if isinstance(summary, dict) and summary.get("run_hash") == run_hash:
            return summary

    client = Stage2ModelClient(pipeline.config)
    semaphore = asyncio.Semaphore(max(1, int(model_concurrency)))
    units: List[Dict[str, Any]] = []

    async def process(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
        if validate_learning_bundle(bundle):
            return []
        async with semaphore:
            generation = await generate_principles(
                client, bundle, pipeline.config.stage2.max_principles_per_question
            )
        results: List[Dict[str, Any]] = []
        for family in generation["families"]:
            generated_dsd = generation.get("routing_semantics", {})
            source_dsd = bundle.get("source_dsd", {})
            if not isinstance(source_dsd, dict) or not all(
                str(source_dsd.get(field, "")).strip()
                for field in ("core_skill", "context_constraints", "task_objective")
            ):
                source_dsd = generated_dsd if isinstance(generated_dsd, dict) else {}
            candidate = {
                **family,
                "candidate_id": _candidate_id(bundle["bundle_id"], family["family_id"]),
                "question_id": bundle["question_id"],
                "source_dsd": source_dsd,
                "generation": generation,
                "status": "generated",
            }
            routing_errors = [
                f"missing_{field}"
                for field in ("core_skill", "context_constraints", "task_objective")
                if not str(source_dsd.get(field, "")).strip()
            ]
            candidate["routing_schema_errors"] = routing_errors
            if routing_errors:
                candidate["status"] = "rejected"
                candidate["quality_audit"] = {
                    "decision": "reject",
                    "protocol_compliant": True,
                    "reasons": routing_errors,
                }
                results.append(candidate)
                continue
            async with semaphore:
                audit = await audit_principle(client, bundle, candidate)
            candidate["quality_audit"] = audit
            if audit["decision"] == "repair":
                async with semaphore:
                    repair = await repair_principle(client, bundle, candidate, audit)
                candidate["repair"] = repair
                if repair["success"]:
                    candidate["content"] = repair["content"]
                    async with semaphore:
                        candidate["quality_audit"] = await audit_principle(client, bundle, candidate)
            candidate["status"] = "quality_pass" if candidate["quality_audit"]["decision"] == "pass" else "rejected"
            results.append(candidate)
        return results

    bank_path = Path(candidate_bank) if candidate_bank else None
    if bank_path and bank_path.exists() and not prepare_candidates_only:
        bank_payload = load_json(str(bank_path))
        if not isinstance(bank_payload, dict) or bank_payload.get("level") != level:
            raise ValueError(f"Candidate bank level mismatch: expected {level}: {bank_path}")
        bank_units = bank_payload.get("candidates", []) if isinstance(bank_payload, dict) else []
        if not isinstance(bank_units, list):
            raise ValueError(f"Invalid candidate bank: {bank_path}")
        units = [dict(item) for item in bank_units if isinstance(item, dict)]
        known_questions = {item["question_id"] for item in bundles}
        unknown = sorted({str(item.get("question_id", "")) for item in units} - known_questions)
        if unknown:
            raise ValueError(f"Candidate bank contains unknown questions: {unknown[:5]}")
    else:
        completed_ids = set(checkpoint.get("completed_question_ids", []))
        if completed_ids and candidate_path.exists():
            persisted = load_json(str(candidate_path))
            if isinstance(persisted, list):
                units.extend(item for item in persisted if isinstance(item, dict))
        pending = [item for item in bundles if item["question_id"] not in completed_ids]

        async def process_with_id(bundle: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
            return bundle["question_id"], await process(bundle)

        tasks = [asyncio.create_task(process_with_id(bundle)) for bundle in pending]
        for task in asyncio.as_completed(tasks):
            try:
                question_id, group = await task
                units.extend(group)
                completed_ids.add(question_id)
            except Exception as exc:
                checkpoint.setdefault("failed_question_ids", []).append(
                    f"{type(exc).__name__}:{exc}"
                )
            units.sort(key=lambda item: item["candidate_id"])
            _save(candidate_path, units)
            checkpoint["completed_question_ids"] = sorted(completed_ids)
            _save(checkpoint_path, checkpoint)
    units.sort(key=lambda item: item["candidate_id"])
    _save(candidate_path, units)
    _save(artifact_root / "06_quality_audits" / "audits.json", [
        {"candidate_id": item["candidate_id"], "audit": item.get("quality_audit", {})} for item in units
    ])
    if prepare_candidates_only:
        if not candidate_bank:
            raise ValueError("--prepare-candidates-only requires --candidate-bank")
        _save(Path(candidate_bank), {"level": level, "candidates": units})
        return {"status": "candidates_ready", "level": level, "candidate_count": len(units)}

    pipeline.memory.load()
    decisions: List[Dict[str, Any]] = []
    bundle_by_question = {item["question_id"]: item for item in bundles}
    for candidate in units:
        if candidate["status"] != "quality_pass":
            decisions.append({"candidate_id": candidate["candidate_id"], "operation": "reject"})
            continue
        decision = await reconcile_candidate(client, pipeline, candidate)
        bundle = bundle_by_question[candidate["question_id"]]
        if decision["operation"] in {"update", "merge"}:
            candidate["content"] = decision["result_content"]
            reaudit = await audit_principle(client, bundle, candidate)
            decision["post_reconciliation_audit"] = reaudit
            if reaudit["decision"] != "pass":
                decision["operation"] = "reject"
        if decision["operation"] == "add":
            record = _memory_record(candidate, bundle, round_id)
            decision["memory_id"] = pipeline.memory.memory.add(record)
        elif decision["operation"] in {"update", "merge"}:
            record = _memory_record(candidate, bundle, round_id)
            pipeline.memory.memory.replace_version(
                decision["target_memory_id"],
                record,
                reason=decision.get("reason", ""),
                operation=decision["operation"],
            )
            decision["memory_id"] = decision["target_memory_id"]
        decisions.append({"candidate_id": candidate["candidate_id"], **decision})

    store_mutated = any(
        item.get("operation") in {"add", "update", "merge"} for item in decisions
    )
    if pipeline.memory.principle_count and (store_mutated or not pipeline.memory.is_indexed):
        pipeline.memory.build_indices()
    if pipeline.memory.is_indexed:
        pipeline.memory.save_indices()
    pipeline.memory.save()
    pipeline.memory.snapshot(round_id)
    _save(artifact_root / "08_memory_reconciliation" / "decisions.json", decisions)
    operation_counts = {
        name: sum(1 for item in decisions if item.get("operation") == name)
        for name in ("add", "covered", "update", "merge", "reject")
    }
    summary = {
        "status": "complete",
        "level": level,
        "input_packet_count": len(packets),
        "learning_bundle_count": len(bundles),
        "evidence_complete_count": sum(1 for item in bundles if item["evidence_complete"]),
        "candidate_count": len(units),
        "quality_pass_count": sum(1 for item in units if item["status"] == "quality_pass"),
        "operation_counts": operation_counts,
        "managed_coverage": sum(operation_counts[name] for name in ("add", "covered", "update", "merge")) / len(units) if units else 0.0,
        "memory_count": pipeline.memory.principle_count,
        "runtime_seconds": time.monotonic() - started,
    }
    summary["run_hash"] = run_hash
    _save(summary_path, summary)
    checkpoint["status"] = "complete"
    checkpoint["store_snapshot"] = f"level{round_id}"
    _save(checkpoint_path, checkpoint)
    return summary
