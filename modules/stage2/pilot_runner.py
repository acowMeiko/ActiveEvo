"""Recoverable, paired Stage-2 generator pilot.

This module never writes to the Memory Store.  It persists every model-facing
stage independently so an interrupted run can resume without repeating a
completed diagnosis, synthesis, or audit call.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from modules.principle_schema import SCHEMA_VERSION
from core.request_context import llm_request_context
from modules.stage2.contracts import build_learning_bundles, validate_learning_bundle
from modules.stage2.generator_comparison import LEVELS, _stable_key
from modules.stage2.model_client import Stage2ModelClient
from modules.stage2.principle_generation import (
    build_diagnosis_prompt,
    build_repair_prompt,
    build_synthesis_prompt,
    diagnose_principles,
    repair_principle,
    synthesize_principles,
)
from modules.stage2.quality_audit import (
    audit_principle,
    build_audit_prompt,
    deterministic_leakage_check,
)
from utils.io_utils import atomic_save_json, load_json
from utils.output_paths import stage1_level_packets_file
from tqdm.asyncio import tqdm as async_tqdm


logger = logging.getLogger(__name__)


TERMINAL_QUESTION_STATUSES = {
    "completed", "no_reliable_principle", "terminal_failed",
}
FINAL_CANDIDATE_LABELS = {
    "dual_pass", "audit_disagreement", "dual_flagged", "hard_invalid",
}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _candidate_id(generator_id: str, bundle_id: str, family_id: str) -> str:
    digest = hashlib.sha256(
        f"{generator_id}:{bundle_id}:{family_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{generator_id}_{digest}"


def _load_packets_by_level(config: Any, backbone: str) -> Dict[str, list[Dict[str, Any]]]:
    result: Dict[str, list[Dict[str, Any]]] = {}
    for level_number, level in enumerate(LEVELS, 1):
        path = stage1_level_packets_file(config.paths.output_dir, level_number, backbone)
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage-1 packets for {level}: {path}")
        payload = load_json(str(path))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Invalid or empty Stage-1 packet file: {path}")
        result[level] = [item for item in payload if isinstance(item, dict)]
    return result


def prepare_pilot_manifest(
    config: Any,
    *,
    output_root: Path,
    dev_manifest_path: Path,
    backbone: str = "llama",
    per_level: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create 2 dev + 23 stable additions per level and freeze bundle hashes."""
    packets_by_level = _load_packets_by_level(config, backbone)
    dev = load_json(str(dev_manifest_path))
    dev_by_level = dev.get("question_ids_by_level", {}) if isinstance(dev, dict) else {}
    ids_by_level: Dict[str, list[str]] = {}
    bundle_hashes: Dict[str, str] = {}
    for level in LEVELS:
        dev_ids = [str(value) for value in dev_by_level.get(level, [])]
        if len(dev_ids) != 2:
            raise ValueError(f"{level} must contribute exactly two frozen dev questions")
        bundles = build_learning_bundles(packets_by_level[level], level)
        by_id = {str(item["question_id"]): item for item in bundles}
        missing = sorted(set(dev_ids) - set(by_id))
        if missing:
            raise ValueError(f"Frozen dev IDs are absent from {level}: {missing}")
        remaining = sorted(
            (question_id for question_id in by_id if question_id not in set(dev_ids)),
            key=lambda value: (_stable_key(value, seed), value),
        )
        needed = per_level - len(dev_ids)
        if len(remaining) < needed:
            raise ValueError(f"{level} has only {len(remaining)} non-dev questions; need {needed}")
        ids_by_level[level] = dev_ids + remaining[:needed]
        for question_id in ids_by_level[level]:
            bundle_hashes[question_id] = by_id[question_id]["source_hash"]
    question_ids = [value for level in LEVELS for value in ids_by_level[level]]
    manifest = {
        "schema_version": "generator_pilot_2.0",
        "policy": "two_frozen_dev_plus_sha256_remainder",
        "seed": seed,
        "backbone": backbone,
        "per_level": per_level,
        "actual_count": len(question_ids),
        "dev_question_ids": [
            value for level in LEVELS for value in ids_by_level[level][:2]
        ],
        "question_ids_by_level": ids_by_level,
        "question_ids": question_ids,
        "bundle_hashes": bundle_hashes,
    }
    manifest["manifest_hash"] = _hash_json({
        "question_ids_by_level": ids_by_level,
        "bundle_hashes": bundle_hashes,
        "seed": seed,
    })
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "sample_manifest.json"
    if path.exists():
        existing = load_json(str(path))
        if existing.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError(f"Refusing to replace a different frozen manifest: {path}")
        return existing
    atomic_save_json(path, manifest)
    return manifest


def _load_selected_bundles(
    config: Any, manifest: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    packets = _load_packets_by_level(config, str(manifest.get("backbone", "llama")))
    selected: Dict[str, Dict[str, Any]] = {}
    for level in LEVELS:
        by_id = {
            str(item["question_id"]): item
            for item in build_learning_bundles(packets[level], level)
        }
        for question_id in manifest["question_ids_by_level"][level]:
            bundle = by_id.get(str(question_id))
            if bundle is None:
                raise ValueError(f"Manifest question missing from Stage 1: {level}/{question_id}")
            expected_hash = manifest["bundle_hashes"].get(str(question_id))
            if expected_hash != bundle["source_hash"]:
                raise ValueError(f"Bundle hash changed for question {question_id}")
            errors = validate_learning_bundle(bundle)
            if errors:
                raise ValueError(f"Invalid bundle {question_id}: {errors}")
            selected[str(question_id)] = bundle
    return selected


def _run_protocol(config: Any, generator_id: str, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    stage2 = config.stage2
    prompt_protocol_hash = _hash_json({
        "diagnosis": inspect.getsource(build_diagnosis_prompt),
        "synthesis": inspect.getsource(build_synthesis_prompt),
        "repair": inspect.getsource(build_repair_prompt),
        "audit": inspect.getsource(build_audit_prompt),
    })
    return {
        "generator_id": generator_id,
        "generator_model": (
            stage2.qwen_generator_model if generator_id == "qwen" else stage2.generator_model
        ),
        "manifest_hash": manifest["manifest_hash"],
        "schema_version": SCHEMA_VERSION,
        "prompt_protocol_hash": prompt_protocol_hash,
        "max_principles": stage2.max_principles_per_question,
        "diagnosis_max_tokens": (
            stage2.qwen_generation_analysis_max_tokens
            if generator_id == "qwen" else stage2.generation_analysis_max_tokens
        ),
        "synthesis_max_tokens": (
            stage2.qwen_generation_max_tokens
            if generator_id == "qwen" else stage2.generation_max_tokens
        ),
        "qwen_retry_max_tokens": (
            stage2.qwen_generation_retry_max_tokens if generator_id == "qwen" else None
        ),
        "qwen_reasoning_effort": (
            stage2.qwen_generator_reasoning_effort if generator_id == "qwen" else None
        ),
        "audit_models": [stage2.generator_model, stage2.auditor_model],
        "audit_max_tokens": stage2.audit_max_tokens,
        "audit_retry_max_tokens": stage2.audit_retry_max_tokens,
        "audit_reasoning_effort": stage2.auditor_reasoning_effort,
        "auditor_timeout": stage2.auditor_timeout,
        "generator_timeout": stage2.generator_timeout,
    }


def _load_stage(path: Path, run_hash: str) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(str(path))
    if not isinstance(payload, dict) or payload.get("run_hash") != run_hash:
        return None
    return payload


def _save_stage(path: Path, run_hash: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    value = {"run_hash": run_hash, "saved_at": time.time(), **dict(payload)}
    atomic_save_json(path, value)
    return value


def _save_retry_stage(
    path: Path, run_hash: str, payload: Mapping[str, Any]
) -> Dict[str, Any]:
    """Replace the current stage value while retaining prior failed calls."""
    previous = _load_stage(path, run_hash)
    history = list((previous or {}).get("attempt_history", []))
    if previous:
        history.append({key: value for key, value in previous.items() if key != "attempt_history"})
    return _save_stage(path, run_hash, {**dict(payload), "attempt_history": history})


def _candidate_label(
    content: Mapping[str, Any], bundle: Mapping[str, Any], audits: Mapping[str, Any]
) -> str:
    leakage = deterministic_leakage_check(
        dict(content),
        source_question=str(bundle["question_text"]),
        expected_answer=str(bundle["expected_answer"]),
    )
    if leakage.get("detected"):
        return "hard_invalid"
    decisions = [str((audits.get(name) or {}).get("decision", "reject")) for name in ("deepseek", "qwen")]
    protocols = [bool((audits.get(name) or {}).get("protocol_compliant")) for name in ("deepseek", "qwen")]
    if not all(protocols):
        return "hard_invalid"
    if decisions == ["pass", "pass"]:
        return "dual_pass"
    if decisions[0] != decisions[1]:
        return "audit_disagreement"
    return "dual_flagged"


def _requires_repair(content: Mapping[str, Any], bundle: Mapping[str, Any], audits: Mapping[str, Any]) -> bool:
    leakage = deterministic_leakage_check(
        dict(content),
        source_question=str(bundle["question_text"]),
        expected_answer=str(bundle["expected_answer"]),
    )
    return bool(
        leakage.get("detected")
        or any((audits.get(name) or {}).get("decision") == "repair" for name in ("deepseek", "qwen"))
    )


async def _persist_dual_audit(
    client: Stage2ModelClient,
    bundle: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    semaphores: Mapping[str, asyncio.Semaphore],
    candidate_root: Path,
    prefix: str,
    run_hash: str,
) -> Dict[str, Any]:
    async def one(auditor_id: str) -> Dict[str, Any]:
        path = candidate_root / f"{prefix}_{auditor_id}_audit.json"
        saved = _load_stage(path, run_hash)
        if saved and (saved.get("audit") or {}).get("protocol_compliant"):
            return saved["audit"]
        async with semaphores[auditor_id]:
            with llm_request_context(
                str(candidate.get("candidate_id", "-")), f"{prefix}_{auditor_id}_audit"
            ):
                audit = await audit_principle(
                    client, bundle, candidate, auditor_id=auditor_id
                )
        _save_retry_stage(path, run_hash, {"audit": audit})
        if not audit.get("protocol_compliant"):
            raise RuntimeError(
                f"{auditor_id} audit protocol failed: {audit.get('protocol_errors', [])}"
            )
        return audit
    results = await asyncio.gather(one("deepseek"), one("qwen"), return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError("; ".join(str(error) for error in failures))
    return {"deepseek": results[0], "qwen": results[1]}


async def _process_question(
    config: Any,
    *,
    generator_id: str,
    bundle: Dict[str, Any],
    branch_root: Path,
    run_hash: str,
    client: Stage2ModelClient,
    generation_semaphore: asyncio.Semaphore,
    audit_semaphores: Mapping[str, asyncio.Semaphore],
    retry_terminal: bool,
    max_question_attempts: int = 3,
) -> Dict[str, Any]:
    question_id = str(bundle["question_id"])
    root = branch_root / "questions" / question_id
    state_path = root / "state.json"
    state = load_json(str(state_path)) if state_path.exists() else {}
    if state and state.get("run_hash") != run_hash:
        raise ValueError(f"Question {question_id} has artifacts from a different protocol")
    if state.get("status") in TERMINAL_QUESTION_STATUSES:
        if state.get("status") != "terminal_failed" or not retry_terminal:
            return state
    if "failure_attempts" in state:
        failure_attempts = int(state.get("failure_attempts", 0))
    else:
        # One-time compatibility for pilot artifacts written before failure and
        # resume counters were separated. A running checkpoint is not a failure.
        failure_attempts = int(bool(
            state.get("status") in {"retryable_failed", "terminal_failed"}
            or state.get("failure_reason")
        ))
    resume_count = int(state.get("resume_count", 0)) + 1
    state = {
        **state,
        "run_hash": run_hash,
        "question_id": question_id,
        "level": bundle["level"],
        "status": "running",
        "current_stage": "bundle_ready",
        "failure_attempts": failure_attempts,
        "resume_count": resume_count,
        "failure_reason": "",
        "updated_at": time.time(),
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_save_json(root / "bundle.json", bundle)
    atomic_save_json(state_path, state)
    max_principles = config.stage2.max_principles_per_question

    diagnosis_path = root / "diagnosis.json"
    diagnosis = _load_stage(diagnosis_path, run_hash)
    if not diagnosis or not diagnosis.get("success"):
        async with generation_semaphore:
            with llm_request_context(question_id, "diagnosis"):
                result = await diagnose_principles(client, bundle, max_principles)
        diagnosis = _save_retry_stage(diagnosis_path, run_hash, result)
    if not diagnosis.get("success"):
        failure_attempts += 1
        status = "terminal_failed" if failure_attempts >= max_question_attempts else "retryable_failed"
        state.update(status=status, current_stage="diagnosis_complete", failure_attempts=failure_attempts, failure_reason=diagnosis.get("failure_reason"))
        atomic_save_json(state_path, state)
        return state
    state.update(current_stage="diagnosis_complete", updated_at=time.time())
    atomic_save_json(state_path, state)

    synthesis_path = root / "synthesis.json"
    synthesis = _load_stage(synthesis_path, run_hash)
    if not synthesis or not synthesis.get("parse_success"):
        async with generation_semaphore:
            with llm_request_context(question_id, "synthesis"):
                result = await synthesize_principles(
                    client, bundle, max_principles, str(diagnosis["diagnosis_text"])
                )
        synthesis = _save_retry_stage(synthesis_path, run_hash, result)
    if synthesis.get("parse_success") and synthesis.get("schema_failures"):
        repair_path = root / "synthesis_schema_repair.json"
        repaired = _load_stage(repair_path, run_hash)
        if not repaired:
            feedback = json.dumps(synthesis["schema_failures"], ensure_ascii=False)
            async with generation_semaphore:
                with llm_request_context(question_id, "synthesis_schema_repair"):
                    result = await synthesize_principles(
                        client,
                        bundle,
                        max_principles,
                        str(diagnosis["diagnosis_text"]),
                        correction_feedback=feedback,
                    )
            repaired = _save_retry_stage(repair_path, run_hash, result)
        if repaired.get("parse_success") and (
            len(repaired.get("families", [])) >= len(synthesis.get("families", []))
        ):
            synthesis = repaired
    if not synthesis.get("parse_success"):
        failure_attempts += 1
        status = "terminal_failed" if failure_attempts >= max_question_attempts else "retryable_failed"
        state.update(status=status, current_stage="synthesis_complete", failure_attempts=failure_attempts, failure_reason=synthesis.get("failure_reason"))
        atomic_save_json(state_path, state)
        return state
    state.update(current_stage="schema_complete", updated_at=time.time())
    atomic_save_json(state_path, state)
    families = synthesis.get("families", [])
    if not families:
        status = "no_reliable_principle" if not synthesis.get("schema_failures") else "terminal_failed"
        state.update(status=status, current_stage="question_terminal", failure_reason=(
            "" if status == "no_reliable_principle" else "no_schema_valid_family"
        ))
        atomic_save_json(root / "result.json", {"status": status, "candidates": []})
        atomic_save_json(state_path, state)
        return state

    candidates: list[Dict[str, Any]] = []
    routing = synthesis.get("routing_semantics", {})
    for family in families:
        candidate_id = _candidate_id(generator_id, bundle["bundle_id"], family["family_id"])
        candidate = {
            **family,
            "candidate_id": candidate_id,
            "generator_id": generator_id,
            "question_id": question_id,
            "level": bundle["level"],
            "source_dsd": bundle.get("source_dsd") or routing,
            "raw_content": family["content"],
            "content": family["content"],
        }
        candidate_root = root / "candidates" / candidate_id
        atomic_save_json(candidate_root / "raw_candidate.json", candidate)
        audits = await _persist_dual_audit(
            client,
            bundle,
            candidate,
            semaphores=audit_semaphores,
            candidate_root=candidate_root,
            prefix="initial",
            run_hash=run_hash,
        )
        candidate["initial_audits"] = audits
        candidate["deterministic_check"] = deterministic_leakage_check(
            candidate["content"],
            source_question=bundle["question_text"],
            expected_answer=bundle["expected_answer"],
        )
        if _requires_repair(candidate["content"], bundle, audits):
            repair_path = candidate_root / "repair.json"
            repair_stage = _load_stage(repair_path, run_hash)
            if not repair_stage:
                async with generation_semaphore:
                    with llm_request_context(candidate_id, "principle_repair"):
                        repaired = await repair_principle(
                            client, bundle, candidate, {"dual_audits": audits}
                        )
                repair_stage = _save_retry_stage(repair_path, run_hash, repaired)
            candidate["repair"] = repair_stage
            if repair_stage.get("success"):
                candidate["content"] = repair_stage["content"]
                candidate["reaudits"] = await _persist_dual_audit(
                    client,
                    bundle,
                    candidate,
                    semaphores=audit_semaphores,
                    candidate_root=candidate_root,
                    prefix="reaudit",
                    run_hash=run_hash,
                )
                audits = candidate["reaudits"]
        candidate["final_audits"] = audits
        candidate["final_label"] = _candidate_label(candidate["content"], bundle, audits)
        if candidate["final_label"] not in FINAL_CANDIDATE_LABELS:
            raise AssertionError(f"Unknown candidate label: {candidate['final_label']}")
        atomic_save_json(candidate_root / "final_candidate.json", candidate)
        candidates.append(candidate)

    state.update(
        status="completed",
        current_stage="question_terminal",
        candidate_count=len(candidates),
        candidate_labels=dict(Counter(item["final_label"] for item in candidates)),
        updated_at=time.time(),
    )
    atomic_save_json(root / "result.json", {"status": "completed", "candidates": candidates})
    atomic_save_json(state_path, state)
    return state


def _phase_ids(manifest: Mapping[str, Any], phase: str) -> list[str]:
    all_ids = [str(value) for value in manifest["question_ids"]]
    dev_ids = {str(value) for value in manifest["dev_question_ids"]}
    if phase == "dev8":
        return [value for value in all_ids if value in dev_ids]
    if phase == "remaining92":
        return [value for value in all_ids if value not in dev_ids]
    if phase == "all100":
        return all_ids
    raise ValueError(f"Unsupported pilot phase: {phase}")


def _aggregate_branch(
    branch_root: Path, manifest: Mapping[str, Any], run_hash: str
) -> Dict[str, Any]:
    states: list[Dict[str, Any]] = []
    candidates_by_level: Dict[str, list[Dict[str, Any]]] = {level: [] for level in LEVELS}
    for question_id in manifest["question_ids"]:
        root = branch_root / "questions" / str(question_id)
        state_path = root / "state.json"
        if not state_path.exists():
            continue
        state = load_json(str(state_path))
        if state.get("run_hash") != run_hash:
            continue
        states.append(state)
        result_path = root / "result.json"
        if result_path.exists():
            result = load_json(str(result_path))
            for candidate in result.get("candidates", []):
                if isinstance(candidate, dict):
                    candidates_by_level[str(candidate.get("level"))].append(candidate)
    for level in LEVELS:
        rows = sorted(candidates_by_level[level], key=lambda item: item["candidate_id"])
        atomic_save_json(branch_root / "candidate_banks" / f"{level}.json", {
            "level": level,
            "run_hash": run_hash,
            "candidates": rows,
        })
    status_counts = Counter(str(item.get("status", "unknown")) for item in states)
    label_counts = Counter()
    for rows in candidates_by_level.values():
        label_counts.update(item.get("final_label", "unknown") for item in rows)
    summary = {
        "run_hash": run_hash,
        "manifest_hash": manifest["manifest_hash"],
        "question_count": len(manifest["question_ids"]),
        "observed_question_count": len(states),
        "status_counts": dict(status_counts),
        "terminal_question_count": sum(
            count for name, count in status_counts.items() if name in TERMINAL_QUESTION_STATUSES
        ),
        "candidate_count": sum(len(rows) for rows in candidates_by_level.values()),
        "candidate_label_counts": dict(label_counts),
        "updated_at": time.time(),
    }
    atomic_save_json(branch_root / "progress.json", summary)
    return summary


def evaluate_dev8_gate(output_root: Path, generator_id: str) -> Dict[str, Any]:
    """Evaluate the fixed eight-question stability gate without model calls."""
    manifest = load_json(str(output_root / "sample_manifest.json"))
    branch_name = "qwen38" if generator_id == "qwen" else "deepseek"
    root = output_root / branch_name
    protocol_path = root / "run_protocol.json"
    if not protocol_path.exists():
        return {"generator_id": generator_id, "passed": False, "reasons": ["branch_not_started"]}
    run_hash = load_json(str(protocol_path)).get("run_hash")
    rows: list[Dict[str, Any]] = []
    for question_id in manifest["dev_question_ids"]:
        question_root = root / "questions" / str(question_id)
        state_path = question_root / "state.json"
        diagnosis_path = question_root / "diagnosis.json"
        synthesis_path = question_root / "synthesis.json"
        state = load_json(str(state_path)) if state_path.exists() else {}
        diagnosis = _load_stage(diagnosis_path, run_hash) if run_hash else None
        synthesis = _load_stage(synthesis_path, run_hash) if run_hash else None
        if synthesis and synthesis.get("schema_failures"):
            repaired = _load_stage(question_root / "synthesis_schema_repair.json", run_hash)
            if repaired and repaired.get("parse_success"):
                synthesis = repaired
        diagnosis_response = (diagnosis or {}).get("response", {})
        rows.append({
            "question_id": str(question_id),
            "status": state.get("status", "missing"),
            "diagnosis_success": bool((diagnosis or {}).get("success")),
            "synthesis_parse_success": bool((synthesis or {}).get("parse_success")),
            "diagnosis_reasoning_nonempty": bool(diagnosis_response.get("reasoning")),
            "diagnosis_final_content_nonempty": bool(diagnosis_response.get("content")),
            "synthesis_reasoning_nonempty": bool((synthesis or {}).get("reasoning_content")),
            "synthesis_final_content_nonempty": bool((synthesis or {}).get("raw_response")),
            "updated_at": state.get("updated_at"),
        })
    reasons: list[str] = []
    terminal_ok = all(
        row["status"] in {"completed", "no_reliable_principle"} for row in rows
    )
    parse_rate = sum(row["synthesis_parse_success"] for row in rows) / max(1, len(rows))
    if len(rows) != 8:
        reasons.append("dev_manifest_does_not_contain_8_questions")
    if not terminal_ok:
        reasons.append("dev_questions_not_successfully_terminal")
    if parse_rate < 0.95:
        reasons.append("final_content_parse_rate_below_95_percent")
    if not all(row["diagnosis_success"] for row in rows):
        reasons.append("diagnosis_incomplete")
    if generator_id == "qwen" and not all(
        row["diagnosis_reasoning_nonempty"]
        and row["diagnosis_final_content_nonempty"]
        and row["synthesis_reasoning_nonempty"]
        and row["synthesis_final_content_nonempty"]
        for row in rows
    ):
        reasons.append("qwen_thinking_and_final_content_not_cleanly_separated")
    report = {
        "generator_id": generator_id,
        "dev_question_count": len(rows),
        "parse_rate": parse_rate,
        "passed": not reasons,
        "reasons": reasons,
        "questions": rows,
    }
    atomic_save_json(root / "dev8_gate.json", report)
    return report


def _require_deepseek_complete(output_root: Path) -> None:
    path = output_root / "deepseek" / "progress.json"
    if not path.exists():
        raise RuntimeError("Qwen branch is blocked until DeepSeek has 100 terminal questions")
    progress = load_json(str(path))
    if int(progress.get("terminal_question_count", 0)) != 100:
        raise RuntimeError(
            "Qwen branch is blocked until DeepSeek has 100 terminal questions; "
            f"current={progress.get('terminal_question_count', 0)}"
        )


async def run_generator_pilot(
    config: Any,
    *,
    output_root: Path,
    generator_id: str,
    phase: str,
    concurrency: int = 2,
    retry_terminal: bool = False,
) -> Dict[str, Any]:
    manifest = load_json(str(output_root / "sample_manifest.json"))
    if generator_id == "qwen":
        _require_deepseek_complete(output_root)
    if phase in {"remaining92", "all100"} and not retry_terminal:
        gate = evaluate_dev8_gate(output_root, generator_id)
        if not gate["passed"]:
            raise RuntimeError(
                f"{generator_id} remaining questions are blocked by the dev8 gate: "
                f"{gate['reasons']}"
            )
    bundles = _load_selected_bundles(config, manifest)
    protocol = _run_protocol(config, generator_id, manifest)
    run_hash = _hash_json(protocol)
    branch_name = "qwen38" if generator_id == "qwen" else "deepseek"
    branch_root = output_root / branch_name
    branch_root.mkdir(parents=True, exist_ok=True)
    protocol_path = branch_root / "run_protocol.json"
    if protocol_path.exists():
        existing = load_json(str(protocol_path))
        if existing.get("run_hash") != run_hash:
            raise ValueError(f"Branch already contains a different protocol: {branch_root}")
    atomic_save_json(protocol_path, {**protocol, "run_hash": run_hash})
    ids = _phase_ids(manifest, phase)
    generation_semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    audit_semaphores = {
        # Commercial API connection setup was unstable at four concurrent
        # audits; serialize that route while both local replicas stay busy.
        "deepseek": asyncio.Semaphore(1),
        "qwen": asyncio.Semaphore(max(1, len(config.stage2.auditor_urls))),
    }
    progress_lock = asyncio.Lock()
    client = Stage2ModelClient(config, generator_id=generator_id)

    async def one(question_id: str) -> Dict[str, Any]:
        while True:
            try:
                state = await _process_question(
                    config,
                    generator_id=generator_id,
                    bundle=bundles[question_id],
                    branch_root=branch_root,
                    run_hash=run_hash,
                    client=client,
                    generation_semaphore=generation_semaphore,
                    audit_semaphores=audit_semaphores,
                    retry_terminal=retry_terminal,
                )
            except Exception as error:  # preserve the exact failing step for resume
                state_path = branch_root / "questions" / question_id / "state.json"
                state = load_json(str(state_path)) if state_path.exists() else {}
                attempts = int(state.get("failure_attempts", 0)) + 1
                status = "terminal_failed" if attempts >= 3 else "retryable_failed"
                state.update({
                    "run_hash": run_hash,
                    "question_id": question_id,
                    "level": bundles[question_id]["level"],
                    "status": status,
                    "failure_attempts": attempts,
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "updated_at": time.time(),
                })
                atomic_save_json(state_path, state)
                logger.exception(
                    "Stage 2 question failed: generator=%s question=%s stage=%s attempt=%d",
                    generator_id, question_id, state.get("current_stage"), attempts,
                )
            if state.get("status") != "retryable_failed":
                break
            await asyncio.sleep(min(30, 2 ** int(state.get("failure_attempts", 1))))
        async with progress_lock:
            _aggregate_branch(branch_root, manifest, run_hash)
        return state

    tasks = [asyncio.create_task(one(question_id)) for question_id in ids]
    for completed in async_tqdm.as_completed(
        tasks,
        total=len(tasks),
        desc=f"Stage 2 {generator_id} {phase}",
    ):
        await completed
    summary = _aggregate_branch(branch_root, manifest, run_hash)
    phase_states = []
    for question_id in ids:
        path = branch_root / "questions" / question_id / "state.json"
        if path.exists():
            phase_states.append(load_json(str(path)))
    phase_complete = len(phase_states) == len(ids) and all(
        item.get("status") in TERMINAL_QUESTION_STATUSES for item in phase_states
    )
    phase_summary = {
        **summary,
        "generator_id": generator_id,
        "phase": phase,
        "phase_question_count": len(ids),
        "phase_terminal_count": sum(
            item.get("status") in TERMINAL_QUESTION_STATUSES for item in phase_states
        ),
        "phase_complete": phase_complete,
    }
    atomic_save_json(branch_root / f"summary_{phase}.json", phase_summary)
    if phase == "dev8":
        phase_summary["dev8_gate"] = evaluate_dev8_gate(output_root, generator_id)
        atomic_save_json(branch_root / f"summary_{phase}.json", phase_summary)
    if phase == "all100" or (phase == "remaining92" and phase_complete):
        atomic_save_json(branch_root / "summary.json", phase_summary)
    return phase_summary


def pilot_status(output_root: Path) -> Dict[str, Any]:
    manifest = load_json(str(output_root / "sample_manifest.json"))
    result: Dict[str, Any] = {
        "manifest_hash": manifest["manifest_hash"],
        "question_count": len(manifest["question_ids"]),
        "branches": {},
    }
    for branch in ("deepseek", "qwen38"):
        branch_root = output_root / branch
        path = branch_root / "progress.json"
        current = load_json(str(path)) if path.exists() else {}
        state_paths = list((branch_root / "questions").glob("*/state.json"))
        states = [load_json(str(value)) for value in state_paths]
        stage_counts = {
            "bundle": len(list((branch_root / "questions").glob("*/bundle.json"))),
            "diagnosis": len(list((branch_root / "questions").glob("*/diagnosis.json"))),
            "synthesis": len(list((branch_root / "questions").glob("*/synthesis.json"))),
            "candidate_final": len(list((branch_root / "questions").glob("*/candidates/*/final_candidate.json"))),
        }
        result["branches"][branch] = {
            **({"status": "not_started"} if not states else {}),
            **current,
            "live_status_counts": dict(Counter(
                str(item.get("status", "unknown")) for item in states
            )),
            "stage_checkpoint_counts": stage_counts,
        }
    return result


def compare_pilot_branches(output_root: Path) -> Dict[str, Any]:
    manifest = load_json(str(output_root / "sample_manifest.json"))
    branches: Dict[str, Dict[str, Any]] = {}
    per_question: Dict[str, Dict[str, Any]] = {
        str(question_id): {} for question_id in manifest["question_ids"]
    }
    for branch in ("deepseek", "qwen38"):
        root = output_root / branch
        summary_path = root / "summary.json"
        if not summary_path.exists():
            raise ValueError(f"Branch is not complete: {branch}")
        summary = load_json(str(summary_path))
        if int(summary.get("terminal_question_count", 0)) != 100:
            raise ValueError(
                f"Branch summary is not a complete 100-question terminal run: {branch}"
            )
        if summary.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError(f"Manifest mismatch in branch: {branch}")
        branches[branch] = summary
        for question_id in manifest["question_ids"]:
            result_path = root / "questions" / str(question_id) / "result.json"
            state_path = root / "questions" / str(question_id) / "state.json"
            state = load_json(str(state_path)) if state_path.exists() else {"status": "missing"}
            result = load_json(str(result_path)) if result_path.exists() else {"candidates": []}
            rows = result.get("candidates", [])
            math_flagged = sum(
                any(
                    ((candidate.get("final_audits") or {}).get(auditor) or {})
                    .get("dimensions", {}).get("mathematical_validity", {}).get("pass") is False
                    for auditor in ("deepseek", "qwen")
                )
                for candidate in rows
            )
            leakage = sum(
                bool((candidate.get("deterministic_check") or {}).get("detected"))
                for candidate in rows
            )
            per_question[str(question_id)][branch] = {
                "status": state.get("status"),
                "candidate_count": len(rows),
                "labels": dict(Counter(item.get("final_label", "unknown") for item in rows)),
                "has_dual_pass": any(item.get("final_label") == "dual_pass" for item in rows),
                "mathematical_error_flag_count": math_flagged,
                "answer_leakage_count": leakage,
            }
    paired = []
    for question_id in manifest["question_ids"]:
        row = {"question_id": str(question_id), **per_question[str(question_id)]}
        paired.append(row)
    quality: Dict[str, Any] = {}
    for branch in ("deepseek", "qwen38"):
        rows = [item[branch] for item in per_question.values()]
        candidate_count = sum(item["candidate_count"] for item in rows)
        label_counts = Counter()
        for item in rows:
            label_counts.update(item["labels"])
        by_level: Dict[str, Any] = {}
        for level in LEVELS:
            level_ids = {str(value) for value in manifest["question_ids_by_level"][level]}
            level_rows = [per_question[value][branch] for value in level_ids]
            by_level[level] = {
                "question_count": len(level_rows),
                "question_with_candidate_count": sum(item["candidate_count"] > 0 for item in level_rows),
                "question_with_dual_pass_count": sum(item["has_dual_pass"] for item in level_rows),
                "terminal_failed_count": sum(item["status"] == "terminal_failed" for item in level_rows),
                "no_reliable_principle_count": sum(
                    item["status"] == "no_reliable_principle" for item in level_rows
                ),
            }
        quality[branch] = {
            "question_with_candidate_count": sum(item["candidate_count"] > 0 for item in rows),
            "legal_principle_question_rate": (
                sum(item["candidate_count"] > 0 for item in rows) / len(rows) if rows else 0.0
            ),
            "question_with_dual_pass_count": sum(item["has_dual_pass"] for item in rows),
            "dual_pass_question_rate": (
                sum(item["has_dual_pass"] for item in rows) / len(rows) if rows else 0.0
            ),
            "candidate_count": candidate_count,
            "candidate_label_counts": dict(label_counts),
            "audit_disagreement_rate": (
                label_counts["audit_disagreement"] / candidate_count if candidate_count else 0.0
            ),
            "answer_leakage_count": sum(item["answer_leakage_count"] for item in rows),
            "answer_leakage_rate": (
                sum(item["answer_leakage_count"] for item in rows) / candidate_count
                if candidate_count else 0.0
            ),
            "mathematical_error_flag_count": sum(
                item["mathematical_error_flag_count"] for item in rows
            ),
            "mathematical_error_flag_rate": (
                sum(item["mathematical_error_flag_count"] for item in rows) / candidate_count
                if candidate_count else 0.0
            ),
            "by_level": by_level,
        }
    report = {
        "manifest_hash": manifest["manifest_hash"],
        "branches": branches,
        "paired_questions": paired,
        "quality_summary": quality,
        "effectiveness_not_validated": True,
        "winner_selected": False,
    }
    comparison_dir = output_root / "comparison"
    atomic_save_json(comparison_dir / "comparison.json", report)
    lines = [
        "# DeepSeek vs Qwen3.8 Stage-2 Generator Pilot", "",
        "> Content quality only. Llama source-effect validation has not run.", "",
        "| Branch | Terminal questions | Candidates | Dual pass | Disagreement | Flagged | Hard invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in ("deepseek", "qwen38"):
        item = branches[branch]
        labels = item.get("candidate_label_counts", {})
        lines.append(
            f"| {branch} | {item.get('terminal_question_count', 0)} | {item.get('candidate_count', 0)} | "
            f"{labels.get('dual_pass', 0)} | {labels.get('audit_disagreement', 0)} | "
            f"{labels.get('dual_flagged', 0)} | {labels.get('hard_invalid', 0)} |"
        )
    lines.extend(["", "## Question-level quality", "",
        "| Branch | Has candidate | Has dual-pass | Leakage | Math flagged |",
        "|---|---:|---:|---:|---:|",
    ])
    for branch in ("deepseek", "qwen38"):
        item = quality[branch]
        lines.append(
            f"| {branch} | {item['question_with_candidate_count']}/100 | "
            f"{item['question_with_dual_pass_count']}/100 | "
            f"{item['answer_leakage_count']} | {item['mathematical_error_flag_count']} |"
        )
    (comparison_dir / "comparison.md").parent.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
