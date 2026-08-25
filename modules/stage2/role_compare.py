"""Fast, recoverable eight-question generator/auditor role comparison.

This harness is intentionally isolated from the production Memory Store and
from Stage 3 retrieval.  It compares raw Principle quality and one-pass blind
cross-auditing before a fixed Llama source-question intervention.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from core.request_context import llm_request_context
from modules.principle_schema import validate_principle_content
from modules.stage2.model_client import Stage2ModelClient, parse_json_object
from modules.stage2.principle_generation import (
    build_diagnosis_prompt,
    diagnose_principles,
    repair_principle,
    synthesize_principles,
    validate_synthesis_payload,
)
from modules.stage2.principle_test import (
    HARD_AUDIT_DIMENSIONS,
    _PairedRunner,
    _audit_protocol_valid,
    _critic_prompt,
    _formatter_prompt,
    _gather_with_progress,
    _healthy_actor_urls,
    _json_hash,
    _paired_counts,
)
from modules.stage2.quality_audit import (
    AUDIT_DIMENSIONS,
    build_audit_prompt,
    deterministic_leakage_check,
)
from utils.io_utils import atomic_save_json, load_json


logger = logging.getLogger(__name__)

ROLE_COMPARE_VERSION = "role_compare.1"
TERMINAL_GENERATION = {"complete", "no_reliable_principle", "terminal_failed"}
KIMI_SYNTHESIS_RECOVERY_VERSION = "kimi_synthesis_recovery.4"
KIMI_SYNTHESIS_RECOVERY_IDS = ("1781", "5622", "580", "2929")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _candidate_id(generator_id: str, question_id: str, family_id: str) -> str:
    digest = hashlib.sha256(
        f"{generator_id}:{question_id}:{family_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"candidate_{generator_id}_{digest}"


def _candidates_from_record(record: Mapping[str, Any], generator_id: str) -> list[dict[str, Any]]:
    result = []
    for family in (record.get("synthesis") or {}).get("families", []):
        family_id = str(family["family_id"])
        result.append({
            "candidate_id": _candidate_id(generator_id, str(record["question_id"]), family_id),
            "question_id": str(record["question_id"]),
            "level": str(record["level"]),
            "family_id": family_id,
            "root_cause": family["root_cause"],
            "supporting_wrong_rollout_ids": family["supporting_wrong_rollout_ids"],
            "contrast_correct_rollout_ids": family["contrast_correct_rollout_ids"],
            "content": family["content"],
            "generator_id": generator_id,
        })
    return result


async def _warmup_map(
    items: Sequence[Any],
    worker: Callable[[Any], Awaitable[dict[str, Any]]],
    *,
    initial: int,
    maximum: int,
    description: str,
) -> list[dict[str, Any]]:
    """Use a small provider warm-up, then raise concurrency after success."""
    values = list(items)
    warmup_count = min(max(1, initial), len(values))

    async def run_group(group: Sequence[Any], concurrency: int, label: str) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def guarded(item: Any) -> dict[str, Any]:
            async with semaphore:
                return await worker(item)

        return await _gather_with_progress(
            [guarded(item) for item in group], description=label, unit="项",
        )

    def rate_limited(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("http_status") == 429:
                return True
            return any(rate_limited(item) for item in value.values())
        if isinstance(value, list):
            return any(rate_limited(item) for item in value)
        return False

    warmup = await run_group(values[:warmup_count], initial, f"{description} 预热")
    stable = all(
        item.get("status") not in {"terminal_failed", "protocol_failed", "infrastructure_failed"}
        for item in warmup
    )
    selected = maximum if stable else initial
    if any(rate_limited(item) for item in warmup):
        stable = False
    if values[warmup_count:]:
        logger.info(
            "[%s] 预热%s，后续并发=%d", description,
            "通过" if stable else "存在失败，维持保守并发", selected,
        )
        tail = []
        remaining = values[warmup_count:]
        offset = 0
        while offset < len(remaining):
            group = remaining[offset : offset + selected]
            completed = await run_group(group, selected, description)
            tail.extend(completed)
            offset += len(group)
            if selected > initial and any(rate_limited(item) for item in completed):
                selected = initial
                logger.warning("[%s] 检测到HTTP 429，后续并发降至%d", description, selected)
    else:
        tail = []
    return warmup + tail


def _generation_hash(config: Any, bundle: Mapping[str, Any], generator_id: str) -> str:
    return _sha({
        "version": ROLE_COMPARE_VERSION,
        "generator_id": generator_id,
        "bundle": bundle,
        "diagnosis_prompt": build_diagnosis_prompt(
            dict(bundle), config.stage2.max_principles_per_question
        ),
        "generator_model": (
            config.stage2.auditor_model if generator_id == "kimi"
            else config.stage2.generator_model
        ),
        "analysis_max_tokens": config.stage2.generation_analysis_max_tokens,
        "synthesis_max_tokens": config.stage2.generation_max_tokens,
    })


def _needs_kimi_length_recovery(config: Any, synthesis: Mapping[str, Any] | None) -> bool:
    """Return true only when a saved Kimi synthesis never tried the larger budget."""
    if not isinstance(synthesis, Mapping):
        return False
    truncated = (
        synthesis.get("failure_reason") == "thinking_truncated"
        or synthesis.get("finish_reason") == "length"
    )
    attempts = synthesis.get("generation_attempts") or []
    maximum_tried = max(
        (int(item.get("max_tokens", 0) or 0) for item in attempts if isinstance(item, Mapping)),
        default=0,
    )
    return bool(
        truncated
        and maximum_tried < config.stage2.role_compare_kimi_generation_retry_max_tokens
    )


def _kimi_recovery_input_hash(config: Any, bundle: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> str:
    """Hash only synthesis inputs, so old diagnosis checkpoints stay reusable."""
    return _sha({
        "version": KIMI_SYNTHESIS_RECOVERY_VERSION,
        "bundle_hash": _sha(bundle),
        "diagnosis_hash": _sha(diagnosis.get("diagnosis_text", "")),
        "model": config.stage2.auditor_model,
        "thinking_enabled": False,
        "max_tokens": 4096,
        "max_principles": config.stage2.max_principles_per_question,
    })


def _is_expected_kimi_synthesis_failure(record: Mapping[str, Any]) -> bool:
    synthesis = record.get("synthesis") or {}
    return bool(
        record.get("status") == "terminal_failed"
        and synthesis.get("finish_reason") == "length"
        and not str(synthesis.get("raw_response", "")).strip()
    )


async def run_kimi_synthesis_recovery(
    config: Any,
    *,
    test_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Retry only the four known Kimi synthesis truncations with frozen diagnoses.

    This helper intentionally performs no audit, no Llama call, and no diagnosis
    request.  It writes to a new artifact root so the original role comparison
    remains a read-only development record.
    """
    test_root = Path(test_root)
    source_root = test_root / "role_compare_v1"
    source_questions = source_root / "kimi_generation" / "questions"
    source_checkpoints = source_root / "checkpoints" / "kimi_generation"
    bundles = load_json(test_root / "source_bundles" / "source40_bundles.json")
    bundle_by_id = {str(item["question_id"]): item for item in bundles}
    missing = [qid for qid in KIMI_SYNTHESIS_RECOVERY_IDS if qid not in bundle_by_id]
    if missing:
        raise RuntimeError(f"Recovery bundles missing: {missing}")

    # Keep v2/v3 immutable. v4 performs a type-only normalization over the
    # complete v3 provider responses and therefore requires no additional API
    # call when those responses are present.
    v3_root = test_root / "kimi_synthesis_recovery_v3"
    output_root = test_root / "kimi_synthesis_recovery_v4"
    synthesis_dir = output_root / "synthesis"
    client = Stage2ModelClient(config, generator_id="kimi")

    async def one(qid: str) -> dict[str, Any]:
        old_path = source_questions / f"{qid}.json"
        diagnosis_path = source_checkpoints / f"{qid}_diagnosis.json"
        if not old_path.exists() or not diagnosis_path.exists():
            raise RuntimeError(f"Recovery source checkpoint missing for question {qid}")
        old_record = load_json(old_path)
        diagnosis_payload = load_json(diagnosis_path)
        diagnosis = diagnosis_payload.get("diagnosis") or {}
        if not _is_expected_kimi_synthesis_failure(old_record):
            raise RuntimeError(f"Question {qid} is not an expected Kimi synthesis truncation")
        if not diagnosis.get("success") or not str(diagnosis.get("diagnosis_text", "")).strip():
            raise RuntimeError(f"Question {qid} lacks a reusable successful diagnosis")

        bundle = bundle_by_id[qid]
        input_hash = _kimi_recovery_input_hash(config, bundle, diagnosis)
        path = synthesis_dir / f"{qid}.json"
        if resume and path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash and saved.get("status") in TERMINAL_GENERATION:
                return saved

        v3_path = v3_root / "synthesis" / f"{qid}.json"
        if v3_path.exists():
            v3_record = load_json(v3_path)
            v3_synthesis = v3_record.get("synthesis") or {}
            raw_content = str(v3_synthesis.get("raw_response") or "").strip()
            if v3_synthesis.get("finish_reason") == "stop" and raw_content:
                parsed = parse_json_object(raw_content)
                validation = validate_synthesis_payload(
                    parsed, bundle, config.stage2.max_principles_per_question,
                )
                synthesis = {
                    **v3_synthesis,
                    **validation,
                    "parse_success": bool(parsed),
                    "format_recovered_from": str(v3_path),
                    "synthesis_api_calls": 0,
                }
                families = synthesis.get("families") or []
                schema_failures = synthesis.get("schema_failures") or []
                status = (
                    "complete" if families and not schema_failures else
                    "no_reliable_principle" if parsed and not schema_failures else
                    "terminal_failed"
                )
                record = {
                    "status": status,
                    "question_id": qid,
                    "level": bundle["level"],
                    "input_hash": input_hash,
                    "source_role_compare_record": str(old_path),
                    "source_diagnosis_checkpoint": str(diagnosis_path),
                    "source_v3_synthesis": str(v3_path),
                    "diagnosis_reused": True,
                    "diagnosis_api_calls": 0,
                    "synthesis_api_calls": 0,
                    "diagnosis_hash": _sha(diagnosis.get("diagnosis_text", "")),
                    "synthesis": synthesis,
                }
                atomic_save_json(path, record)
                logger.info(
                    "[Kimi合成恢复] 题目=%s，复用v3最终JSON，格式归一化=%d",
                    qid, len(validation.get("format_normalizations") or []),
                )
                return record

        logger.info("[Kimi合成恢复] 题目=%s，复用诊断，开始非Thinking JSON合成", qid)
        with llm_request_context(qid, "kimi_synthesis_recovery/synthesis"):
            synthesis = await synthesize_principles(
                client,
                bundle,
                config.stage2.max_principles_per_question,
                diagnosis["diagnosis_text"],
                max_tokens_override=4096,
                compact_synthesis=True,
                thinking_enabled=False,
            )
        families = synthesis.get("families") or []
        schema_failures = synthesis.get("schema_failures") or []
        status = (
            "complete" if families and synthesis.get("parse_success") and not schema_failures else
            "no_reliable_principle" if synthesis.get("parse_success") and not schema_failures else
            "terminal_failed"
        )
        record = {
            "status": status,
            "question_id": qid,
            "level": bundle["level"],
            "input_hash": input_hash,
            "source_role_compare_record": str(old_path),
            "source_diagnosis_checkpoint": str(diagnosis_path),
            "diagnosis_reused": True,
            "diagnosis_api_calls": 0,
            "synthesis_api_calls": 1,
            "diagnosis_hash": _sha(diagnosis.get("diagnosis_text", "")),
            "synthesis": synthesis,
        }
        atomic_save_json(path, record)
        return record

    records = await _gather_with_progress(
        [one(qid) for qid in KIMI_SYNTHESIS_RECOVERY_IDS],
        description="Kimi合成恢复", unit="道题",
    )
    candidates = []
    for record in records:
        candidates.extend(_candidates_from_record(record, "kimi"))
    candidates.sort(key=lambda item: item["candidate_id"])
    manifest = {
        "schema_version": KIMI_SYNTHESIS_RECOVERY_VERSION,
        "source_role_compare_root": str(source_root),
        "question_ids": list(KIMI_SYNTHESIS_RECOVERY_IDS),
        "question_count": len(records),
        "diagnosis_api_calls": 0,
        "synthesis_api_calls": sum(int(record.get("synthesis_api_calls", 0)) for record in records),
        "thinking_enabled_for_synthesis": False,
        "synthesis_max_tokens": 4096,
    }
    manifest["manifest_hash"] = _sha(manifest)
    bank = {
        "schema_version": KIMI_SYNTHESIS_RECOVERY_VERSION,
        "generator_id": "kimi",
        "manifest_hash": manifest["manifest_hash"],
        "question_count": len(records),
        "synthesis_api_calls": manifest["synthesis_api_calls"],
        "candidate_count": len(candidates),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "candidates": candidates,
    }
    bank["candidate_bank_hash"] = _sha(bank)
    summary = {
        "question_count": len(records),
        "diagnosis_api_calls": 0,
        "synthesis_api_calls": bank["synthesis_api_calls"],
        "format_normalization_count": sum(
            len((record.get("synthesis") or {}).get("format_normalizations") or [])
            for record in records
        ),
        "status_counts": bank["status_counts"],
        "candidate_count": len(candidates),
        "all_stop": all((record.get("synthesis") or {}).get("finish_reason") == "stop" for record in records),
        "all_nonempty_final_content": all(bool((record.get("synthesis") or {}).get("raw_response", "")) for record in records),
        "all_schema_valid": all(not ((record.get("synthesis") or {}).get("schema_failures") or []) for record in records),
    }
    lines = [
        "# Kimi 合成恢复报告", "",
        "- 诊断 API 调用：0（全部复用旧Checkpoint）",
        f"- 合成 API 调用：{summary['synthesis_api_calls']}（优先复用v3最终JSON）",
        f"- 确定性格式归一化：{summary['format_normalization_count']}处",
        f"- 题目数：{len(records)}",
        f"- 状态：{json.dumps(summary['status_counts'], ensure_ascii=False)}",
        f"- 全部正常停止：{summary['all_stop']}",
        f"- 全部具有最终内容：{summary['all_nonempty_final_content']}",
        f"- 全部Schema合法：{summary['all_schema_valid']}",
    ]
    atomic_save_json(output_root / "manifest.json", manifest)
    atomic_save_json(output_root / "candidate_bank.json", bank)
    atomic_save_json(output_root / "summary.json", summary)
    (output_root / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


async def _generate_kimi_bank(
    config: Any,
    bundles: Sequence[dict[str, Any]],
    output_root: Path,
    *,
    initial_concurrency: int,
    maximum_concurrency: int,
) -> dict[str, Any]:
    client = Stage2ModelClient(config, generator_id="kimi")
    question_dir = output_root / "questions"
    checkpoint_dir = output_root.parent / "checkpoints" / "kimi_generation"

    async def one(bundle: dict[str, Any]) -> dict[str, Any]:
        qid = str(bundle["question_id"])
        input_hash = _generation_hash(config, bundle, "kimi")
        final_path = question_dir / f"{qid}.json"
        if final_path.exists():
            saved = load_json(final_path)
            recover_length = _needs_kimi_length_recovery(
                config, saved.get("synthesis")
            )
            if (
                saved.get("input_hash") == input_hash
                and saved.get("status") in TERMINAL_GENERATION
                and not recover_length
            ):
                logger.info("[Kimi生成] 题目=%s，复用终态=%s", qid, saved["status"])
                return saved

        diagnosis_path = checkpoint_dir / f"{qid}_diagnosis.json"
        diagnosis = None
        if diagnosis_path.exists():
            saved = load_json(diagnosis_path)
            if saved.get("input_hash") == input_hash and saved.get("diagnosis", {}).get("success"):
                diagnosis = saved["diagnosis"]
        if diagnosis is None:
            logger.info("[Kimi生成] 题目=%s，开始Thinking诊断", qid)
            with llm_request_context(qid, "role_compare/kimi/diagnosis"):
                diagnosis = await diagnose_principles(
                    client, bundle, config.stage2.max_principles_per_question
                )
            atomic_save_json(diagnosis_path, {
                "input_hash": input_hash, "diagnosis": diagnosis,
            })
        if not diagnosis.get("success"):
            record = {
                "status": "terminal_failed", "failure_stage": "diagnosis",
                "question_id": qid, "level": bundle["level"],
                "input_hash": input_hash, "diagnosis": diagnosis,
            }
            atomic_save_json(final_path, record)
            return record

        synthesis_path = checkpoint_dir / f"{qid}_synthesis.json"
        synthesis = None
        recover_with_large_budget = False
        if synthesis_path.exists():
            saved = load_json(synthesis_path)
            if saved.get("input_hash") == input_hash:
                saved_synthesis = saved.get("synthesis")
                if _needs_kimi_length_recovery(config, saved_synthesis):
                    recover_with_large_budget = True
                else:
                    synthesis = saved_synthesis
        if synthesis is None:
            selected_budget = (
                config.stage2.role_compare_kimi_generation_retry_max_tokens
                if recover_with_large_budget else None
            )
            logger.info(
                "[Kimi生成] 题目=%s，诊断完成，开始Thinking合成%s", qid,
                f"（截断恢复，max_tokens={selected_budget}）" if selected_budget else "",
            )
            with llm_request_context(qid, "role_compare/kimi/synthesis"):
                synthesis = await synthesize_principles(
                    client, bundle, config.stage2.max_principles_per_question,
                    diagnosis["diagnosis_text"],
                    max_tokens_override=selected_budget,
                )
            atomic_save_json(synthesis_path, {
                "input_hash": input_hash, "synthesis": synthesis,
            })

        if not synthesis.get("parse_success") or synthesis.get("schema_failures"):
            retry_path = checkpoint_dir / f"{qid}_format_retry.json"
            saved_retry = load_json(retry_path) if retry_path.exists() else {}
            if (
                saved_retry.get("input_hash") == input_hash
                and not _needs_kimi_length_recovery(config, saved_retry.get("synthesis"))
            ):
                synthesis = saved_retry["synthesis"]
            else:
                feedback = json.dumps({
                    "parse_success": synthesis.get("parse_success"),
                    "schema_failures": synthesis.get("schema_failures", []),
                }, ensure_ascii=False)
                with llm_request_context(qid, "role_compare/kimi/synthesis_format_retry"):
                    synthesis = await synthesize_principles(
                        client, bundle, config.stage2.max_principles_per_question,
                        diagnosis["diagnosis_text"], correction_feedback=feedback,
                    )
                atomic_save_json(retry_path, {
                    "input_hash": input_hash, "synthesis": synthesis,
                })

        families = synthesis.get("families") or []
        status = "complete" if families else (
            "no_reliable_principle" if synthesis.get("parse_success") else "terminal_failed"
        )
        record = {
            "status": status, "question_id": qid, "level": bundle["level"],
            "input_hash": input_hash, "bundle_hash": _sha(bundle),
            "diagnosis": diagnosis, "synthesis": synthesis,
        }
        atomic_save_json(final_path, record)
        logger.info("[Kimi生成] 题目=%s，完成，状态=%s，候选=%d", qid, status, len(families))
        return record

    records = await _warmup_map(
        bundles, one, initial=initial_concurrency, maximum=maximum_concurrency,
        description="Kimi Principle生成",
    )
    candidates = []
    for record in records:
        candidates.extend(_candidates_from_record(record, "kimi"))
    candidates.sort(key=lambda item: item["candidate_id"])
    bank = {
        "schema_version": ROLE_COMPARE_VERSION,
        "generator_id": "kimi", "question_count": len(records),
        "candidate_count": len(candidates), "candidates": candidates,
        "status_counts": dict(Counter(item["status"] for item in records)),
    }
    bank["candidate_bank_hash"] = _sha(bank)
    atomic_save_json(output_root / "candidate_bank.json", bank)
    atomic_save_json(output_root / "summary.json", {
        key: bank[key] for key in (
            "question_count", "candidate_count", "status_counts", "candidate_bank_hash"
        )
    })
    return bank


def deterministic_candidate_check(
    candidate: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    content, schema_errors = validate_principle_content(
        candidate.get("content"), require_extended=True
    )
    if content and not 2 <= len(content.get("required_evidence", [])) <= 5:
        schema_errors.append("required_evidence_must_contain_2_to_5_items")
    if content and not 1 <= len(content.get("failure_signals", [])) <= 4:
        schema_errors.append("failure_signals_must_contain_1_to_4_items")

    rollout_map = {
        str(item.get("rollout_id")): item
        for item in bundle.get("rollouts", []) if isinstance(item, dict)
    }
    reference_errors = []
    wrong_ids = [str(value) for value in candidate.get("supporting_wrong_rollout_ids", [])]
    correct_ids = [str(value) for value in candidate.get("contrast_correct_rollout_ids", [])]
    if not wrong_ids:
        reference_errors.append("missing_supporting_wrong_rollout_ids")
    if not correct_ids:
        reference_errors.append("missing_contrast_correct_rollout_ids")
    for rollout_id in wrong_ids:
        if rollout_id not in rollout_map or rollout_map[rollout_id].get("correct") is not False:
            reference_errors.append(f"invalid_wrong_rollout:{rollout_id}")
    for rollout_id in correct_ids:
        if rollout_id not in rollout_map or rollout_map[rollout_id].get("correct") is not True:
            reference_errors.append(f"invalid_correct_rollout:{rollout_id}")

    leakage = deterministic_leakage_check(
        content or dict(candidate.get("content") or {}),
        source_question=str(bundle.get("question_text", "")),
        expected_answer=str(bundle.get("expected_answer", "")),
    )
    hard_errors = list(schema_errors) + reference_errors
    if leakage.get("detected"):
        hard_errors.append("source_instance_or_answer_leakage")
    repair_flags = []
    if leakage.get("post_solution_trigger_detected"):
        repair_flags.append("post_solution_trigger")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "schema_errors": schema_errors,
        "reference_errors": reference_errors,
        "leakage": leakage,
        "repair_flags": repair_flags,
        "hard_invalid": bool(hard_errors),
        "hard_errors": hard_errors,
        "normalized_content": content,
    }


def _audit_decision(audit: Mapping[str, Any], deterministic: Mapping[str, Any]) -> str:
    if deterministic.get("hard_invalid"):
        return "reject"
    dimensions = audit.get("dimensions", {})
    failed = {
        name for name in AUDIT_DIMENSIONS
        if isinstance(dimensions.get(name), dict) and dimensions[name].get("pass") is False
    }
    if HARD_AUDIT_DIMENSIONS & failed:
        return "reject"
    if failed or deterministic.get("repair_flags"):
        return "repair"
    return "pass"


async def _semantic_review(
    client: Stage2ModelClient,
    bundle: dict[str, Any],
    candidate: dict[str, Any],
    *, reviewer_id: str,
) -> dict[str, Any]:
    evidence_ids = set(candidate.get("supporting_wrong_rollout_ids", [])) | set(
        candidate.get("contrast_correct_rollout_ids", [])
    )
    if reviewer_id == "kimi":
        prompt = _critic_prompt(bundle, candidate)
        with llm_request_context(candidate["candidate_id"], "role_compare/audit/kimi_critic"):
            critic = await client.auditor(
                prompt, auditor_id="kimi", thinking_enabled=True,
                max_tokens=client.config.role_compare_kimi_critic_max_tokens,
                json_mode=False,
            )
        critique = "\n".join(
            str(value) for value in (critic.get("reasoning", ""), critic.get("content", ""))
            if str(value).strip()
        )
        if not critique or critic.get("finish_reason") == "length":
            return {
                "status": "protocol_failed", "reviewer_id": reviewer_id,
                "critic_response": critic, "formatter_response": {},
                "protocol_errors": ["empty_or_truncated_critique"], "audit": {},
            }
        formatter_prompt = _formatter_prompt(critique, evidence_ids)
        with llm_request_context(candidate["candidate_id"], "role_compare/audit/deepseek_formatter"):
            final = await client.format_audit(formatter_prompt, formatter_id="deepseek")
        audit, errors = _audit_protocol_valid(final.get("parsed"), evidence_ids)
        return {
            "status": "complete" if not errors else "protocol_failed",
            "reviewer_id": reviewer_id, "critic_response": critic,
            "formatter_response": final, "protocol_errors": errors, "audit": audit,
        }

    evidence = [
        item for item in bundle.get("rollouts", [])
        if isinstance(item, dict) and item.get("rollout_id") in evidence_ids
    ]
    prompt = build_audit_prompt(bundle, candidate, evidence_ids, evidence)
    with llm_request_context(candidate["candidate_id"], "role_compare/audit/deepseek"):
        response = await client.auditor(
            prompt, auditor_id="deepseek", thinking_enabled=True,
            max_tokens=client.config.role_compare_audit_max_tokens, json_mode=True,
        )
    audit, errors = _audit_protocol_valid(response.get("parsed"), evidence_ids)
    formatter = {}
    if errors and response.get("raw"):
        formatter_prompt = _formatter_prompt(response["raw"], evidence_ids)
        with llm_request_context(candidate["candidate_id"], "role_compare/audit/deepseek_format_retry"):
            formatter = await client.format_audit(formatter_prompt, formatter_id="deepseek")
        audit, errors = _audit_protocol_valid(formatter.get("parsed"), evidence_ids)
    return {
        "status": "complete" if not errors else "protocol_failed",
        "reviewer_id": reviewer_id, "critic_response": response,
        "formatter_response": formatter, "protocol_errors": errors, "audit": audit,
    }


async def _cross_audit_bank(
    config: Any,
    *,
    bank: Mapping[str, Any],
    bundle_by_id: Mapping[str, dict[str, Any]],
    output_root: Path,
    initial_concurrency: int,
    maximum_concurrency: int,
) -> list[dict[str, Any]]:
    generator_id = str(bank["generator_id"])
    reviewer_id = "kimi" if generator_id == "deepseek" else "deepseek"
    reviewer_client = Stage2ModelClient(config, generator_id="deepseek")
    repair_client = Stage2ModelClient(config, generator_id=generator_id)

    async def one(candidate: dict[str, Any]) -> dict[str, Any]:
        path = output_root / generator_id / f"{candidate['candidate_id']}.json"
        bundle = bundle_by_id[str(candidate["question_id"])]
        input_hash = _sha({
            "version": ROLE_COMPARE_VERSION, "candidate": candidate,
            "bundle": bundle, "reviewer_id": reviewer_id,
        })
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash and saved.get("status") == "terminal":
                return saved

        deterministic = deterministic_candidate_check(candidate, bundle)
        check_path = output_root.parent / "deterministic_checks" / generator_id / f"{candidate['candidate_id']}.json"
        atomic_save_json(check_path, deterministic)
        if deterministic["hard_invalid"]:
            record = {
                "status": "terminal", "input_hash": input_hash,
                "candidate_id": candidate["candidate_id"], "generator_id": generator_id,
                "reviewer_id": reviewer_id, "original_candidate": candidate,
                "deterministic": deterministic, "initial_review": None,
                "initial_decision": "reject", "repair": None,
                "reaudit": None, "final_candidate": candidate,
                "final_decision": "reject", "model_calls": 0,
            }
            atomic_save_json(path, record)
            return record

        review = await _semantic_review(
            reviewer_client, bundle, candidate, reviewer_id=reviewer_id
        )
        if review["status"] != "complete":
            decision = "protocol_failed"
        else:
            decision = _audit_decision(review["audit"], deterministic)

        repair = None
        reaudit = None
        final_candidate = candidate
        final_decision = decision
        if decision == "repair":
            with llm_request_context(candidate["candidate_id"], f"role_compare/repair/{generator_id}"):
                repair = await repair_principle(
                    repair_client, bundle, candidate, review["audit"]
                )
            if repair.get("success"):
                final_candidate = {
                    **candidate,
                    "content": repair["content"],
                    "parent_content": candidate["content"],
                    "version": "repaired",
                }
                atomic_save_json(
                    output_root.parent / "repaired_candidates" / generator_id /
                    f"{candidate['candidate_id']}.json",
                    final_candidate,
                )
                repaired_check = deterministic_candidate_check(final_candidate, bundle)
                if repaired_check["hard_invalid"]:
                    final_decision = "reject"
                    reaudit = {"status": "skipped_hard_invalid", "deterministic": repaired_check}
                else:
                    reaudit = await _semantic_review(
                        reviewer_client, bundle, final_candidate, reviewer_id=reviewer_id
                    )
                    if reaudit["status"] == "complete":
                        next_decision = _audit_decision(reaudit["audit"], repaired_check)
                        final_decision = "pass" if next_decision == "pass" else "reject"
                    else:
                        final_decision = "protocol_failed"
            else:
                final_decision = "reject"

        record = {
            "status": "terminal", "input_hash": input_hash,
            "candidate_id": candidate["candidate_id"], "generator_id": generator_id,
            "reviewer_id": reviewer_id, "original_candidate": candidate,
            "deterministic": deterministic, "initial_review": review,
            "initial_decision": decision, "repair": repair, "reaudit": reaudit,
            "final_candidate": final_candidate, "final_decision": final_decision,
            "model_calls": 1 + int(bool(review.get("formatter_response"))) +
                int(repair is not None) + int(reaudit is not None),
        }
        atomic_save_json(path, record)
        return record

    return await _warmup_map(
        list(bank["candidates"]), one,
        initial=initial_concurrency, maximum=maximum_concurrency,
        description=f"交叉盲审 {generator_id}→{reviewer_id}",
    )


def render_six_field_card(content: Mapping[str, Any]) -> str:
    evidence = "\n".join(
        f"  - {item}" for item in content.get("required_evidence", [])
    ) or "  - None"
    return "\n".join([
        "[Optional Conditional Experience]",
        "This is fallible advice. First decide from the current problem whether its conditions hold; ignore it otherwise.",
        f"When to consider: {content.get('trigger_condition', '')}",
        "Required evidence to check:", evidence,
        f"Do not use if: {content.get('exclusion_condition', '')}",
        f"Failure to avoid: {content.get('failure_mode', '')}",
        f"Suggested action: {content.get('action_guidance', '')}",
        f"Verification: {content.get('verification_step', '')}",
    ])


async def _evaluate_role_candidates(
    config: Any,
    *,
    bundles: Sequence[dict[str, Any]],
    banks: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    output_root: Path,
    ports: Sequence[int],
    model: str,
    concurrency: int,
    timeout: int,
) -> dict[str, Any]:
    bundle_by_id = {str(item["question_id"]): item for item in bundles}
    seeds = tuple(config.stage2.role_compare_seeds)
    temperature = float(config.stage2.role_compare_temperature)
    urls = _healthy_actor_urls(ports)
    runner = _PairedRunner(
        config, output_dir=output_root, urls=urls, model=model,
        concurrency=concurrency, timeout=timeout,
    )
    baseline_tasks = {}
    for bundle in bundles:
        qid = str(bundle["question_id"])
        for seed in seeds:
            baseline_tasks[(qid, seed)] = asyncio.create_task(runner.solve(
                phase="role_compare", arm="baseline", question_id=qid,
                question=bundle["question_text"], expected_answer=bundle["expected_answer"],
                temperature=temperature, seed=seed,
            ))
    await _gather_with_progress(
        list(baseline_tasks.values()), description="Llama Baseline", unit="次推理"
    )

    audit_by_id = {str(item["candidate_id"]): item for item in audits}
    variants = []
    for bank in banks:
        for candidate in bank["candidates"]:
            variants.append(("raw", candidate))
            audit = audit_by_id.get(str(candidate["candidate_id"]), {})
            repaired = audit.get("final_candidate")
            if (
                isinstance(repaired, dict)
                and repaired.get("content") != candidate.get("content")
            ):
                variants.append(("repaired", repaired))

    async def evaluate(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        version, candidate = item
        qid = str(candidate["question_id"])
        bundle = bundle_by_id[qid]
        baseline = [await baseline_tasks[(qid, seed)] for seed in seeds]
        principle_text = render_six_field_card(candidate["content"])
        treatment = await asyncio.gather(*(
            runner.solve(
                phase="role_compare", arm=f"{candidate['generator_id']}_{version}",
                question_id=qid, question=bundle["question_text"],
                expected_answer=bundle["expected_answer"],
                temperature=temperature, seed=seed,
                principle_text=principle_text, candidate=candidate,
            )
            for seed in seeds
        ))
        baseline_correct = sum(bool(row["correct"]) for row in baseline)
        guided_correct = sum(bool(row["correct"]) for row in treatment)
        delta = guided_correct - baseline_correct
        return {
            "candidate_id": candidate["candidate_id"],
            "generator_id": candidate["generator_id"], "version": version,
            "question_id": qid, "level": candidate["level"],
            "baseline_correct_at_3": baseline_correct,
            "guided_correct_at_3": guided_correct, "delta_correct": delta,
            "effect_label": "positive" if delta > 0 else "negative" if delta < 0 else "unchanged",
            "paired": _paired_counts(baseline, treatment),
            "solver_use_count": sum(row.get("experience_decision") == "use" for row in treatment),
            "solver_ignore_count": sum(row.get("experience_decision") == "ignore" for row in treatment),
            "privacy_violations": sorted({
                violation for row in treatment for violation in row.get("privacy_violations", [])
            }),
        }

    effects = await _gather_with_progress(
        [evaluate(item) for item in variants],
        description="Llama Principle配对验证", unit="个版本",
    )
    atomic_save_json(output_root / "candidate_effects.json", effects)
    return {"effects": effects, "rollout_count": len(runner.records)}


def _summarize_generator(
    generator_id: str,
    bank: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = [item for item in bank["candidates"] if item["generator_id"] == generator_id]
    candidate_ids = {item["candidate_id"] for item in candidates}
    raw = [
        item for item in effects
        if item["candidate_id"] in candidate_ids and item["version"] == "raw"
    ]
    accepted_ids = {
        item["candidate_id"] for item in audits
        if item.get("generator_id") == generator_id and item.get("final_decision") == "pass"
    }
    accepted_effects = []
    by_id_version = {(item["candidate_id"], item["version"]): item for item in effects}
    for candidate_id in accepted_ids:
        accepted_effects.append(
            by_id_version.get((candidate_id, "repaired"))
            or by_id_version.get((candidate_id, "raw"))
        )
    accepted_effects = [item for item in accepted_effects if item]

    def effect_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        positive_questions = {item["question_id"] for item in rows if item["delta_correct"] > 0}
        negative_questions = {item["question_id"] for item in rows if item["delta_correct"] < 0}
        return {
            "candidate_count": len(rows),
            "positive_candidates": sum(item["delta_correct"] > 0 for item in rows),
            "negative_candidates": sum(item["delta_correct"] < 0 for item in rows),
            "unchanged_candidates": sum(item["delta_correct"] == 0 for item in rows),
            "positive_question_count": len(positive_questions),
            "negative_question_count": len(negative_questions),
            "delta_correct_sum": sum(int(item["delta_correct"]) for item in rows),
            "gain": sum(int(item["paired"]["gain"]) for item in rows),
            "loss": sum(int(item["paired"]["loss"]) for item in rows),
        }

    branch_audits = [item for item in audits if item.get("generator_id") == generator_id]
    false_rejects = sum(
        by_id_version.get((item["candidate_id"], "raw"), {}).get("delta_correct", 0) > 0
        and item.get("final_decision") == "reject"
        for item in branch_audits
    )
    return {
        "question_count": int(bank.get("question_count", 0)),
        "candidate_count": len(candidates),
        "generation_status_counts": bank.get("status_counts", {}),
        "audit_decision_counts": dict(Counter(
            item.get("final_decision", "unknown") for item in branch_audits
        )),
        "hard_invalid_count": sum(item.get("deterministic", {}).get("hard_invalid", False) for item in branch_audits),
        "audit_false_reject_positive_raw_count": false_rejects,
        "raw_effect": effect_stats(raw),
        "accepted_effect": effect_stats(accepted_effects),
    }


def _usage_totals(paths: Sequence[Path]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen: set[str] = set()

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict):
                prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
                total = int(usage.get("total_tokens", prompt + completion) or prompt + completion)
                provider_response = value.get("provider_response")
                identity = (
                    "provider:" + _sha(provider_response)
                    if isinstance(provider_response, dict) and provider_response
                    else f"path:{path}:{prompt}:{completion}:{total}"
                )
                if total and identity not in seen:
                    seen.add(identity)
                    totals["prompt_tokens"] += prompt
                    totals["completion_tokens"] += completion
                    totals["total_tokens"] += total
            for key, item in value.items():
                if key != "provider_response":
                    walk(item, f"{path}/{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}/{index}")

    for file_path in paths:
        if file_path.exists():
            walk(load_json(file_path), str(file_path))
    return totals


def _write_report(root: Path, summary: Mapping[str, Any]) -> None:
    d = summary["generators"]["deepseek"]
    k = summary["generators"]["kimi"]
    lines = [
        "# 8题双生成器交叉审查报告", "",
        "> 本报告仅用于协议筛选，不是论文正式统计结论。", "",
        "## 总览", "",
        "| 路线 | 候选数 | 审查后接受 | Raw正向题 | Raw负向题 | 审查后净变化 |", "|---|---:|---:|---:|---:|---:|",
        f"| DeepSeek生成→Kimi审查 | {d['candidate_count']} | {d['audit_decision_counts'].get('pass', 0)} | {d['raw_effect']['positive_question_count']} | {d['raw_effect']['negative_question_count']} | {d['accepted_effect']['delta_correct_sum']} |",
        f"| Kimi生成→DeepSeek审查 | {k['candidate_count']} | {k['audit_decision_counts'].get('pass', 0)} | {k['raw_effect']['positive_question_count']} | {k['raw_effect']['negative_question_count']} | {k['accepted_effect']['delta_correct_sum']} |",
        "", "## 解释规则", "",
        "- Raw有效但被拒绝：审查可能错杀。",
        "- 修复版优于Raw：审查与修复链路有正向作用。",
        "- Raw与修复版均无效：优先检查Principle生成内容或固定注入模板。",
        "- 8题只能得到临时优胜路线，40题复验后才能冻结正式协议。",
        "", f"临时选择：**{summary['provisional_selection']['winner'] or '未产生明确胜者'}**",
        f"原因：{summary['provisional_selection']['reason']}", "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _select_provisional(deepseek: Mapping[str, Any], kimi: Mapping[str, Any]) -> dict[str, str]:
    eligible = []
    for name, stats in (("deepseek", deepseek), ("kimi", kimi)):
        if stats["hard_invalid_count"] == 0 or stats["audit_decision_counts"].get("pass", 0):
            eligible.append((name, stats))
    if not eligible:
        return {"winner": "", "reason": "两条路线均没有通过硬质量门槛的候选"}
    ranked = sorted(eligible, key=lambda pair: (
        -pair[1]["accepted_effect"]["positive_question_count"],
        pair[1]["accepted_effect"]["negative_question_count"],
        -pair[1]["accepted_effect"]["delta_correct_sum"],
        0 if pair[0] == "deepseek" else 1,
    ))
    winner, stats = ranked[0]
    if len(ranked) == 2:
        first_key = (
            stats["accepted_effect"]["positive_question_count"],
            stats["accepted_effect"]["negative_question_count"],
            stats["accepted_effect"]["delta_correct_sum"],
        )
        other = ranked[1][1]
        other_key = (
            other["accepted_effect"]["positive_question_count"],
            other["accepted_effect"]["negative_question_count"],
            other["accepted_effect"]["delta_correct_sum"],
        )
        if first_key == other_key:
            return {"winner": "deepseek", "reason": "效果并列，按成本规则暂选DeepSeek，等待40题复验"}
    return {"winner": winner, "reason": "按正向题覆盖、负向题数量、净变化和成本顺序选择"}


async def run_role_compare(
    config: Any,
    *,
    test_root: Path,
    ports: Sequence[int],
    model: str,
    model_concurrency: int = 8,
    validation_concurrency: int = 4,
    timeout: int = 900,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the complete eight-question role comparison with checkpoints."""
    del resume  # Hash-compatible checkpoint reuse is always safe and mandatory.
    test_root = Path(test_root)
    role_root = test_root / "role_compare_v1"
    for relative in (
        "kimi_generation/questions", "deterministic_checks/deepseek",
        "deterministic_checks/kimi", "cross_audits/deepseek",
        "cross_audits/kimi", "repaired_candidates/deepseek",
        "repaired_candidates/kimi", "llama_evaluation", "checkpoints",
    ):
        (role_root / relative).mkdir(parents=True, exist_ok=True)
    bundles_path = test_root / "source_bundles" / "source40_bundles.json"
    deepseek_bank_path = test_root / "raw_candidates" / "candidate_bank.json"
    if not bundles_path.exists() or not deepseek_bank_path.exists():
        raise FileNotFoundError("role-compare requires completed smoke8 bundles and DeepSeek Candidate Bank")
    bundles = load_json(bundles_path)
    deepseek_bank = load_json(deepseek_bank_path)
    if len(bundles) != 8 or deepseek_bank.get("candidate_count") != 17:
        raise RuntimeError(
            f"Expected frozen 8-question/17-candidate DeepSeek input; got "
            f"questions={len(bundles)} candidates={deepseek_bank.get('candidate_count')}"
        )
    if not os.environ.get(config.stage2.generator_key_env):
        raise RuntimeError(f"Missing {config.stage2.generator_key_env}; source stage2.env first")
    if not os.environ.get(config.stage2.auditor_key_env):
        raise RuntimeError(f"Missing {config.stage2.auditor_key_env}; source stage2.env first")
    # Fail before any paid remote call when the local causal validator is not
    # ready.  A resumed run performs the same check but reuses every completed
    # remote checkpoint.
    _healthy_actor_urls(ports)

    # Preserve the original bank byte-for-byte.  The comparison manifest pins
    # its hash instead of copying or rewriting it.
    manifest = {
        "schema_version": ROLE_COMPARE_VERSION,
        "question_count": 8, "seeds": list(config.stage2.role_compare_seeds),
        "temperature": config.stage2.role_compare_temperature,
        "bundle_hash": _sha(bundles),
        "deepseek_candidate_bank_path": str(deepseek_bank_path),
        "deepseek_candidate_bank_hash": deepseek_bank["candidate_bank_hash"],
        "deepseek_candidate_count": deepseek_bank["candidate_count"],
        "ports": list(ports), "model": model,
        "model_concurrency_initial": min(
            config.stage2.role_compare_initial_concurrency, model_concurrency
        ),
        "model_concurrency_max": model_concurrency,
        "validation_concurrency": validation_concurrency,
    }
    atomic_save_json(role_root / "manifest.json", manifest)
    logger.info("[角色对照 1/3] 生成Kimi候选；DeepSeek 17条候选只读复用")
    kimi_bank = await _generate_kimi_bank(
        config, bundles, role_root / "kimi_generation",
        initial_concurrency=min(config.stage2.role_compare_initial_concurrency, model_concurrency),
        maximum_concurrency=model_concurrency,
    )

    # Canonicalize only the comparison view; never mutate the source bank.
    deepseek_compare_bank = {
        **deepseek_bank,
        "generator_id": "deepseek",
        "question_count": 8,
        "status_counts": {"complete": 8},
        "candidates": [
            {**item, "generator_id": "deepseek"}
            for item in deepseek_bank["candidates"]
        ],
    }
    bundle_by_id = {str(item["question_id"]): item for item in bundles}
    logger.info("[角色对照 2/3] 两条路线并行交叉盲审；每候选只审查一次")
    deepseek_audits, kimi_audits = await asyncio.gather(
        _cross_audit_bank(
            config, bank=deepseek_compare_bank, bundle_by_id=bundle_by_id,
            output_root=role_root / "cross_audits",
            initial_concurrency=min(config.stage2.role_compare_initial_concurrency, model_concurrency),
            maximum_concurrency=model_concurrency,
        ),
        _cross_audit_bank(
            config, bank=kimi_bank, bundle_by_id=bundle_by_id,
            output_root=role_root / "cross_audits",
            initial_concurrency=min(config.stage2.role_compare_initial_concurrency, model_concurrency),
            maximum_concurrency=model_concurrency,
        ),
    )
    audits = deepseek_audits + kimi_audits
    atomic_save_json(role_root / "cross_audits" / "all_records.json", audits)

    logger.info("[角色对照 3/3] Llama运行Baseline、Raw及实际修复版本，seeds=42,43,44")
    evaluation = await _evaluate_role_candidates(
        config, bundles=bundles, banks=(deepseek_compare_bank, kimi_bank), audits=audits,
        output_root=role_root / "llama_evaluation", ports=ports, model=model,
        concurrency=validation_concurrency, timeout=timeout,
    )
    effects = evaluation["effects"]
    ds_summary = _summarize_generator("deepseek", deepseek_compare_bank, audits, effects)
    kimi_summary = _summarize_generator("kimi", kimi_bank, audits, effects)
    usage_paths = list((role_root / "kimi_generation").rglob("*.json"))
    usage_paths += list((role_root / "checkpoints").rglob("*.json"))
    usage_paths += list((role_root / "cross_audits").glob("*/*.json"))
    usage = _usage_totals(usage_paths)
    atomic_save_json(role_root / "usage_summary.json", usage)
    summary = {
        "schema_version": ROLE_COMPARE_VERSION,
        "manifest_hash": _sha(manifest),
        "generators": {"deepseek": ds_summary, "kimi": kimi_summary},
        "llama_rollout_count": evaluation["rollout_count"],
        "usage": usage,
        "provisional_selection": _select_provisional(ds_summary, kimi_summary),
        "formal_claim_allowed": False,
        "next_gate": "confirm_provisional_winner_on_40_source_questions",
    }
    atomic_save_json(role_root / "summary.json", summary)
    _write_report(role_root, summary)
    return summary
