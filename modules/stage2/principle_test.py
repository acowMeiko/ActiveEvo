"""Recoverable 40-source/40-variant Principle calibration.

This is an experiment harness, not the production Memory Factory.  It never
writes to ``memory/`` and deliberately bypasses SPD/Neighbor Validation because
source/variant pairs are frozen before Principle generation.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from tqdm.auto import tqdm

from benchmarks.matcher import check_answer
from core.request_context import llm_request_context
from modules.execute_module import (
    ExecuteModule,
    SOLVER_V1_INITIAL_PLAN_PROMPT,
    SOLVER_V1_PROMPT_HASHES,
    SOLVER_V1_PROTOCOL,
    filter_healthy_urls,
)
from modules.principle_schema import (
    SCHEMA_VERSION,
    render_solver_condition_card,
    render_solver_memory,
    validate_condition_card,
    validate_principle_content,
)
from modules.stage2.contracts import build_learning_bundles, validate_learning_bundle
from modules.stage2.model_client import Stage2ModelClient, parse_json_object
from modules.stage2.principle_generation import (
    build_diagnosis_prompt,
    build_repair_prompt,
    build_synthesis_prompt,
    diagnose_principles,
    repair_principle,
    synthesize_principles,
)
from modules.stage2.quality_audit import (
    AUDIT_DIMENSIONS,
    build_audit_prompt,
    deterministic_leakage_check,
)
from utils.io_utils import atomic_save_json, load_json


logger = logging.getLogger(__name__)

LEVELS = ("level1", "level2", "level3", "level4")
AUDIT_PROTOCOLS = ("raw_control", "kimi_single", "kimi_critic_formatter")
INTERVENTION_ARMS = ("direct_short", "pre_condition", "post_plan")
HARD_AUDIT_DIMENSIONS = {
    "source_fidelity", "mathematical_validity", "instance_leakage",
}
DEV8_BY_LEVEL = {
    "level1": ("544", "566"),
    "level2": ("5622", "1781"),
    "level3": ("580", "2437"),
    "level4": ("2929", "352"),
}

PHASE_LABELS_ZH = {
    "source": "来源题测试",
    "transfer": "结构变式题迁移测试",
    "robustness_source": "来源题温度稳健性测试",
    "robustness_transfer": "变式题温度稳健性测试",
}


def _phase_label_zh(phase: str) -> str:
    return PHASE_LABELS_ZH.get(phase, phase)


async def _gather_with_progress(
    awaitables: Sequence[Any],
    *,
    description: str,
    unit: str,
) -> list[Any]:
    """Gather concurrent work while preserving input order and showing real progress."""
    items = list(awaitables)
    if not items:
        return []

    async def indexed(index: int, awaitable: Any) -> tuple[int, Any]:
        return index, await awaitable

    tasks = [asyncio.create_task(indexed(index, item)) for index, item in enumerate(items)]
    results: list[Any] = [None] * len(tasks)
    interactive = sys.stderr.isatty()
    logger.info("[%s] 开始，总数=%d", description, len(tasks))
    progress = tqdm(
        total=len(tasks), desc=description, unit=unit, dynamic_ncols=True,
        disable=not interactive, leave=True,
    )
    try:
        for completed in asyncio.as_completed(tasks):
            index, result = await completed
            results[index] = result
            progress.update(1)
            if not interactive and (progress.n == 1 or progress.n % 10 == 0):
                logger.info("[%s] 进度=%d/%d", description, progress.n, len(tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        progress.close()
    logger.info("[%s] 完成=%d/%d", description, len(tasks), len(tasks))
    return results


def current_stage2_protocol(
    config: Any,
    *,
    audit_protocol: str,
    intervention_arm: str = "",
) -> dict[str, Any]:
    """Describe the exact executable protocol without including API keys."""
    if audit_protocol not in AUDIT_PROTOCOLS:
        raise ValueError(f"Unknown audit protocol: {audit_protocol}")
    if intervention_arm and intervention_arm not in INTERVENTION_ARMS:
        raise ValueError(f"Unknown intervention arm: {intervention_arm}")
    prompt_sources = {
        "diagnosis": inspect.getsource(build_diagnosis_prompt),
        "synthesis": inspect.getsource(build_synthesis_prompt),
        "repair": inspect.getsource(build_repair_prompt),
        "quality_audit": inspect.getsource(build_audit_prompt),
        "kimi_critic": inspect.getsource(_critic_prompt),
        "condition_card": CONDITION_CARD_PROMPT,
    }
    descriptor = {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "provider": "deepseek",
            "model": config.stage2.generator_model,
            "analysis_max_tokens": config.stage2.generation_analysis_max_tokens,
            "synthesis_max_tokens": config.stage2.generation_max_tokens,
            "max_principles_per_question": config.stage2.max_principles_per_question,
        },
        "auditor": {
            "provider": "kimi",
            "model": config.stage2.auditor_model,
            "protocol": audit_protocol,
            "repeats": config.stage2.principle_test_audit_repeats,
            "critic_max_tokens": config.stage2.audit_max_tokens,
            "formatter_max_tokens": config.stage2.audit_formatter_max_tokens,
        },
        "intervention_arm": intervention_arm,
        "prompt_hashes": {
            name: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for name, source in prompt_sources.items()
        },
    }
    descriptor["generation_protocol_hash"] = _json_hash({
        "schema_version": descriptor["schema_version"],
        "generator": descriptor["generator"],
        "prompt_hashes": {
            name: descriptor["prompt_hashes"][name]
            for name in ("diagnosis", "synthesis")
        },
    })
    descriptor["candidate_protocol_hash"] = _json_hash({
        "generation_protocol_hash": descriptor["generation_protocol_hash"],
        "auditor": descriptor["auditor"],
        "prompt_hashes": {
            name: descriptor["prompt_hashes"][name]
            for name in ("repair", "quality_audit", "kimi_critic")
        },
    })
    descriptor["protocol_hash"] = _json_hash(descriptor)
    return descriptor


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_key(question_id: str, salt: str) -> str:
    return hashlib.sha256(f"{question_id}{salt}".encode("utf-8")).hexdigest()


def _candidate_id(question_id: str, family_id: str) -> str:
    digest = hashlib.sha256(f"deepseek:{question_id}:{family_id}".encode()).hexdigest()[:16]
    return f"candidate_{digest}"


def generation_input_hash(config: Any, bundle: Mapping[str, Any]) -> str:
    """Hash every input that can change a question's Principle generation."""
    diagnosis_hash = hashlib.sha256(build_diagnosis_prompt(
        dict(bundle), config.stage2.max_principles_per_question
    ).encode("utf-8")).hexdigest()
    return _json_hash({
        "bundle": bundle,
        "diagnosis_prompt_hash": diagnosis_hash,
        "generator_model": config.stage2.generator_model,
        "analysis_max_tokens": config.stage2.generation_analysis_max_tokens,
        "synthesis_max_tokens": config.stage2.generation_max_tokens,
        "max_principles": config.stage2.max_principles_per_question,
    })


def candidates_from_generation(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive canonical candidates from one terminal generation record."""
    candidates = []
    for family in (record.get("synthesis") or {}).get("families", []):
        candidates.append({
            "candidate_id": _candidate_id(str(record["question_id"]), family["family_id"]),
            "question_id": str(record["question_id"]),
            "level": record["level"],
            "family_id": family["family_id"],
            "root_cause": family["root_cause"],
            "supporting_wrong_rollout_ids": family["supporting_wrong_rollout_ids"],
            "contrast_correct_rollout_ids": family["contrast_correct_rollout_ids"],
            "content": family["content"],
            "generator_id": "deepseek",
        })
    return candidates


def audit_input_hash(
    config: Any,
    candidate: Mapping[str, Any],
    bundle: Mapping[str, Any],
    protocol: str,
) -> str:
    """Hash every candidate, evidence item, and setting that can change an audit."""
    return _json_hash({
        "candidate": candidate,
        "bundle": bundle,
        "protocol": protocol,
        "audit_dimensions": AUDIT_DIMENSIONS,
        "repeat_count": config.stage2.principle_test_audit_repeats,
        "kimi_model": config.stage2.auditor_model,
        "deepseek_model": config.stage2.generator_model,
    })


def _archive_stale(path: Path, *, output_root: Path, reason: str) -> None:
    if not path.exists():
        return
    payload = load_json(path)
    digest = _json_hash(payload)[:12]
    atomic_save_json(output_root / f"{path.stem}_{digest}.json", {
        "reason": reason,
        "original_path": str(path),
        "payload": payload,
    })


def _stage1_packet_path(config: Any, backbone: str, level: str) -> Path:
    return (
        config.paths.output_dir / "stage1" / "02_learning_packets" /
        backbone / f"{level}_learning_packets.json"
    )


def _eligible_bundle(bundle: Mapping[str, Any], level: str) -> tuple[bool, list[str]]:
    reasons = validate_learning_bundle(dict(bundle))
    if not bundle.get("evidence_complete"):
        reasons.append("evidence_incomplete")
    signal = bundle.get("capability_signal") or {}
    assigned = str(signal.get("assigned_level") or bundle.get("level") or "")
    if assigned != level:
        reasons.append(f"assigned_level_mismatch:{assigned}")
    if any(
        str(item.get("match_reason", "")) == "unsafe_symbolic_expected"
        for item in bundle.get("rollouts", []) if isinstance(item, dict)
    ):
        reasons.append("unsafe_expected_answer")
    if not str(bundle.get("expected_answer", "")).strip():
        reasons.append("missing_expected_answer")
    return not reasons, reasons


def _load_bundles_by_level(config: Any, backbone: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    result: dict[str, list[dict]] = {}
    hashes: dict[str, str] = {}
    for level in LEVELS:
        path = _stage1_packet_path(config, backbone, level)
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage-1 packets: {path}")
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Stage-1 packet file must contain a list: {path}")
        bundles = build_learning_bundles(payload, level)
        eligible = [bundle for bundle in bundles if _eligible_bundle(bundle, level)[0]]
        result[level] = eligible
        hashes[str(path)] = _file_hash(path)
    return result, hashes


def prepare_principle_test(
    config: Any,
    *,
    test_root: Path,
    backbone: str = "llama",
    sources_per_level: int = 10,
    reserve_per_level: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Freeze 10 sources and 5 reserves per Level, preserving the historical dev8."""
    test_root = Path(test_root)
    manifest_dir = test_root / "manifests"
    source_dir = test_root / "source_bundles"
    manifest_path = manifest_dir / "source40.json"
    bundles_path = source_dir / "source40_bundles.json"
    if manifest_path.exists() and bundles_path.exists():
        return load_json(manifest_path)

    by_level, input_hashes = _load_bundles_by_level(config, backbone)
    selected_by_level: dict[str, list[str]] = {}
    reserve_by_level: dict[str, list[str]] = {}
    selected_bundles: list[dict] = []
    reserve_bundles: list[dict] = []
    salt = f"stage2_principle_test_{seed}"
    for level in LEVELS:
        pool = {str(item["question_id"]): item for item in by_level[level]}
        required = list(DEV8_BY_LEVEL[level])
        missing = [qid for qid in required if qid not in pool]
        if missing:
            raise ValueError(f"Historical dev questions are not eligible in {level}: {missing}")
        remainder = sorted(
            (qid for qid in pool if qid not in required),
            key=lambda qid: _stable_key(qid, salt),
        )
        need = sources_per_level - len(required)
        if len(remainder) < need + reserve_per_level:
            raise ValueError(
                f"{level} has only {len(remainder)} additional eligible bundles; "
                f"need {need + reserve_per_level}"
            )
        selected_ids = required + remainder[:need]
        reserve_ids = remainder[need:need + reserve_per_level]
        selected_by_level[level] = selected_ids
        reserve_by_level[level] = reserve_ids
        selected_bundles.extend(pool[qid] for qid in selected_ids)
        reserve_bundles.extend(pool[qid] for qid in reserve_ids)

    manifest = {
        "schema_version": "principle_test.1",
        "backbone": backbone,
        "seed": seed,
        "selection_salt": salt,
        "sources_per_level": sources_per_level,
        "reserve_per_level": reserve_per_level,
        "source_question_ids_by_level": selected_by_level,
        "reserve_question_ids_by_level": reserve_by_level,
        "source_question_ids": [qid for level in LEVELS for qid in selected_by_level[level]],
        "reserve_question_ids": [qid for level in LEVELS for qid in reserve_by_level[level]],
        "stage1_input_hashes": input_hashes,
        "source_bundle_hashes": {
            str(item["question_id"]): _json_hash(item) for item in selected_bundles
        },
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    atomic_save_json(manifest_path, manifest)
    atomic_save_json(bundles_path, selected_bundles)
    atomic_save_json(source_dir / "reserve20_bundles.json", reserve_bundles)
    atomic_save_json(manifest_dir / "reserve20.json", {
        "question_ids_by_level": reserve_by_level,
        "question_ids": manifest["reserve_question_ids"],
        "bundle_hashes": {str(item["question_id"]): _json_hash(item) for item in reserve_bundles},
    })
    atomic_save_json(manifest_dir / "source40_hashes.json", {
        "manifest_hash": manifest["manifest_hash"],
        "stage1_input_hashes": input_hashes,
        "source_bundle_hashes": manifest["source_bundle_hashes"],
    })
    return manifest


def _required_keys_present() -> dict[str, bool]:
    return {
        "deepseek": bool(os.environ.get("ACTIVEEVO_GENERATOR_API_KEY", "").strip()),
        "kimi": bool(os.environ.get("ACTIVEEVO_AUDITOR_API_KEY", "").strip()),
    }


async def run_provider_probe(config: Any, *, test_root: Path, repeats: int = 3) -> dict[str, Any]:
    keys = _required_keys_present()
    if not all(keys.values()):
        raise RuntimeError(f"Missing Stage-2 API key(s): {[name for name, ok in keys.items() if not ok]}")
    client = Stage2ModelClient(config, generator_id="deepseek")
    records = []
    for index in range(repeats):
        deepseek = await client.generator(
            'Return raw JSON only: {"provider":"deepseek","protocol_ok":true}',
            thinking_enabled=False,
            max_tokens=512,
        )
        kimi = await client.auditor(
            'Return JSON only: {"provider":"kimi","protocol_ok":true}',
            auditor_id="kimi", thinking_enabled=True, max_tokens=1024, json_mode=True,
        )
        formatter = await client.format_audit(
            'Format this fixed judgment as raw JSON without changing it: protocol_ok=true.'
        )
        record = {
            "repeat": index + 1,
            "deepseek": {
                "passed": deepseek.get("parsed", {}).get("protocol_ok") is True,
                "reasoning_nonempty": bool(deepseek.get("reasoning")),
                "content_nonempty": bool(deepseek.get("content")),
                "finish_reason": deepseek.get("finish_reason"),
                "response": deepseek,
            },
            "kimi_critic": {
                "passed": kimi.get("parsed", {}).get("protocol_ok") is True,
                "reasoning_nonempty": bool(kimi.get("reasoning")),
                "content_nonempty": bool(kimi.get("content")),
                "finish_reason": kimi.get("finish_reason"),
                "response": kimi,
            },
            "kimi_formatter": {
                "passed": formatter.get("parsed", {}).get("protocol_ok") is True,
                "reasoning_nonempty": bool(formatter.get("reasoning")),
                "content_nonempty": bool(formatter.get("content")),
                "finish_reason": formatter.get("finish_reason"),
                "response": formatter,
            },
        }
        records.append(record)
    summary = {
        "repeats": repeats,
        "all_passed": all(
            row[name]["passed"]
            for row in records for name in ("deepseek", "kimi_critic", "kimi_formatter")
        ),
        "records": records,
    }
    probe_dir = Path(test_root) / "provider_probe"
    atomic_save_json(probe_dir / "probe_records.json", records)
    atomic_save_json(probe_dir / "probe_summary.json", {
        "repeats": repeats, "all_passed": summary["all_passed"],
        "counts": {
            name: sum(row[name]["passed"] for row in records)
            for name in ("deepseek", "kimi_critic", "kimi_formatter")
        },
    })
    if not summary["all_passed"]:
        raise RuntimeError("DeepSeek/Kimi provider probe did not pass 3/3")
    return summary


VARIANT_PROMPT = """You construct one blind transfer-test problem. You may inspect only
the source problem below. Do not infer or mention any stored Principle, model trajectory,
or experiment result. Preserve the mathematical operation, constraint topology, objective
type, and reasoning skeleton, while changing the surface setting, entities, wording, and
all non-essential numbers. The new problem must be self-contained, unambiguous, solvable,
and have an independently checkable exact final answer. Do not merely paraphrase.

Return raw JSON only:
{{
  "variant_question": "...",
  "variant_answer": "...",
  "structure_spec": {{
    "core_skill": "...",
    "context_constraints": "...",
    "task_objective": "...",
    "reasoning_skeleton": ["..."]
  }},
  "changed_elements": ["..."]
}}

Source problem:
{question}
"""


VARIANT_VERIFY_PROMPT = """Independently verify a proposed transfer-test problem. You do
not have access to any Principle or model trajectory. Solve the proposed problem yourself,
compare its mathematical structure with the source problem, and judge whether it is a
genuine surface/parameter change rather than a paraphrase. The proposed answer must match
your independent solution.

Return raw JSON only:
{{
  "structure_match": true,
  "answer_correct": true,
  "surface_difference_sufficient": true,
  "source_answer_memorization_risk": false,
  "independent_solution": "...",
  "computed_answer": "...",
  "reason": "..."
}}

Source problem:
{source_question}

Proposed variant:
{variant_question}

Proposed answer:
{variant_answer}
"""


def validate_variant_payload(value: Any, *, source_question: str) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict):
        return {}, ["variant_must_be_object"]
    errors: list[str] = []
    question = str(value.get("variant_question", "")).strip()
    answer = str(value.get("variant_answer", "")).strip()
    structure = value.get("structure_spec")
    changed = value.get("changed_elements")
    if not question:
        errors.append("missing_variant_question")
    if not answer:
        errors.append("missing_variant_answer")
    if not isinstance(structure, dict):
        errors.append("structure_spec_must_be_object")
        structure = {}
    for field in ("core_skill", "context_constraints", "task_objective"):
        if not str(structure.get(field, "")).strip():
            errors.append(f"missing_structure_{field}")
    skeleton = structure.get("reasoning_skeleton", [])
    if not isinstance(skeleton, list) or not skeleton or not all(
        isinstance(item, str) and item.strip() for item in skeleton
    ):
        errors.append("reasoning_skeleton_must_be_nonempty_string_list")
    if not isinstance(changed, list) or not changed or not all(
        isinstance(item, str) and item.strip() for item in changed
    ):
        errors.append("changed_elements_must_be_nonempty_string_list")
    ratio = SequenceMatcher(None, " ".join(source_question.split()), " ".join(question.split())).ratio()
    if question and ratio >= 0.90:
        errors.append("surface_copy_ratio_too_high")
    normalized = {
        "variant_question": question,
        "variant_answer": answer,
        "structure_spec": {
            "core_skill": str(structure.get("core_skill", "")).strip(),
            "context_constraints": str(structure.get("context_constraints", "")).strip(),
            "task_objective": str(structure.get("task_objective", "")).strip(),
            "reasoning_skeleton": [str(item).strip() for item in skeleton] if isinstance(skeleton, list) else [],
        },
        "changed_elements": [str(item).strip() for item in changed] if isinstance(changed, list) else [],
        "surface_similarity_ratio": ratio,
    }
    return normalized, errors


def validate_variant_verification(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["verification_must_be_object"]
    errors = []
    for field in (
        "structure_match", "answer_correct", "surface_difference_sufficient",
        "source_answer_memorization_risk",
    ):
        if not isinstance(value.get(field), bool):
            errors.append(f"{field}_must_be_boolean")
    if value.get("structure_match") is not True:
        errors.append("structure_mismatch")
    if value.get("answer_correct") is not True:
        errors.append("answer_incorrect")
    if value.get("surface_difference_sufficient") is not True:
        errors.append("surface_difference_insufficient")
    if value.get("source_answer_memorization_risk") is not False:
        errors.append("source_answer_memorization_risk")
    if not str(value.get("independent_solution", "")).strip():
        errors.append("missing_independent_solution")
    return errors


async def _construct_variant(
    client: Stage2ModelClient,
    bundle: dict[str, Any],
    *,
    output_dir: Path,
    attempts: int,
) -> dict[str, Any]:
    question_id = str(bundle["question_id"])
    completed_path = output_dir / "verification" / f"{question_id}.json"
    protocol_hash = _json_hash({
        "bundle": bundle,
        "variant_prompt": VARIANT_PROMPT,
        "verify_prompt": VARIANT_VERIFY_PROMPT,
        "kimi_model": client.config.auditor_model,
        "deepseek_model": client.config.generator_model,
        "attempts": attempts,
    })
    if completed_path.exists():
        saved = load_json(completed_path)
        if saved.get("status") == "verified" and saved.get("protocol_hash") == protocol_hash:
            return saved
        _archive_stale(
            completed_path, output_root=output_dir / "stale_checkpoints",
            reason="variant_input_or_protocol_hash_changed",
        )
    history = []
    for attempt in range(1, attempts + 1):
        prompt = VARIANT_PROMPT.format(question=bundle["question_text"])
        with llm_request_context(question_id, f"principle_test/variant_generation/attempt={attempt}"):
            generated = await client.kimi_generate(prompt, max_tokens=8192, json_mode=True)
        normalized, schema_errors = validate_variant_payload(
            generated.get("parsed"), source_question=bundle["question_text"]
        )
        generation_record = {
            "attempt": attempt,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "response": generated,
            "normalized": normalized,
            "schema_errors": schema_errors,
        }
        atomic_save_json(output_dir / "generation" / f"{question_id}_attempt{attempt}.json", generation_record)
        if schema_errors:
            history.append({"attempt": attempt, "stage": "generation", "errors": schema_errors})
            continue
        verify_prompt = VARIANT_VERIFY_PROMPT.format(
            source_question=bundle["question_text"],
            variant_question=normalized["variant_question"],
            variant_answer=normalized["variant_answer"],
        )
        with llm_request_context(question_id, f"principle_test/variant_verification/attempt={attempt}"):
            verified = await client.generator(verify_prompt, thinking_enabled=True, max_tokens=8192)
        verification = verified.get("parsed", {})
        verify_errors = validate_variant_verification(verification)
        answer_match = check_answer(
            str(verification.get("computed_answer", "")), normalized["variant_answer"]
        ) if str(verification.get("computed_answer", "")).strip() else False
        if not answer_match:
            verify_errors.append("computed_answer_mismatch")
        record = {
            "status": "verified" if not verify_errors else "verification_failed",
            "source_question_id": question_id,
            "paired_source_level": bundle["level"],
            **normalized,
            "generation_attempt": attempt,
            "generation_response": generated,
            "verification": verification,
            "verification_response": verified,
            "verification_errors": sorted(set(verify_errors)),
            "history": history,
            "protocol_hash": protocol_hash,
        }
        atomic_save_json(completed_path, record)
        if not verify_errors:
            return record
        history.append({"attempt": attempt, "stage": "verification", "errors": verify_errors})
    return {
        "status": "variant_unavailable",
        "source_question_id": question_id,
        "paired_source_level": bundle["level"],
        "history": history,
        "protocol_hash": protocol_hash,
    }


def _write_variant_review(test_root: Path, variants: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Principle Test Variant Review", "",
        "> 请逐题核对数学结构、题目完整性和标准答案。全部批准后，将 `variant_approvals.json` 中对应的 `approved` 改为 `true`。", "",
    ]
    approvals_path = test_root / "variants" / "variant_approvals.json"
    previous = load_json(approvals_path) if approvals_path.exists() else []
    prior_by_id = {str(item.get("source_question_id")): item for item in previous if isinstance(item, dict)}
    approvals = []
    for index, item in enumerate(variants, 1):
        qid = str(item["source_question_id"])
        lines.extend([
            f"## {index}. {item['paired_source_level']} / source {qid}", "",
            "### 变式题", "", str(item["variant_question"]), "",
            f"**候选答案：** `{item['variant_answer']}`", "",
            "### 结构", "", "```json",
            json.dumps(item["structure_spec"], ensure_ascii=False, indent=2), "```", "",
            "### DeepSeek验算", "", "```json",
            json.dumps(item["verification"], ensure_ascii=False, indent=2), "```", "",
        ])
        old = prior_by_id.get(qid, {})
        approvals.append({
            "source_question_id": qid,
            "approved": old.get("approved") is True,
            "reviewer_note": str(old.get("reviewer_note", "")),
        })
    (test_root / "variants").mkdir(parents=True, exist_ok=True)
    (test_root / "variants" / "variant_review.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_save_json(approvals_path, approvals)


async def generate_variants(
    config: Any,
    *,
    test_root: Path,
    concurrency: int = 2,
) -> dict[str, Any]:
    """Generate and verify 40 variants; stop for explicit human approval."""
    test_root = Path(test_root)
    bundles_path = test_root / "source_bundles" / "source40_bundles.json"
    reserves_path = test_root / "source_bundles" / "reserve20_bundles.json"
    bundles = load_json(bundles_path)
    reserves = load_json(reserves_path)
    if len(bundles) != 40:
        raise ValueError(f"Expected 40 frozen source bundles, found {len(bundles)}")
    client = Stage2ModelClient(config, generator_id="deepseek")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(bundle: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _construct_variant(
                client, bundle, output_dir=test_root / "variants",
                attempts=config.stage2.principle_test_variant_attempts,
            )

    variants = list(await _gather_with_progress(
        [one(bundle) for bundle in bundles],
        description="Stage2 结构变式题生成", unit="道题",
    ))

    # A failed variant says nothing about the source Principle. Before the Raw
    # Candidate Bank is frozen, replace only that source slot with deterministic
    # same-Level reserves. Every failed attempt remains in the audit trail.
    replacements = []
    used_source_ids = {str(item["question_id"]) for item in bundles}
    reserve_by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reserve in reserves:
        reserve_by_level[str(reserve["level"])].append(reserve)
    for index, failed_record in enumerate(list(variants)):
        if failed_record.get("status") == "verified":
            continue
        failed_bundle = bundles[index]
        level = str(failed_bundle["level"])
        while reserve_by_level[level]:
            replacement = reserve_by_level[level].pop(0)
            replacement_id = str(replacement["question_id"])
            if replacement_id in used_source_ids:
                continue
            replacement_record = await one(replacement)
            replacements.append({
                "level": level,
                "replaced_question_id": str(failed_bundle["question_id"]),
                "replacement_question_id": replacement_id,
                "failed_variant": failed_record,
                "replacement_status": replacement_record.get("status"),
            })
            if replacement_record.get("status") == "verified":
                used_source_ids.remove(str(failed_bundle["question_id"]))
                used_source_ids.add(replacement_id)
                bundles[index] = replacement
                variants[index] = replacement_record
                break

    unavailable = [item for item in variants if item.get("status") != "verified"]
    verified = [item for item in variants if item.get("status") == "verified"]
    atomic_save_json(test_root / "variants" / "replacement_history.json", replacements)
    if replacements:
        manifest_path = test_root / "manifests" / "source40.json"
        manifest = load_json(manifest_path)
        selected_by_level = {
            level: [
                str(item["question_id"]) for item in bundles if str(item["level"]) == level
            ] for level in LEVELS
        }
        manifest["source_question_ids_by_level"] = selected_by_level
        manifest["source_question_ids"] = [
            qid for level in LEVELS for qid in selected_by_level[level]
        ]
        manifest["source_bundle_hashes"] = {
            str(item["question_id"]): _json_hash(item) for item in bundles
        }
        manifest["variant_source_replacements"] = [
            {
                "level": item["level"],
                "replaced_question_id": item["replaced_question_id"],
                "replacement_question_id": item["replacement_question_id"],
                "replacement_status": item["replacement_status"],
            }
            for item in replacements
        ]
        manifest.pop("manifest_hash", None)
        manifest["manifest_hash"] = _json_hash(manifest)
        atomic_save_json(manifest_path, manifest)
        atomic_save_json(bundles_path, bundles)
        atomic_save_json(test_root / "manifests" / "source40_hashes.json", {
            "manifest_hash": manifest["manifest_hash"],
            "stage1_input_hashes": manifest["stage1_input_hashes"],
            "source_bundle_hashes": manifest["source_bundle_hashes"],
        })
    atomic_save_json(test_root / "manifests" / "variant40.json", verified)
    _write_variant_review(test_root, verified)
    summary = {
        "source_count": len(bundles),
        "verified_count": len(verified),
        "unavailable_count": len(unavailable),
        "unavailable_question_ids": [item["source_question_id"] for item in unavailable],
        "replacement_attempt_count": len(replacements),
        "human_approved_count": 0,
        "ready_for_transfer": False,
    }
    atomic_save_json(test_root / "variants" / "summary.json", summary)
    if unavailable:
        raise RuntimeError(
            "Some variants and all same-Level reserves exhausted three attempts: "
            + ", ".join(summary["unavailable_question_ids"])
        )
    return summary


def require_variant_approvals(test_root: Path) -> dict[str, Any]:
    variants = load_json(Path(test_root) / "manifests" / "variant40.json")
    approvals = load_json(Path(test_root) / "variants" / "variant_approvals.json")
    approved = {
        str(item.get("source_question_id"))
        for item in approvals if isinstance(item, dict) and item.get("approved") is True
    }
    expected = {str(item.get("source_question_id")) for item in variants}
    missing = sorted(expected - approved)
    if len(expected) != 40 or missing:
        raise RuntimeError(
            f"Transfer evaluation requires 40 human-approved variants; "
            f"approved={len(approved & expected)}/40 missing={missing[:10]}"
        )
    return {"approved_count": 40, "approved_question_ids": sorted(approved)}


async def generate_raw_candidates(
    config: Any,
    *,
    test_root: Path,
    concurrency: int = 2,
    registry_root: Path | None = None,
    generation_protocol_hash: str = "",
) -> dict[str, Any]:
    """Generate one immutable Candidate Bank after variant pairs are frozen."""
    test_root = Path(test_root)
    bundles = load_json(test_root / "source_bundles" / "source40_bundles.json")
    client = Stage2ModelClient(config, generator_id="deepseek")
    semaphore = asyncio.Semaphore(max(1, concurrency))
    result_dir = test_root / "raw_candidates" / "questions"

    async def one(bundle: dict[str, Any]) -> dict[str, Any]:
        qid = str(bundle["question_id"])
        path = result_dir / f"{qid}.json"
        expected_generation_input_hash = generation_input_hash(config, bundle)
        registry_path = (
            Path(registry_root) / "questions" / qid / generation_protocol_hash / "generation.json"
            if registry_root is not None and generation_protocol_hash else None
        )
        if registry_path is not None and registry_path.exists():
            saved = load_json(registry_path)
            if (
                saved.get("status") in {"complete", "no_reliable_principle", "terminal_failed"}
                and saved.get("generation_input_hash") == expected_generation_input_hash
            ):
                atomic_save_json(path, saved)
                logger.info(
                    "[Principle生成] 题目=%s，复用已有结果，状态=%s", qid, saved.get("status")
                )
                return saved
        if path.exists():
            saved = load_json(path)
            if (
                saved.get("status") in {"complete", "no_reliable_principle", "terminal_failed"}
                and saved.get("generation_input_hash") == expected_generation_input_hash
            ):
                if registry_path is not None:
                    atomic_save_json(registry_path, saved)
                logger.info(
                    "[Principle生成] 题目=%s，断点恢复，状态=%s", qid, saved.get("status")
                )
                return saved
            _archive_stale(
                path, output_root=result_dir.parent / "stale_checkpoints",
                reason="generation_input_or_prompt_hash_changed",
            )
        async with semaphore:
            started = time.perf_counter()
            logger.info("[Principle生成] 题目=%s，开始轨迹诊断", qid)
            with llm_request_context(qid, "principle_test/diagnosis"):
                diagnosis = await diagnose_principles(
                    client, bundle, config.stage2.max_principles_per_question
                )
            if not diagnosis.get("success"):
                record = {
                    "status": "terminal_failed", "question_id": qid,
                    "level": bundle["level"], "diagnosis": diagnosis,
                    "generation_input_hash": expected_generation_input_hash,
                }
                atomic_save_json(path, record)
                if registry_path is not None:
                    atomic_save_json(registry_path, record)
                logger.warning(
                    "[Principle生成] 题目=%s，轨迹诊断失败，用时=%.1f秒，原因=%s",
                    qid, time.perf_counter() - started, diagnosis.get("failure_reason"),
                )
                return record
            logger.info(
                "[Principle生成] 题目=%s，轨迹诊断完成；开始合成，用时=%.1f秒",
                qid, time.perf_counter() - started,
            )
            with llm_request_context(qid, "principle_test/synthesis"):
                synthesis = await synthesize_principles(
                    client, bundle, config.stage2.max_principles_per_question,
                    diagnosis["diagnosis_text"],
                )
            if not synthesis.get("parse_success") or synthesis.get("schema_failures"):
                logger.warning(
                    "[Principle生成] 题目=%s，合成结果需要格式修复，可解析=%s，"
                    "Schema错误数=%d",
                    qid, synthesis.get("parse_success"),
                    len(synthesis.get("schema_failures") or []),
                )
                feedback = json.dumps({
                    "parse_success": synthesis.get("parse_success"),
                    "schema_failures": synthesis.get("schema_failures", []),
                }, ensure_ascii=False)
                with llm_request_context(qid, "principle_test/synthesis_format_retry"):
                    synthesis = await synthesize_principles(
                        client, bundle, config.stage2.max_principles_per_question,
                        diagnosis["diagnosis_text"], correction_feedback=feedback,
                    )
            families = synthesis.get("families", [])
            status = "complete" if families else "no_reliable_principle"
            record = {
                "status": status, "question_id": qid, "level": bundle["level"],
                "bundle_hash": _json_hash(bundle), "diagnosis": diagnosis,
                "synthesis": synthesis,
                "generation_input_hash": expected_generation_input_hash,
            }
            atomic_save_json(path, record)
            if registry_path is not None:
                atomic_save_json(registry_path, record)
            logger.info(
                "[Principle生成] 题目=%s，完成，状态=%s，合法Principle=%d，"
                "丢弃候选=%d，用时=%.1f秒",
                qid, status, len(families), len(synthesis.get("schema_failures") or []),
                time.perf_counter() - started,
            )
            return record

    records = await _gather_with_progress(
        [one(bundle) for bundle in bundles],
        description="Stage2 Principle生成", unit="道题",
    )
    candidates = []
    for record in records:
        candidates.extend(candidates_from_generation(record))
    candidates.sort(key=lambda item: item["candidate_id"])
    bank = {
        "schema_version": "principle_test_candidates.1",
        "source_manifest_hash": load_json(test_root / "manifests" / "source40.json")["manifest_hash"],
        "candidate_count": len(candidates), "candidates": candidates,
    }
    bank["candidate_bank_hash"] = _json_hash(bank)
    bank_dir = test_root / "raw_candidates" / "candidate_banks"
    bank_path = bank_dir / f"target_{len(bundles):04d}.json"
    if bank_path.exists():
        previous = load_json(bank_path)
        if previous.get("candidate_bank_hash") != bank["candidate_bank_hash"]:
            raise RuntimeError(
                f"Frozen target Candidate Bank differs from regenerated content: {bank_path}"
            )
    atomic_save_json(bank_path, bank)
    # This is only a convenience pointer for downstream phases. Immutable
    # target snapshots above are never overwritten.
    atomic_save_json(test_root / "raw_candidates" / "candidate_bank.json", bank)
    summary = {
        "question_count": len(records), "candidate_count": len(candidates),
        "status_counts": dict(Counter(item["status"] for item in records)),
        "candidate_bank_hash": bank["candidate_bank_hash"],
    }
    atomic_save_json(test_root / "raw_candidates" / "summary.json", summary)
    return summary


def _audit_protocol_valid(parsed: Any, evidence_ids: set[str]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(parsed, dict):
        return {}, ["audit_must_be_object"]
    dimensions = parsed.get("dimensions")
    errors = []
    if not isinstance(dimensions, dict):
        dimensions = {}
        errors.append("dimensions_must_be_object")
    normalized = {}
    for name in AUDIT_DIMENSIONS:
        raw = dimensions.get(name)
        if not isinstance(raw, dict) or not isinstance(raw.get("pass"), bool):
            errors.append(f"invalid_{name}")
            continue
        refs = raw.get("evidence_refs", [])
        if not isinstance(refs, list) or not set(map(str, refs)).issubset(evidence_ids):
            errors.append(f"invalid_{name}_evidence_refs")
            refs = []
        normalized[name] = {
            "pass": raw["pass"], "reason": str(raw.get("reason", "")).strip(),
            "evidence_refs": [str(value) for value in refs],
        }
    decision = str(parsed.get("decision", "")).lower()
    if decision not in {"pass", "repair", "reject"}:
        errors.append("invalid_decision")
    return {
        "dimensions": normalized, "decision": decision,
        "reasons": parsed.get("reasons", []) if isinstance(parsed.get("reasons"), list) else [],
    }, errors


def aggregate_audits(
    runs: Sequence[Mapping[str, Any]],
    *,
    deterministic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    valid = [item for item in runs if item.get("protocol_compliant")]
    failed_counts = {
        name: sum(
            run.get("audit", {}).get("dimensions", {}).get(name, {}).get("pass") is False
            for run in valid
        )
        for name in AUDIT_DIMENSIONS
    }
    majority_failed = sorted(name for name, count in failed_counts.items() if count >= 2)
    disagreements = sorted(name for name, count in failed_counts.items() if count == 1)
    deterministic = dict(deterministic or {})
    if deterministic.get("detected"):
        decision = "reject"
    elif len(valid) < 2:
        decision = "protocol_failed"
    elif HARD_AUDIT_DIMENSIONS.intersection(majority_failed):
        decision = "reject"
    elif majority_failed:
        decision = "repair"
    else:
        decision = "pass"
    return {
        "decision": decision,
        "valid_run_count": len(valid),
        "protocol_success_rate": len(valid) / len(runs) if runs else 0.0,
        "failed_counts": failed_counts,
        "majority_failed_dimensions": majority_failed,
        "disagreement_dimensions": disagreements,
        "audit_disagreement": bool(disagreements),
        "deterministic_leakage": deterministic,
    }


def _critic_prompt(bundle: dict[str, Any], candidate: dict[str, Any]) -> str:
    evidence_ids = set(candidate.get("supporting_wrong_rollout_ids", [])) | set(
        candidate.get("contrast_correct_rollout_ids", [])
    )
    evidence = [
        item for item in bundle.get("rollouts", [])
        if isinstance(item, dict) and item.get("rollout_id") in evidence_ids
    ]
    return f"""Act as an independent mathematical Principle critic. Do not predict whether
the weak solver will improve. Inspect source fidelity, mathematical validity, instance
leakage, trigger/evidence/failure/exclusion decidability, action and verification
executability, and supported generalization. Discuss every dimension explicitly and cite
only exact rollout IDs from the evidence. This is a semantic critique, not a JSON task.

Question: {bundle['question_text']}
Principle: {json.dumps(candidate['content'], ensure_ascii=False)}
Evidence: {json.dumps(evidence, ensure_ascii=False)}
Allowed evidence IDs: {json.dumps(sorted(evidence_ids), ensure_ascii=False)}
"""


def _formatter_prompt(critique: str, evidence_ids: set[str]) -> str:
    return f"""Convert the completed critique below into the exact JSON contract. Do not
re-audit, add facts, change judgments, or resolve uncertainty. For each dimension return
pass:boolean, reason:string, evidence_refs:list[string]. Evidence references must be drawn
only from the allowed IDs. Return all ten dimensions and decision pass|repair|reject.

Allowed IDs: {json.dumps(sorted(evidence_ids), ensure_ascii=False)}
Dimensions: {json.dumps(list(AUDIT_DIMENSIONS))}

Critique:
{critique}
"""


async def _run_audit_once(
    client: Stage2ModelClient,
    bundle: dict[str, Any],
    candidate: dict[str, Any],
    protocol: str,
    repeat: int,
) -> dict[str, Any]:
    evidence_ids = set(candidate.get("supporting_wrong_rollout_ids", [])) | set(
        candidate.get("contrast_correct_rollout_ids", [])
    )
    evidence = [
        item for item in bundle.get("rollouts", [])
        if isinstance(item, dict) and item.get("rollout_id") in evidence_ids
    ]
    if protocol == "kimi_single":
        prompt = build_audit_prompt(bundle, candidate, evidence_ids, evidence)
        response = await client.auditor(
            prompt, auditor_id="kimi", thinking_enabled=True,
            max_tokens=8192, json_mode=True,
        )
        critique_response = None
        final_response = response
    elif protocol == "kimi_critic_formatter":
        prompt = _critic_prompt(bundle, candidate)
        critique_response = await client.auditor(
            prompt, auditor_id="kimi", thinking_enabled=True,
            max_tokens=8192, json_mode=False,
        )
        critique = "\n".join(
            str(value) for value in (
                critique_response.get("reasoning", ""), critique_response.get("content", "")
            ) if str(value).strip()
        )
        formatter_prompt = _formatter_prompt(critique, evidence_ids)
        final_response = await client.format_audit(formatter_prompt) if critique else {
            "parsed": {}, "raw": "", "failure_reason": "empty_critique",
        }
    else:
        raise ValueError(f"Unknown audit protocol: {protocol}")
    audit, errors = _audit_protocol_valid(final_response.get("parsed"), evidence_ids)
    return {
        "repeat": repeat, "protocol": protocol,
        "protocol_compliant": not errors, "protocol_errors": errors,
        "audit": audit, "critic_response": critique_response,
        "final_response": final_response,
    }


async def run_audit_ablation(
    config: Any,
    *,
    test_root: Path,
    concurrency: int = 2,
    protocols: Sequence[str] = AUDIT_PROTOCOLS,
    registry_root: Path | None = None,
    candidate_protocol_hash: str = "",
) -> dict[str, Any]:
    test_root = Path(test_root)
    bank = load_json(test_root / "raw_candidates" / "candidate_bank.json")
    bundles = load_json(test_root / "source_bundles" / "source40_bundles.json")
    by_question = {str(item["question_id"]): item for item in bundles}
    client = Stage2ModelClient(config, generator_id="deepseek")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def process(candidate: dict[str, Any], protocol: str) -> dict[str, Any]:
        if protocol not in AUDIT_PROTOCOLS:
            raise ValueError(f"Unknown audit protocol: {protocol}")
        path = test_root / "audit_variants" / protocol / f"{candidate['candidate_id']}.json"
        registry_path = (
            Path(registry_root) / "questions" / str(candidate["question_id"]) /
            candidate_protocol_hash / "audits" / protocol /
            f"{candidate['candidate_id']}.json"
            if registry_root is not None and candidate_protocol_hash else None
        )
        bundle = by_question[candidate["question_id"]]
        expected_audit_input_hash = audit_input_hash(config, candidate, bundle, protocol)
        if registry_path is not None and registry_path.exists():
            saved = load_json(registry_path)
            if (
                saved.get("status") == "terminal"
                and saved.get("audit_input_hash") == expected_audit_input_hash
            ):
                atomic_save_json(path, saved)
                return saved
        if path.exists():
            saved = load_json(path)
            if (
                saved.get("status") == "terminal"
                and saved.get("audit_input_hash") == expected_audit_input_hash
            ):
                if registry_path is not None:
                    atomic_save_json(registry_path, saved)
                return saved
            _archive_stale(
                path, output_root=path.parent / "stale_checkpoints",
                reason="audit_input_or_protocol_hash_changed",
            )
        deterministic = deterministic_leakage_check(
            candidate["content"], source_question=bundle["question_text"],
            expected_answer=bundle["expected_answer"],
        )
        if protocol == "raw_control":
            aggregate = {
                "decision": "reject" if deterministic.get("detected") else "pass",
                "valid_run_count": 0, "protocol_success_rate": 1.0,
                "majority_failed_dimensions": [], "disagreement_dimensions": [],
                "audit_disagreement": False, "deterministic_leakage": deterministic,
            }
            record = {
                "status": "terminal", "protocol": protocol,
                "candidate_id": candidate["candidate_id"], "original_candidate": candidate,
                "runs": [], "aggregate": aggregate, "final_candidate": candidate,
                "final_decision": aggregate["decision"], "repair": None, "reaudit_runs": [],
                "audit_input_hash": expected_audit_input_hash,
            }
            atomic_save_json(path, record)
            if registry_path is not None:
                atomic_save_json(registry_path, record)
            return record
        runs = []
        for repeat in range(1, config.stage2.principle_test_audit_repeats + 1):
            async with semaphore:
                with llm_request_context(
                    candidate["candidate_id"], f"principle_test/audit/{protocol}/repeat={repeat}"
                ):
                    runs.append(await _run_audit_once(client, bundle, candidate, protocol, repeat))
        aggregate = aggregate_audits(runs, deterministic=deterministic)
        repair = None
        final_candidate = candidate
        reaudit_runs = []
        final_aggregate = aggregate
        if aggregate["decision"] == "repair":
            async with semaphore:
                repair = await repair_principle(client, bundle, candidate, aggregate)
            if repair.get("success"):
                final_candidate = {**candidate, "content": repair["content"], "parent_content": candidate["content"]}
                for repeat in range(1, config.stage2.principle_test_audit_repeats + 1):
                    async with semaphore:
                        reaudit_runs.append(await _run_audit_once(
                            client, bundle, final_candidate, protocol, repeat
                        ))
                repaired_leakage = deterministic_leakage_check(
                    final_candidate["content"], source_question=bundle["question_text"],
                    expected_answer=bundle["expected_answer"],
                )
                final_aggregate = aggregate_audits(reaudit_runs, deterministic=repaired_leakage)
            else:
                final_aggregate = {**aggregate, "decision": "reject", "repair_failed": True}
        record = {
            "status": "terminal", "protocol": protocol,
            "candidate_id": candidate["candidate_id"], "original_candidate": candidate,
            "runs": runs, "aggregate": aggregate, "repair": repair,
            "reaudit_runs": reaudit_runs, "final_candidate": final_candidate,
            "final_aggregate": final_aggregate,
            "final_decision": final_aggregate["decision"],
            "audit_input_hash": expected_audit_input_hash,
        }
        atomic_save_json(path, record)
        if registry_path is not None:
            atomic_save_json(registry_path, record)
        return record

    records = []
    protocols = tuple(protocols)
    for protocol in protocols:
        logger.info(
            "[质量审查] 协议=%s，候选数=%d，每个候选重复审查=%d次",
            protocol, len(bank["candidates"]), config.stage2.principle_test_audit_repeats,
        )
        records.extend(await _gather_with_progress(
            [process(candidate, protocol) for candidate in bank["candidates"]],
            description=f"Stage2质量审查 {protocol}", unit="个候选",
        ))
    summary = {
        "candidate_bank_hash": bank["candidate_bank_hash"],
        "candidate_count": bank["candidate_count"],
        "protocols": {},
    }
    for protocol in protocols:
        group = [item for item in records if item["protocol"] == protocol]
        summary["protocols"][protocol] = {
            "count": len(group),
            "decision_counts": dict(Counter(item["final_decision"] for item in group)),
            "parse_success_rate": (
                sum(
                    item.get("final_aggregate", item["aggregate"]).get("protocol_success_rate", 0)
                    for item in group
                ) / len(group) if group else 0.0
            ),
            "repair_count": sum(item.get("repair") is not None for item in group),
            "disagreement_count": sum(
                item.get("final_aggregate", item["aggregate"]).get("audit_disagreement", False)
                for item in group
            ),
        }
    atomic_save_json(test_root / "audit_variants" / "summary.json", summary)
    return summary


CONDITION_CARD_PROMPT = """Adapt one candidate mathematical Principle to the CURRENT
problem. Do not solve the problem and do not predict its final answer. Every item in
evidence_found must pair one Principle requirement with a fact explicitly observable in
the current problem. Similar topic or vocabulary is not sufficient. If a required fact
cannot be confirmed, an exclusion applies, or the optional initial plan reveals a
different mathematical structure, return status=reject. Advice must remain optional.

Return raw JSON only:
{{
  "status": "use|reject",
  "when_to_consider": "...",
  "evidence_found": [{{"requirement": "...", "problem_fact": "..."}}],
  "do_not_use_if": "...",
  "failure_to_avoid": "...",
  "suggested_action": "...",
  "verification": "...",
  "reason": "..."
}}

Current problem:
{question}

Candidate Principle fields:
{principle}

Optional initial plan (may be empty):
{initial_plan}
"""


def _solver_safe_principle(content: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trigger_condition": content.get("trigger_condition", ""),
        "required_evidence": content.get("required_evidence", []),
        "exclusion_condition": content.get("exclusion_condition", ""),
        "failure_mode": content.get("failure_mode", ""),
        "action_guidance": content.get("action_guidance", ""),
        "verification_step": content.get("verification_step", ""),
    }


async def _make_condition_card(
    client: Stage2ModelClient,
    *,
    question: str,
    content: Mapping[str, Any],
    initial_plan: str = "",
) -> dict[str, Any]:
    prompt = CONDITION_CARD_PROMPT.format(
        question=question,
        principle=json.dumps(_solver_safe_principle(content), ensure_ascii=False, indent=2),
        initial_plan=initial_plan,
    )
    input_hash = _json_hash({
        "prompt": prompt,
        "model": client.config.auditor_model,
        "max_tokens": 3072,
    })
    response = await client.kimi_generate(prompt, max_tokens=3072, json_mode=True)
    card, errors = validate_condition_card(response.get("parsed"))
    status = card.get("status", "reject")
    if errors:
        status = "reject"
        card["status"] = "reject"
    return {
        "status": status,
        "condition_card": card if status == "use" else {},
        "schema_errors": errors,
        "reason": str((response.get("parsed") or {}).get("reason", "")).strip(),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_hash": input_hash,
        "response": response,
    }


def _record_id(
    phase: str, arm: str, question_id: str, temperature: float, seed: int,
    candidate_id: str = "",
) -> str:
    return f"{phase}:{arm}:{question_id}:t{temperature:.6f}:s{seed}:{candidate_id or '-'}"


def _stable_endpoint_index(question_id: str, temperature: float, seed: int, size: int) -> int:
    digest = hashlib.sha256(f"{question_id}:{temperature:.6f}:{seed}".encode()).hexdigest()
    return int(digest[:12], 16) % max(1, size)


def _paired_counts(baseline: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    base = {int(item["seed"]): bool(item["correct"]) for item in baseline}
    guided = {int(item["seed"]): bool(item["correct"]) for item in treatment}
    if set(base) != set(guided):
        raise ValueError("Baseline and treatment seed sets differ")
    counts = {"gain": 0, "loss": 0, "both_correct": 0, "both_wrong": 0}
    for seed in sorted(base):
        if not base[seed] and guided[seed]:
            counts["gain"] += 1
        elif base[seed] and not guided[seed]:
            counts["loss"] += 1
        elif base[seed]:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _effect_label(delta: int) -> str:
    if delta >= 2:
        return "helpful"
    if delta <= -2:
        return "harmful"
    return "inconclusive"


def _privacy_violations(prompt: str, candidate: Mapping[str, Any]) -> list[str]:
    lowered = (prompt or "").lower()
    violations = []
    for label, marker in {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "retrieval_score": "retrieval_score",
        "audit_decision": "audit_decision",
        "source_rollout": "source rollout",
        "expected_answer": "expected_answer",
    }.items():
        if marker and marker.lower() in lowered:
            violations.append(label)
    return sorted(set(violations))


class _PairedRunner:
    def __init__(
        self,
        config: Any,
        *,
        output_dir: Path,
        urls: Sequence[str],
        model: str,
        concurrency: int,
        timeout: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.urls = list(urls)
        self.model = model
        self.solver_fingerprint = _json_hash({
            "protocol_version": SOLVER_V1_PROTOCOL,
            "prompt_hashes": SOLVER_V1_PROMPT_HASHES,
            "model": model,
        })
        self.execute = ExecuteModule(config, urls=self.urls, model_override=model, timeout=timeout)
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.timeout = timeout
        self.rollout_path = self.output_dir / "rollouts.json"
        previous = load_json(self.rollout_path) if self.rollout_path.exists() else []
        self.records = {}
        for item in previous:
            if not isinstance(item, dict) or not item.get("record_id"):
                continue
            response = item.get("response", {})
            guided = bool(response.get("guided"))
            expected_hashes = {
                "reasoner": SOLVER_V1_PROMPT_HASHES[
                    "reasoner_guided" if guided else "reasoner_base"
                ],
                "finalizer": SOLVER_V1_PROMPT_HASHES["finalizer"],
            }
            compatible = (
                item.get("status") in {"complete", "infrastructure_failed"}
                and item.get("solver_fingerprint") == self.solver_fingerprint
                and (
                    item.get("status") == "infrastructure_failed"
                    or (
                        response.get("protocol_version") == SOLVER_V1_PROTOCOL
                        and response.get("prompt_hashes") == expected_hashes
                    )
                )
            )
            if compatible:
                self.records[str(item["record_id"])] = item
        self.lock = asyncio.Lock()

    async def _persist(self, record: dict[str, Any]) -> dict[str, Any]:
        async with self.lock:
            self.records[record["record_id"]] = record
            atomic_save_json(self.rollout_path, [self.records[key] for key in sorted(self.records)])
            if len(self.records) == 1 or len(self.records) % 10 == 0:
                logger.info(
                    "Principle效果推理已保存=%d，最近记录=%s",
                    len(self.records), record["record_id"],
                )
        return record

    async def draft_plan(
        self, *, phase: str, question_id: str, question: str,
        temperature: float, seed: int,
    ) -> dict[str, Any]:
        path = self.output_dir / "initial_plans" / f"{question_id}_t{temperature:.1f}_s{seed}.json"
        if path.exists():
            saved = load_json(path)
            expected_hash = hashlib.sha256(
                SOLVER_V1_INITIAL_PLAN_PROMPT.strip().encode("utf-8")
            ).hexdigest()
            if saved.get("prompt_hash") == expected_hash and saved.get("model") == self.model:
                return saved
            _archive_stale(
                path, output_root=path.parent / "stale_checkpoints",
                reason="initial_plan_model_or_prompt_hash_changed",
            )
        primary = _stable_endpoint_index(question_id, temperature, seed, len(self.urls))
        endpoints = [self.urls[primary]]
        if len(self.urls) > 1:
            endpoints.append(self.urls[(primary + 1) % len(self.urls)])
        result = {}
        async with self.semaphore:
            for endpoint in endpoints:
                with llm_request_context(question_id, f"{phase}/initial_plan/seed={seed}"):
                    result = await self.execute.draft_plan_v1(
                        question, seed=seed, url_override=endpoint,
                        temperature=temperature, timeout=min(self.timeout, 120),
                    )
                if result.get("infrastructure_success"):
                    break
        result["model"] = self.model
        atomic_save_json(path, result)
        return result

    async def solve(
        self,
        *,
        phase: str,
        arm: str,
        question_id: str,
        question: str,
        expected_answer: str,
        temperature: float,
        seed: int,
        principle_text: str = "",
        candidate: Mapping[str, Any] | None = None,
        derived_from: str = "",
    ) -> dict[str, Any]:
        candidate_id = str((candidate or {}).get("candidate_id", "")) if arm not in {"baseline", "plan_only"} else ""
        record_id = _record_id(phase, arm, question_id, temperature, seed, candidate_id)
        if record_id in self.records:
            return self.records[record_id]
        if derived_from:
            source = self.records[derived_from]
            return await self._persist({
                **source,
                "record_id": record_id,
                "arm": arm,
                "candidate_id": candidate_id,
                "injected": False,
                "derived_from": derived_from,
                "experience_decision": "not_injected",
            })
        primary = _stable_endpoint_index(question_id, temperature, seed, len(self.urls))
        endpoints = [self.urls[primary]]
        if len(self.urls) > 1:
            endpoints.append(self.urls[(primary + 1) % len(self.urls)])
        response: dict[str, Any] = {}
        trace = []
        async with self.semaphore:
            for endpoint_attempt, endpoint in enumerate(endpoints, 1):
                with llm_request_context(
                    candidate_id or question_id,
                    f"{phase}/{arm}/t={temperature}/seed={seed}/attempt={endpoint_attempt}",
                ):
                    response = await self.execute.solve_v1(
                        question, principle_text=principle_text, seed=seed,
                        url_override=endpoint, reasoner_temperature=temperature,
                        reasoner_timeout=min(self.timeout, 300),
                        finalizer_timeout=min(self.timeout, 60),
                    )
                trace.append({
                    "endpoint": endpoint, "endpoint_attempt": endpoint_attempt,
                    "success": bool(response.get("infrastructure_success")),
                })
                if response.get("infrastructure_success"):
                    break
        success = bool(response.get("infrastructure_success"))
        composed = str(response.get("composed_output", ""))
        solver_prompt = str((response.get("reasoner") or {}).get("prompt", ""))
        reasoner_output = str((response.get("reasoner") or {}).get("output", ""))
        decision_match = re.search(
            r"Experience Decision:\s*(use|ignore)\b", reasoner_output, flags=re.IGNORECASE
        )
        return await self._persist({
            "record_id": record_id, "phase": phase, "arm": arm,
            "question_id": question_id, "candidate_id": candidate_id,
            "temperature": temperature, "seed": seed,
            "model": self.model,
            "solver_fingerprint": self.solver_fingerprint,
            "status": "complete" if success else "infrastructure_failed",
            "correct": check_answer(composed, expected_answer) if success else False,
            "injected": bool(principle_text),
            "experience_decision": decision_match.group(1).lower() if decision_match else (
                "baseline" if not principle_text else "unparsed"
            ),
            "response": response, "solver_prompt": solver_prompt,
            "privacy_violations": _privacy_violations(solver_prompt, candidate or {}),
            "transport_trace": trace,
        })


def _healthy_actor_urls(ports: Sequence[int]) -> list[str]:
    urls = [f"http://127.0.0.1:{int(port)}/v1/chat/completions" for port in ports]
    healthy = filter_healthy_urls(urls)
    if len(healthy) != len(urls):
        raise RuntimeError(f"Expected {len(urls)} healthy Llama endpoints, found {len(healthy)}")
    return healthy


def _load_audit_records(test_root: Path, protocol: str) -> list[dict[str, Any]]:
    paths = sorted((Path(test_root) / "audit_variants" / protocol).glob("candidate_*.json"))
    return [load_json(path) for path in paths]


def _accepted_candidates(test_root: Path, protocol: str) -> list[dict[str, Any]]:
    return [
        item["final_candidate"] for item in _load_audit_records(test_root, protocol)
        if item.get("final_decision") == "pass"
    ]


def _summarize_effect_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in rows if item.get("evaluated")]
    by_question: dict[str, list[float]] = defaultdict(list)
    by_level: dict[str, list[float]] = defaultdict(list)
    for item in eligible:
        normalized_delta = float(item.get("delta_correct", 0)) / 10.0
        by_question[str(item.get("question_id", ""))].append(normalized_delta)
        by_level[str(item.get("level", "unknown"))].append(normalized_delta)
    question_deltas = [sum(values) / len(values) for values in by_question.values() if values]
    harmful_questions = {
        str(item.get("question_id")) for item in eligible
        if item.get("effect_label") == "harmful"
    }
    helpful_questions = {
        str(item.get("question_id")) for item in eligible
        if item.get("effect_label") == "helpful"
    }
    bootstrap_ci = [0.0, 0.0]
    if question_deltas:
        rng = random.Random(42)
        estimates = []
        for _ in range(10_000):
            sample = [rng.choice(question_deltas) for _ in question_deltas]
            estimates.append(sum(sample) / len(sample))
        estimates.sort()
        bootstrap_ci = [
            estimates[int(0.025 * (len(estimates) - 1))],
            estimates[int(0.975 * (len(estimates) - 1))],
        ]
    return {
        "candidate_count": len(rows),
        "evaluated_count": len(eligible),
        "question_count": len(question_deltas),
        "correct_delta_sum": sum(int(item.get("delta_correct", 0)) for item in eligible),
        "gain": sum(int(item.get("paired", {}).get("gain", 0)) for item in eligible),
        "loss": sum(int(item.get("paired", {}).get("loss", 0)) for item in eligible),
        "helpful_count": sum(item.get("effect_label") == "helpful" for item in eligible),
        "harmful_count": sum(item.get("effect_label") == "harmful" for item in eligible),
        "inconclusive_count": sum(item.get("effect_label") == "inconclusive" for item in eligible),
        "harmful_rate": (
            sum(item.get("effect_label") == "harmful" for item in eligible) / len(eligible)
            if eligible else 0.0
        ),
        "helpful_question_count": len(helpful_questions),
        "harmful_question_count": len(harmful_questions),
        "harmful_question_rate": (
            len(harmful_questions) / len(by_question) if by_question else 0.0
        ),
        "prompt_privacy_violation_count": sum(
            bool(item.get("privacy_violations")) for item in eligible
        ),
        "solver_use_count": sum(int(item.get("solver_use_count", 0)) for item in eligible),
        "solver_ignore_count": sum(int(item.get("solver_ignore_count", 0)) for item in eligible),
        "not_injected_count": sum(int(item.get("not_injected_count", 0)) for item in eligible),
        "question_mean_accuracy_delta": (
            sum(question_deltas) / len(question_deltas) if question_deltas else 0.0
        ),
        "paired_bootstrap_question_95ci": bootstrap_ci,
        "level_mean_accuracy_delta": {
            level: sum(values) / len(values)
            for level, values in sorted(by_level.items()) if values
        },
    }


def _write_effect_diagnostics(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    rollout_records: Sequence[Mapping[str, Any]],
) -> None:
    """Materialize failures without changing any experimental decision."""
    transport = [
        item for item in rollout_records
        if item.get("status") == "infrastructure_failed"
    ]
    atomic_save_json(output_dir / "transport_failures.json", transport)
    failures = {
        "harmful": [item for item in rows if item.get("effect_label") == "harmful"],
        "inconclusive": [item for item in rows if item.get("effect_label") == "inconclusive"],
        "not_evaluated": [item for item in rows if not item.get("evaluated")],
        "prompt_privacy_violations": [
            item for item in rows if item.get("privacy_violations")
        ],
    }
    atomic_save_json(output_dir / "failure_analysis.json", failures)


async def _evaluate_direct_protocols(
    config: Any,
    *,
    test_root: Path,
    ports: Sequence[int],
    model: str,
    concurrency: int,
    timeout: int,
    temperature: float,
) -> dict[str, Any]:
    bundles = load_json(test_root / "source_bundles" / "source40_bundles.json")
    bundle_by_id = {str(item["question_id"]): item for item in bundles}
    urls = _healthy_actor_urls(ports)
    runner = _PairedRunner(
        config, output_dir=test_root / "source_evaluation" / "audit_protocols",
        urls=urls, model=model, concurrency=concurrency, timeout=timeout,
    )
    seeds = tuple(config.stage2.principle_test_solver_seeds)
    baseline_tasks = {}
    for bundle in bundles:
        qid = str(bundle["question_id"])
        for seed in seeds:
            baseline_tasks[(qid, seed)] = asyncio.create_task(runner.solve(
                phase="source_audit", arm="baseline", question_id=qid,
                question=bundle["question_text"], expected_answer=bundle["expected_answer"],
                temperature=temperature, seed=seed,
            ))
    await _gather_with_progress(
        list(baseline_tasks.values()),
        description="来源题基线推理", unit="次推理",
    )

    rows = []
    audit_effect_total = sum(
        len(_load_audit_records(test_root, protocol)) for protocol in AUDIT_PROTOCOLS
    )
    audit_effect_progress = tqdm(
        total=audit_effect_total, desc="来源题审查协议效果", unit="个候选",
        dynamic_ncols=True, disable=not sys.stderr.isatty(), leave=True,
    )
    for protocol in AUDIT_PROTOCOLS:
        for audit_record in _load_audit_records(test_root, protocol):
            candidate = audit_record["final_candidate"]
            qid = str(candidate["question_id"])
            bundle = bundle_by_id[qid]
            baseline = [await baseline_tasks[(qid, seed)] for seed in seeds]
            if audit_record.get("final_decision") != "pass":
                rows.append({
                    "protocol": protocol, "candidate_id": candidate["candidate_id"],
                    "question_id": qid, "evaluated": False,
                    "final_decision": audit_record.get("final_decision"),
                    "baseline_correct_at_10": sum(item["correct"] for item in baseline),
                })
                audit_effect_progress.update(1)
                continue
            text = render_solver_memory(candidate["content"])
            treatment = await asyncio.gather(*(
                runner.solve(
                    phase="source_audit", arm=protocol, question_id=qid,
                    question=bundle["question_text"], expected_answer=bundle["expected_answer"],
                    temperature=temperature, seed=seed, principle_text=text, candidate=candidate,
                ) for seed in seeds
            ))
            paired = _paired_counts(baseline, treatment)
            baseline_correct = sum(item["correct"] for item in baseline)
            guided_correct = sum(item["correct"] for item in treatment)
            rows.append({
                "protocol": protocol, "candidate_id": candidate["candidate_id"],
                "question_id": qid, "level": candidate["level"], "evaluated": True,
                "final_decision": "pass", "baseline_correct_at_10": baseline_correct,
                "guided_correct_at_10": guided_correct,
                "delta_correct": guided_correct - baseline_correct,
                "paired": paired, "effect_label": _effect_label(guided_correct - baseline_correct),
                "privacy_violations": sorted({
                    violation for item in treatment for violation in item.get("privacy_violations", [])
                }),
            })
            audit_effect_progress.update(1)
    audit_effect_progress.close()
    atomic_save_json(test_root / "source_evaluation" / "audit_protocols" / "candidate_effects.json", rows)
    _write_effect_diagnostics(
        test_root / "source_evaluation" / "audit_protocols",
        rows,
        list(runner.records.values()),
    )
    by_protocol = {
        protocol: _summarize_effect_rows([item for item in rows if item["protocol"] == protocol])
        for protocol in AUDIT_PROTOCOLS
    }
    audit_summary = load_json(test_root / "audit_variants" / "summary.json")["protocols"]
    eligible = []
    for protocol in ("kimi_single", "kimi_critic_formatter"):
        stats = by_protocol[protocol]
        protocol_parse = audit_summary[protocol]["parse_success_rate"]
        if protocol_parse >= 0.95 and stats["prompt_privacy_violation_count"] == 0:
            eligible.append(protocol)
    if not eligible:
        winner = ""
        reason = "no_audit_protocol_met_parse_and_privacy_floor"
    else:
        winner = sorted(eligible, key=lambda protocol: (
            by_protocol[protocol]["harmful_rate"],
            -by_protocol[protocol]["correct_delta_sum"],
            -by_protocol[protocol]["gain"] + by_protocol[protocol]["loss"],
            0 if protocol == "kimi_critic_formatter" else 1,
        ))[0]
        reason = "lexicographic_quality_harm_net_gain_stability"
    selection = {
        "winner": winner, "eligible_protocols": eligible, "reason": reason,
        "protocol_effects": by_protocol, "audit_protocol_stats": audit_summary,
        "temperature": temperature,
    }
    atomic_save_json(test_root / "source_evaluation" / "audit_winner.json", selection)
    if not winner:
        raise RuntimeError(reason)
    return selection


async def _prepare_pre_cards(
    client: Stage2ModelClient,
    *,
    candidates: Sequence[Mapping[str, Any]],
    questions: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, dict[str, Any]]:
    cards = {}
    for candidate in candidates:
        qid = str(candidate["question_id"])
        path = output_dir / f"{candidate['candidate_id']}.json"
        expected_input_hash = _json_hash({
            "prompt": CONDITION_CARD_PROMPT.format(
                question=str(questions[qid]["question_text"]),
                principle=json.dumps(
                    _solver_safe_principle(candidate["content"]), ensure_ascii=False, indent=2
                ),
                initial_plan="",
            ),
            "model": client.config.auditor_model,
            "max_tokens": 3072,
        })
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == expected_input_hash:
                cards[candidate["candidate_id"]] = saved
                continue
            _archive_stale(
                path, output_root=output_dir / "stale_checkpoints",
                reason="condition_card_input_or_prompt_hash_changed",
            )
        async with semaphore:
            with llm_request_context(candidate["candidate_id"], "principle_test/pre_condition"):
                record = await _make_condition_card(
                    client, question=str(questions[qid]["question_text"]),
                    content=candidate["content"],
                )
        atomic_save_json(path, record)
        cards[candidate["candidate_id"]] = record
    return cards


async def _evaluate_interventions_on_questions(
    config: Any,
    *,
    test_root: Path,
    phase: str,
    questions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    ports: Sequence[int],
    model: str,
    concurrency: int,
    timeout: int,
    temperature: float,
    arms: Sequence[str],
) -> dict[str, Any]:
    question_by_id = {str(item["question_id"]): item for item in questions}
    urls = _healthy_actor_urls(ports)
    if phase == "source":
        phase_root = test_root / "source_evaluation"
    elif phase == "transfer":
        phase_root = test_root / "transfer_evaluation"
    elif phase == "robustness_source":
        phase_root = test_root / "robustness_07" / "source"
    elif phase == "robustness_transfer":
        phase_root = test_root / "robustness_07" / "transfer"
    else:
        raise ValueError(f"Unknown Principle-test phase: {phase}")
    output_dir = phase_root / "interventions"
    runner = _PairedRunner(
        config, output_dir=output_dir, urls=urls, model=model,
        concurrency=concurrency, timeout=timeout,
    )
    client = Stage2ModelClient(config, generator_id="deepseek")
    card_semaphore = asyncio.Semaphore(max(1, min(concurrency, 4)))
    candidate_by_question = defaultdict(list)
    for candidate in candidates:
        candidate_by_question[str(candidate["question_id"])].append(candidate)
    pre_cards = await _prepare_pre_cards(
        client, candidates=candidates, questions=question_by_id,
        output_dir=output_dir / "pre_cards", semaphore=card_semaphore,
    ) if "pre_condition" in arms else {}
    seeds = tuple(config.stage2.principle_test_solver_seeds)

    baseline_tasks = {}
    for qid, question in question_by_id.items():
        for seed in seeds:
            baseline_tasks[(qid, seed)] = asyncio.create_task(runner.solve(
                phase=phase, arm="baseline", question_id=qid,
                question=str(question["question_text"]),
                expected_answer=str(question["expected_answer"]),
                temperature=temperature, seed=seed,
            ))
    await _gather_with_progress(
        list(baseline_tasks.values()),
        description=f"{_phase_label_zh(phase)}：基线推理", unit="次推理",
    )

    plan_tasks = {}
    if "post_plan" in arms:
        for qid, question in question_by_id.items():
            for seed in seeds:
                plan_tasks[(qid, seed)] = asyncio.create_task(runner.draft_plan(
                    phase=phase, question_id=qid, question=str(question["question_text"]),
                    temperature=temperature, seed=seed,
                ))
        await _gather_with_progress(
            list(plan_tasks.values()),
            description=f"{_phase_label_zh(phase)}：初步计划", unit="个计划",
        )

    rows = []
    intervention_total = sum(
        len(grouped) for qid, grouped in candidate_by_question.items()
        if qid in question_by_id
    ) * len(arms)
    intervention_progress = tqdm(
        total=intervention_total,
        desc=f"{_phase_label_zh(phase)}：Principle介入",
        unit="个候选-方案",
        dynamic_ncols=True, disable=not sys.stderr.isatty(), leave=True,
    )
    for qid, grouped_candidates in candidate_by_question.items():
        if qid not in question_by_id:
            continue
        question = question_by_id[qid]
        baseline = [await baseline_tasks[(qid, seed)] for seed in seeds]
        for candidate in grouped_candidates:
            for arm in arms:
                if arm == "direct_short":
                    contexts = {seed: render_solver_memory(candidate["content"]) for seed in seeds}
                    controls = baseline
                elif arm == "pre_condition":
                    card_record = pre_cards[candidate["candidate_id"]]
                    card = card_record.get("condition_card", {}) if card_record.get("status") == "use" else {}
                    contexts = {
                        seed: render_solver_condition_card(card) if card else "" for seed in seeds
                    }
                    controls = baseline
                elif arm == "post_plan":
                    contexts = {}
                    controls = []
                    for seed in seeds:
                        plan = await plan_tasks[(qid, seed)]
                        plan_text = str(plan.get("output", ""))
                        plan_only_context = f"[Initial Solver Plan]\n{plan_text}"
                        control = await runner.solve(
                            phase=phase, arm="plan_only", question_id=qid,
                            question=str(question["question_text"]),
                            expected_answer=str(question["expected_answer"]),
                            temperature=temperature, seed=seed,
                            principle_text=plan_only_context,
                        )
                        controls.append(control)
                        card_path = output_dir / "post_cards" / (
                            f"{candidate['candidate_id']}_s{seed}.json"
                        )
                        expected_card_hash = _json_hash({
                            "prompt": CONDITION_CARD_PROMPT.format(
                                question=str(question["question_text"]),
                                principle=json.dumps(
                                    _solver_safe_principle(candidate["content"]),
                                    ensure_ascii=False, indent=2,
                                ),
                                initial_plan=plan_text,
                            ),
                            "model": client.config.auditor_model,
                            "max_tokens": 3072,
                        })
                        if card_path.exists():
                            card_record = load_json(card_path)
                            if card_record.get("input_hash") != expected_card_hash:
                                _archive_stale(
                                    card_path,
                                    output_root=output_dir / "post_cards" / "stale_checkpoints",
                                    reason="post_plan_card_input_or_prompt_hash_changed",
                                )
                                card_record = {}
                        else:
                            card_record = {}
                        if not card_record:
                            async with card_semaphore:
                                card_record = await _make_condition_card(
                                    client, question=str(question["question_text"]),
                                    content=candidate["content"], initial_plan=plan_text,
                                )
                            atomic_save_json(card_path, card_record)
                        card = card_record.get("condition_card", {}) if card_record.get("status") == "use" else {}
                        contexts[seed] = (
                            plan_only_context + "\n\n" + render_solver_condition_card(card)
                            if card else ""
                        )
                else:
                    raise ValueError(f"Unknown intervention arm: {arm}")
                treatment = []
                for index, seed in enumerate(seeds):
                    if contexts[seed]:
                        treatment.append(await runner.solve(
                            phase=phase, arm=arm, question_id=qid,
                            question=str(question["question_text"]),
                            expected_answer=str(question["expected_answer"]),
                            temperature=temperature, seed=seed,
                            principle_text=contexts[seed], candidate=candidate,
                        ))
                    else:
                        source = controls[index]
                        treatment.append(await runner.solve(
                            phase=phase, arm=arm, question_id=qid,
                            question=str(question["question_text"]),
                            expected_answer=str(question["expected_answer"]),
                            temperature=temperature, seed=seed, candidate=candidate,
                            derived_from=source["record_id"],
                        ))
                paired = _paired_counts(controls, treatment)
                control_correct = sum(item["correct"] for item in controls)
                guided_correct = sum(item["correct"] for item in treatment)
                rows.append({
                    "phase": phase, "arm": arm, "candidate_id": candidate["candidate_id"],
                    "question_id": qid, "level": candidate["level"], "evaluated": True,
                    "control": "plan_only" if arm == "post_plan" else "baseline",
                    "baseline_correct_at_10": control_correct,
                    "guided_correct_at_10": guided_correct,
                    "delta_correct": guided_correct - control_correct,
                    "paired": paired, "effect_label": _effect_label(guided_correct - control_correct),
                    "solver_use_count": sum(item.get("experience_decision") == "use" for item in treatment),
                    "solver_ignore_count": sum(item.get("experience_decision") == "ignore" for item in treatment),
                    "not_injected_count": sum(not item.get("injected") for item in treatment),
                    "privacy_violations": sorted({
                        violation for item in treatment for violation in item.get("privacy_violations", [])
                    }),
                })
                intervention_progress.update(1)
    intervention_progress.close()
    atomic_save_json(output_dir / "candidate_effects.json", rows)
    summary = {
        "phase": phase, "temperature": temperature,
        "question_count": len(question_by_id), "candidate_count": len(candidates),
        "arms": {
            arm: _summarize_effect_rows([item for item in rows if item["arm"] == arm])
            for arm in arms
        },
    }
    atomic_save_json(output_dir / "summary.json", summary)
    _write_effect_diagnostics(output_dir, rows, list(runner.records.values()))
    return summary


async def run_source_effect_test(
    config: Any,
    *,
    test_root: Path,
    ports: Sequence[int],
    model: str,
    concurrency: int = 4,
    timeout: int = 600,
) -> dict[str, Any]:
    test_root = Path(test_root)
    temperature = config.stage2.principle_test_primary_temperature
    audit_selection = await _evaluate_direct_protocols(
        config, test_root=test_root, ports=ports, model=model,
        concurrency=concurrency, timeout=timeout, temperature=temperature,
    )
    winner = audit_selection["winner"]
    candidates = _accepted_candidates(test_root, winner)
    questions = load_json(test_root / "source_bundles" / "source40_bundles.json")
    intervention = await _evaluate_interventions_on_questions(
        config, test_root=test_root, phase="source", questions=questions,
        candidates=candidates, ports=ports, model=model, concurrency=concurrency,
        timeout=timeout, temperature=temperature, arms=INTERVENTION_ARMS,
    )
    eligible_arms = [
        arm for arm, stats in intervention["arms"].items()
        if stats["prompt_privacy_violation_count"] == 0 and stats["evaluated_count"]
    ]
    if not eligible_arms:
        raise RuntimeError("No intervention arm met privacy and completion requirements")
    winner_arm = sorted(eligible_arms, key=lambda arm: (
        intervention["arms"][arm]["harmful_rate"],
        -intervention["arms"][arm]["correct_delta_sum"],
        -intervention["arms"][arm]["gain"] + intervention["arms"][arm]["loss"],
        INTERVENTION_ARMS.index(arm),
    ))[0]
    selection = {
        "audit_protocol": winner,
        "intervention_arm": winner_arm,
        "temperature": temperature,
        "audit_selection": audit_selection,
        "intervention_summary": intervention,
    }
    atomic_save_json(test_root / "source_evaluation" / "frozen_protocol.json", selection)
    return selection


def _variant_questions(test_root: Path) -> list[dict[str, Any]]:
    require_variant_approvals(test_root)
    variants = load_json(test_root / "manifests" / "variant40.json")
    return [{
        "question_id": str(item["source_question_id"]),
        "question_text": item["variant_question"],
        "expected_answer": item["variant_answer"],
        "level": item["paired_source_level"],
        "variant": True,
    } for item in variants]


async def run_transfer_effect_test(
    config: Any,
    *,
    test_root: Path,
    ports: Sequence[int],
    model: str,
    concurrency: int = 4,
    timeout: int = 600,
) -> dict[str, Any]:
    test_root = Path(test_root)
    frozen = load_json(test_root / "source_evaluation" / "frozen_protocol.json")
    candidates = _accepted_candidates(test_root, frozen["audit_protocol"])
    return await _evaluate_interventions_on_questions(
        config, test_root=test_root, phase="transfer", questions=_variant_questions(test_root),
        candidates=candidates, ports=ports, model=model, concurrency=concurrency,
        timeout=timeout, temperature=config.stage2.principle_test_primary_temperature,
        arms=(frozen["intervention_arm"],),
    )


async def run_robustness_effect_test(
    config: Any,
    *,
    test_root: Path,
    ports: Sequence[int],
    model: str,
    concurrency: int = 4,
    timeout: int = 600,
) -> dict[str, Any]:
    test_root = Path(test_root)
    frozen = load_json(test_root / "source_evaluation" / "frozen_protocol.json")
    candidates = _accepted_candidates(test_root, frozen["audit_protocol"])
    source_questions = load_json(test_root / "source_bundles" / "source40_bundles.json")
    source = await _evaluate_interventions_on_questions(
        config, test_root=test_root, phase="robustness_source", questions=source_questions,
        candidates=candidates, ports=ports, model=model, concurrency=concurrency,
        timeout=timeout, temperature=config.stage2.principle_test_robustness_temperature,
        arms=(frozen["intervention_arm"],),
    )
    transfer = await _evaluate_interventions_on_questions(
        config, test_root=test_root, phase="robustness_transfer", questions=_variant_questions(test_root),
        candidates=candidates, ports=ports, model=model, concurrency=concurrency,
        timeout=timeout, temperature=config.stage2.principle_test_robustness_temperature,
        arms=(frozen["intervention_arm"],),
    )
    combined = {"source": source, "transfer": transfer, "frozen_protocol": frozen}
    atomic_save_json(test_root / "robustness_07" / "summary_combined.json", combined)
    return combined


def build_principle_test_report(test_root: Path) -> dict[str, Any]:
    test_root = Path(test_root)
    source = load_json(test_root / "source_evaluation" / "interventions" / "summary.json")
    transfer = load_json(test_root / "transfer_evaluation" / "interventions" / "summary.json")
    frozen = load_json(test_root / "source_evaluation" / "frozen_protocol.json")
    robustness_path = test_root / "robustness_07" / "summary_combined.json"
    robustness = load_json(robustness_path) if robustness_path.exists() else {}
    arm = frozen["intervention_arm"]
    source_stats = source["arms"][arm]
    transfer_stats = transfer["arms"][arm]
    source_rows = load_json(test_root / "source_evaluation" / "interventions" / "candidate_effects.json")
    transfer_rows = load_json(test_root / "transfer_evaluation" / "interventions" / "candidate_effects.json")
    helpful_source_questions = {
        item["question_id"] for item in source_rows
        if item["arm"] == arm and item["effect_label"] == "helpful"
    }
    positive_transfer_questions = {
        item["question_id"] for item in transfer_rows
        if item["arm"] == arm and int(item["delta_correct"]) > 0
    }
    negative_transfer_questions = {
        item["question_id"] for item in transfer_rows
        if item["arm"] == arm and int(item["delta_correct"]) < 0
    }
    source_level_positive = sum(
        value > 0 for value in source_stats.get("level_mean_accuracy_delta", {}).values()
    )
    robustness_gates = {
        "robust_source_direction_gate": False,
        "robust_transfer_direction_gate": False,
        "robust_source_harm_gate": False,
        "robust_transfer_harm_gate": False,
    }
    if robustness:
        robust_source = robustness["source"]["arms"][arm]
        robust_transfer = robustness["transfer"]["arms"][arm]
        robustness_gates = {
            "robust_source_direction_gate": (
                source_stats["question_mean_accuracy_delta"] > 0
                and robust_source["question_mean_accuracy_delta"] > 0
            ),
            "robust_transfer_direction_gate": (
                transfer_stats["question_mean_accuracy_delta"] > 0
                and robust_transfer["question_mean_accuracy_delta"] > 0
            ),
            "robust_source_harm_gate": robust_source["harmful_question_rate"] <= 0.10,
            "robust_transfer_harm_gate": robust_transfer["harmful_question_rate"] <= 0.10,
        }
    audit_summary = load_json(test_root / "audit_variants" / "summary.json")
    raw_summary = load_json(test_root / "raw_candidates" / "summary.json")
    acceptance = {
        "source_helpful_question_count": len(helpful_source_questions),
        "source_helpful_question_gate": len(helpful_source_questions) >= 24,
        "source_harmful_rate": source_stats["harmful_rate"],
        "source_harm_gate": source_stats["harmful_rate"] <= 0.10,
        "source_net_gain_gate": source_stats["gain"] > source_stats["loss"],
        "source_level_direction_gate": source_level_positive >= 3,
        "transfer_delta_gate": transfer_stats["correct_delta_sum"] > 0,
        "transfer_positive_vs_negative_gate": len(positive_transfer_questions) > len(negative_transfer_questions),
        "transfer_net_gain_gate": transfer_stats["gain"] > transfer_stats["loss"],
        "privacy_gate": (
            source_stats["prompt_privacy_violation_count"] == 0
            and transfer_stats["prompt_privacy_violation_count"] == 0
        ),
        "robustness_complete": bool(robustness),
        **robustness_gates,
    }
    acceptance["ready_to_scale"] = all(
        value for key, value in acceptance.items()
        if key.endswith("_gate") or key == "robustness_complete"
    )
    report = {
        "frozen_protocol": frozen,
        "raw_generation": raw_summary,
        "audit": audit_summary,
        "source": source,
        "transfer": transfer,
        "robustness": robustness,
        "acceptance": acceptance,
    }
    atomic_save_json(test_root / "report" / "summary.json", report)
    lines = [
        "# Principle Test 80 Report", "",
        f"- Audit protocol: `{frozen['audit_protocol']}`",
        f"- Intervention arm: `{arm}`",
        f"- Source helpful questions: {len(helpful_source_questions)} / 40",
        f"- Source gain/loss: {source_stats['gain']} / {source_stats['loss']}",
        f"- Source Levels with positive mean direction: {source_level_positive} / 4",
        f"- Source question-level mean delta: {source_stats['question_mean_accuracy_delta']:.4f}",
        f"- Source paired-bootstrap 95% CI: {source_stats['paired_bootstrap_question_95ci']}",
        f"- Transfer positive/negative questions: {len(positive_transfer_questions)} / {len(negative_transfer_questions)}",
        f"- Transfer gain/loss: {transfer_stats['gain']} / {transfer_stats['loss']}",
        f"- Transfer question-level mean delta: {transfer_stats['question_mean_accuracy_delta']:.4f}",
        f"- Transfer paired-bootstrap 95% CI: {transfer_stats['paired_bootstrap_question_95ci']}",
        f"- Solver use/ignore/not-injected: {source_stats['solver_use_count']} / {source_stats['solver_ignore_count']} / {source_stats['not_injected_count']}",
        f"- Ready to scale: {acceptance['ready_to_scale']}", "",
        "## Acceptance", "", "```json",
        json.dumps(acceptance, ensure_ascii=False, indent=2), "```", "",
    ]
    (test_root / "report" / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (test_root / "report" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def principle_test_status(test_root: Path) -> dict[str, Any]:
    root = Path(test_root)
    files = {
        "source_manifest": root / "manifests" / "source40.json",
        "variants": root / "manifests" / "variant40.json",
        "approvals": root / "variants" / "variant_approvals.json",
        "candidate_bank": root / "raw_candidates" / "candidate_bank.json",
        "audit_summary": root / "audit_variants" / "summary.json",
        "frozen_protocol": root / "source_evaluation" / "frozen_protocol.json",
        "transfer_summary": root / "transfer_evaluation" / "interventions" / "summary.json",
        "robustness": root / "robustness_07" / "summary_combined.json",
        "report": root / "report" / "summary.json",
    }
    return {
        "test_root": str(root),
        "artifacts": {name: {"exists": path.exists(), "path": str(path)} for name, path in files.items()},
        "human_approved_variants": (
            sum(
                item.get("approved") is True
                for item in load_json(files["approvals"])
                if isinstance(item, dict)
            ) if files["approvals"].exists() else 0
        ),
    }


async def run_principle_test_smoke(
    config: Any,
    *,
    test_root: Path,
    backbone: str,
    ports: Sequence[int],
    model: str,
    concurrency: int,
    timeout: int,
) -> dict[str, Any]:
    """Run the new protocols on the historical two-per-Level development set."""
    smoke_root = Path(test_root) / "smoke8"
    logger.info(
        "[Stage2最小流程测试] 开始：题目数=8，backbone=%s，端口=%s，模型=%s，并发=%d",
        backbone, ",".join(map(str, ports)), model, concurrency,
    )
    prepare_principle_test(
        config, test_root=smoke_root, backbone=backbone,
        sources_per_level=2, reserve_per_level=0,
        seed=config.stage2.principle_test_seed,
    )
    logger.info("[Stage2最小流程测试 1/3] 生成Principle")
    await generate_raw_candidates(
        config, test_root=smoke_root, concurrency=min(2, max(1, concurrency))
    )
    logger.info("[Stage2最小流程测试 2/3] 比较三种质量审查协议")
    audit = await run_audit_ablation(
        config, test_root=smoke_root, concurrency=min(2, max(1, concurrency))
    )
    logger.info("[Stage2最小流程测试 3/3] 使用Llama验证来源题纠偏效果")
    source = await run_source_effect_test(
        config, test_root=smoke_root, ports=ports, model=model,
        concurrency=concurrency, timeout=timeout,
    )
    summary = {
        "status": "complete", "question_count": 8,
        "audit": audit, "source": source,
    }
    atomic_save_json(smoke_root / "smoke_summary.json", summary)
    logger.info("[Stage2最小流程测试] 全部完成：题目数=8，结果目录=%s", smoke_root)
    return summary
