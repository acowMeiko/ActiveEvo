"""Qwen recovery and sampled Kimi quality audit for Stage-2 production.

The original DeepSeek artifacts are immutable inputs.  Failed or malformed
DeepSeek questions are regenerated into an isolated Qwen branch, then merged
with explicit per-question provenance.  Kimi audits a reproducible stratified
sample only; it never certifies unreviewed candidates.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.request_context import llm_request_context
from modules.execute_module import _repetition_diagnostics
from modules.stage2.model_client import Stage2ModelClient, parse_json_object
from modules.stage2.principle_generation import (
    build_compact_synthesis_prompt,
    build_diagnosis_prompt,
    diagnose_principles,
    synthesize_principles,
)
from modules.stage2.principle_test import _gather_with_progress, _json_hash
from modules.stage2.protocol_calibration import (
    _audit_prompt,
    _decision,
    _format_check,
    _parse_audit,
)
from utils.io_utils import atomic_save_json, load_json


logger = logging.getLogger(__name__)

RECOVERY_VERSION = "stage2_qwen_recovery.1"
RECOVERY_SALT = "stage2_qwen_recovery_v1_seed42"
KIMI_SAMPLE_SALT = "kimi_quality_sample_v1_seed42"
EXPECTED_RECOVERY_BY_LEVEL = {
    "level1": 74,
    "level2": 246,
    "level3": 291,
    "level4": 15,
}
VALID_QUESTION_STATUSES = {"complete", "no_reliable_principle"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


def _explicit_no_reliable(record: Mapping[str, Any]) -> bool:
    if record.get("status") != "no_reliable_principle":
        return False
    synthesis = record.get("synthesis") or {}
    if not synthesis.get("parse_success") or synthesis.get("finish_reason") != "stop":
        return False
    parsed = parse_json_object(str(synthesis.get("raw_response", "")))
    return isinstance(parsed.get("failure_families"), list) and not parsed["failure_families"]


def _keep_deepseek_record(record: Mapping[str, Any]) -> bool:
    return record.get("status") == "complete" or _explicit_no_reliable(record)


def _round_robin_recovery_order(
    ids_by_level: Mapping[str, Sequence[str]],
) -> list[str]:
    pools = {
        level: sorted(ids, key=lambda qid: _stable_hash(qid, RECOVERY_SALT))
        for level, ids in ids_by_level.items()
    }
    ordered: list[str] = []
    levels = ("level1", "level2", "level3", "level4")
    cursor = 0
    while any(pools.get(level) for level in levels):
        level = levels[cursor % len(levels)]
        cursor += 1
        if pools.get(level):
            ordered.append(pools[level].pop(0))
    return ordered


def _fit_qwen_context(bundle: Mapping[str, Any], max_principles: int) -> tuple[dict, dict]:
    """Remove duplicated pairing prose only when a diagnosis prompt is oversized.

    Every original rollout remains intact.  ``divergence_step_text`` is merely
    a deterministic pairing hint and often duplicates the same correct trace
    once for every wrong rollout; capping that duplicate prevents otherwise
    valid 32K-context requests from being rejected.
    """
    original = dict(bundle)
    prompt_chars = len(build_diagnosis_prompt(original, max_principles))
    if prompt_chars <= 70_000:
        return original, {"applied": False, "original_prompt_chars": prompt_chars}
    compact = dict(original)
    compact["trajectory_pairs"] = [
        {
            **dict(pair),
            "divergence_step_text": str(pair.get("divergence_step_text", ""))[:600],
        }
        for pair in original.get("trajectory_pairs", [])
        if isinstance(pair, Mapping)
    ]
    compact_chars = len(build_diagnosis_prompt(compact, max_principles))
    if compact_chars > 70_000:
        raise RuntimeError(
            f"Qwen diagnosis remains too large after deduplicating pairing hints: "
            f"question={bundle.get('question_id')} chars={compact_chars}"
        )
    return compact, {
        "applied": True,
        "operation": "cap_duplicate_divergence_step_text",
        "original_prompt_chars": prompt_chars,
        "compact_prompt_chars": compact_chars,
        "rollouts_preserved": len(compact.get("rollouts", [])),
    }


def prepare_qwen_recovery(target_root: Path) -> dict[str, Any]:
    """Freeze exactly the malformed/failed DeepSeek records to regenerate."""
    root = Path(target_root)
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    source_dir = root / "raw_candidates" / "questions"
    if len(bundles) != 1200:
        raise RuntimeError(f"Qwen recovery requires 1200 frozen bundles; found {len(bundles)}")

    keep_ids: list[str] = []
    recovery_by_level: dict[str, list[str]] = defaultdict(list)
    source_records: dict[str, dict[str, str]] = {}
    for qid, bundle in bundles.items():
        path = source_dir / f"{qid}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing DeepSeek generation record: {path}")
        record = load_json(path)
        source_records[qid] = {
            "status": str(record.get("status", "missing")),
            "record_sha256": _file_sha256(path),
        }
        if _keep_deepseek_record(record):
            keep_ids.append(qid)
        else:
            recovery_by_level[str(bundle["level"])].append(qid)

    observed = {level: len(recovery_by_level[level]) for level in EXPECTED_RECOVERY_BY_LEVEL}
    if observed != EXPECTED_RECOVERY_BY_LEVEL:
        raise RuntimeError(
            f"Unexpected Qwen recovery distribution: expected={EXPECTED_RECOVERY_BY_LEVEL} "
            f"observed={observed}"
        )
    recovery_ids = _round_robin_recovery_order(recovery_by_level)
    if len(keep_ids) != 574 or len(recovery_ids) != 626:
        raise RuntimeError(
            f"Expected 574 retained and 626 recovery questions; got {len(keep_ids)} and "
            f"{len(recovery_ids)}"
        )

    manifest = {
        "schema_version": RECOVERY_VERSION,
        "source_question_count": len(bundles),
        "retained_deepseek_question_ids": sorted(keep_ids),
        "recovery_question_ids": recovery_ids,
        "recovery_question_ids_by_level": {
            level: [qid for qid in recovery_ids if str(bundles[qid]["level"]) == level]
            for level in EXPECTED_RECOVERY_BY_LEVEL
        },
        "smoke8_question_ids": recovery_ids[:8],
        "monitor30_question_ids": recovery_ids[:30],
        "source_record_fingerprints": source_records,
        "source_bundle_hashes": {qid: _json_hash(bundle) for qid, bundle in bundles.items()},
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    path = root / "qwen_recovery" / "manifest.json"
    if path.exists() and load_json(path).get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError(f"Frozen Qwen recovery manifest differs: {path}")
    atomic_save_json(path, manifest)
    return manifest


def _qwen_generation_hash(config: Any, bundle: Mapping[str, Any]) -> str:
    prompt = build_diagnosis_prompt(
        dict(bundle), config.stage2.max_principles_per_question,
    )
    return _json_hash({
        "version": RECOVERY_VERSION,
        "generator_id": "qwen",
        "bundle": bundle,
        "diagnosis_prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model": config.stage2.qwen_generator_model,
        "reasoning_effort": config.stage2.qwen_generator_reasoning_effort,
        "analysis_max_tokens": config.stage2.qwen_generation_analysis_max_tokens,
        "synthesis_max_tokens": config.stage2.qwen_generation_max_tokens,
        "retry_max_tokens": config.stage2.qwen_generation_retry_max_tokens,
        "max_principles": config.stage2.max_principles_per_question,
    })


def _checkpoint_matches(path: Path, expected_hash: str, *, require_success: bool) -> dict | None:
    if not path.exists():
        return None
    try:
        saved = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if saved.get("input_hash") != expected_hash:
        return None
    if require_success and not saved.get("success"):
        return None
    return saved


def _save_attempt(root: Path, qid: str, stage: str, payload: Mapping[str, Any]) -> None:
    attempt_dir = root / "qwen_recovery" / "attempt_history" / qid
    attempt_dir.mkdir(parents=True, exist_ok=True)
    number = len(list(attempt_dir.glob(f"{stage}_*.json"))) + 1
    atomic_save_json(attempt_dir / f"{stage}_{number:02d}.json", dict(payload))


def _qwen_candidate(record: Mapping[str, Any], family: Mapping[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"qwen:{record['question_id']}:{family['family_id']}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "candidate_id": f"qwen_{digest}",
        "question_id": str(record["question_id"]),
        "level": str(record["level"]),
        "family_id": str(family["family_id"]),
        "root_cause": str(family["root_cause"]),
        "supporting_wrong_rollout_ids": list(family["supporting_wrong_rollout_ids"]),
        "contrast_correct_rollout_ids": list(family["contrast_correct_rollout_ids"]),
        "content": dict(family["content"]),
        "generator_id": "qwen",
    }


def _valid_qwen_terminal(record: Mapping[str, Any], expected_hash: str) -> bool:
    if record.get("generation_input_hash") != expected_hash:
        return False
    if record.get("generator_id") != "qwen":
        return False
    if record.get("status") == "complete":
        return bool((record.get("synthesis") or {}).get("families"))
    return _explicit_no_reliable(record)


async def recover_qwen_generation(
    config: Any,
    *,
    target_root: Path,
    concurrency: int = 8,
    limit: int = 0,
) -> dict[str, Any]:
    """Regenerate a deterministic prefix of the frozen Qwen recovery set."""
    root = Path(target_root)
    manifest = prepare_qwen_recovery(root)
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    recovery_ids = list(manifest["recovery_question_ids"])
    selected_ids = recovery_ids[:limit] if limit > 0 else recovery_ids
    client = Stage2ModelClient(config, generator_id="qwen")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(qid: str) -> dict[str, Any]:
        bundle = bundles[qid]
        generation_bundle, context_compaction = _fit_qwen_context(
            bundle, config.stage2.max_principles_per_question,
        )
        generation_hash = _qwen_generation_hash(config, generation_bundle)
        record_path = root / "qwen_recovery" / "questions" / f"{qid}.json"
        if record_path.exists():
            try:
                saved = load_json(record_path)
            except (OSError, ValueError, json.JSONDecodeError):
                saved = {}
            if _valid_qwen_terminal(saved, generation_hash):
                return saved

        checkpoint_dir = root / "qwen_recovery" / "checkpoints" / qid
        diagnosis_hash = _json_hash({
            "generation_input_hash": generation_hash,
            "stage": "diagnosis",
        })
        diagnosis_path = checkpoint_dir / "diagnosis.json"
        diagnosis_checkpoint = _checkpoint_matches(
            diagnosis_path, diagnosis_hash, require_success=True,
        )
        async with semaphore:
            if diagnosis_checkpoint is None:
                logger.info("[Qwen恢复] 题目=%s，开始Thinking轨迹诊断", qid)
                with llm_request_context(qid, "qwen_recovery/diagnosis"):
                    diagnosis = await diagnose_principles(
                        client, generation_bundle, config.stage2.max_principles_per_question,
                    )
                diagnosis_checkpoint = {"input_hash": diagnosis_hash, **diagnosis}
                _save_attempt(root, qid, "diagnosis", diagnosis_checkpoint)
                atomic_save_json(diagnosis_path, diagnosis_checkpoint)
            diagnosis = diagnosis_checkpoint
            if not diagnosis.get("success"):
                record = {
                    "status": "retryable_failed",
                    "failure_stage": "diagnosis",
                    "question_id": qid,
                    "level": bundle["level"],
                    "generator_id": "qwen",
                    "generation_input_hash": generation_hash,
                    "diagnosis": diagnosis,
                }
                atomic_save_json(record_path, record)
                return record

            synthesis_hash = _json_hash({
                "generation_input_hash": generation_hash,
                "stage": "synthesis",
                "diagnosis_hash": _json_hash(diagnosis.get("diagnosis_text", "")),
                "prompt_hash": hashlib.sha256(build_compact_synthesis_prompt(
                    generation_bundle,
                    config.stage2.max_principles_per_question,
                    str(diagnosis.get("diagnosis_text", "")),
                ).encode("utf-8")).hexdigest(),
            })
            synthesis_path = checkpoint_dir / "synthesis.json"
            synthesis_checkpoint = _checkpoint_matches(
                synthesis_path, synthesis_hash, require_success=True,
            )
            if synthesis_checkpoint is None:
                logger.info("[Qwen恢复] 题目=%s，复用诊断并开始Thinking合成", qid)
                with llm_request_context(qid, "qwen_recovery/synthesis"):
                    synthesis = await synthesize_principles(
                        client, generation_bundle, config.stage2.max_principles_per_question,
                        str(diagnosis["diagnosis_text"]),
                        compact_synthesis=True,
                    )
                success = bool(synthesis.get("parse_success")) and not synthesis.get(
                    "schema_failures"
                )
                synthesis_checkpoint = {
                    "input_hash": synthesis_hash,
                    "success": success,
                    **synthesis,
                }
                _save_attempt(root, qid, "synthesis", synthesis_checkpoint)
                atomic_save_json(synthesis_path, synthesis_checkpoint)

            synthesis = synthesis_checkpoint
            if not synthesis.get("success"):
                feedback = json.dumps({
                    "parse_success": synthesis.get("parse_success"),
                    "schema_failures": synthesis.get("schema_failures", []),
                }, ensure_ascii=False)
                retry_hash = _json_hash({
                    "synthesis_hash": synthesis_hash,
                    "stage": "format_retry",
                    "feedback": feedback,
                })
                retry_path = checkpoint_dir / "format_retry.json"
                retry_checkpoint = _checkpoint_matches(
                    retry_path, retry_hash, require_success=True,
                )
                if retry_checkpoint is None:
                    with llm_request_context(qid, "qwen_recovery/synthesis_format_retry"):
                        retried = await synthesize_principles(
                            client, generation_bundle, config.stage2.max_principles_per_question,
                            str(diagnosis["diagnosis_text"]), correction_feedback=feedback,
                            compact_synthesis=True,
                        )
                    success = bool(retried.get("parse_success")) and not retried.get(
                        "schema_failures"
                    )
                    retry_checkpoint = {
                        "input_hash": retry_hash,
                        "success": success,
                        **retried,
                    }
                    _save_attempt(root, qid, "format_retry", retry_checkpoint)
                    atomic_save_json(retry_path, retry_checkpoint)
                synthesis = retry_checkpoint

        raw_parsed = parse_json_object(str(synthesis.get("raw_response", "")))
        explicit_empty = (
            synthesis.get("success")
            and isinstance(raw_parsed.get("failure_families"), list)
            and not raw_parsed["failure_families"]
        )
        families = synthesis.get("families") or []
        if synthesis.get("success") and families:
            status = "complete"
        elif explicit_empty:
            status = "no_reliable_principle"
        else:
            status = "retryable_failed"
        record = {
            "status": status,
            "question_id": qid,
            "level": bundle["level"],
            "bundle_hash": _json_hash(bundle),
            "generator_id": "qwen",
            "generation_input_hash": generation_hash,
            "context_compaction": context_compaction,
            "diagnosis": diagnosis,
            "synthesis": synthesis,
        }
        if status == "retryable_failed":
            record["failure_stage"] = "synthesis"
        atomic_save_json(record_path, record)
        return record

    records = await _gather_with_progress(
        [one(qid) for qid in selected_ids],
        description=f"Stage2 Qwen恢复 {len(selected_ids)}/{len(recovery_ids)}",
        unit="道题",
    )
    summary = build_mixed_candidate_bank(root)
    monitor = qwen_recovery_monitor(root, expected_count=len(selected_ids))
    result = {
        "requested_question_count": len(selected_ids),
        "status_counts": dict(Counter(row.get("status", "missing") for row in records)),
        "monitor": monitor,
        "mixed_bank": summary,
    }
    atomic_save_json(root / "qwen_recovery" / "summary.json", result)
    return result


def build_mixed_candidate_bank(target_root: Path) -> dict[str, Any]:
    root = Path(target_root)
    manifest = prepare_qwen_recovery(root)
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    deepseek_bank = load_json(root / "raw_candidates" / "candidate_bank.json")
    candidates = [dict(row) for row in deepseek_bank.get("candidates", [])]
    for candidate in candidates:
        candidate["generator_id"] = "deepseek"

    qwen_status = Counter()
    provenance: dict[str, dict[str, Any]] = {}
    for qid in manifest["retained_deepseek_question_ids"]:
        source = root / "raw_candidates" / "questions" / f"{qid}.json"
        record = load_json(source)
        provenance[qid] = {
            "generator_id": "deepseek",
            "status": record["status"],
            "record_sha256": _file_sha256(source),
        }
    for qid in manifest["recovery_question_ids"]:
        path = root / "qwen_recovery" / "questions" / f"{qid}.json"
        if not path.exists():
            qwen_status["pending"] += 1
            provenance[qid] = {"generator_id": "qwen", "status": "pending"}
            continue
        record = load_json(path)
        # The stored generator hash is validated during generation; merge only
        # explicit valid terminals and leave all other states pending.
        status = str(record.get("status", "retryable_failed"))
        qwen_status[status] += 1
        provenance[qid] = {
            "generator_id": "qwen",
            "status": status,
            "record_sha256": _file_sha256(path),
        }
        if status == "complete":
            for family in (record.get("synthesis") or {}).get("families", []):
                candidates.append(_qwen_candidate(record, family))

    ids = [str(row["candidate_id"]) for row in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Mixed Candidate Bank contains duplicate candidate IDs")
    candidates.sort(key=lambda row: str(row["candidate_id"]))
    mixed = {
        "schema_version": "stage2_mixed_candidates.1",
        "description": "Mixed DeepSeek/Qwen generation; generator provenance is mandatory.",
        "source_recovery_manifest_hash": manifest["manifest_hash"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    mixed["candidate_bank_hash"] = _json_hash(mixed)
    atomic_save_json(root / "raw_candidates" / "mixed_candidate_bank.json", mixed)

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for candidate in candidates:
        check = _format_check(candidate, bundles[str(candidate["question_id"])])
        row = {**candidate, "quality_status": "format_only_unreviewed"}
        if check["hard_invalid"]:
            invalid.append({"candidate_id": candidate["candidate_id"], "check": check})
        else:
            valid.append(row)
    valid_bank = {
        "schema_version": "stage2_format_valid_mixed_candidates.1",
        "description": "Format-valid only; candidates are not content-audit certified.",
        "source_candidate_bank_hash": mixed["candidate_bank_hash"],
        "candidate_count": len(valid),
        "candidates": valid,
    }
    valid_bank["candidate_bank_hash"] = _json_hash(valid_bank)
    atomic_save_json(root / "raw_candidates" / "format_valid_mixed_bank.json", valid_bank)

    complete = qwen_status.get("complete", 0) + qwen_status.get("no_reliable_principle", 0)
    summary = {
        "schema_version": RECOVERY_VERSION,
        "retained_deepseek_question_count": len(manifest["retained_deepseek_question_ids"]),
        "qwen_recovery_target_count": len(manifest["recovery_question_ids"]),
        "qwen_status_counts": dict(qwen_status),
        "qwen_terminal_count": complete,
        "all_questions_terminal": complete == len(manifest["recovery_question_ids"]),
        "mixed_candidate_count": len(candidates),
        "format_valid_candidate_count": len(valid),
        "format_invalid_candidate_count": len(invalid),
        "question_provenance": provenance,
        "mixed_candidate_bank_hash": mixed["candidate_bank_hash"],
        "format_valid_candidate_bank_hash": valid_bank["candidate_bank_hash"],
    }
    atomic_save_json(root / "mixed_generation_summary.json", summary)
    return summary

def qwen_recovery_monitor(target_root: Path, *, expected_count: int) -> dict[str, Any]:
    root = Path(target_root)
    manifest = prepare_qwen_recovery(root)
    ids = manifest["recovery_question_ids"][:expected_count]
    counts = Counter()
    final_nonempty = reasoning_nonempty = finish_stop = loops = 0
    candidate_total = candidate_valid = 0
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    for qid in ids:
        path = root / "qwen_recovery" / "questions" / f"{qid}.json"
        if not path.exists():
            counts["pending"] += 1
            continue
        record = load_json(path)
        counts[str(record.get("status", "missing"))] += 1
        diagnosis_response = (record.get("diagnosis") or {}).get("response") or {}
        synthesis = record.get("synthesis") or {}
        synthesis_stage = synthesis.get("synthesis_stage") or {}
        if str(diagnosis_response.get("reasoning", "")).strip():
            reasoning_nonempty += 1
        if str(synthesis.get("raw_response", "")).strip():
            final_nonempty += 1
        if synthesis.get("finish_reason") == "stop":
            finish_stop += 1
        repetition = _repetition_diagnostics(str(synthesis.get("raw_response", "")))
        # Pretty-printed JSON necessarily repeats braces, list delimiters and
        # schema field lines.  The generic solver heuristic treats three such
        # repeated lines as a likely loop, which is inappropriate for a
        # structured Candidate Bank.  Require a contiguous repeated block (or
        # an extreme repeated 8-gram ratio) for synthesis output instead.
        if repetition["hard_loop"] or repetition["repeated_8gram_ratio"] >= 0.08:
            loops += 1
        for family in synthesis.get("families", []) or []:
            candidate = _qwen_candidate(record, family)
            candidate_total += 1
            if not _format_check(candidate, bundles[qid])["hard_invalid"]:
                candidate_valid += 1
    terminal = counts["complete"] + counts["no_reliable_principle"]
    valid_rate = candidate_valid / candidate_total if candidate_total else 1.0
    passed = (
        terminal == len(ids)
        and counts["retryable_failed"] == 0
        and final_nonempty == len(ids)
        and reasoning_nonempty == len(ids)
        and finish_stop == len(ids)
        and loops == 0
        and valid_rate >= 0.95
    )
    result = {
        "expected_count": len(ids),
        "status_counts": dict(counts),
        "terminal_count": terminal,
        "diagnosis_reasoning_nonempty": reasoning_nonempty,
        "synthesis_final_nonempty": final_nonempty,
        "synthesis_finish_stop": finish_stop,
        "likely_loop_count": loops,
        "candidate_count": candidate_total,
        "format_valid_candidate_count": candidate_valid,
        "format_valid_rate": valid_rate,
        "passed": passed,
    }
    atomic_save_json(root / "qwen_recovery" / f"monitor_{len(ids):04d}.json", result)
    return result


def _interleave_correct_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pools: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[int(row["production_correct_count"])].append(row)
    for values in pools.values():
        values.sort(key=lambda row: row["sample_key"])
    result: list[dict[str, Any]] = []
    while any(pools.values()):
        for count in sorted(pools):
            if pools[count]:
                result.append(pools[count].pop(0))
    return result


def prepare_kimi_quality_sample(
    target_root: Path, *, sample_size: int = 80,
) -> dict[str, Any]:
    root = Path(target_root)
    summary = build_mixed_candidate_bank(root)
    if not summary["all_questions_terminal"]:
        raise RuntimeError("Kimi sampling is blocked until all Qwen recovery questions are terminal")
    bank = load_json(root / "raw_candidates" / "format_valid_mixed_bank.json")
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in bank["candidates"]:
        by_question[str(candidate["question_id"])].append(candidate)

    representatives = []
    for qid, rows in by_question.items():
        chosen = min(
            rows,
            key=lambda row: _stable_hash(
                row["generator_id"], qid, row["candidate_id"], KIMI_SAMPLE_SALT,
            ),
        )
        representatives.append({
            **chosen,
            "production_correct_count": int(bundles[qid]["production_correct_count"]),
            "sample_key": _stable_hash(
                chosen["generator_id"], qid, chosen["candidate_id"], KIMI_SAMPLE_SALT,
            ),
        })

    generators = ("deepseek", "qwen")
    levels = ("level1", "level2", "level3", "level4")
    quota = sample_size // (len(generators) * len(levels))
    if sample_size != quota * len(generators) * len(levels):
        raise ValueError("Kimi sample size must be divisible by 8")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    allocations: dict[str, int] = {}
    shortages: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for generator in generators:
        for level in levels:
            rows = [
                row for row in representatives
                if row["generator_id"] == generator and row["level"] == level
            ]
            ordered = (
                _interleave_correct_counts(rows)
                if level in {"level2", "level3"}
                else sorted(rows, key=lambda row: row["sample_key"])
            )
            grouped[(generator, level)] = ordered
            chosen = ordered[:quota]
            selected.extend(chosen)
            selected_ids.update(str(row["candidate_id"]) for row in chosen)
            allocations[f"{generator}:{level}"] = len(chosen)
            if len(chosen) < quota:
                shortages.append({
                    "generator_id": generator,
                    "level": level,
                    "requested": quota,
                    "available": len(chosen),
                })

    for generator in generators:
        target = sample_size // len(generators)
        current = sum(1 for row in selected if row["generator_id"] == generator)
        while current < target:
            remaining_by_level = []
            for level in levels:
                remaining = [
                    row for row in grouped[(generator, level)]
                    if str(row["candidate_id"]) not in selected_ids
                ]
                if remaining:
                    remaining_by_level.append((len(remaining), level, remaining[0]))
            if not remaining_by_level:
                break
            _, level, row = sorted(
                remaining_by_level, key=lambda item: (-item[0], item[1], item[2]["sample_key"])
            )[0]
            selected.append(row)
            selected_ids.add(str(row["candidate_id"]))
            allocations[f"{generator}:{level}"] += 1
            current += 1

    if len(selected) < sample_size:
        remaining = sorted(
            [row for row in representatives if str(row["candidate_id"]) not in selected_ids],
            key=lambda row: row["sample_key"],
        )
        selected.extend(remaining[: sample_size - len(selected)])
    if len(selected) != sample_size:
        raise RuntimeError(f"Only {len(selected)} candidates available for Kimi sample")

    items = [{
        "candidate_id": row["candidate_id"],
        "question_id": row["question_id"],
        "generator_id": row["generator_id"],
        "level": row["level"],
        "production_correct_count": row["production_correct_count"],
        "candidate_hash": _json_hash(row["content"]),
        "sample_key": row["sample_key"],
    } for row in selected]
    manifest = {
        "schema_version": "kimi_quality_sample.1",
        "purpose": "quality_estimation_only",
        "sample_size": sample_size,
        "sample_salt": KIMI_SAMPLE_SALT,
        "source_bank_hash": bank["candidate_bank_hash"],
        "allocations": allocations,
        "shortages": shortages,
        "items": items,
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    path = root / "quality_sample" / "kimi80_manifest.json"
    if path.exists() and load_json(path).get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError(f"Frozen Kimi sample differs: {path}")
    atomic_save_json(path, manifest)
    return manifest


def _wilson_interval(failures: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = failures / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


async def run_kimi_quality_sample(
    config: Any,
    *,
    target_root: Path,
    sample_size: int = 80,
    concurrency: int = 8,
    max_calls: int = 90,
) -> dict[str, Any]:
    root = Path(target_root)
    manifest = prepare_kimi_quality_sample(root, sample_size=sample_size)
    bank = load_json(root / "raw_candidates" / "format_valid_mixed_bank.json")
    candidates = {str(row["candidate_id"]): row for row in bank["candidates"]}
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    client = Stage2ModelClient(config, generator_id="deepseek")
    semaphore = asyncio.Semaphore(max(1, concurrency))
    call_lock = asyncio.Lock()
    audit_dir = root / "quality_sample" / "audits"
    calls = 0
    if audit_dir.exists():
        for saved_path in audit_dir.glob("*.json"):
            try:
                calls += int(load_json(saved_path).get("attempt_count", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    async def invoke(item: Mapping[str, Any], *, retry: bool = False) -> dict[str, Any]:
        nonlocal calls
        cid = str(item["candidate_id"])
        candidate = candidates[cid]
        bundle = bundles[str(item["question_id"])]
        evidence_ids = set(map(str, candidate["supporting_wrong_rollout_ids"])) | set(
            map(str, candidate["contrast_correct_rollout_ids"])
        )
        check = _format_check(candidate, bundle)
        prompt = _audit_prompt(bundle, candidate, check["semantic_hints"])
        input_hash = _json_hash({
            "manifest_hash": manifest["manifest_hash"],
            "candidate_content": candidate["content"],
            "question_id": item["question_id"],
            "evidence_ids": sorted(evidence_ids),
            "auditor_model": config.stage2.auditor_model,
        })
        path = root / "quality_sample" / "audits" / f"{cid}.json"
        previous = load_json(path) if path.exists() else None
        if previous and previous.get("input_hash") == input_hash:
            if previous.get("status") in {"audited_pass", "audited_flagged"}:
                return previous
            if previous.get("attempt_count", 0) >= 2:
                return previous
        async with call_lock:
            if calls >= max_calls:
                return {
                    "input_hash": input_hash,
                    "candidate_id": cid,
                    "status": "protocol_failed",
                    "protocol_errors": ["kimi_call_budget_exhausted"],
                    "attempt_count": int((previous or {}).get("attempt_count", 0)),
                }
            calls += 1
        async with semaphore:
            with llm_request_context(cid, "production/kimi_quality_sample"):
                response = await client.auditor(
                    prompt, auditor_id="kimi", thinking_enabled=False,
                    max_tokens=2048, json_mode=True,
                )
        audit, errors = _parse_audit(response.get("parsed"), evidence_ids)
        derived = _decision(audit) if not errors else "protocol_failed"
        reported = str(audit.get("decision", ""))
        if not errors and reported != derived:
            errors.append(f"decision_dimension_inconsistent:{reported}:{derived}")
        status = (
            "protocol_failed" if errors else
            "audited_pass" if derived == "pass" else
            "audited_flagged"
        )
        attempts = list((previous or {}).get("attempts", []))
        attempts.append({"retry": retry, "response": response})
        record = {
            "input_hash": input_hash,
            "candidate_id": cid,
            "question_id": item["question_id"],
            "status": status,
            "decision": derived,
            "audit": audit,
            "protocol_errors": errors,
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        atomic_save_json(path, record)
        return record

    initial = await _gather_with_progress(
        [invoke(item) for item in manifest["items"]],
        description="Stage2 Kimi分层质量抽样",
        unit="条",
    )
    retry_items = [
        item for item, record in zip(manifest["items"], initial)
        if record.get("status") == "protocol_failed"
    ][:max(0, max_calls - calls)]
    if retry_items:
        await _gather_with_progress(
            [invoke(item, retry=True) for item in retry_items],
            description="Stage2 Kimi协议失败补调用",
            unit="条",
        )

    records = []
    for item in manifest["items"]:
        path = root / "quality_sample" / "audits" / f"{item['candidate_id']}.json"
        records.append(load_json(path))
    status_counts = Counter(str(row["status"]) for row in records)
    dimension_failures = Counter()
    for row in records:
        for name, value in (row.get("audit") or {}).get("dimensions", {}).items():
            if isinstance(value, Mapping) and value.get("pass") is False:
                dimension_failures[name] += 1

    by_group: dict[str, dict[str, Any]] = {}
    for generator in ("deepseek", "qwen"):
        for level in ("level1", "level2", "level3", "level4"):
            ids = {
                str(item["candidate_id"]) for item in manifest["items"]
                if item["generator_id"] == generator and item["level"] == level
            }
            subset = [row for row in records if str(row["candidate_id"]) in ids]
            completed = [row for row in subset if row["status"] != "protocol_failed"]
            flagged = sum(row["status"] == "audited_flagged" for row in completed)
            by_group[f"{generator}:{level}"] = {
                "sample_count": len(subset),
                "completed_audit_count": len(completed),
                "flagged_count": flagged,
                "flagged_rate": flagged / len(completed) if completed else None,
                "wilson_95": _wilson_interval(flagged, len(completed)),
            }

    labels = {
        str(candidate["candidate_id"]): "format_only_unreviewed"
        for candidate in bank["candidates"]
    }
    for row in records:
        labels[str(row["candidate_id"])] = str(row["status"])
    atomic_save_json(root / "quality_sample" / "candidate_quality_labels.json", labels)
    summary = {
        "schema_version": "kimi_quality_sample_summary.1",
        "purpose": "quality_estimation_only",
        "sample_count": sample_size,
        "logical_provider_call_count": calls,
        "hard_call_limit": max_calls,
        "status_counts": dict(status_counts),
        "dimension_failure_counts": dict(dimension_failures),
        "by_generator_and_level": by_group,
        "unreviewed_candidate_count": len(bank["candidates"]) - sample_size,
    }
    atomic_save_json(root / "quality_sample" / "audit_summary.json", summary)
    report_lines = [
        "# Kimi分层抽样质量报告",
        "",
        "> 本报告只估计候选质量，不代表未抽样候选通过Kimi审查。",
        "",
        f"- 抽样数量：{sample_size}",
        f"- Kimi逻辑调用：{calls}/{max_calls}",
        f"- 通过：{status_counts.get('audited_pass', 0)}",
        f"- 标记问题：{status_counts.get('audited_flagged', 0)}",
        f"- 协议失败：{status_counts.get('protocol_failed', 0)}",
        "",
        "## 分层结果",
        "",
        "| 生成器×Level | 样本 | 有效审查 | 标记问题 | 比例 | 95% Wilson区间 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in by_group.items():
        rate = row["flagged_rate"]
        interval = row["wilson_95"]
        report_lines.append(
            f"| {name} | {row['sample_count']} | {row['completed_audit_count']} | "
            f"{row['flagged_count']} | {rate:.1%} | "
            f"[{interval[0]:.1%}, {interval[1]:.1%}] |"
            if rate is not None else
            f"| {name} | {row['sample_count']} | 0 | 0 | N/A | N/A |"
        )
    (root / "quality_sample" / "audit_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8",
    )
    return summary


def qwen_recovery_status(target_root: Path) -> dict[str, Any]:
    root = Path(target_root)
    result: dict[str, Any] = {"target_root": str(root)}
    for name, path in {
        "manifest": root / "qwen_recovery" / "manifest.json",
        "recovery": root / "qwen_recovery" / "summary.json",
        "mixed": root / "mixed_generation_summary.json",
        "kimi_sample": root / "quality_sample" / "audit_summary.json",
    }.items():
        result[name] = load_json(path) if path.exists() else {"status": "not_started"}
    return result
