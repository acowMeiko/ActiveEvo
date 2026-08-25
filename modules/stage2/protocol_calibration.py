"""Fresh Level-2/3 calibration for the production-cost Stage-2 route.

This runner is deliberately isolated from the Memory Store.  It tests the
planned production division of labor before that protocol is allowed to build
the 600-question Candidate Bank.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.request_context import llm_request_context
from modules.principle_schema import validate_principle_content
from modules.stage2.model_client import Stage2ModelClient
from modules.stage2.principle_generation import repair_principle
from modules.stage2.principle_test import (
    DEV8_BY_LEVEL,
    _PairedRunner,
    _eligible_bundle,
    _gather_with_progress,
    _healthy_actor_urls,
    _paired_counts,
    _stable_key,
    _load_bundles_by_level,
    _json_hash,
    generate_raw_candidates,
)
from modules.stage2.quality_audit import deterministic_leakage_check
from utils.io_utils import atomic_save_json, load_json


logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "stage2_protocol_l23.1"
LEVELS = ("level2", "level3")
SIX_AUDIT_DIMENSIONS = (
    "source_fidelity",
    "mathematical_validity",
    "instance_leakage",
    "condition_decidability",
    "action_executability",
    "verification_executability",
)
HARD_AUDIT_DIMENSIONS = {
    "source_fidelity", "mathematical_validity", "instance_leakage",
}


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _source_paths(root: Path) -> tuple[Path, Path]:
    return root / "manifests" / "source40.json", root / "source_bundles" / "source40_bundles.json"


def _load_prior_protocol_exclusions(
    roots: Sequence[Path],
) -> tuple[set[str], list[dict[str, str]]]:
    """Load frozen question IDs from earlier protocol-calibration rounds."""
    excluded: set[str] = set()
    parents: list[dict[str, str]] = []
    for raw_root in roots:
        root = Path(raw_root).resolve()
        manifest_path, _ = _source_paths(root)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing prior protocol manifest: {manifest_path}"
            )
        manifest = load_json(manifest_path)
        if manifest.get("schema_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"Incompatible prior protocol manifest: {manifest_path}"
            )
        question_ids = {
            str(value) for value in manifest.get("source_question_ids", [])
        }
        if not question_ids:
            raise RuntimeError(
                f"Prior protocol manifest contains no source questions: {manifest_path}"
            )
        excluded.update(question_ids)
        parents.append({
            "root": str(root),
            "manifest_hash": str(manifest.get("manifest_hash", "")),
            "question_count": str(len(question_ids)),
        })
    return excluded, parents


def prepare_l23_protocol(
    config: Any,
    *,
    test_root: Path,
    backbone: str = "llama",
    per_level: int = 10,
    seed: int = 42,
    exclude_protocol_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Freeze a fresh, middle-Learning-Zone sample with no development overlap."""
    root = Path(test_root)
    manifest_path, bundles_path = _source_paths(root)
    if manifest_path.exists() and bundles_path.exists():
        return load_json(manifest_path)

    by_level, stage1_hashes = _load_bundles_by_level(config, backbone)
    excluded_dev = {qid for values in DEV8_BY_LEVEL.values() for qid in values}
    excluded_prior, parent_rounds = _load_prior_protocol_exclusions(
        exclude_protocol_roots
    )
    excluded = excluded_dev | excluded_prior
    selected: dict[str, list[str]] = {}
    bundles: list[dict] = []
    salt = "stage2_protocol_l23_v1"
    expected_counts = {"level2": set(range(5, 8)), "level3": set(range(3, 5))}
    for level in LEVELS:
        pool = []
        for bundle in by_level[level]:
            qid = str(bundle["question_id"])
            signal = bundle.get("capability_signal") or {}
            correct_count = int(signal.get("correct_count", -1))
            eligible, _ = _eligible_bundle(bundle, level)
            if eligible and qid not in excluded and correct_count in expected_counts[level]:
                pool.append(bundle)
        pool.sort(key=lambda item: _stable_key(str(item["question_id"]), salt))
        if len(pool) < per_level:
            raise RuntimeError(f"{level} only has {len(pool)} fresh eligible bundles; need {per_level}")
        chosen = pool[:per_level]
        selected[level] = [str(item["question_id"]) for item in chosen]
        bundles.extend(chosen)

    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "backbone": backbone,
        "selection_salt": salt,
        "seed": seed,
        "levels": list(LEVELS),
        "per_level": per_level,
        "source_question_ids_by_level": selected,
        "source_question_ids": [qid for level in LEVELS for qid in selected[level]],
        "excluded_development_question_ids": sorted(excluded_dev),
        "excluded_prior_protocol_question_ids": sorted(excluded_prior),
        "parent_protocol_rounds": parent_rounds,
        "stage1_input_hashes": stage1_hashes,
        "source_bundle_hashes": {str(item["question_id"]): _hash(item) for item in bundles},
        "requires_variants": False,
    }
    manifest["manifest_hash"] = _hash(manifest)
    atomic_save_json(manifest_path, manifest)
    atomic_save_json(bundles_path, bundles)
    atomic_save_json(root / "manifests" / "source40_hashes.json", {
        "manifest_hash": manifest["manifest_hash"],
        "stage1_input_hashes": stage1_hashes,
        "source_bundle_hashes": manifest["source_bundle_hashes"],
    })
    return manifest


def _format_check(candidate: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    content, schema_errors = validate_principle_content(candidate.get("content"), require_extended=True)
    if content and not 2 <= len(content.get("required_evidence", [])) <= 5:
        schema_errors.append("required_evidence_must_contain_2_to_5_items")
    if content and not 1 <= len(content.get("failure_signals", [])) <= 4:
        schema_errors.append("failure_signals_must_contain_1_to_4_items")
    rollout_map = {str(row.get("rollout_id")): row for row in bundle.get("rollouts", []) if isinstance(row, dict)}
    reference_errors = []
    for field, expected in (("supporting_wrong_rollout_ids", False), ("contrast_correct_rollout_ids", True)):
        values = candidate.get(field)
        if not isinstance(values, list) or not values:
            reference_errors.append(f"missing_or_invalid_{field}")
            continue
        for value in values:
            row = rollout_map.get(str(value))
            if row is None or row.get("correct") is not expected:
                reference_errors.append(f"invalid_{field}:{value}")
    # These signals are audit evidence only.  They never decide accept/reject.
    semantic_hints = deterministic_leakage_check(
        content or dict(candidate.get("content") or {}),
        source_question=str(bundle.get("question_text", "")),
        expected_answer=str(bundle.get("expected_answer", "")),
    )
    hard_errors = list(schema_errors) + reference_errors
    return {
        "schema_errors": schema_errors,
        "reference_errors": reference_errors,
        "semantic_hints": semantic_hints,
        "hard_invalid": bool(hard_errors),
        "hard_errors": hard_errors,
        "normalized_content": content,
    }


def _audit_prompt(bundle: Mapping[str, Any], candidate: Mapping[str, Any], hints: Mapping[str, Any]) -> str:
    evidence_ids = set(candidate.get("supporting_wrong_rollout_ids", [])) | set(candidate.get("contrast_correct_rollout_ids", []))
    evidence = [row for row in bundle.get("rollouts", []) if isinstance(row, dict) and row.get("rollout_id") in evidence_ids]
    return f"""You are an independent mathematical quality auditor. Assess the Principle
only; do not predict whether a downstream solver will improve. Return raw JSON only and
keep every reason to one sentence. Do not provide a solution to the question.

For each of these six dimensions output {{"pass": true|false, "reason": "...",
"evidence_refs": ["exact rollout IDs"]}}:
{json.dumps(list(SIX_AUDIT_DIMENSIONS))}

Definitions:
- source_fidelity: the stated root cause follows from the cited correct and incorrect trajectories.
- mathematical_validity: the advice contains no false mathematical claim.
- instance_leakage: reject copied source answer, source-only numerical payload, entity, or conclusion; an abstract operation is not leakage.
- condition_decidability: trigger, required evidence, exclusions, and failure mode can be checked from the current question or a draft.
- action_executability: action is concrete enough to perform without source-only details.
- verification_executability: checks can be performed on a solution.

Set decision to reject only when source_fidelity, mathematical_validity, or instance_leakage fails.
Set decision to repair when only one of the last three dimensions fails. Otherwise set pass.

Question: {bundle['question_text']}
Reference answer (audit-only, never repeat it in a Principle): {bundle['expected_answer']}
Candidate Principle: {json.dumps(candidate['content'], ensure_ascii=False)}
Cited trajectories: {json.dumps(evidence, ensure_ascii=False)}
Non-binding heuristic flags (verify them; they are not verdicts): {json.dumps(hints, ensure_ascii=False)}

Return exactly:
{{"dimensions": {{...}}, "decision": "pass|repair|reject", "reasons": ["..."]}}
"""


def _parse_audit(value: Any, evidence_ids: set[str]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict) or not isinstance(value.get("dimensions"), dict):
        return {}, ["audit_must_contain_dimensions_object"]
    dimensions = {}
    errors = []
    for name in SIX_AUDIT_DIMENSIONS:
        item = value["dimensions"].get(name)
        if not isinstance(item, dict) or not isinstance(item.get("pass"), bool):
            errors.append(f"invalid_{name}")
            continue
        refs = item.get("evidence_refs", [])
        if not isinstance(refs, list) or not set(map(str, refs)).issubset(evidence_ids):
            errors.append(f"invalid_{name}_evidence_refs")
            refs = []
        dimensions[name] = {"pass": item["pass"], "reason": str(item.get("reason", "")).strip(), "evidence_refs": [str(x) for x in refs]}
    decision = str(value.get("decision", "")).lower()
    if decision not in {"pass", "repair", "reject"}:
        errors.append("invalid_decision")
    return {"dimensions": dimensions, "decision": decision, "reasons": value.get("reasons", []) if isinstance(value.get("reasons"), list) else []}, errors


def _decision(audit: Mapping[str, Any]) -> str:
    failed = {name for name, value in audit.get("dimensions", {}).items() if isinstance(value, dict) and value.get("pass") is False}
    if failed & HARD_AUDIT_DIMENSIONS:
        return "reject"
    return "repair" if failed else "pass"


async def _audit_candidate(client: Stage2ModelClient, bundle: dict, candidate: dict) -> dict[str, Any]:
    format_check = _format_check(candidate, bundle)
    if format_check["hard_invalid"]:
        return {"status": "terminal", "candidate_id": candidate["candidate_id"], "original_candidate": candidate, "format_check": format_check, "final_candidate": candidate, "final_decision": "reject", "audit": None, "repair": None, "reaudit": None}
    evidence_ids = set(candidate["supporting_wrong_rollout_ids"]) | set(candidate["contrast_correct_rollout_ids"])

    async def one_audit(target: dict, stage: str) -> dict:
        prompt = _audit_prompt(bundle, target, format_check["semantic_hints"])
        with llm_request_context(target["candidate_id"], stage):
            response = await client.auditor(prompt, auditor_id="kimi", thinking_enabled=False, max_tokens=2048, json_mode=True)
        audit, errors = _parse_audit(response.get("parsed"), evidence_ids)
        return {"response": response, "audit": audit, "protocol_errors": errors, "status": "complete" if not errors else "protocol_failed"}

    initial = await one_audit(candidate, "protocol_l23/kimi_audit")
    if initial["status"] != "complete":
        return {"status": "terminal", "candidate_id": candidate["candidate_id"], "original_candidate": candidate, "format_check": format_check, "audit": initial, "final_candidate": candidate, "final_decision": "protocol_failed", "repair": None, "reaudit": None}
    decision = _decision(initial["audit"])
    if decision != "repair":
        return {"status": "terminal", "candidate_id": candidate["candidate_id"], "original_candidate": candidate, "format_check": format_check, "audit": initial, "final_candidate": candidate, "final_decision": decision, "repair": None, "reaudit": None}
    with llm_request_context(candidate["candidate_id"], "protocol_l23/deepseek_repair"):
        repair = await repair_principle(client, bundle, candidate, initial["audit"])
    if not repair.get("success"):
        return {"status": "terminal", "candidate_id": candidate["candidate_id"], "original_candidate": candidate, "format_check": format_check, "audit": initial, "final_candidate": candidate, "final_decision": "reject", "repair": repair, "reaudit": None}
    repaired = {**candidate, "content": repair["content"], "parent_content": candidate["content"], "version": "repaired"}
    repaired_format = _format_check(repaired, bundle)
    if repaired_format["hard_invalid"]:
        return {"status": "terminal", "candidate_id": candidate["candidate_id"], "original_candidate": candidate, "format_check": format_check, "audit": initial, "final_candidate": repaired, "final_decision": "reject", "repair": repair, "reaudit": {"status": "skipped_hard_invalid", "format_check": repaired_format}}
    reaudit = await one_audit(repaired, "protocol_l23/kimi_reaudit")
    final_decision = _decision(reaudit["audit"]) if reaudit["status"] == "complete" else "protocol_failed"
    return {"status": "terminal", "candidate_id": candidate["candidate_id"], "original_candidate": candidate, "format_check": format_check, "audit": initial, "final_candidate": repaired, "final_decision": final_decision, "repair": repair, "reaudit": reaudit}


def _six_field_card(content: Mapping[str, Any]) -> str:
    evidence = "\n".join(f"  - {item}" for item in content.get("required_evidence", []))
    return "\n".join([
        "[Optional Conditional Experience]",
        "This is fallible advice. Use it only if the current problem satisfies its conditions.",
        f"When to consider: {content.get('trigger_condition', '')}",
        "Required evidence to check:", evidence,
        f"Do not use if: {content.get('exclusion_condition', '')}",
        f"Failure to avoid: {content.get('failure_mode', '')}",
        f"Suggested action: {content.get('action_guidance', '')}",
        f"Verification: {content.get('verification_step', '')}",
    ])


async def run_l23_protocol_calibration(
    config: Any,
    *,
    test_root: Path,
    ports: Sequence[int],
    model: str,
    generation_concurrency: int,
    validation_concurrency: int,
    timeout: int,
) -> dict[str, Any]:
    """Run DeepSeek generation, one Kimi audit, and paired Llama pass@10."""
    root = Path(test_root)
    manifest_path, bundles_path = _source_paths(root)
    if not manifest_path.exists() or not bundles_path.exists():
        raise RuntimeError("Run protocol-prepare before protocol-run")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != PROTOCOL_VERSION:
        raise RuntimeError("protocol-run requires a stage2_protocol_l23 manifest")
    # Existing generation helper is checkpointed and does not require variants.
    bank_summary = await generate_raw_candidates(config, test_root=root, concurrency=generation_concurrency)
    bank = load_json(root / "raw_candidates" / "candidate_bank.json")
    bundles = load_json(bundles_path)
    by_question = {str(item["question_id"]): item for item in bundles}
    client = Stage2ModelClient(config, generator_id="deepseek")
    audit_root = root / "production_audit"
    semaphore = asyncio.Semaphore(max(1, generation_concurrency))

    async def audit_one(candidate: dict) -> dict:
        path = audit_root / f"{candidate['candidate_id']}.json"
        input_hash = _hash({"candidate": candidate, "bundle": by_question[candidate["question_id"]], "protocol": PROTOCOL_VERSION})
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash and saved.get("status") == "terminal":
                return saved
        async with semaphore:
            record = await _audit_candidate(client, by_question[candidate["question_id"]], candidate)
        record["input_hash"] = input_hash
        atomic_save_json(path, record)
        return record

    audits = await _gather_with_progress([audit_one(candidate) for candidate in bank["candidates"]], description="Kimi六维审查", unit="条Principle")
    atomic_save_json(audit_root / "all_records.json", audits)

    urls = _healthy_actor_urls(ports)
    runner = _PairedRunner(config, output_dir=root / "llama_validation", urls=urls, model=model, concurrency=validation_concurrency, timeout=timeout)
    seeds = tuple(config.stage2.principle_test_solver_seeds)
    baseline_tasks = {(str(bundle["question_id"]), seed): asyncio.create_task(runner.solve(phase="protocol_l23", arm="baseline", question_id=str(bundle["question_id"]), question=bundle["question_text"], expected_answer=bundle["expected_answer"], temperature=config.stage2.principle_test_primary_temperature, seed=seed)) for bundle in bundles for seed in seeds}
    await _gather_with_progress(list(baseline_tasks.values()), description="Llama L2/L3 Baseline", unit="次推理")

    async def validate_one(record: dict) -> dict:
        candidate = record["final_candidate"]
        qid = str(candidate["question_id"])
        baseline = [await baseline_tasks[(qid, seed)] for seed in seeds]
        if record.get("final_decision") != "pass":
            return {"candidate_id": candidate["candidate_id"], "question_id": qid, "level": candidate["level"], "evaluated": False, "decision": record.get("final_decision"), "baseline_correct_at_10": sum(bool(x["correct"]) for x in baseline)}
        treatment = await asyncio.gather(*[runner.solve(phase="protocol_l23", arm="guided", question_id=qid, question=by_question[qid]["question_text"], expected_answer=by_question[qid]["expected_answer"], temperature=config.stage2.principle_test_primary_temperature, seed=seed, principle_text=_six_field_card(candidate["content"]), candidate=candidate) for seed in seeds])
        base_correct = sum(bool(x["correct"]) for x in baseline)
        guided_correct = sum(bool(x["correct"]) for x in treatment)
        return {"candidate_id": candidate["candidate_id"], "question_id": qid, "level": candidate["level"], "evaluated": True, "decision": "pass", "baseline_correct_at_10": base_correct, "guided_correct_at_10": guided_correct, "delta_correct": guided_correct - base_correct, "paired": _paired_counts(baseline, treatment)}

    effects = await _gather_with_progress([validate_one(record) for record in audits], description="Llama L2/L3 Principle验证", unit="条Principle")
    atomic_save_json(root / "llama_validation" / "candidate_effects.json", effects)
    decision_counts = Counter(record.get("final_decision") for record in audits)
    evaluated = [row for row in effects if row.get("evaluated")]
    by_level = {}
    for level in LEVELS:
        rows = [row for row in evaluated if row["level"] == level]
        by_level[level] = {"candidate_count": len(rows), "mean_delta": sum(row["delta_correct"] for row in rows) / len(rows) if rows else 0.0}
    summary = {"schema_version": PROTOCOL_VERSION, "manifest_hash": manifest["manifest_hash"], "generation": bank_summary, "audit_decisions": dict(decision_counts), "audit_parse_rate": sum(record.get("final_decision") != "protocol_failed" for record in audits) / len(audits) if audits else 0.0, "effects": {"evaluated_candidate_count": len(evaluated), "gain": sum(row.get("paired", {}).get("gain", 0) for row in evaluated), "loss": sum(row.get("paired", {}).get("loss", 0) for row in evaluated), "helpful_candidate_count": sum(row.get("delta_correct", 0) >= 2 for row in evaluated), "harmful_candidate_count": sum(row.get("delta_correct", 0) <= -2 for row in evaluated), "by_level": by_level}}
    atomic_save_json(root / "summary.json", summary)
    return summary
