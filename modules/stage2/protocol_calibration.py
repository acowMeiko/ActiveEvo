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
import re
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
    _effect_label,
    _eligible_bundle,
    _gather_with_progress,
    _healthy_actor_urls,
    _paired_counts,
    _record_id,
    _stable_endpoint_index,
    _stable_key,
    _load_bundles_by_level,
    _json_hash,
    _summarize_effect_rows,
    generate_raw_candidates,
)
from modules.stage2.model_client import parse_json_object
from modules.stage2.quality_audit import deterministic_leakage_check
from modules.execute_module import (
    SOLVER_V1_FINALIZER_MAX_TOKENS,
    SOLVER_V1_FINALIZER_TEMPERATURE,
    SOLVER_V1_REASONER_MAX_TOKENS,
    SOLVER_V1_TOP_P,
    _completion_tokens,
    _finalizer_format_valid,
    _finish_reason,
    _repetition_diagnostics,
)
from template.prompt_template import (
    FORCED_TIMING_AFTER_DRAFT_PROMPT,
    FORCED_TIMING_AFTER_PLAN_PROMPT,
    FORCED_TIMING_PLAN_PROMPT,
    SOLVER_V1_FINALIZER_PROMPT,
)
from benchmarks.matcher import check_answer, extract_final_answer
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


INTERVENTION_COMPARE_VERSION = "principle_intervention_compare.1"
INTERVENTION_ARMS = ("p1_pre_solve", "p2_after_plan", "p3_after_draft")
INTERVENTION_JUDGE_PROMPT = """You are deciding whether one past mathematical
experience should be used on the current problem.

Read the current problem, the candidate experience routing fields, and the
optional solver draft. Decide from current-problem facts only. The experience
is fallible; reject it when the trigger is not present, required evidence is
missing, or the exclusion applies.

Return raw JSON only:
{{
  "assessment": ["short factual checks"],
  "decision": "use|reject",
  "matched_evidence": [
    {{"requirement": "...", "current_problem_fact": "..."}}
  ],
  "exclusion_found": "",
  "detected_failure_signal": "",
  "guidance": "One concise paragraph of optional advice for the solver. It must not mention source problems, memory, routing, retrieval, scores, or confidence."
}}

Current problem:
{question}

Candidate experience routing fields:
{principle}

Optional current solver draft or plan:
{draft}
"""


GENERIC_REVIEW_CARD = "\n".join([
    "[Optional Current-Draft Review]",
    "Review the previous draft for arithmetic mistakes, missing conditions, invalid shortcuts, and final-answer consistency.",
    "If the draft is already valid, keep its method. If a mistake is found, correct it using only the current problem facts.",
])


def _compact_current_draft(text: str, *, max_chars: int = 4500) -> str:
    """Keep P3 inside the served context while preserving the draft's main signal."""
    cleaned = "\n".join(line.rstrip() for line in str(text or "").splitlines()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head = cleaned[: max_chars // 2].rstrip()
    tail = cleaned[-max_chars // 2 :].lstrip()
    return "\n".join([
        head,
        "[... draft truncated to fit the model context ...]",
        tail,
    ])


def _plain_solver_guidance(judgment: Mapping[str, Any]) -> str:
    guidance = " ".join(str(judgment.get("guidance", "")).split()).strip()
    failure = " ".join(str(judgment.get("detected_failure_signal", "")).split()).strip()
    verification = ""
    matched = judgment.get("matched_evidence")
    if isinstance(matched, list) and matched:
        facts = []
        for item in matched[:3]:
            if isinstance(item, Mapping):
                fact = " ".join(str(item.get("current_problem_fact", "")).split()).strip()
                if fact:
                    facts.append(fact)
        if facts:
            verification = "Check these current-problem facts before applying it: " + "; ".join(facts) + "."
    pieces = [
        "[Optional Current-Problem Guidance]",
        "Use this only if it remains mathematically valid during the solution.",
    ]
    if failure:
        pieces.append(f"Avoid this failure mode: {failure}.")
    if guidance:
        pieces.append(guidance)
    if verification:
        pieces.append(verification)
    return "\n".join(pieces).strip()


def _validate_intervention_judgment(value: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict):
        return {}, ["judgment_must_be_object"]
    errors: list[str] = []
    decision = str(value.get("decision", "")).strip().lower()
    if decision not in {"use", "reject"}:
        errors.append("decision_must_be_use_or_reject")
        decision = "reject"
    assessment = value.get("assessment", [])
    if not isinstance(assessment, list):
        errors.append("assessment_must_be_list")
        assessment = []
    matched = value.get("matched_evidence", [])
    if not isinstance(matched, list):
        errors.append("matched_evidence_must_be_list")
        matched = []
    normalized_matched = []
    for index, item in enumerate(matched):
        if not isinstance(item, Mapping):
            errors.append(f"matched_evidence_{index}_must_be_object")
            continue
        requirement = " ".join(str(item.get("requirement", "")).split()).strip()
        fact = " ".join(str(item.get("current_problem_fact", "")).split()).strip()
        if requirement or fact:
            normalized_matched.append({
                "requirement": requirement,
                "current_problem_fact": fact,
            })
    guidance = " ".join(str(value.get("guidance", "")).split()).strip()
    if decision == "use" and not guidance:
        errors.append("guidance_required_when_use")
    return {
        "assessment": [" ".join(str(item).split()).strip() for item in assessment if str(item).strip()],
        "decision": decision,
        "matched_evidence": normalized_matched,
        "exclusion_found": " ".join(str(value.get("exclusion_found", "")).split()).strip(),
        "detected_failure_signal": " ".join(str(value.get("detected_failure_signal", "")).split()).strip(),
        "guidance": guidance,
    }, errors


def _source_root_name(path: Path) -> str:
    return Path(path).resolve().name


def _load_intervention_sources(source_roots: Sequence[Path]) -> dict[str, Any]:
    bundles: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    p0_rows: list[dict[str, Any]] = []
    p0_rollouts: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for raw_root in source_roots:
        root = Path(raw_root)
        manifest_path, bundles_path = _source_paths(root)
        effects_path = root / "llama_validation" / "candidate_effects.json"
        rollouts_path = root / "llama_validation" / "rollouts.json"
        audits_path = root / "production_audit" / "all_records.json"
        if not all(path.exists() for path in (manifest_path, bundles_path, effects_path, rollouts_path, audits_path)):
            missing = [
                str(path) for path in (manifest_path, bundles_path, effects_path, rollouts_path, audits_path)
                if not path.exists()
            ]
            raise FileNotFoundError(f"Missing completed protocol artifacts: {missing}")
        manifest = load_json(manifest_path)
        manifests.append({
            "root": str(root),
            "manifest_hash": manifest.get("manifest_hash", ""),
            "question_count": len(manifest.get("source_question_ids", [])),
        })
        for bundle in load_json(bundles_path):
            qid = str(bundle["question_id"])
            if qid in bundles:
                raise RuntimeError(f"Duplicate source question across roots: {qid}")
            bundles[qid] = bundle
        for record in load_json(audits_path):
            if record.get("final_decision") != "pass":
                continue
            candidate = dict(record["final_candidate"])
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in candidates:
                raise RuntimeError(f"Duplicate candidate_id across roots: {candidate_id}")
            candidate["source_root"] = _source_root_name(root)
            candidates[candidate_id] = candidate
        for row in load_json(effects_path):
            if row.get("evaluated"):
                row = dict(row)
                row["source_root"] = _source_root_name(root)
                p0_rows.append(row)
        for row in load_json(rollouts_path):
            row = dict(row)
            row["source_root"] = _source_root_name(root)
            p0_rollouts.append(row)
    p0_ids = {str(row["candidate_id"]) for row in p0_rows}
    candidates = {cid: item for cid, item in candidates.items() if cid in p0_ids}
    return {
        "bundles": bundles,
        "candidates": candidates,
        "p0_rows": p0_rows,
        "p0_rollouts": p0_rollouts,
        "manifests": manifests,
    }


async def _judge_intervention(
    runner: _PairedRunner,
    *,
    phase: str,
    arm: str,
    question_id: str,
    question: str,
    candidate: Mapping[str, Any],
    temperature: float,
    seed: int,
    draft: str = "",
) -> dict[str, Any]:
    output_dir = runner.output_dir / "judgments" / arm
    path = output_dir / f"{candidate['candidate_id']}_s{seed}.json"
    prompt = INTERVENTION_JUDGE_PROMPT.format(
        question=question,
        principle=json.dumps(_solver_safe(candidate["content"]), ensure_ascii=False, indent=2),
        draft=draft or "None.",
    )
    input_hash = _json_hash({
        "version": INTERVENTION_COMPARE_VERSION,
        "arm": arm,
        "prompt": prompt,
        "model": runner.model,
        "temperature": temperature,
        "seed": seed,
    })
    if path.exists():
        saved = load_json(path)
        if saved.get("input_hash") == input_hash:
            return saved
    primary = _stable_endpoint_index(question_id, temperature, seed, len(runner.urls))
    endpoints = [runner.urls[primary]]
    if len(runner.urls) > 1:
        endpoints.append(runner.urls[(primary + 1) % len(runner.urls)])
    response = {}
    endpoint_used = ""
    async with runner.semaphore:
        for attempt, endpoint in enumerate(endpoints, 1):
            with llm_request_context(candidate["candidate_id"], f"{phase}/{arm}/judge/seed={seed}/attempt={attempt}"):
                response, endpoint_used = await runner.execute._call_with_router_metadata(
                    prompt=prompt,
                    temperature=0.0,
                    top_p=0.95,
                    max_tokens=768,
                    repetition_penalty=1.0,
                    seed=seed,
                    url_override=endpoint,
                    timeout_override=min(runner.timeout, 120),
                )
            if response.get("text"):
                break
    parsed = parse_json_object(str(response.get("text", "")))
    judgment, errors = _validate_intervention_judgment(parsed)
    # One format-only repair on the same endpoint budget. It sees only the invalid
    # answer and schema, not hidden source information.
    repair_response = None
    if errors and response.get("text"):
        repair_prompt = "\n".join([
            "Convert the following answer into valid raw JSON with exactly the required schema.",
            "Do not change the substantive decision unless the original text explicitly says use or reject.",
            "Required schema:",
            '{"assessment": [], "decision": "use|reject", "matched_evidence": [{"requirement": "", "current_problem_fact": ""}], "exclusion_found": "", "detected_failure_signal": "", "guidance": ""}',
            "Original answer:",
            str(response.get("text", "")),
        ])
        with llm_request_context(candidate["candidate_id"], f"{phase}/{arm}/judge_format_repair/seed={seed}"):
            repair_response, endpoint_used = await runner.execute._call_with_router_metadata(
                prompt=repair_prompt,
                temperature=0.0,
                top_p=0.95,
                max_tokens=512,
                repetition_penalty=1.0,
                seed=seed,
                url_override=endpoint_used or endpoints[0],
                timeout_override=min(runner.timeout, 120),
            )
        parsed = parse_json_object(str(repair_response[0].get("text", "") if isinstance(repair_response, tuple) else repair_response.get("text", "")))
        judgment, errors = _validate_intervention_judgment(parsed)
    if errors:
        judgment = {
            "assessment": [],
            "decision": "reject",
            "matched_evidence": [],
            "exclusion_found": "",
            "detected_failure_signal": "",
            "guidance": "",
        }
    record = {
        "input_hash": input_hash,
        "phase": phase,
        "arm": arm,
        "candidate_id": candidate["candidate_id"],
        "question_id": question_id,
        "seed": seed,
        "decision": judgment["decision"],
        "judgment": judgment,
        "schema_errors": errors,
        "parse_success": not errors,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "response": response,
        "repair_response": repair_response,
        "endpoint": endpoint_used,
    }
    atomic_save_json(path, record)
    return record


def _solver_safe(content: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trigger_condition": content.get("trigger_condition", ""),
        "required_evidence": content.get("required_evidence", []),
        "exclusion_condition": content.get("exclusion_condition", ""),
        "failure_mode": content.get("failure_mode", ""),
        "failure_signals": content.get("failure_signals", []),
        "action_guidance": content.get("action_guidance", ""),
        "verification_step": content.get("verification_step", ""),
    }


def _baseline_by_seed(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for row in records:
        if row.get("arm") == "baseline" and row.get("status") == "complete":
            result[(str(row["question_id"]), int(row["seed"]))] = dict(row)
    return result


def _write_intervention_report(root: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# 40题 Principle 介入时机对比",
        "",
        f"- 题目数：{summary['question_count']}",
        f"- 候选数：{summary['candidate_count']}",
        f"- seeds：{summary['seeds']}",
        f"- temperature：{summary['temperature']}",
        "",
        "## 结果汇总",
        "",
        "| Arm | Control | Evaluated | Delta Sum | Gain | Loss | Harmful Rate | Question Mean Delta | Parse Success | Not Injected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, stats in summary["arms"].items():
        lines.append(
            f"| {arm} | {stats.get('control', '')} | {stats.get('evaluated_count', 0)} | "
            f"{stats.get('correct_delta_sum', 0)} | {stats.get('gain', 0)} | "
            f"{stats.get('loss', 0)} | {stats.get('harmful_rate', 0):.3f} | "
            f"{stats.get('question_mean_accuracy_delta', 0):.4f} | "
            f"{stats.get('judge_parse_success_rate', 1.0):.3f} | "
            f"{stats.get('not_injected_count', 0)} |"
        )
    lines.extend([
        "",
        "## 选择规则",
        "",
        "优先选择有害率最低、gain 大于 loss、题目级平均 delta 为正、且超过等成本 control 的方案。P3 必须超过 generic self-check 才能归因于 Principle。",
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


async def run_intervention_compare(
    config: Any,
    *,
    test_root: Path,
    source_roots: Sequence[Path],
    ports: Sequence[int],
    model: str,
    validation_concurrency: int,
    timeout: int,
) -> dict[str, Any]:
    """Compare P1/P2/P3 intervention timing using frozen 40-question artifacts."""
    root = Path(test_root)
    data = _load_intervention_sources(source_roots)
    bundles = data["bundles"]
    candidates = data["candidates"]
    prior_baseline = _baseline_by_seed(data["p0_rollouts"])
    p0_rows = [row for row in data["p0_rows"] if str(row.get("candidate_id")) in candidates]
    if not candidates:
        raise RuntimeError("No evaluated pass candidates found in source roots")
    seeds = tuple(config.stage2.principle_test_solver_seeds)
    temperature = float(config.stage2.principle_test_primary_temperature)
    for candidate in candidates.values():
        qid = str(candidate["question_id"])
        for seed in seeds:
            if (qid, int(seed)) not in prior_baseline:
                raise RuntimeError(f"Missing imported baseline for question={qid} seed={seed}")

    urls = _healthy_actor_urls(ports)
    runner = _PairedRunner(
        config,
        output_dir=root / "rollouts",
        urls=urls,
        model=model,
        concurrency=validation_concurrency,
        timeout=timeout,
    )
    atomic_save_json(root / "manifest.json", {
        "schema_version": INTERVENTION_COMPARE_VERSION,
        "source_manifests": data["manifests"],
        "question_count": len(bundles),
        "candidate_count": len(candidates),
        "candidate_ids": sorted(candidates),
        "seeds": list(seeds),
        "temperature": temperature,
        "model": model,
        "ports": list(ports),
    })
    atomic_save_json(root / "imported_p0" / "candidate_effects.json", p0_rows)
    atomic_save_json(root / "imported_p0" / "rollouts.json", data["p0_rollouts"])

    async def persist_derived(
        *,
        phase: str,
        arm: str,
        candidate: Mapping[str, Any],
        seed: int,
        control: Mapping[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        record_id = _record_id(
            phase, arm, str(candidate["question_id"]), temperature, seed,
            str(candidate["candidate_id"]),
        )
        if record_id in runner.records:
            return runner.records[record_id]
        return await runner._persist({
            **dict(control),
            "record_id": record_id,
            "phase": phase,
            "arm": arm,
            "candidate_id": str(candidate["candidate_id"]),
            "injected": False,
            "derived_from": control.get("record_id", ""),
            "experience_decision": reason,
        })

    plan_tasks: dict[tuple[str, int], asyncio.Task] = {}
    for qid, bundle in bundles.items():
        for seed in seeds:
            plan_tasks[(qid, int(seed))] = asyncio.create_task(runner.draft_plan(
                phase="intervention_compare",
                question_id=qid,
                question=str(bundle["question_text"]),
                temperature=temperature,
                seed=int(seed),
            ))
    await _gather_with_progress(
        list(plan_tasks.values()),
        description="P2初步计划生成",
        unit="个计划",
    )

    async def plan_only_for(qid: str, seed: int, plan_text: str) -> dict[str, Any]:
        return await runner.solve(
            phase="intervention_compare",
            arm="p2_plan_only",
            question_id=qid,
            question=str(bundles[qid]["question_text"]),
            expected_answer=str(bundles[qid]["expected_answer"]),
            temperature=temperature,
            seed=seed,
            principle_text=f"[Initial Solver Plan]\n{plan_text}",
        )

    plan_only_tasks: dict[tuple[str, int], asyncio.Task] = {}
    for qid in bundles:
        for seed in seeds:
            plan = await plan_tasks[(qid, int(seed))]
            plan_only_tasks[(qid, int(seed))] = asyncio.create_task(
                plan_only_for(qid, int(seed), str(plan.get("output", "")))
            )
    await _gather_with_progress(
        list(plan_only_tasks.values()),
        description="P2 plan-only对照",
        unit="次推理",
    )

    async def p3_control_for(qid: str, seed: int, baseline: Mapping[str, Any]) -> dict[str, Any]:
        draft = _compact_current_draft(
            str(((baseline.get("response") or {}).get("reasoner") or {}).get("output", ""))
        )
        return await runner.solve(
            phase="intervention_compare",
            arm="p3_generic_self_check",
            question_id=qid,
            question=str(bundles[qid]["question_text"]),
            expected_answer=str(bundles[qid]["expected_answer"]),
            temperature=temperature,
            seed=seed,
            principle_text="\n\n".join([
                "[Previous Solver Draft]",
                draft,
                GENERIC_REVIEW_CARD,
            ]),
        )

    p3_control_tasks: dict[tuple[str, int], asyncio.Task] = {}
    for qid in bundles:
        for seed in seeds:
            p3_control_tasks[(qid, int(seed))] = asyncio.create_task(
                p3_control_for(qid, int(seed), prior_baseline[(qid, int(seed))])
            )
    await _gather_with_progress(
        list(p3_control_tasks.values()),
        description="P3 generic self-check对照",
        unit="次推理",
    )

    rows: list[dict[str, Any]] = []

    async def run_candidate_arm(candidate: Mapping[str, Any], arm: str) -> dict[str, Any]:
        qid = str(candidate["question_id"])
        bundle = bundles[qid]
        controls: list[dict[str, Any]] = []
        treatment: list[dict[str, Any]] = []
        judgments: list[dict[str, Any]] = []
        for seed in seeds:
            seed = int(seed)
            baseline = prior_baseline[(qid, seed)]
            if arm == "p1_pre_solve":
                control = baseline
                draft = ""
            elif arm == "p2_after_plan":
                plan = await plan_tasks[(qid, seed)]
                control = await plan_only_tasks[(qid, seed)]
                draft = str(plan.get("output", ""))
            elif arm == "p3_after_draft":
                control = await p3_control_tasks[(qid, seed)]
                draft = _compact_current_draft(
                    str(((baseline.get("response") or {}).get("reasoner") or {}).get("output", ""))
                )
            else:
                raise ValueError(f"Unknown intervention arm: {arm}")
            controls.append(control)
            judgment = await _judge_intervention(
                runner,
                phase="intervention_compare",
                arm=arm,
                question_id=qid,
                question=str(bundle["question_text"]),
                candidate=candidate,
                temperature=temperature,
                seed=seed,
                draft=draft,
            )
            judgments.append(judgment)
            if judgment["decision"] != "use":
                treatment.append(await persist_derived(
                    phase="intervention_compare",
                    arm=arm,
                    candidate=candidate,
                    seed=seed,
                    control=control,
                    reason="judge_reject",
                ))
                continue
            guidance = _plain_solver_guidance(judgment["judgment"])
            if arm == "p2_after_plan":
                guidance = "\n\n".join(["[Initial Solver Plan]", draft, guidance])
            elif arm == "p3_after_draft":
                guidance = "\n\n".join(["[Previous Solver Draft]", draft, guidance])
            treatment.append(await runner.solve(
                phase="intervention_compare",
                arm=arm,
                question_id=qid,
                question=str(bundle["question_text"]),
                expected_answer=str(bundle["expected_answer"]),
                temperature=temperature,
                seed=seed,
                principle_text=guidance,
                candidate=candidate,
            ))
        paired = _paired_counts(controls, treatment)
        control_correct = sum(bool(item["correct"]) for item in controls)
        guided_correct = sum(bool(item["correct"]) for item in treatment)
        return {
            "phase": "intervention_compare",
            "arm": arm,
            "candidate_id": candidate["candidate_id"],
            "question_id": qid,
            "level": candidate["level"],
            "source_root": candidate.get("source_root", ""),
            "evaluated": True,
            "control": {
                "p1_pre_solve": "imported_baseline",
                "p2_after_plan": "plan_only",
                "p3_after_draft": "generic_self_check",
            }[arm],
            "control_correct_at_10": control_correct,
            "guided_correct_at_10": guided_correct,
            "delta_correct": guided_correct - control_correct,
            "paired": paired,
            "effect_label": _effect_label(guided_correct - control_correct),
            "judge_use_count": sum(item["decision"] == "use" for item in judgments),
            "judge_reject_count": sum(item["decision"] == "reject" for item in judgments),
            "judge_parse_success_count": sum(bool(item.get("parse_success")) for item in judgments),
            "not_injected_count": sum(not item.get("injected") for item in treatment),
            "solver_use_count": sum(item.get("experience_decision") == "use" for item in treatment),
            "solver_ignore_count": sum(item.get("experience_decision") == "ignore" for item in treatment),
            "privacy_violations": sorted({
                violation for item in treatment for violation in item.get("privacy_violations", [])
            }),
        }

    for arm in INTERVENTION_ARMS:
        arm_rows = await _gather_with_progress(
            [run_candidate_arm(candidate, arm) for candidate in candidates.values()],
            description=f"{arm}介入对比",
            unit="条Principle",
        )
        rows.extend(arm_rows)
        atomic_save_json(root / "arms" / arm / "candidate_effects.json", arm_rows)

    atomic_save_json(root / "effects" / "candidate_effects.json", rows)
    atomic_save_json(root / "rollouts" / "all_rollouts.json", list(runner.records.values()))
    p0_summary = _summarize_effect_rows(p0_rows)
    arms_summary = {"p0_direct_reused": {**p0_summary, "control": "imported_baseline"}}
    for arm in INTERVENTION_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        stats = _summarize_effect_rows(arm_rows)
        parse_total = len(arm_rows) * len(seeds)
        parse_success = sum(int(row.get("judge_parse_success_count", 0)) for row in arm_rows)
        stats.update({
            "control": arm_rows[0]["control"] if arm_rows else "",
            "judge_parse_success_rate": parse_success / parse_total if parse_total else 0.0,
            "judge_use_count": sum(int(row.get("judge_use_count", 0)) for row in arm_rows),
            "judge_reject_count": sum(int(row.get("judge_reject_count", 0)) for row in arm_rows),
        })
        arms_summary[arm] = stats
    summary = {
        "schema_version": INTERVENTION_COMPARE_VERSION,
        "source_manifests": data["manifests"],
        "question_count": len(bundles),
        "candidate_count": len(candidates),
        "seeds": list(seeds),
        "temperature": temperature,
        "arms": arms_summary,
        "output_root": str(root),
    }
    atomic_save_json(root / "summary.json", summary)
    _write_intervention_report(root, summary)
    return summary


# ---------------------------------------------------------------------------
# Forced timing comparison: every candidate is exposed at every seed.  This is
# intentionally separate from the historical gated intervention experiment.

FORCED_TIMING_VERSION = "principle_forced_timing.1"
FORCED_TIMING_ARMS = (
    "a_before_plan",
    "b_after_plan",
    "c_after_draft",
)
FORCED_TIMING_FIELDS = (
    "introspection",
    "scope",
    "trigger_condition",
    "required_evidence",
    "exclusion_condition",
    "failure_mode",
    "failure_signals",
    "action_guidance",
    "verification_step",
)


def _card_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).split()).strip() for item in value if str(item).strip()]


def render_forced_principle_card(content: Mapping[str, Any]) -> str:
    """Render the immutable, source-free card shared by all timing arms."""
    missing = [name for name in FORCED_TIMING_FIELDS if name not in content]
    if missing:
        raise ValueError(f"Principle is missing forced-timing fields: {missing}")
    required = _card_list(content.get("required_evidence"))
    signals = _card_list(content.get("failure_signals"))
    if not required or not signals:
        raise ValueError("required_evidence and failure_signals must be non-empty lists")

    def text_field(name: str) -> str:
        value = " ".join(str(content.get(name, "")).split()).strip()
        if not value:
            raise ValueError(f"Principle field is empty: {name}")
        return value

    return "\n".join([
        "[Reasoning Principle]",
        "",
        "Lesson:",
        text_field("introspection"),
        "",
        "Scope:",
        text_field("scope"),
        "",
        "Relevant when:",
        text_field("trigger_condition"),
        "",
        "Evidence to check:",
        *[f"- {item}" for item in required],
        "",
        "Do not apply mechanically when:",
        text_field("exclusion_condition"),
        "",
        "Failure pattern to avoid:",
        text_field("failure_mode"),
        "",
        "Observable warning signs:",
        *[f"- {item}" for item in signals],
        "",
        "Recommended procedure:",
        text_field("action_guidance"),
        "",
        "Verification procedure:",
        text_field("verification_step"),
    ]).strip()


def _forced_prompt_privacy_violations(
    prompt: str,
    *,
    candidate: Mapping[str, Any],
) -> list[str]:
    lowered = prompt.lower()
    violations: list[str] = []
    forbidden_labels = {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "supporting_rollout_id": "supporting_wrong_rollout_ids",
        "contrast_rollout_id": "contrast_correct_rollout_ids",
        "retrieval_score": "retrieval_score",
        "audit_decision": "audit_decision",
        "expected_answer_label": "expected_answer",
    }
    for name, marker in forbidden_labels.items():
        if marker and marker.lower() in lowered:
            violations.append(name)
    for rollout_id in (
        list(candidate.get("supporting_wrong_rollout_ids", []))
        + list(candidate.get("contrast_correct_rollout_ids", []))
    ):
        if str(rollout_id) and str(rollout_id).lower() in lowered:
            violations.append("rollout_id")
    return sorted(set(violations))


def _load_forced_timing_cache(
    cache_root: Path,
    *,
    bundles: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    rollout_path = cache_root / "rollouts" / "all_rollouts.json"
    plan_dir = cache_root / "rollouts" / "initial_plans"
    manifest_path = cache_root / "manifest.json"
    if not rollout_path.exists() or not plan_dir.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Incomplete intervention cache under {cache_root}; expected manifest, "
            "rollouts/all_rollouts.json, and rollouts/initial_plans/"
        )
    rows = load_json(rollout_path)
    plan_only = {
        (str(row["question_id"]), int(row["seed"])): row
        for row in rows
        if row.get("arm") == "p2_plan_only" and row.get("status") == "complete"
    }
    self_check = {
        (str(row["question_id"]), int(row["seed"])): row
        for row in rows
        if row.get("arm") == "p3_generic_self_check" and row.get("status") == "complete"
    }
    plans: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(plan_dir.glob("*.json")):
        record = load_json(path)
        if not record.get("infrastructure_success"):
            continue
        match = re.match(r"(.+)_t[-0-9.]+_s(\d+)\.json$", path.name)
        if match:
            plans[(match.group(1), int(match.group(2)))] = record

    required = {(str(qid), int(seed)) for qid in bundles for seed in seeds}
    missing = {
        "initial_plan": sorted(required - set(plans)),
        "plan_only": sorted(required - set(plan_only)),
        "generic_self_check": sorted(required - set(self_check)),
    }
    if any(missing.values()):
        counts = {name: len(values) for name, values in missing.items()}
        raise RuntimeError(f"Forced-timing cache is incomplete: {counts}")
    manifest = load_json(manifest_path)
    manifest_ids = set(map(str, manifest.get("candidate_ids", [])))
    return {
        "manifest": manifest,
        "manifest_candidate_ids": manifest_ids,
        "plans": plans,
        "plan_only": plan_only,
        "generic_self_check": self_check,
        "counts": {
            "initial_plan": len(plans),
            "plan_only": len(plan_only),
            "generic_self_check": len(self_check),
        },
    }


def _forced_question_effects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), str(row["question_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (arm, question_id), items in sorted(grouped.items()):
        deltas = [int(item.get("delta_correct", 0)) for item in items]
        output.append({
            "arm": arm,
            "question_id": question_id,
            "level": str(items[0].get("level", "")),
            "candidate_count": len(items),
            "mean_candidate_delta": sum(deltas) / len(deltas),
            "mean_candidate_accuracy_delta": sum(
                float(item.get("accuracy_delta", 0.0)) for item in items
            ) / len(items),
            "positive_principle_count": sum(delta > 0 for delta in deltas),
            "negative_principle_count": sum(delta < 0 for delta in deltas),
            "neutral_principle_count": sum(delta == 0 for delta in deltas),
            "any_helpful_principle": any(delta >= 2 for delta in deltas),
            "all_principles_harmful": bool(deltas) and all(delta <= -2 for delta in deltas),
            "best_candidate_delta_diagnostic_only": max(deltas),
        })
    return output


def _processing_status(text: str) -> str:
    match = re.search(
        r"Principle Processing:\s*\n?\s*([a-z_]+)", text or "", flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else "unparsed"


async def run_forced_timing_compare(
    config: Any,
    *,
    test_root: Path,
    source_roots: Sequence[Path],
    comparison_cache_root: Path,
    ports: Sequence[int],
    model: str,
    validation_concurrency: int,
    timeout: int,
    smoke_only: bool = False,
) -> dict[str, Any]:
    """Compare fixed Principle exposure at plan, execution, and review time."""
    root = Path(test_root)
    data = _load_intervention_sources(source_roots)
    bundles = data["bundles"]
    all_candidates = data["candidates"]
    full_seeds = tuple(int(seed) for seed in config.stage2.principle_test_solver_seeds)
    temperature = float(config.stage2.principle_test_primary_temperature)
    baseline = _baseline_by_seed(data["p0_rollouts"])
    if not all_candidates:
        raise RuntimeError("No audited Principle candidates are available")
    for candidate in all_candidates.values():
        for seed in full_seeds:
            key = (str(candidate["question_id"]), seed)
            if key not in baseline:
                raise RuntimeError(f"Missing baseline question={key[0]} seed={seed}")

    cache = _load_forced_timing_cache(
        comparison_cache_root, bundles=bundles, seeds=full_seeds,
    )
    if cache["manifest_candidate_ids"] != set(all_candidates):
        raise RuntimeError("Candidate IDs differ between source roots and comparison cache")

    cards: dict[str, dict[str, str]] = {}
    for candidate_id, candidate in sorted(all_candidates.items()):
        card = render_forced_principle_card(candidate["content"])
        cards[candidate_id] = {
            "candidate_id": candidate_id,
            "question_id": str(candidate["question_id"]),
            "text": card,
            "sha256": hashlib.sha256(card.encode("utf-8")).hexdigest(),
        }
        atomic_save_json(root / "principle_cards" / f"{candidate_id}.json", cards[candidate_id])

    cache_validation = {
        "valid": True,
        "comparison_cache_root": str(comparison_cache_root),
        "question_count": len(bundles),
        "candidate_count": len(all_candidates),
        "seed_count": len(full_seeds),
        "expected_question_seed_controls": len(bundles) * len(full_seeds),
        "cache_counts": cache["counts"],
        "candidate_hash_match": True,
    }
    atomic_save_json(root / "cache_validation.json", cache_validation)

    # A full run always proves the live path first.  The recursive smoke call
    # writes the exact same hashed checkpoints used by the full run, so those
    # 18 treatment rollouts are reused rather than sampled again.
    if not smoke_only:
        logger.info("[固定注入时机] 先运行2条Principle×3 seeds的真实Smoke Test")
        smoke_summary = await run_forced_timing_compare(
            config,
            test_root=root,
            source_roots=source_roots,
            comparison_cache_root=comparison_cache_root,
            ports=ports,
            model=model,
            validation_concurrency=validation_concurrency,
            timeout=timeout,
            smoke_only=True,
        )
        smoke_errors = []
        for arm, stats in smoke_summary.get("arms", {}).items():
            if int(stats.get("infrastructure_failure_count", 0)):
                smoke_errors.append(f"{arm}:infrastructure_failure")
            if int(stats.get("not_injected_count", 0)):
                smoke_errors.append(f"{arm}:principle_not_injected")
            if int(stats.get("privacy_violation_count", 0)):
                smoke_errors.append(f"{arm}:prompt_privacy_violation")
            if int(stats.get("paired_seed_count", 0)) != 6:
                smoke_errors.append(f"{arm}:expected_6_pairs")
        smoke_validation = {
            "passed": not smoke_errors,
            "errors": smoke_errors,
            "expected_candidates": 2,
            "expected_seeds": [42, 43, 44],
            "summary": smoke_summary,
        }
        atomic_save_json(root / "smoke_validation.json", smoke_validation)
        if smoke_errors:
            raise RuntimeError(
                "Forced-timing Smoke Test failed; full run stopped: "
                + ", ".join(smoke_errors)
            )
        logger.info("[固定注入时机] Smoke Test通过，开始完整74条×10 seeds实验")

    selected_candidates = list(sorted(all_candidates.values(), key=lambda item: str(item["candidate_id"])))
    seeds = full_seeds
    if smoke_only:
        selected_candidates = selected_candidates[:2]
        seeds = full_seeds[:3]

    manifest = {
        "schema_version": FORCED_TIMING_VERSION,
        "run_scope": "smoke" if smoke_only else "full",
        "source_manifests": data["manifests"],
        "comparison_cache_root": str(comparison_cache_root),
        "question_count": len(bundles),
        "candidate_count": len(all_candidates),
        "run_candidate_count": len(selected_candidates),
        "candidate_ids": sorted(all_candidates),
        "run_candidate_ids": [str(item["candidate_id"]) for item in selected_candidates],
        "full_seeds": list(full_seeds),
        "run_seeds": list(seeds),
        "temperature": temperature,
        "model": model,
        "ports": list(ports),
        "principle_field_names": list(FORCED_TIMING_FIELDS),
    }
    atomic_save_json(root / "manifest.json", manifest)
    atomic_save_json(root / "imported_p0" / "candidate_effects.json", data["p0_rows"])

    urls = _healthy_actor_urls(ports)
    solver_runner = _PairedRunner(
        config,
        output_dir=root / "rollouts" / "a_solver",
        urls=urls,
        model=model,
        concurrency=validation_concurrency,
        timeout=timeout,
    )

    async def endpoint_pair(question_id: str, seed: int) -> list[str]:
        primary = _stable_endpoint_index(question_id, temperature, seed, len(urls))
        values = [urls[primary]]
        if len(urls) > 1:
            values.append(urls[(primary + 1) % len(urls)])
        return values

    async def forced_plan(candidate: Mapping[str, Any], seed: int) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        qid = str(candidate["question_id"])
        card_info = cards[candidate_id]
        prompt = FORCED_TIMING_PLAN_PROMPT.format(
            question=bundles[qid]["question_text"], principle_card=card_info["text"],
        )
        path = root / "arms" / "a_before_plan" / "plans" / f"{candidate_id}_s{seed}.json"
        input_hash = _json_hash({
            "version": FORCED_TIMING_VERSION,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": 768,
        })
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash and saved.get("status") == "complete":
                return saved
        response: dict[str, Any] = {}
        endpoint_used = ""
        trace = []
        async with solver_runner.semaphore:
            for attempt, endpoint in enumerate(await endpoint_pair(qid, seed), 1):
                with llm_request_context(candidate_id, f"forced_timing/a_plan/seed={seed}/attempt={attempt}"):
                    response, endpoint_used = await solver_runner.execute._call_with_router_metadata(
                        prompt=prompt,
                        temperature=temperature,
                        top_p=SOLVER_V1_TOP_P,
                        max_tokens=768,
                        repetition_penalty=1.0,
                        seed=seed,
                        url_override=endpoint,
                        timeout_override=min(timeout, 120),
                    )
                trace.append({"endpoint": endpoint, "success": bool(response.get("text"))})
                if response.get("text"):
                    break
        output = str(response.get("text") or "").strip()
        required_headings = (
            "Principle Processing:", "Problem Structure:", "Target:",
            "Constraints:", "Planned Method:", "Ordered Steps:", "Critical Checks:",
        )
        record = {
            "input_hash": input_hash,
            "status": "complete" if output else "infrastructure_failed",
            "arm": "a_before_plan",
            "stage": "plan",
            "candidate_id": candidate_id,
            "question_id": qid,
            "seed": seed,
            "principle_injected": True,
            "principle_card_hash": card_info["sha256"],
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output": output,
            "processing_status": _processing_status(output),
            "format_valid": all(heading in output for heading in required_headings),
            "word_count": len(output.split()),
            "finish_reason": _finish_reason(response),
            "completion_tokens": _completion_tokens(response),
            "endpoint": endpoint_used,
            "transport_trace": trace,
            "privacy_violations": _forced_prompt_privacy_violations(prompt, candidate=candidate),
        }
        atomic_save_json(path, record)
        return record

    async def custom_solution(
        candidate: Mapping[str, Any], seed: int, arm: str,
    ) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        qid = str(candidate["question_id"])
        card_info = cards[candidate_id]
        if arm == "b_after_plan":
            context = str(cache["plans"][(qid, seed)].get("output", ""))
            prompt = FORCED_TIMING_AFTER_PLAN_PROMPT.format(
                question=bundles[qid]["question_text"],
                initial_plan=context,
                principle_card=card_info["text"],
            )
        elif arm == "c_after_draft":
            context = _compact_current_draft(str(
                ((baseline[(qid, seed)].get("response") or {}).get("reasoner") or {}).get("output", "")
            ))
            prompt = FORCED_TIMING_AFTER_DRAFT_PROMPT.format(
                question=bundles[qid]["question_text"],
                previous_draft=context,
                principle_card=card_info["text"],
            )
        else:
            raise ValueError(f"Unsupported custom arm: {arm}")
        path = root / "arms" / arm / "rollouts" / f"{candidate_id}_s{seed}.json"
        input_hash = _json_hash({
            "version": FORCED_TIMING_VERSION,
            "arm": arm,
            "prompt": prompt,
            "finalizer_prompt": SOLVER_V1_FINALIZER_PROMPT,
            "model": model,
            "temperature": temperature,
            "seed": seed,
        })
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash and saved.get("status") == "complete":
                return saved

        reasoner_meta: dict[str, Any] = {}
        finalizer_meta: dict[str, Any] = {}
        reasoner_output = ""
        finalizer_output = ""
        endpoint_used = ""
        trace = []
        async with solver_runner.semaphore:
            for attempt, endpoint in enumerate(await endpoint_pair(qid, seed), 1):
                with llm_request_context(candidate_id, f"forced_timing/{arm}/reasoner/seed={seed}/attempt={attempt}"):
                    reasoner_meta, endpoint_used = await solver_runner.execute._call_with_router_metadata(
                        prompt=prompt,
                        temperature=temperature,
                        top_p=SOLVER_V1_TOP_P,
                        max_tokens=SOLVER_V1_REASONER_MAX_TOKENS,
                        repetition_penalty=1.0,
                        seed=seed,
                        url_override=endpoint,
                        timeout_override=min(timeout, 300),
                    )
                reasoner_output = str(reasoner_meta.get("text") or "").strip()
                if not reasoner_output:
                    trace.append({"endpoint": endpoint, "stage": "reasoner", "success": False})
                    continue
                finalizer_prompt = SOLVER_V1_FINALIZER_PROMPT.format(
                    question=bundles[qid]["question_text"], reasoner_output=reasoner_output,
                )
                with llm_request_context(candidate_id, f"forced_timing/{arm}/finalizer/seed={seed}/attempt={attempt}"):
                    finalizer_meta, _ = await solver_runner.execute._call_with_router_metadata(
                        prompt=finalizer_prompt,
                        temperature=SOLVER_V1_FINALIZER_TEMPERATURE,
                        top_p=SOLVER_V1_TOP_P,
                        max_tokens=SOLVER_V1_FINALIZER_MAX_TOKENS,
                        repetition_penalty=1.0,
                        seed=seed,
                        url_override=endpoint_used or endpoint,
                        timeout_override=min(timeout, 60),
                    )
                finalizer_output = str(finalizer_meta.get("text") or "").strip()
                trace.append({
                    "endpoint": endpoint,
                    "stage": "transaction",
                    "reasoner_success": True,
                    "finalizer_success": bool(finalizer_output),
                })
                if finalizer_output:
                    break
                reasoner_output = ""

        success = bool(reasoner_output and finalizer_output)
        composed = reasoner_output
        if finalizer_output:
            composed = f"{reasoner_output}\n\n{finalizer_output}".strip()
        record = {
            "input_hash": input_hash,
            "status": "complete" if success else "infrastructure_failed",
            "arm": arm,
            "stage": "solve",
            "candidate_id": candidate_id,
            "question_id": qid,
            "level": candidate.get("level", ""),
            "seed": seed,
            "temperature": temperature,
            "principle_injected": True,
            "principle_card_hash": card_info["sha256"],
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "processing_status": _processing_status(reasoner_output),
            "correct": check_answer(composed, str(bundles[qid]["expected_answer"])) if success else False,
            "final_answer": extract_final_answer(finalizer_output, allow_short_direct=False),
            "reasoner": {
                "output": reasoner_output,
                "finish_reason": _finish_reason(reasoner_meta),
                "completion_tokens": _completion_tokens(reasoner_meta),
                "hit_token_limit": _finish_reason(reasoner_meta) == "length",
                "repetition": _repetition_diagnostics(reasoner_output),
                "usage": reasoner_meta.get("usage"),
            },
            "finalizer": {
                "output": finalizer_output,
                "finish_reason": _finish_reason(finalizer_meta),
                "completion_tokens": _completion_tokens(finalizer_meta),
                "format_valid": _finalizer_format_valid(finalizer_output),
                "usage": finalizer_meta.get("usage"),
            },
            "transport_trace": trace,
            "privacy_violations": _forced_prompt_privacy_violations(prompt, candidate=candidate),
        }
        atomic_save_json(path, record)
        return record

    async def run_pair(candidate: Mapping[str, Any], seed: int, arm: str) -> dict[str, Any]:
        qid = str(candidate["question_id"])
        card_hash = cards[str(candidate["candidate_id"])]["sha256"]
        if arm == "a_before_plan":
            plan = await forced_plan(candidate, seed)
            if plan.get("status") != "complete":
                return {
                    "arm": arm, "candidate_id": candidate["candidate_id"],
                    "question_id": qid, "seed": seed, "status": "infrastructure_failed",
                    "principle_injected": True, "principle_card_hash": card_hash,
                    "failure_stage": "plan",
                }
            raw = await solver_runner.solve(
                phase="forced_timing",
                arm=arm,
                question_id=qid,
                question=str(bundles[qid]["question_text"]),
                expected_answer=str(bundles[qid]["expected_answer"]),
                temperature=temperature,
                seed=seed,
                principle_text=f"[Initial Solver Plan]\n{plan['output']}",
                candidate=candidate,
            )
            return {
                **raw,
                "principle_injected": True,
                "principle_injection_stage": "plan",
                "principle_card_hash": card_hash,
                "plan_record": str(
                    root / "arms" / "a_before_plan" / "plans"
                    / f"{candidate['candidate_id']}_s{seed}.json"
                ),
                "plan_processing_status": plan.get("processing_status", "unparsed"),
                "plan_privacy_violations": plan.get("privacy_violations", []),
            }
        return await custom_solution(candidate, seed, arm)

    all_rollouts: list[dict[str, Any]] = []
    for arm, description in (
        ("a_before_plan", "A：Principle参与计划"),
        ("b_after_plan", "B：Principle参与计划执行"),
        ("c_after_draft", "C：Principle参与草稿修正"),
    ):
        arm_rollouts = await _gather_with_progress(
            [run_pair(candidate, seed, arm) for candidate in selected_candidates for seed in seeds],
            description=description,
            unit="次配对推理",
        )
        all_rollouts.extend(arm_rollouts)
        atomic_save_json(root / "arms" / arm / "rollouts.json", arm_rollouts)

    by_rollout = {
        (str(item["arm"]), str(item["candidate_id"]), int(item["seed"])): item
        for item in all_rollouts
    }
    effect_rows: list[dict[str, Any]] = []
    control_names = {
        "a_before_plan": "plan_only",
        "b_after_plan": "plan_only",
        "c_after_draft": "generic_self_check",
    }
    for arm in FORCED_TIMING_ARMS:
        for candidate in selected_candidates:
            cid = str(candidate["candidate_id"])
            qid = str(candidate["question_id"])
            control_map = cache[
                "generic_self_check" if arm == "c_after_draft" else "plan_only"
            ]
            valid_control = []
            valid_treatment = []
            infrastructure_failures = 0
            for seed in seeds:
                treatment = by_rollout[(arm, cid, seed)]
                control = control_map[(qid, seed)]
                if treatment.get("status") != "complete" or control.get("status") != "complete":
                    infrastructure_failures += 1
                    continue
                valid_control.append(control)
                valid_treatment.append(treatment)
            paired = _paired_counts(valid_control, valid_treatment) if valid_control else {
                "gain": 0, "loss": 0, "both_correct": 0, "both_wrong": 0,
            }
            control_correct = sum(bool(item.get("correct")) for item in valid_control)
            guided_correct = sum(bool(item.get("correct")) for item in valid_treatment)
            delta = guided_correct - control_correct
            denominator = len(valid_control)
            effect_rows.append({
                "arm": arm,
                "control": control_names[arm],
                "candidate_id": cid,
                "question_id": qid,
                "level": candidate.get("level", ""),
                "evaluated": denominator > 0,
                "paired_seed_count": denominator,
                "infrastructure_failure_count": infrastructure_failures,
                "control_correct": control_correct,
                "guided_correct": guided_correct,
                "delta_correct": delta,
                "accuracy_delta": delta / denominator if denominator else 0.0,
                "paired": paired,
                "effect_label": _effect_label(delta),
                "principle_card_hash": cards[cid]["sha256"],
                "not_injected_count": sum(
                    not bool(item.get("principle_injected")) for item in valid_treatment
                ),
                "privacy_violations": sorted({
                    value for item in valid_treatment
                    for value in (
                        list(item.get("privacy_violations", []))
                        + list(item.get("plan_privacy_violations", []))
                    )
                }),
            })

    question_rows = _forced_question_effects(effect_rows)
    atomic_save_json(root / "rollouts" / "forced_rollouts.json", all_rollouts)
    atomic_save_json(root / "effects" / "candidate_effects.json", effect_rows)
    atomic_save_json(root / "effects" / "question_effects.json", question_rows)

    arms_summary: dict[str, Any] = {}
    for arm in FORCED_TIMING_ARMS:
        rows = [row for row in effect_rows if row["arm"] == arm]
        questions = [row for row in question_rows if row["arm"] == arm]
        pair_total = sum(row["paired_seed_count"] for row in rows)
        arms_summary[arm] = {
            "control": control_names[arm],
            "candidate_count": len(rows),
            "question_count": len(questions),
            "paired_seed_count": pair_total,
            "gain": sum(row["paired"]["gain"] for row in rows),
            "loss": sum(row["paired"]["loss"] for row in rows),
            "both_correct": sum(row["paired"]["both_correct"] for row in rows),
            "both_wrong": sum(row["paired"]["both_wrong"] for row in rows),
            "candidate_delta_sum": sum(row["delta_correct"] for row in rows),
            "question_macro_accuracy_delta": (
                sum(row["mean_candidate_accuracy_delta"] for row in questions) / len(questions)
                if questions else 0.0
            ),
            "positive_candidate_count": sum(row["delta_correct"] > 0 for row in rows),
            "negative_candidate_count": sum(row["delta_correct"] < 0 for row in rows),
            "neutral_candidate_count": sum(row["delta_correct"] == 0 for row in rows),
            "helpful_candidate_count": sum(row["delta_correct"] >= 2 for row in rows),
            "harmful_candidate_count": sum(row["delta_correct"] <= -2 for row in rows),
            "not_injected_count": sum(row["not_injected_count"] for row in rows),
            "privacy_violation_count": sum(bool(row["privacy_violations"]) for row in rows),
            "infrastructure_failure_count": sum(row["infrastructure_failure_count"] for row in rows),
        }

    summary = {
        "schema_version": FORCED_TIMING_VERSION,
        "run_scope": "smoke" if smoke_only else "full",
        "question_count": len(bundles),
        "candidate_count": len(all_candidates),
        "run_candidate_count": len(selected_candidates),
        "seeds": list(seeds),
        "temperature": temperature,
        "cache_validation": cache_validation,
        "arms": arms_summary,
        "p0_direct_reused": _summarize_effect_rows(data["p0_rows"]),
        "output_root": str(root),
    }
    atomic_save_json(root / "summary.json", summary)

    example_lines = ["# Principle固定注入Prompt示例", ""]
    if selected_candidates:
        example = selected_candidates[0]
        cid = str(example["candidate_id"])
        qid = str(example["question_id"])
        seed = int(seeds[0])
        example_lines.extend([
            f"- question_id: `{qid}`",
            f"- candidate_id: `{cid}`",
            f"- seed: `{seed}`",
            f"- Principle hash: `{cards[cid]['sha256']}`",
            "",
            "## A：计划前",
            "```text",
            FORCED_TIMING_PLAN_PROMPT.format(
                question=bundles[qid]["question_text"], principle_card=cards[cid]["text"],
            ),
            "```",
            "",
            "## B：计划后",
            "```text",
            FORCED_TIMING_AFTER_PLAN_PROMPT.format(
                question=bundles[qid]["question_text"],
                initial_plan=cache["plans"][(qid, seed)]["output"],
                principle_card=cards[cid]["text"],
            ),
            "```",
            "",
            "## C：草稿后",
            "```text",
            FORCED_TIMING_AFTER_DRAFT_PROMPT.format(
                question=bundles[qid]["question_text"],
                previous_draft=_compact_current_draft(str(
                    ((baseline[(qid, seed)].get("response") or {}).get("reasoner") or {}).get("output", "")
                )),
                principle_card=cards[cid]["text"],
            ),
            "```",
        ])
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "representative_examples.md").write_text(
        "\n".join(example_lines) + "\n", encoding="utf-8",
    )

    report_lines = [
        "# Principle 固定注入时机对比",
        "",
        f"- 运行范围：{'Smoke Test' if smoke_only else '完整实验'}",
        f"- 来源题：{len(bundles)}",
        f"- 可测试 Principle：{len(all_candidates)}",
        f"- 本次 Principle：{len(selected_candidates)}",
        f"- seeds：{list(seeds)}",
        "",
        "| 节点 | 对照 | Principle | Gain | Loss | Macro Delta | 未注入 | Infra失败 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in FORCED_TIMING_ARMS:
        stats = arms_summary[arm]
        report_lines.append(
            f"| {arm} | {stats['control']} | {stats['candidate_count']} | "
            f"{stats['gain']} | {stats['loss']} | "
            f"{stats['question_macro_accuracy_delta']:.4f} | "
            f"{stats['not_injected_count']} | {stats['infrastructure_failure_count']} |"
        )
    report_lines.extend([
        "",
        "结论选择必须先比较同成本对照，不能只与旧Baseline比较。",
        "`best_candidate_delta_diagnostic_only`仅供案例分析，不作为总体效果。",
    ])
    (root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary
