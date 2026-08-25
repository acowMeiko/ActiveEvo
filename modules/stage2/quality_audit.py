"""Layered deterministic and model-based Principle audit."""
from __future__ import annotations

import json
import hashlib
import re
from typing import Any, Dict

from modules.principle_schema import PRINCIPLE_FIELDS


AUDIT_DIMENSIONS = (
    "source_fidelity",
    "mathematical_validity",
    "instance_leakage",
    "trigger_decidability",
    "evidence_grounding",
    "failure_signal_decidability",
    "exclusion_decidability",
    "action_executability",
    "verification_executability",
    "generalization_support",
)


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if left == right:
        return True
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = j = differences = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            differences += 1
            j += 1
            if differences > 1:
                return False
    return True


def normalize_audit_evidence_refs(
    dimensions: Dict[str, Any], evidence_ids: set[str]
) -> list[Dict[str, str]]:
    """Correct only uniquely identifiable one-character/prefix ID transcription errors."""
    repairs: list[Dict[str, str]] = []
    for value in dimensions.values():
        if not isinstance(value, dict) or not isinstance(value.get("evidence_refs"), list):
            continue
        normalized = []
        for raw_ref in value["evidence_refs"]:
            ref = str(raw_ref)
            if ref in evidence_ids:
                normalized.append(ref)
                continue
            matches = [
                allowed for allowed in evidence_ids
                if allowed.startswith(ref)
                or ref.startswith(allowed)
                or _edit_distance_at_most_one(ref, allowed)
            ]
            if len(matches) == 1:
                normalized.append(matches[0])
                repairs.append({"from": ref, "to": matches[0]})
            else:
                normalized.append(ref)
        value["evidence_refs"] = normalized
    return repairs


def deterministic_leakage_check(
    content: Dict[str, Any], *, source_question: str, expected_answer: str
) -> Dict[str, Any]:
    principle_text = " ".join(str(content.get(field, "")) for field in PRINCIPLE_FIELDS)
    source_numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?", source_question or ""))
    significant = {value for value in source_numbers if len(re.sub(r"\D", "", value)) >= 2}
    copied = sorted(value for value in significant if value in principle_text)
    def normalize_math(text: str) -> str:
        value = str(text or "").lower()
        value = value.replace("\\dfrac", "\\frac")
        value = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", value)
        value = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", value)
        return re.sub(r"[\s,$]", "", value)

    answer = normalize_math(expected_answer)
    normalized_principle = normalize_math(principle_text)
    answer_copied = bool(answer and answer not in {"0", "1", "-1"} and answer in normalized_principle)
    trigger = str(content.get("trigger_condition", "")).lower()
    post_solution_trigger = bool(re.search(
        r"\b(solver|rollout|response|draft|answer)\b.{0,40}\b(claims?|says?|assumes?|computes?|fails?|omits?|uses?)\b",
        trigger,
    ))
    return {
        "detected": bool(copied or answer_copied),
        "copied_source_numbers": copied,
        "answer_copied": answer_copied,
        "normalized_expected_answer": answer,
        "post_solution_trigger_detected": post_solution_trigger,
    }


def build_audit_prompt(
    bundle: Dict[str, Any],
    candidate: Dict[str, Any],
    evidence_ids: set[str],
    evidence: list[Dict[str, Any]],
    format_repair_raw: str = "",
) -> str:
    if format_repair_raw:
        return f"""You repair only the JSON protocol of an existing audit. Preserve every
semantic pass/fail judgment, reason, and final decision. The only allowed evidence_refs
are the exact IDs listed below. Correct a malformed ID only when it uniquely matches one
allowed ID; otherwise replace it with an empty list. Return raw JSON only with all ten
dimensions, each containing pass:boolean, reason:string, evidence_refs:list[string], plus
decision:pass|repair|reject and reasons:list[string].

Allowed evidence IDs: {json.dumps(sorted(evidence_ids), ensure_ascii=False)}
Malformed audit:
{format_repair_raw}
"""
    return f"""You are an independent quality auditor. Judge each dimension separately
against the cited trajectories. Do not judge whether the Principle improves downstream
accuracy; Stage 3 measures that. A reusable mathematical operation is not answer leakage
unless source-specific payload is copied.
For instance_leakage, pass=true means no source-specific answer, constant, entity,
or source-only conclusion is embedded in the Principle.

For every dimension output {{"pass": boolean, "reason": "...", "evidence_refs": ["rollout_id"]}}.
Final decision must be pass, repair, or reject. Use reject only for unsupported root cause,
contradiction with successful evidence, answer leakage, or irreparable schema/content.
Use repair for vague trigger/evidence/exclusion/scope/action/checks.

Output raw JSON only:
{{"dimensions": {{"source_fidelity": {{...}}, "mathematical_validity": {{...}},
"instance_leakage": {{...}}, "trigger_decidability": {{...}},
"evidence_grounding": {{...}}, "failure_signal_decidability": {{...}}, "exclusion_decidability": {{...}},
"action_executability": {{...}}, "verification_executability": {{...}},
"generalization_support": {{...}}}}, "decision": "pass|repair|reject", "reasons": ["..."]}}

Question: {bundle['question_text']}
Principle: {json.dumps(candidate['content'], ensure_ascii=False)}
Cited evidence: {json.dumps(evidence, ensure_ascii=False)}
"""


async def audit_principle(
    client: Any,
    bundle: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    auditor_id: str = "qwen",
    _format_repair_raw: str = "",
) -> Dict[str, Any]:
    deterministic = deterministic_leakage_check(
        candidate["content"],
        source_question=bundle["question_text"],
        expected_answer=bundle["expected_answer"],
    )
    evidence_ids = set(candidate.get("supporting_wrong_rollout_ids", [])) | set(
        candidate.get("contrast_correct_rollout_ids", [])
    )
    evidence = [
        item for item in bundle.get("rollouts", [])
        if isinstance(item, dict) and item.get("rollout_id") in evidence_ids
    ]
    prompt = build_audit_prompt(
        bundle, candidate, evidence_ids, evidence, _format_repair_raw
    )
    response = await client.auditor(prompt, auditor_id=auditor_id)
    parsed = response["parsed"]
    dimensions = parsed.get("dimensions", {}) if isinstance(parsed.get("dimensions"), dict) else {}
    deterministic_id_repairs = normalize_audit_evidence_refs(dimensions, evidence_ids)
    protocol_errors = []
    for name in AUDIT_DIMENSIONS:
        value = dimensions.get(name)
        if not isinstance(value, dict) or not isinstance(value.get("pass"), bool):
            protocol_errors.append(f"invalid_{name}")
            continue
        refs = value.get("evidence_refs", [])
        if not isinstance(refs, list) or not set(refs).issubset(evidence_ids):
            protocol_errors.append(f"invalid_{name}_evidence_refs")
    decision = str(parsed.get("decision", "")).lower()
    if decision not in {"pass", "repair", "reject"}:
        protocol_errors.append("invalid_decision")
        decision = "reject"
    failed_dimensions = [
        name for name in AUDIT_DIMENSIONS
        if isinstance(dimensions.get(name), dict) and dimensions[name].get("pass") is False
    ]
    if any(
        name in {"source_fidelity", "mathematical_validity", "instance_leakage"}
        for name in failed_dimensions
    ):
        decision = "reject"
    elif failed_dimensions and decision == "pass":
        decision = "repair"
    if deterministic.get("detected"):
        decision = "reject"
    elif deterministic.get("post_solution_trigger_detected") and decision == "pass":
        decision = "repair"
    if protocol_errors:
        decision = "reject"
    result = {
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "decision": decision,
        "protocol_compliant": not protocol_errors,
        "protocol_errors": protocol_errors,
        "dimensions": dimensions,
        "failed_dimensions": failed_dimensions,
        "deterministic_leakage": deterministic,
        "reasons": parsed.get("reasons", []),
        "raw_response": response["raw"],
        "reasoning_content": response.get("reasoning", ""),
        "provider_response": response.get("provider_response", {}),
        "auditor_endpoint": response.get("endpoint", ""),
        "auditor_id": auditor_id,
        "attempted_endpoints": response.get("attempted_endpoints", []),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "attempts": response.get("attempts", []),
        "parse_source": "message.content",
        "deterministic_id_repairs": deterministic_id_repairs,
    }
    if protocol_errors and not _format_repair_raw and response.get("raw"):
        repaired = await audit_principle(
            client,
            bundle,
            candidate,
            auditor_id=auditor_id,
            _format_repair_raw=response["raw"],
        )
        repaired["format_repair"] = {
            "attempted": True,
            "original_protocol_errors": protocol_errors,
            "original_raw_response": response["raw"],
            "success": repaired.get("protocol_compliant", False),
        }
        return repaired
    result["format_repair"] = {
        "attempted": bool(_format_repair_raw),
        "success": bool(_format_repair_raw) and not protocol_errors,
    }
    return result
