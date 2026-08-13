"""Layered deterministic and model-based Principle audit."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from modules.principle_schema import PRINCIPLE_FIELDS


AUDIT_DIMENSIONS = (
    "source_fidelity",
    "instance_leakage",
    "trigger_decidability",
    "evidence_grounding",
    "exclusion_decidability",
    "action_executability",
    "verification_executability",
    "generalization_support",
)


def deterministic_leakage_check(
    content: Dict[str, Any], *, source_question: str, expected_answer: str
) -> Dict[str, Any]:
    principle_text = " ".join(str(content.get(field, "")) for field in PRINCIPLE_FIELDS)
    source_numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?", source_question or ""))
    significant = {value for value in source_numbers if len(re.sub(r"\D", "", value)) >= 2}
    copied = sorted(value for value in significant if value in principle_text)
    answer = (expected_answer or "").strip()
    answer_copied = bool(answer and len(answer) >= 2 and answer in principle_text)
    return {"detected": bool(copied or answer_copied), "copied_source_numbers": copied, "answer_copied": answer_copied}


async def audit_principle(client: Any, bundle: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
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
    prompt = f"""You are an independent quality auditor. Judge each dimension separately
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
{{"dimensions": {{"source_fidelity": {{...}}, "instance_leakage": {{...}}, "trigger_decidability": {{...}},
"evidence_grounding": {{...}}, "exclusion_decidability": {{...}},
"action_executability": {{...}}, "verification_executability": {{...}},
"generalization_support": {{...}}}}, "decision": "pass|repair|reject", "reasons": ["..."]}}

Question: {bundle['question_text']}
Principle: {json.dumps(candidate['content'], ensure_ascii=False)}
Cited evidence: {json.dumps(evidence, ensure_ascii=False)}
"""
    response = await client.auditor(prompt)
    parsed = response["parsed"]
    dimensions = parsed.get("dimensions", {}) if isinstance(parsed.get("dimensions"), dict) else {}
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
    if any(name in {"source_fidelity", "instance_leakage"} for name in failed_dimensions):
        decision = "reject"
    elif failed_dimensions and decision == "pass":
        decision = "repair"
    if deterministic.get("detected"):
        decision = "reject"
    if protocol_errors:
        decision = "reject"
    return {
        "decision": decision,
        "protocol_compliant": not protocol_errors,
        "protocol_errors": protocol_errors,
        "dimensions": dimensions,
        "failed_dimensions": failed_dimensions,
        "deterministic_leakage": deterministic,
        "reasons": parsed.get("reasons", []),
        "raw_response": response["raw"],
    }
