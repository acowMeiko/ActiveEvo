"""Multi-trajectory diagnosis and Principle generation."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from modules.principle_schema import validate_principle_content


def _rollout_payload(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "rollout_id": item.get("rollout_id"),
            "correct": item.get("correct"),
            "response": item.get("response"),
        }
        for item in bundle.get("rollouts", [])
        if isinstance(item, dict)
    ]


def build_generation_prompt(bundle: Dict[str, Any], max_principles: int) -> str:
    """Build the evidence prompt without capability ranks or entropy metadata."""
    pairs = [
        {
            "wrong_rollout_id": item.get("wrong_rollout_id"),
            "closest_correct_rollout_id": item.get("closest_correct_rollout_id"),
            "divergence_step": item.get("divergence_step"),
            "divergence_step_text": item.get("divergence_step_text"),
        }
        for item in bundle.get("trajectory_pairs", []) if isinstance(item, dict)
    ]
    return f"""You construct reusable mathematical memories for a weaker solver.

Analyze ALL rollouts together. First infer the stable success consensus. For every wrong
rollout, locate its first causal divergence from that consensus. A closest correct rollout
may be cited as a concrete contrast, but it is not the sole ground truth. Group wrong
rollouts with the same root cause into at most {max_principles} failure families and produce
one Principle per family.

Each Principle must be advice, not a theorem or mandatory instruction. Do not copy the
source answer, source-specific numbers, names, or conclusions. required_evidence must be
2-5 observable facts. action_guidance must contain 3-6 executable sentences and
verification_step 2-4 concrete checks.

Output raw JSON only:
{{
  "routing_semantics": {{
    "core_skill": "mathematical operator or invariant",
    "context_constraints": "conditions controlling valid use",
    "task_objective": "what the solver must derive"
  }},
  "success_consensus": ["..."],
  "failure_families": [
    {{
      "family_id": "family_1",
      "root_cause": "...",
      "supporting_wrong_rollout_ids": ["..."],
      "contrast_correct_rollout_ids": ["..."],
      "principle": {{
        "introspection": "...",
        "trigger_condition": "...",
        "required_evidence": ["..."],
        "exclusion_condition": "...",
        "failure_mode": "...",
        "action_guidance": "...",
        "verification_step": "...",
        "generalization_level": "instance|family|universal",
        "scope": "..."
      }}
    }}
  ]
}}

Question:
{bundle['question_text']}

Rollouts:
{json.dumps(_rollout_payload(bundle), ensure_ascii=False)}

Deterministic trajectory pairing hints:
{json.dumps(pairs, ensure_ascii=False)}
"""


async def generate_principles(client: Any, bundle: Dict[str, Any], max_principles: int) -> Dict[str, Any]:
    """Generate 1-3 principles from failure families, not one per trajectory pair."""
    prompt = build_generation_prompt(bundle, max_principles)
    response = await client.generator(prompt)
    parsed = response["parsed"]
    families = parsed.get("failure_families", [])
    if not isinstance(families, list):
        families = []
    valid: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    known_rollout_ids = {item.get("rollout_id") for item in bundle.get("rollouts", [])}
    for index, family in enumerate(families[:max_principles]):
        if not isinstance(family, dict):
            errors.append({"index": index, "errors": ["family_must_be_object"]})
            continue
        content, schema_errors = validate_principle_content(family.get("principle"), require_extended=True)
        if not schema_errors and not 2 <= len(content.get("required_evidence", [])) <= 5:
            schema_errors.append("required_evidence_must_contain_2_to_5_items")
        wrong_ids = family.get("supporting_wrong_rollout_ids", [])
        correct_ids = family.get("contrast_correct_rollout_ids", [])
        reference_errors = []
        if not isinstance(wrong_ids, list) or not set(wrong_ids).issubset(known_rollout_ids):
            reference_errors.append("invalid_wrong_rollout_reference")
        if not isinstance(correct_ids, list) or not set(correct_ids).issubset(known_rollout_ids):
            reference_errors.append("invalid_correct_rollout_reference")
        all_errors = schema_errors + reference_errors
        if all_errors:
            errors.append({"index": index, "errors": all_errors, "family": family})
            continue
        valid.append({
            "family_id": str(family.get("family_id") or f"family_{index + 1}"),
            "root_cause": str(family.get("root_cause", "")).strip(),
            "supporting_wrong_rollout_ids": [str(value) for value in wrong_ids],
            "contrast_correct_rollout_ids": [str(value) for value in correct_ids],
            "content": content,
        })
    return {
        "routing_semantics": parsed.get("routing_semantics", {}),
        "success_consensus": parsed.get("success_consensus", []),
        "families": valid,
        "schema_failures": errors,
        "raw_response": response["raw"],
        "parse_success": bool(parsed),
    }


async def repair_principle(client: Any, bundle: Dict[str, Any], candidate: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Repair one mathematical Principle using the audit below. Preserve its
evidence-supported root cause. Do not broaden its scope, add source-specific payload, or
invent evidence. Output raw JSON only as {{"principle": {{...exact canonical fields...}}}}.

Question: {bundle['question_text']}
Evidence rollout IDs: {candidate.get('supporting_wrong_rollout_ids', [])}
Current Principle: {json.dumps(candidate['content'], ensure_ascii=False)}
Audit: {json.dumps(audit, ensure_ascii=False)}
"""
    response = await client.generator(prompt)
    content, errors = validate_principle_content(response["parsed"].get("principle"), require_extended=True)
    return {"success": not errors, "content": content, "errors": errors, "raw_response": response["raw"]}
