"""Multi-trajectory diagnosis and Principle generation."""
from __future__ import annotations

import json
import hashlib
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
    return f"""You construct reusable post-draft mathematical memories for a weaker solver.

Analyze ALL rollouts together. First infer the stable success consensus. For every wrong
rollout, locate its first consequential error: the earliest step that changes the later
derivation or answer. Do not mistake a downstream symptom, final wrong answer, formatting
problem, truncation, or isolated arithmetic slip for the root cause. Cite an exact wrong
step and contrast it with a successful step. A closest correct rollout is a hint, not the
sole ground truth. State the valid invariant or method that a later reviewer must preserve.

Group wrong rollouts with the same causal root into at most {max_principles} independent
failure families. Prefer one well-supported family over several overlapping lessons. Emit
multiple families only when each is causally independent and supported by at least one
wrong and one correct rollout. Prefer no Principle over a weak or invented generalization.

Each Principle must be advice, not a theorem or mandatory instruction. Do not copy the
source answer, source-specific numbers, names, or conclusions. required_evidence must be
2-5 facts observable directly in the problem statement. trigger_condition must describe
only pre-solution mathematical structure; it must never say that a solver/rollout already
made a claim or error. Express reusable structure through variables, mathematical roles,
relations, or invariants rather than source constants.

The memory will be used AFTER a solver has produced a complete draft. failure_signals must
therefore contain 1-4 concrete draft-observable behaviors. action_guidance must contain 3-6
minimal corrective steps: preserve every valid part of the draft, change only the diagnosed
step, and never demand a restart merely to follow the Principle. verification_step must
contain 2-4 answer-independent checks that can be applied to the revised draft. The
exclusion_condition must explicitly cover cases where the warning sign is absent or the
draft already passes the verification. Neither required_evidence nor verification_step may
state the requested answer or a source-only intermediate conclusion.

Bad abstraction: "When the problem contains 7 objects, subtract 2^7."
Good abstraction: "When assignments to two labeled subsets are counted and both subsets
must be nonempty, check the overlap of the two empty-subset exclusion events before applying
inclusion-exclusion."

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
        "failure_signals": ["..."],
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


def build_diagnosis_prompt(bundle: Dict[str, Any], max_principles: int) -> str:
    """Ask the thinking model to diagnose evidence without spending tokens on JSON syntax."""
    pairs = [
        {
            "wrong_rollout_id": item.get("wrong_rollout_id"),
            "closest_correct_rollout_id": item.get("closest_correct_rollout_id"),
            "divergence_step": item.get("divergence_step"),
            "divergence_step_text": item.get("divergence_step_text"),
        }
        for item in bundle.get("trajectory_pairs", []) if isinstance(item, dict)
    ]
    return f"""Analyze evidence for reusable post-draft mathematical memories. This is the
diagnosis stage; a separate synthesis stage will convert your analysis to JSON.

Work in this order:
1. Infer the stable success consensus from all correct rollouts and identify the invariant,
   formula, case split, or valid step that must be preserved.
2. For every wrong rollout, quote or precisely locate the first consequential error and
   explain why it changes the downstream derivation. Separate root causes from symptoms.
3. Contrast each root cause with evidence from at least one correct rollout.
4. Merge the same cause across wrong rollouts. Keep at most {max_principles} causally
   independent families and prefer fewer, stronger families.
5. Decide whether each family can become a source-independent draft-repair rule. Reject
   formatting failures, truncation, isolated random slips, unsupported mathematics, and
   lessons that require the source answer or source-only constants.

For each retained family identify supporting wrong rollout IDs, contrasting correct rollout
IDs, pre-solution structure, required problem facts, exact draft-observable warning signs,
exclusions, minimal corrective actions, preservation requirements, answer-independent
verification checks, and scope. Do not solve the question again. Do not copy the source
answer or source-specific payload. Prefer no Principle over an unsupported generalization.

Question:
{bundle['question_text']}

Rollouts:
{json.dumps(_rollout_payload(bundle), ensure_ascii=False)}

Deterministic trajectory pairing hints:
{json.dumps(pairs, ensure_ascii=False)}
"""


def build_synthesis_prompt(
    bundle: Dict[str, Any], max_principles: int, diagnosis: str
) -> str:
    """Turn the thinking transcript into canonical JSON while retaining source evidence."""
    return f"""{build_generation_prompt(bundle, max_principles)}

Diagnostic notes from the preceding thinking stage follow. Treat them as analysis, not
as additional evidence. Verify every claim against the rollouts above. Do not add a
failure family merely because it appears in the notes. Your only task now is to emit the
complete raw JSON object requested above; do not include commentary or hidden reasoning.

Diagnostic notes:
{diagnosis}
"""


def build_compact_synthesis_prompt(
    bundle: Dict[str, Any], max_principles: int, diagnosis: str
) -> str:
    """Serialize a frozen diagnosis without re-sending the long trajectories.

    This prompt is deliberately for the second stage only.  The diagnosis stage
    already read every rollout and is the sole place where long-form reasoning
    is allowed.  Keeping this input small prevents thinking models from
    restarting the trajectory analysis instead of returning a final JSON object.
    """
    rollout_ids = [
        str(item.get("rollout_id")) for item in bundle.get("rollouts", [])
        if isinstance(item, dict) and item.get("rollout_id")
    ]
    return f"""Convert the frozen diagnosis below into one raw JSON object. This is a
serialization task, not a new analysis task.

Do not solve the question. Do not re-read or reconstruct trajectories. Do not introduce
a failure family, numerical example, answer, or mathematical claim that is absent from
the frozen diagnosis. You may emit zero to {max_principles} families when the diagnosis
says no reusable Principle is supported.

Use only these rollout IDs when citing evidence: {json.dumps(rollout_ids, ensure_ascii=False)}

Return raw JSON only:
{{
  "routing_semantics": {{"core_skill": "...", "context_constraints": "...", "task_objective": "..."}},
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
        "required_evidence": ["2-5 observable facts from the current problem"],
        "exclusion_condition": "...",
        "failure_mode": "...",
        "failure_signals": ["1-4 draft-observable behaviors"],
        "action_guidance": "3-6 advisory executable sentences",
        "verification_step": "2-4 concrete checks",
        "generalization_level": "instance|family|universal",
        "scope": "..."
      }}
    }}
  ]
}}

Question:
{bundle['question_text']}

Frozen diagnosis:
{diagnosis}
"""


def validate_synthesis_payload(
    parsed: Dict[str, Any], bundle: Dict[str, Any], max_principles: int,
) -> Dict[str, Any]:
    """Validate generated families after lossless, type-only normalization.

    Some providers render multi-step ``action_guidance`` or
    ``verification_step`` as ``list[str]`` even though the canonical Memory
    schema stores each field as one string.  Joining those strings is a format
    repair, not a semantic rewrite.  The caller keeps the raw provider content
    for audit, and this function records every normalized field.
    """
    families = parsed.get("failure_families", [])
    if not isinstance(families, list):
        families = []
    valid: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    normalizations: List[Dict[str, Any]] = []
    known_rollout_ids = {
        str(item.get("rollout_id")) for item in bundle.get("rollouts", [])
        if isinstance(item, dict) and item.get("rollout_id")
    }
    for index, family in enumerate(families[:max_principles]):
        if not isinstance(family, dict):
            errors.append({"index": index, "errors": ["family_must_be_object"]})
            continue
        principle = family.get("principle")
        if isinstance(principle, dict):
            principle = dict(principle)
            for field in ("action_guidance", "verification_step"):
                value = principle.get(field)
                if isinstance(value, list) and value and all(
                    isinstance(item, str) and item.strip() for item in value
                ):
                    principle[field] = " ".join(
                        f"{position}. {item.strip()}"
                        for position, item in enumerate(value, 1)
                    )
                    normalizations.append({
                        "family_index": index,
                        "field": field,
                        "operation": "join_string_list_as_numbered_text",
                        "item_count": len(value),
                    })
        content, schema_errors = validate_principle_content(
            principle, require_extended=True,
        )
        if not schema_errors and not 2 <= len(content.get("required_evidence", [])) <= 5:
            schema_errors.append("required_evidence_must_contain_2_to_5_items")
        if not schema_errors and not 1 <= len(content.get("failure_signals", [])) <= 4:
            schema_errors.append("failure_signals_must_contain_1_to_4_items")
        wrong_ids = family.get("supporting_wrong_rollout_ids", [])
        correct_ids = family.get("contrast_correct_rollout_ids", [])
        reference_errors = []
        if not isinstance(wrong_ids, list) or not set(map(str, wrong_ids)).issubset(known_rollout_ids):
            reference_errors.append("invalid_wrong_rollout_reference")
        if not isinstance(correct_ids, list) or not set(map(str, correct_ids)).issubset(known_rollout_ids):
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
        "format_normalizations": normalizations,
    }


async def diagnose_principles(
    client: Any, bundle: Dict[str, Any], max_principles: int
) -> Dict[str, Any]:
    """Run and return the independently checkpointable diagnosis stage."""
    diagnosis_prompt = build_diagnosis_prompt(bundle, max_principles)
    diagnosis_response = await client.generator(
        diagnosis_prompt,
        thinking_enabled=True,
        max_tokens=(
            client.config.qwen_generation_analysis_max_tokens
            if client.generator_id == "qwen"
            else client.config.generation_analysis_max_tokens
        ),
    )
    diagnosis = "\n".join(
        value for value in (
            diagnosis_response.get("reasoning", ""),
            diagnosis_response.get("content", ""),
        ) if str(value).strip()
    ).strip()
    return {
        "prompt": diagnosis_prompt,
        "prompt_hash": hashlib.sha256(diagnosis_prompt.encode("utf-8")).hexdigest(),
        "diagnosis_text": diagnosis,
        "response": diagnosis_response,
        "success": bool(diagnosis),
        "failure_reason": "" if diagnosis else "empty_diagnosis",
    }


async def synthesize_principles(
    client: Any,
    bundle: Dict[str, Any],
    max_principles: int,
    diagnosis: str,
    *,
    correction_feedback: str = "",
    max_tokens_override: int | None = None,
    compact_synthesis: bool = False,
    thinking_enabled: bool | None = None,
) -> Dict[str, Any]:
    """Run synthesis from a persisted diagnosis and validate its final JSON."""
    synthesis_diagnosis = diagnosis
    if correction_feedback:
        synthesis_diagnosis += (
            "\n\nThe preceding synthesis was invalid. Correct only these reported issues "
            "while rechecking every claim against the original trajectories:\n"
            + correction_feedback
        )
    prompt_builder = build_compact_synthesis_prompt if compact_synthesis else build_synthesis_prompt
    prompt = prompt_builder(bundle, max_principles, synthesis_diagnosis)
    response = await client.generator(
        prompt,
        thinking_enabled=(client.requires_full_thinking if thinking_enabled is None else thinking_enabled),
        max_tokens=max_tokens_override or (
            client.config.qwen_generation_max_tokens
            if client.generator_id == "qwen"
            else client.config.generation_max_tokens
        ),
    )
    parsed = response["parsed"]
    validation = validate_synthesis_payload(parsed, bundle, max_principles)
    return {
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        **validation,
        "raw_response": response["raw"],
        "parse_success": bool(parsed),
        "generator_endpoint": response.get("endpoint", ""),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "reasoning_content": response.get("reasoning", ""),
        "provider_response": response.get("provider_response", {}),
        "synthesis_stage": response,
        "failure_reason": "" if parsed else response.get("failure_reason", "synthesis_parse_failed"),
        "generator_id": client.generator_id,
        "parse_source": response.get("parse_source", "message.content"),
        "generation_attempts": response.get("attempts", []),
    }


async def generate_principles(client: Any, bundle: Dict[str, Any], max_principles: int) -> Dict[str, Any]:
    """Generate 1-3 principles from failure families, not one per trajectory pair."""
    diagnosis_stage = await diagnose_principles(client, bundle, max_principles)
    diagnosis = diagnosis_stage["diagnosis_text"]
    diagnosis_response = diagnosis_stage["response"]
    if not diagnosis:
        return {
            "routing_semantics": {},
            "success_consensus": [],
            "families": [],
            "schema_failures": [],
            "raw_response": "",
            "parse_success": False,
            "generator_endpoint": diagnosis_response.get("endpoint", ""),
            "finish_reason": diagnosis_response.get("finish_reason"),
            "usage": diagnosis_response.get("usage"),
            "reasoning_content": "",
            "provider_response": diagnosis_response.get("provider_response", {}),
            "analysis_stage": diagnosis_response,
            "synthesis_stage": {},
            "failure_reason": "empty_diagnosis",
        }
    synthesis = await synthesize_principles(client, bundle, max_principles, diagnosis)
    return {
        **synthesis,
        "analysis_stage": diagnosis_response,
    }


def build_repair_prompt(
    bundle: Dict[str, Any], candidate: Dict[str, Any], audit: Dict[str, Any]
) -> str:
    return f"""Repair one mathematical Principle using the audit below. Preserve its
evidence-supported root cause. Do not broaden its scope, add source-specific payload, or
invent evidence. Output raw JSON only. The principle object must contain exactly these
fields and types:
{{"principle": {{
  "introspection": "string",
  "trigger_condition": "string",
  "required_evidence": ["2-5 observable facts"],
  "exclusion_condition": "string",
  "failure_mode": "string",
  "failure_signals": ["1-4 draft-observable behaviors"],
  "action_guidance": "3-6 advisory executable sentences",
  "verification_step": "2-4 concrete checks",
  "generalization_level": "instance|family|universal",
  "scope": "string"
}}}}

Question: {bundle['question_text']}
Evidence rollout IDs: {candidate.get('supporting_wrong_rollout_ids', [])}
Current Principle: {json.dumps(candidate['content'], ensure_ascii=False)}
Audit: {json.dumps(audit, ensure_ascii=False)}
"""


async def repair_principle(client: Any, bundle: Dict[str, Any], candidate: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_repair_prompt(bundle, candidate, audit)
    response = await client.generator(prompt)
    content, errors = validate_principle_content(response["parsed"].get("principle"), require_extended=True)
    return {
        "success": not errors,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "content": content,
        "errors": errors,
        "raw_response": response["raw"],
        "generator_endpoint": response.get("endpoint", ""),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "reasoning_content": response.get("reasoning", ""),
        "provider_response": response.get("provider_response", {}),
        "parse_source": response.get("parse_source", "message.content"),
        "generation_attempts": response.get("attempts", []),
        "generator_id": client.generator_id,
    }
