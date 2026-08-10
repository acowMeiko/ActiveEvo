"""Canonical schema and rendering helpers for ActiveEvo memory principles.

The fields have different consumers. Keeping them structured prevents the
old failure mode where a list/dict was silently converted to a Python string and
stored as solver guidance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple


SCHEMA_VERSION = "3.0"


LEGACY_PRINCIPLE_FIELDS = (
    "introspection",
    "trigger_condition",
    "action_guidance",
    "scope",
    "failure_mode",
)

EXTENDED_PRINCIPLE_FIELDS = (
    "required_evidence",
    "exclusion_condition",
    "verification_step",
    "generalization_level",
)

PRINCIPLE_FIELDS = LEGACY_PRINCIPLE_FIELDS + EXTENDED_PRINCIPLE_FIELDS


def _normalize_evidence_list(value: Any, *, compatibility: bool) -> Tuple[List[str], List[str]]:
    """Normalize required evidence without silently stringifying containers."""
    if isinstance(value, list):
        normalized = [" ".join(item.split()).strip() for item in value if isinstance(item, str)]
        errors = []
        if len(normalized) != len(value):
            errors.append("required_evidence items must be strings")
        normalized = [item for item in normalized if item]
        if not normalized:
            errors.append("required_evidence must contain at least one non-empty string")
        return normalized, errors
    if compatibility and isinstance(value, str) and value.strip():
        return [" ".join(value.split()).strip()], []
    return [], ["required_evidence must be a list of non-empty strings"]


def _extended_defaults(content: Dict[str, Any]) -> Dict[str, Any]:
    """Provide auditable compatibility defaults for pre-v3 memories."""
    normalized = dict(content)
    normalized.setdefault("required_evidence", [normalized.get("trigger_condition", "")])
    normalized.setdefault(
        "exclusion_condition",
        "Reject when any required trigger, mathematical structure, constraint, or objective is absent.",
    )
    normalized.setdefault(
        "verification_step",
        "Verify the recommended operation against the current problem's original constraints.",
    )
    normalized.setdefault("generalization_level", "family")
    return normalized


def validate_principle_content(
    value: Any,
    *,
    require_extended: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a normalized principle and validation errors.

    Values must be non-empty strings.  Containers are rejected rather than
    stringified because their apparent readability hides a malformed schema.
    """
    errors: List[str] = []
    if not isinstance(value, dict):
        return {}, ["principle must be an object"]

    normalized: Dict[str, Any] = {}
    required_fields = PRINCIPLE_FIELDS if require_extended else LEGACY_PRINCIPLE_FIELDS
    for field in required_fields:
        raw = value.get(field)
        if field == "required_evidence":
            evidence, evidence_errors = _normalize_evidence_list(raw, compatibility=not require_extended)
            normalized[field] = evidence
            errors.extend(evidence_errors)
            continue
        if not isinstance(raw, str):
            errors.append(f"{field} must be a string")
            continue
        text = " ".join(raw.split()).strip()
        if not text:
            errors.append(f"{field} must not be empty")
            continue
        normalized[field] = text

    for field in EXTENDED_PRINCIPLE_FIELDS:
        if field in value and field not in normalized:
            raw = value.get(field)
            if field == "required_evidence":
                evidence, evidence_errors = _normalize_evidence_list(raw, compatibility=True)
                normalized[field] = evidence
                errors.extend(evidence_errors)
                continue
            if not isinstance(raw, str):
                errors.append(f"{field} must be a string")
            else:
                text = " ".join(raw.split()).strip()
                if not text:
                    errors.append(f"{field} must not be empty")
                else:
                    normalized[field] = text

    extras = sorted(set(value) - set(PRINCIPLE_FIELDS) - {"repair_status"})
    if extras:
        errors.append("unexpected fields: " + ", ".join(extras))
    if not errors:
        normalized = _extended_defaults(normalized)
        if normalized["generalization_level"] not in {"instance", "family", "universal"}:
            errors.append("generalization_level must be instance, family, or universal")
    return normalized, errors


def principle_content_from_item(item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Read canonical content, with an explicit compatibility view for legacy memory."""
    raw = item.get("content")
    if not isinstance(raw, dict):
        raw = {field: item.get(field) for field in PRINCIPLE_FIELDS}
    content, errors = validate_principle_content(raw)
    if not errors:
        schema_source = (
            "canonical_extended"
            if all(field in raw for field in EXTENDED_PRINCIPLE_FIELDS)
            else "canonical_legacy_defaults"
        )
        return content, schema_source

    # Legacy stores only retained action_guidance as ``principle``.  The derived
    # routing fields are marked as such in traces and must not be mistaken for
    # original model-generated fields during paper case analysis.
    action = item.get("principle") or item.get("action_guidance")
    if not isinstance(action, str) or not action.strip():
        return {}, "invalid"
    core = str(item.get("core_skill") or "the corresponding reasoning skill").strip()
    condition = str(item.get("context_constraints") or "the stated constraints match").strip()
    objective = str(item.get("task_objective") or "the requested objective matches").strip()
    return _extended_defaults({
        "introspection": "Legacy memory did not retain root-cause introspection.",
        "trigger_condition": condition,
        "action_guidance": " ".join(action.split()).strip(),
        "scope": f"Problems involving {core} with objective: {objective}",
        "failure_mode": "Do not apply this experience when its trigger or scope does not match the problem.",
    }), "legacy_derived"


def build_memory_views(
    content: Dict[str, Any],
    *,
    source_dsd: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Build explicit routing, solving, validation, and audit views.

    Routing metadata is deliberately unavailable in the solver view.  This
    prevents retrieval similarity and applicability language from being
    mistaken for evidence that a memory is reliable.
    """
    source_dsd = source_dsd or {}
    validation = validation or {}
    audit = audit or {}
    return {
        "routing_view": {
            "core_skill": str(source_dsd.get("core_skill", "")),
            "context_constraints": str(source_dsd.get("context_constraints", "")),
            "task_objective": str(source_dsd.get("task_objective", "")),
            "trigger_condition": content["trigger_condition"],
            "required_evidence": content.get("required_evidence", content["trigger_condition"]),
            "exclusion_condition": content.get("exclusion_condition", ""),
            "scope": content["scope"],
            "failure_mode": content["failure_mode"],
            "action_guidance": content["action_guidance"],
            "verification_step": content.get("verification_step", ""),
            "generalization_level": content.get("generalization_level", "family"),
        },
        "solver_view": {
            "lesson": content["introspection"],
            "action_guidance": content["action_guidance"],
            "verification_step": content.get("verification_step", ""),
        },
        "validation_view": dict(validation),
        "audit_view": {
            "schema_version": SCHEMA_VERSION,
            "content": dict(content),
            **dict(audit),
        },
    }


def render_solver_memory(content: Dict[str, Any], index: int = 1) -> str:
    """Render only the experience itself; never expose scores or routing fields."""
    return "\n".join(
        [
            f"[Past Experience {index}]",
            f"- Lesson learned: {content['introspection']}",
            f"- Recommended action: {content['action_guidance']}",
            f"- Verification: {content.get('verification_step', '')}",
        ]
    )
