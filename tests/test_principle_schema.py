from modules.principle_schema import (
    EXTENDED_PRINCIPLE_FIELDS,
    build_memory_views,
    principle_content_from_item,
    render_solver_memory,
    validate_principle_content,
)
from modules.plan_module import _normalize_principle_fields


LEGACY = {
    "introspection": "The solver transferred a source-only formula.",
    "trigger_condition": "When a row pattern has a known size rule.",
    "action_guidance": "Derive the preceding row count before using an offset.",
    "scope": "Row-pattern indexing problems.",
    "failure_mode": "Using an incompatible source recurrence.",
}


def test_legacy_memory_gets_auditable_extended_defaults():
    content, errors = validate_principle_content(LEGACY)
    assert not errors
    assert all(content[field] for field in EXTENDED_PRINCIPLE_FIELDS)
    assert content["generalization_level"] == "family"
    views = build_memory_views(content)
    assert "exclusion_condition" in views["routing_view"]
    assert "required_evidence" not in views["solver_view"]
    rendered = render_solver_memory(content)
    assert "Recommended action" in rendered
    assert "Reject when" not in rendered


def test_stage2_extended_schema_is_required_when_requested():
    _, errors = validate_principle_content(LEGACY, require_extended=True)
    assert errors
    extended = {
        **LEGACY,
        "required_evidence": ["The current row-size recurrence is explicitly established."],
        "exclusion_condition": "Reject if the current recurrence differs from the source recurrence.",
        "verification_step": "Recompute the requested index from the current recurrence.",
        "generalization_level": "family",
    }
    content, errors = validate_principle_content(extended, require_extended=True)
    assert not errors
    assert content == extended


def test_stage2_required_evidence_rejects_string_but_legacy_reader_normalizes_it():
    extended = {
        **LEGACY,
        "required_evidence": "The current recurrence is explicit.",
        "exclusion_condition": "Reject if the recurrence changes.",
        "verification_step": "Recompute using the current recurrence.",
        "generalization_level": "family",
    }
    _, strict_errors = validate_principle_content(extended, require_extended=True)
    assert "required_evidence must be a list of non-empty strings" in strict_errors
    compatible, compatible_errors = validate_principle_content(extended)
    assert not compatible_errors
    assert compatible["required_evidence"] == ["The current recurrence is explicit."]


def test_principle_content_reports_compatibility_source():
    content, source = principle_content_from_item({"content": LEGACY})
    assert source == "canonical_legacy_defaults"
    assert content["exclusion_condition"]


def test_stage2_parser_preserves_exact_extended_schema():
    extended = {
        **LEGACY,
        "required_evidence": ["The current recurrence is established."],
        "exclusion_condition": "Reject if the current recurrence changes.",
        "verification_step": "Recompute from the current recurrence.",
        "generalization_level": "family",
    }
    parsed = _normalize_principle_fields({"principles": [{**extended, "status": "candidate"}]})
    assert parsed["principles"] == [extended]
    _, errors = validate_principle_content(parsed["principles"][0], require_extended=True)
    assert not errors
