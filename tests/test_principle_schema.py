from modules.principle_schema import (
    EXTENDED_PRINCIPLE_FIELDS,
    build_memory_views,
    principle_content_from_item,
    render_solver_condition_card,
    render_solver_memory,
    validate_condition_card,
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
        "failure_signals": ["The draft transfers a row offset without deriving the current recurrence."],
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
        "failure_signals": ["The draft copies a source recurrence."],
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
        "failure_signals": ["The draft uses an unsupported row-size formula."],
        "exclusion_condition": "Reject if the current recurrence changes.",
        "verification_step": "Recompute from the current recurrence.",
        "generalization_level": "family",
    }
    parsed = _normalize_principle_fields({"principles": [{**extended, "status": "candidate"}]})
    assert parsed["principles"] == [extended]
    _, errors = validate_principle_content(parsed["principles"][0], require_extended=True)
    assert not errors


def test_condition_card_schema_and_solver_rendering_hide_routing_metadata():
    card, errors = validate_condition_card({
        "status": "use",
        "when_to_consider": "Consider this when a current equation contains a rational expression.",
        "evidence_found": [{
            "requirement": "A denominator restriction is observable.",
            "problem_fact": "The current problem contains a denominator that cannot be zero.",
        }],
        "do_not_use_if": "Ignore it if no denominator can vanish.",
        "failure_to_avoid": "Avoid introducing extraneous roots after clearing denominators.",
        "suggested_action": "State the denominator restriction before clearing fractions. Solve the transformed equation. Then test candidate roots against the original denominator.",
        "verification": "Substitute candidates into the original expression. Reject any value that makes a denominator zero.",
        "reason": "The required current fact is present.",
    })
    assert not errors
    rendered = render_solver_condition_card(card)
    assert "Current fact" in rendered
    assert "retrieval" not in rendered.lower()
    assert "memory_id" not in rendered

    _, extra_errors = validate_condition_card({**card, "memory_id": "mem_should_not_be_allowed"})
    assert "unexpected condition card fields: memory_id" in extra_errors


def test_condition_card_rejects_missing_current_problem_fact():
    card, errors = validate_condition_card({
        "status": "use",
        "when_to_consider": "Consider it.",
        "evidence_found": [{"requirement": "Need a recurrence.", "problem_fact": ""}],
        "do_not_use_if": "Ignore otherwise.",
        "failure_to_avoid": "Avoid recurrence transfer.",
        "suggested_action": "Derive the recurrence from the current problem.",
        "verification": "Check it.",
        "reason": "Incomplete evidence.",
    })
    assert card["status"] == "use"
    assert "evidence_found[0].problem_fact must be a non-empty string" in errors
