from modules.stage2.adaptive_tradeoff import (
    ADAPTIVE_EXECUTION_PROMPT,
    NO_MEMORY_ROUTE_PROMPT,
    PRINCIPLE_ROUTE_PROMPT,
    _answer_supported,
    parse_route_output,
    principle_action_view,
    principle_routing_view,
    validate_route,
)


def _content():
    return {
        "introspection": "lesson",
        "trigger_condition": "trigger",
        "required_evidence": ["evidence"],
        "exclusion_condition": "exclude",
        "failure_mode": "failure",
        "failure_signals": ["signal"],
        "action_guidance": "take this action",
        "verification_step": "verify this result",
        "expected_answer": "secret",
        "candidate_id": "secret-id",
    }


def _valid_route(decision="local_repair", scope="local"):
    return {
        "decision": decision,
        "trigger_satisfied": True,
        "exclusion_hit": False,
        "matched_evidence": [{
            "principle_requirement": "requirement",
            "problem_fact": "fact",
            "draft_span": "quoted draft error",
            "why_incorrect": "it contradicts the equation",
        }],
        "error_scope": scope,
        "reason": "specific error",
        "repair_instruction": "repair it",
    }


def test_principle_views_are_strictly_separated():
    routing = principle_routing_view(_content())
    action = principle_action_view(_content())
    assert set(routing) == {
        "trigger_condition", "required_evidence", "exclusion_condition",
        "failure_mode", "failure_signals",
    }
    assert set(action) == {"action_guidance", "verification_step"}
    assert "take this action" not in str(routing)
    assert "trigger" not in str(action)
    assert "secret" not in str(routing)
    assert "secret" not in str(action)


def test_valid_local_and_restart_routes_are_preserved():
    local, local_errors = validate_route(_valid_route())
    restart, restart_errors = validate_route(_valid_route("restart", "method_level"))
    assert local_errors == [] and local["decision"] == "local_repair"
    assert restart_errors == [] and restart["decision"] == "restart"


def test_route_fails_closed_without_exact_evidence():
    value = _valid_route("restart", "method_level")
    value["matched_evidence"] = []
    route, errors = validate_route(value)
    assert errors == []
    assert route["decision"] == "keep"
    assert route["repair_instruction"] == ""


def test_route_fails_closed_on_exclusion_or_schema_error():
    excluded = _valid_route()
    excluded["exclusion_hit"] = True
    route, _ = validate_route(excluded)
    assert route["decision"] == "keep"
    malformed, errors = validate_route(["restart"])
    assert errors == ["route_must_be_object"]
    assert malformed["decision"] == "keep"


def test_prompts_do_not_mix_routing_and_action_views():
    route_prompt = PRINCIPLE_ROUTE_PROMPT.format(
        question="Q", draft="D",
        routing_view=str(principle_routing_view(_content())),
    )
    execution_prompt = ADAPTIVE_EXECUTION_PROMPT.format(
        decision="local_repair", diagnosis="diagnosis",
        guidance=str(principle_action_view(_content())), question="Q", draft="D",
    )
    control_prompt = NO_MEMORY_ROUTE_PROMPT.format(question="Q", draft="D")
    assert "take this action" not in route_prompt
    assert "trigger_condition" not in execution_prompt
    assert "Principle diagnostic fields" not in control_prompt
    for prompt in (route_prompt, execution_prompt, control_prompt):
        assert "secret-id" not in prompt
        assert "expected_answer" not in prompt


def test_answer_support_requires_reasoner_derived_result():
    finalizer = r"Final Answer: \boxed{16}"
    supported = "Reasoning:\n...\n\nDerived Result:\n16"
    unsupported = "Reasoning:\nI expanded the expression."
    assert _answer_supported(supported, finalizer)
    assert not _answer_supported(unsupported, finalizer)


def test_route_parser_repairs_only_invalid_latex_backslashes():
    raw = r'''{
      "decision": "keep",
      "trigger_satisfied": true,
      "exclusion_hit": false,
      "matched_evidence": [{
        "principle_requirement": "check cosine",
        "problem_fact": "angle",
        "draft_span": "use \cos x",
        "why_incorrect": "none"
      }],
      "error_scope": "none",
      "reason": "valid",
      "repair_instruction": ""
    }'''
    parsed, stage = parse_route_output(raw)
    assert stage in {"raw_json", "escaped_invalid_backslashes"}
    assert parsed["decision"] == "keep"
