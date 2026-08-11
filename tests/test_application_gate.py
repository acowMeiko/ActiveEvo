import asyncio
import json
from types import MethodType

from modules.plan_module import PlanModule


def _candidate():
    return {
        "memory_id": "m1",
        "content": {
            "introspection": "The source solver transferred a fixed-width row formula.",
            "trigger_condition": "Rows have a fixed width.",
            "required_evidence": ["The same number of entries occurs in every row."],
            "exclusion_condition": "Reject when row width varies or doubles.",
            "action_guidance": "Use the fixed row width to derive the first entry of the target row.",
            "verification_step": "Check the inferred entry against the row boundaries.",
            "generalization_level": "family",
            "scope": "Fixed-width consecutive-number lattices.",
            "failure_mode": "Transferring a fixed-width formula to another recurrence.",
        },
        "source_item": {
            "question": "Each row contains five consecutive integers. Find row nine.",
            "core_skill": "sequence indexing",
            "context_constraints": "fixed width of five",
            "task_objective": "locate an entry",
        },
    }


def test_application_gate_fails_closed_on_structural_mismatch():
    plan = PlanModule.__new__(PlanModule)

    async def fake_call(self, prompt, **kwargs):
        assert "doubling" in prompt
        return json.dumps({
            "judgments": [{
                "memory_id": "m1",
                "trigger_match": False,
                "required_evidence_match": False,
                "structural_match": False,
                "scope_match": False,
                "contraindicated": True,
                "violated_conditions": ["Current row widths double."],
                "source_specific_elements": ["fixed width of five"],
                "confidence": 0.08,
                "reason": "The source recurrence is not preserved.",
            }]
        })

    plan._call_llm = MethodType(fake_call, plan)
    result = asyncio.run(plan.judge_memory_applicability(
        "The number of entries doubles in every row. Find an entry.",
        [_candidate()],
        confidence_threshold=0.8,
    ))
    judgment = result["judgments"][0]
    assert not judgment["accepted"]
    assert judgment["contraindicated"]
    assert result["selected_ids"] == []


def test_adapter_runs_only_after_acceptance_and_keeps_audit_text():
    plan = PlanModule.__new__(PlanModule)
    calls = []

    async def fake_call(self, prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({
            "status": "use",
            "when_to_consider": "Consider this when the current problem asks for an indexed row entry.",
            "evidence_found": [{
                "requirement": "The row-size rule is observable.",
                "problem_fact": "The current question states how entries are arranged in rows.",
            }],
            "do_not_use_if": "Ignore it if row widths are not governed by a stable current-problem rule.",
            "failure_to_avoid": "Avoid transferring a source indexing formula before deriving the current row rule.",
            "suggested_action": "First derive the row-size rule from the current statement. Then compute the offset using that current rule. Treat any source row width as irrelevant unless it is explicitly present here.",
            "verification": "Check the inferred row boundary against the current problem. Verify the target entry lies inside that row.",
            "reason": "The action now depends only on current evidence.",
        })

    plan._call_llm = MethodType(fake_call, plan)
    rejected = asyncio.run(plan.adapt_memory_for_question(
        "Current question", _candidate(), {"accepted": False}
    ))
    assert rejected["status"] == "reject"
    assert calls == []

    accepted = asyncio.run(plan.adapt_memory_for_question(
        "Current question", _candidate(), {"accepted": True}
    ))
    assert accepted["status"] == "use"
    assert accepted["condition_card"]["suggested_action"].startswith("First derive")
    assert accepted["condition_card"]["evidence_found"][0]["problem_fact"]
    assert accepted["raw_response"]
    assert len(calls) == 1


def test_application_gate_normalizes_direct_list_without_calling_it_strictly_compliant():
    plan = PlanModule.__new__(PlanModule)

    async def fake_call(self, prompt, **kwargs):
        return json.dumps([{
            "memory_id": "m1",
            "trigger_match": True,
            "required_evidence_match": True,
            "structural_match": True,
            "scope_match": True,
            "contraindicated": False,
            "violated_conditions": [],
            "source_specific_elements": [],
            "confidence": 0.93,
            "reason": "The required structure is preserved.",
        }]) + "\nThe candidates were checked conservatively."

    plan._call_llm = MethodType(fake_call, plan)
    result = asyncio.run(plan.judge_memory_applicability(
        "Each row has a fixed width.", [_candidate()], confidence_threshold=0.8
    ))
    assert result["status"] == "success"
    assert result["parse_stage"] == "direct_list_normalized"
    assert result["format_parsed"]
    assert not result["protocol_compliant"]
    assert "top_level_must_be_raw_object_with_judgments_array" in result["protocol_errors"]
    assert result["judgments"][0]["accepted"]


def test_gate_parser_preserves_full_batched_list_before_trailing_explanation():
    plan = PlanModule.__new__(PlanModule)
    response = json.dumps([
        {"memory_id": "m1"},
        {"memory_id": "m2"},
        {"memory_id": "m3"},
    ]) + "\nExtra explanation that violates the strict protocol."
    parsed, stage = plan._extract_gate_json(response)
    assert stage == "direct_list_normalized"
    assert [item["memory_id"] for item in parsed["judgments"]] == ["m1", "m2", "m3"]
