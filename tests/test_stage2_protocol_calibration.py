from __future__ import annotations

import json

import pytest

from modules.stage2.protocol_calibration import (
    PROTOCOL_VERSION,
    _decision,
    _format_check,
    _load_prior_protocol_exclusions,
    _parse_audit,
)
from modules.stage2.principle_generation import validate_synthesis_payload


def _bundle() -> dict:
    return {
        "question_id": "q1",
        "question_text": "Solve a general integer constraint problem.",
        "expected_answer": "17",
        "rollouts": [
            {"rollout_id": "w1", "correct": False, "response": "wrong"},
            {"rollout_id": "c1", "correct": True, "response": "correct"},
        ],
    }


def _candidate() -> dict:
    return {
        "candidate_id": "candidate_1",
        "question_id": "q1",
        "supporting_wrong_rollout_ids": ["w1"],
        "contrast_correct_rollout_ids": ["c1"],
        "content": {
            "introspection": "One condition was treated as sufficient.",
            "trigger_condition": "There are multiple discrete constraints.",
            "required_evidence": ["Each constraint is explicit.", "The variable is integral."],
            "exclusion_condition": "The domain is continuous.",
            "failure_mode": "A candidate is not checked against every constraint.",
            "failure_signals": ["A constraint is never substituted back."],
            "action_guidance": "List every constraint. Produce candidates. Check each candidate against all constraints.",
            "verification_step": "Substitute the selected candidate into every original constraint.",
            "generalization_level": "family",
            "scope": "Discrete constrained problems.",
        },
    }


def test_format_check_keeps_semantic_leakage_as_audit_hint() -> None:
    candidate = _candidate()
    candidate["content"]["action_guidance"] = "Return 17 after checking the constraints."
    result = _format_check(candidate, _bundle())

    assert result["hard_invalid"] is False
    assert result["semantic_hints"].get("detected") is True
    assert result["semantic_hints"].get("answer_copied") is True


def test_format_check_hard_rejects_only_schema_or_bad_references() -> None:
    candidate = _candidate()
    candidate["supporting_wrong_rollout_ids"] = ["unknown"]
    result = _format_check(candidate, _bundle())

    assert result["hard_invalid"] is True
    assert "invalid_supporting_wrong_rollout_ids:unknown" in result["hard_errors"]


def test_six_dimension_audit_protocol_and_decisions() -> None:
    evidence_ids = {"w1", "c1"}
    parsed, errors = _parse_audit({
        "dimensions": {
            name: {"pass": True, "reason": "ok", "evidence_refs": []}
            for name in (
                "source_fidelity", "mathematical_validity", "instance_leakage",
                "condition_decidability", "action_executability", "verification_executability",
            )
        },
        "decision": "pass",
        "reasons": [],
    }, evidence_ids)
    assert not errors
    assert _decision(parsed) == "pass"

    parsed["dimensions"]["action_executability"]["pass"] = False
    assert _decision(parsed) == "repair"
    parsed["dimensions"]["mathematical_validity"]["pass"] = False
    assert _decision(parsed) == "reject"


def test_synthesis_normalizes_only_step_lists_without_semantic_rewrite() -> None:
    candidate = _candidate()
    principle = candidate["content"]
    principle["action_guidance"] = ["List every constraint.", "Check every candidate."]
    principle["verification_step"] = ["Substitute the result.", "Check the domain."]
    parsed = {
        "routing_semantics": {},
        "success_consensus": [],
        "failure_families": [{
            "family_id": "family_1",
            "root_cause": "unchecked constraint",
            "supporting_wrong_rollout_ids": ["w1"],
            "contrast_correct_rollout_ids": ["c1"],
            "principle": principle,
        }],
    }

    result = validate_synthesis_payload(parsed, _bundle(), 3)

    assert not result["schema_failures"]
    assert len(result["families"]) == 1
    content = result["families"][0]["content"]
    assert content["action_guidance"] == "1. List every constraint. 2. Check every candidate."
    assert content["verification_step"] == "1. Substitute the result. 2. Check the domain."
    assert {item["field"] for item in result["format_normalizations"]} == {
        "action_guidance", "verification_step",
    }


def test_prior_protocol_rounds_are_loaded_as_stable_exclusions(tmp_path) -> None:
    first = tmp_path / "round1"
    manifest_dir = first / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "source40.json").write_text(json.dumps({
        "schema_version": PROTOCOL_VERSION,
        "manifest_hash": "hash1",
        "source_question_ids": ["q1", "q2"],
    }), encoding="utf-8")

    excluded, parents = _load_prior_protocol_exclusions([first])

    assert excluded == {"q1", "q2"}
    assert parents == [{
        "root": str(first.resolve()),
        "manifest_hash": "hash1",
        "question_count": "2",
    }]


def test_prior_protocol_exclusion_requires_a_frozen_manifest(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing prior protocol manifest"):
        _load_prior_protocol_exclusions([tmp_path / "missing"])
