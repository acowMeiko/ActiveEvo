import json
import tempfile
from pathlib import Path

from modules.stage2.source_validation import (
    effect_label,
    inherited_adapted_record,
    load_validation_inputs,
    paired_outcomes,
    robust_effect_label,
)
from modules.principle_schema import render_solver_memory


def _content():
    return {
        "introspection": "The failed solution treated a diagonal as a perpendicular height.",
        "trigger_condition": "A triangle area uses a candidate diagonal as its height.",
        "required_evidence": ["The base and height must be perpendicular."],
        "exclusion_condition": "Do not apply when the diagonal is proved perpendicular to the base.",
        "failure_mode": "Using a non-perpendicular segment as an altitude.",
        "failure_signals": ["The draft calls a segment an altitude without establishing perpendicularity."],
        "action_guidance": "Identify the selected base. Derive its perpendicular height from the current constraints. Use those two quantities in the area formula.",
        "verification_step": "Verify the base-height dot product is zero.",
        "generalization_level": "family",
        "scope": "Triangle area problems with ambiguous auxiliary segments.",
    }


def test_paired_outcome_and_effect_labels():
    baseline = [
        {"seed": 42, "correct": False},
        {"seed": 43, "correct": True},
        {"seed": 44, "correct": False},
        {"seed": 45, "correct": True},
    ]
    treatment = [
        {"seed": 42, "correct": True},
        {"seed": 43, "correct": False},
        {"seed": 44, "correct": True},
        {"seed": 45, "correct": True},
    ]
    outcomes = paired_outcomes(baseline, treatment)
    assert outcomes == {"gain": 2, "loss": 1, "both_correct": 1, "both_wrong": 0}
    assert effect_label(outcomes) == "source_helpful"
    assert robust_effect_label(["source_helpful", "source_helpful"]) == "robust_source_helpful"
    assert robust_effect_label(["source_helpful", "source_harmful"]) == "temperature_sensitive_or_harmful"


def test_fail_closed_adapted_record_reuses_baseline_without_injection():
    baseline = {
        "record_id": "baseline:q1:t0.200000:s42",
        "question_id": "q1",
        "arm": "baseline",
        "seed": 42,
        "temperature": 0.2,
        "correct": True,
        "injected": False,
    }
    inherited = inherited_adapted_record(
        baseline, candidate_id="candidate_1", temperature=0.2, seed=42
    )
    assert inherited["correct"] is True
    assert inherited["arm"] == "adapted"
    assert inherited["injected"] is False
    assert inherited["experience_decision"] == "gate_or_adapter_reject"
    assert inherited["derived_from_baseline"] == baseline["record_id"]


def test_loader_keeps_audit_pass_and_rejected_candidates():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "stage2"
        (root / "candidate_banks").mkdir(parents=True)
        (root / "level1" / "00_learning_bundles").mkdir(parents=True)
        bundle = {
            "question_id": "q1", "question_text": "Find the area.",
            "expected_answer": "2", "rollouts": [], "source_dsd": {},
        }
        (root / "level1" / "00_learning_bundles" / "bundles.json").write_text(
            json.dumps([bundle]), encoding="utf-8"
        )
        candidates = [
            {"candidate_id": "c_pass", "question_id": "q1", "status": "quality_pass", "content": _content()},
            {"candidate_id": "c_reject", "question_id": "q1", "status": "rejected", "content": _content()},
        ]
        (root / "candidate_banks" / "level1.json").write_text(
            json.dumps({"level": "level1", "candidates": candidates}), encoding="utf-8"
        )
        loaded, hashes = load_validation_inputs(root)
        assert [item["candidate_id"] for item in loaded] == ["c_pass", "c_reject"]
        assert {item["status"] for item in loaded} == {"quality_pass", "rejected"}
        assert len(hashes) == 2


def test_direct_solver_view_excludes_routing_and_audit_metadata():
    rendered = render_solver_memory(_content())
    assert "Lesson learned:" in rendered
    assert "Recommended action:" in rendered
    assert "Verification:" in rendered
    assert "trigger_condition" not in rendered
    assert "required_evidence" not in rendered
    assert "candidate_id" not in rendered
    assert "quality_audit" not in rendered
