from modules.stage2.contracts import build_learning_bundles, validate_learning_bundle
from modules.stage2.quality_audit import deterministic_leakage_check
from modules.stage1_miner import mine_pairs


def test_learning_bundle_keeps_all_rollouts() -> None:
    packets = [{
        "idx": 7,
        "question": "Find x.",
        "expected_answer": "11",
        "success_rate": 0.5,
        "trial_records": [
            {"trial": 1, "response": "correct path", "is_correct": True},
            {"trial": 2, "response": "wrong path A", "is_correct": False},
            {"trial": 3, "response": "wrong path B", "is_correct": False},
        ],
        "evidence_complete": True,
    }]
    bundles = build_learning_bundles(packets, "level2")
    assert len(bundles) == 1
    assert len(bundles[0]["rollouts"]) == 3
    assert len(bundles[0]["wrong_rollout_ids"]) == 2
    assert bundles[0]["evidence_complete"] is True
    assert validate_learning_bundle(bundles[0]) == []


def test_legacy_pair_is_explicitly_partial() -> None:
    packets = [{
        "idx": 8,
        "question": "Compute y.",
        "expected_answer": "5",
        "chosen_path": "a complete correct path",
        "rejected_path": "a complete wrong path",
    }]
    bundle = build_learning_bundles(packets, "level1")[0]
    assert bundle["evidence_complete"] is False
    assert bundle["evidence_warning"] == "partial_trajectory_evidence"


def test_literal_source_payload_leak_is_rejected() -> None:
    content = {
        "introspection": "Do not copy 137 into another problem.",
        "trigger_condition": "A matching recurrence is present.",
        "required_evidence": ["The recurrence is visible."],
        "exclusion_condition": "The recurrence differs.",
        "failure_mode": "Copying a source constant.",
        "action_guidance": "Derive the value from the current recurrence.",
        "verification_step": "Substitute the result.",
        "generalization_level": "family",
        "scope": "Recurrence problems.",
    }
    result = deterministic_leakage_check(
        content, source_question="A recurrence starts from 137.", expected_answer="274"
    )
    assert result["detected"] is True


def test_every_wrong_trajectory_keeps_a_correct_contrast() -> None:
    sample = {
        "correct_trajectories": ["Correct approach starts by defining the invariant and then derives the requested value."],
        "wrong_trajectories": [
            "Wrong approach starts with an unrelated formula and reaches an unsupported conclusion.",
            "Another wrong approach assumes the requested result before checking any constraint.",
        ],
    }
    pairs = mine_pairs(sample)
    assert len(pairs) == 2
    assert all(pair["anchor"] for pair in pairs)
