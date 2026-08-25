import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.stage2.principle_test import (
    AUDIT_PROTOCOLS,
    CONDITION_CARD_PROMPT,
    VARIANT_PROMPT,
    VARIANT_VERIFY_PROMPT,
    _audit_protocol_valid,
    _effect_label,
    _gather_with_progress,
    _paired_counts,
    _solver_safe_principle,
    _summarize_effect_rows,
    aggregate_audits,
    require_variant_approvals,
    validate_variant_payload,
    validate_variant_verification,
)
from modules.stage2.quality_audit import AUDIT_DIMENSIONS
from utils.io_utils import atomic_save_json
import modules.stage2.principle_test as principle_test_module


def _audit_run(*failed: str, compliant: bool = True) -> dict:
    dimensions = {
        name: {"pass": name not in failed, "reason": name, "evidence_refs": ["r1"]}
        for name in AUDIT_DIMENSIONS
    }
    return {
        "protocol_compliant": compliant,
        "audit": {"dimensions": dimensions, "decision": "repair" if failed else "pass"},
    }


def test_audit_protocols_keep_raw_as_control_only() -> None:
    assert AUDIT_PROTOCOLS == ("raw_control", "kimi_single", "kimi_critic_formatter")


def test_progress_gather_preserves_input_order() -> None:
    async def delayed(value: int) -> int:
        await asyncio.sleep(0.001 * (3 - value))
        return value

    result = asyncio.run(_gather_with_progress(
        [delayed(1), delayed(2), delayed(3)],
        description="contract progress", unit="item",
    ))
    assert result == [1, 2, 3]


def test_majority_hard_failure_rejects_but_one_vote_is_disagreement() -> None:
    rejected = aggregate_audits([
        _audit_run("mathematical_validity"),
        _audit_run("mathematical_validity"),
        _audit_run(),
    ])
    disputed = aggregate_audits([
        _audit_run("mathematical_validity"),
        _audit_run(),
        _audit_run(),
    ])
    assert rejected["decision"] == "reject"
    assert disputed["decision"] == "pass"
    assert disputed["audit_disagreement"] is True


def test_protocol_failure_is_not_converted_to_quality_failure() -> None:
    result = aggregate_audits([
        _audit_run(compliant=False),
        _audit_run(compliant=False),
        _audit_run(),
    ])
    assert result["decision"] == "protocol_failed"
    assert result["valid_run_count"] == 1
    assert result["majority_failed_dimensions"] == []


def test_audit_schema_rejects_unknown_evidence_id() -> None:
    payload = {
        "dimensions": {
            name: {"pass": True, "reason": "ok", "evidence_refs": ["unknown"]}
            for name in AUDIT_DIMENSIONS
        },
        "decision": "pass",
        "reasons": [],
    }
    _, errors = _audit_protocol_valid(payload, {"r1"})
    assert errors
    assert any("evidence_refs" in error for error in errors)


def test_variant_prompts_are_blind_to_principles_and_trajectories() -> None:
    combined = (VARIANT_PROMPT + VARIANT_VERIFY_PROMPT).lower()
    assert "source problem" in combined
    assert "principle:" not in combined
    assert "correct trajectory" not in combined
    assert "wrong trajectory" not in combined


def test_variant_validation_separates_schema_and_semantic_verification() -> None:
    payload = {
        "variant_question": "A changed, self-contained problem?",
        "variant_answer": "7",
        "structure_spec": {
            "core_skill": "algebra",
            "context_constraints": "positive integers",
            "task_objective": "find a value",
            "reasoning_skeleton": ["form an equation", "solve and check"],
        },
        "changed_elements": ["entities", "numbers"],
    }
    normalized, errors = validate_variant_payload(payload, source_question="Original geometry problem")
    assert not errors
    assert normalized["variant_answer"] == "7"
    verify_errors = validate_variant_verification({
        "structure_match": True,
        "answer_correct": False,
        "surface_difference_sufficient": True,
        "source_answer_memorization_risk": False,
        "independent_solution": "The computed answer is 8.",
    })
    assert "answer_incorrect" in verify_errors


def test_transfer_requires_all_40_explicit_human_approvals(tmp_path: Path) -> None:
    variants = [{"source_question_id": str(index)} for index in range(40)]
    approvals = [
        {"source_question_id": str(index), "approved": index < 39}
        for index in range(40)
    ]
    atomic_save_json(tmp_path / "manifests" / "variant40.json", variants)
    atomic_save_json(tmp_path / "variants" / "variant_approvals.json", approvals)
    with pytest.raises(RuntimeError):
        require_variant_approvals(tmp_path)
    approvals[-1]["approved"] = True
    atomic_save_json(tmp_path / "variants" / "variant_approvals.json", approvals)
    assert require_variant_approvals(tmp_path)["approved_count"] == 40


def test_solver_safe_principle_excludes_source_and_routing_metadata() -> None:
    safe = _solver_safe_principle({
        "trigger_condition": "T",
        "required_evidence": ["E"],
        "exclusion_condition": "X",
        "failure_mode": "F",
        "action_guidance": "A",
        "verification_step": "V",
        "source_question": "secret",
        "expected_answer": "42",
        "candidate_id": "candidate_secret",
        "retrieval_score": 0.99,
    })
    assert set(safe) == {
        "trigger_condition", "required_evidence", "exclusion_condition",
        "failure_mode", "action_guidance", "verification_step",
    }
    assert "secret" not in json.dumps(safe)


def test_effect_summary_bootstraps_at_question_level() -> None:
    rows = [
        {"evaluated": True, "question_id": "q1", "level": "level1", "delta_correct": 2,
         "effect_label": "helpful", "paired": {"gain": 2, "loss": 0}, "privacy_violations": []},
        {"evaluated": True, "question_id": "q1", "level": "level1", "delta_correct": 0,
         "effect_label": "inconclusive", "paired": {"gain": 0, "loss": 0}, "privacy_violations": []},
        {"evaluated": True, "question_id": "q2", "level": "level2", "delta_correct": -2,
         "effect_label": "harmful", "paired": {"gain": 0, "loss": 2}, "privacy_violations": []},
    ]
    summary = _summarize_effect_rows(rows)
    assert summary["question_count"] == 2
    assert summary["question_mean_accuracy_delta"] == pytest.approx(-0.05)
    assert len(summary["paired_bootstrap_question_95ci"]) == 2


def test_paired_outcomes_and_effect_thresholds() -> None:
    baseline = [{"seed": 42, "correct": False}, {"seed": 43, "correct": True}]
    guided = [{"seed": 42, "correct": True}, {"seed": 43, "correct": False}]
    assert _paired_counts(baseline, guided) == {
        "gain": 1, "loss": 1, "both_correct": 0, "both_wrong": 0,
    }
    assert _effect_label(2) == "helpful"
    assert _effect_label(-2) == "harmful"
    assert _effect_label(1) == "inconclusive"


def test_condition_card_prompt_says_not_to_solve() -> None:
    assert "Do not solve the problem" in CONDITION_CARD_PROMPT


def test_variant_phase_uses_same_level_reserve_and_rewrites_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    bundles = []
    reserves = []
    for level_index in range(1, 5):
        for item_index in range(10):
            bundles.append({
                "question_id": f"q{level_index}_{item_index}",
                "level": f"level{level_index}",
                "question_text": "source",
            })
        for item_index in range(5):
            reserves.append({
                "question_id": f"r{level_index}_{item_index}",
                "level": f"level{level_index}",
                "question_text": "reserve",
            })
    atomic_save_json(tmp_path / "source_bundles" / "source40_bundles.json", bundles)
    atomic_save_json(tmp_path / "source_bundles" / "reserve20_bundles.json", reserves)
    atomic_save_json(tmp_path / "manifests" / "source40.json", {
        "source_question_ids_by_level": {},
        "source_question_ids": [],
        "source_bundle_hashes": {},
        "stage1_input_hashes": {"packets": "hash"},
        "manifest_hash": "old",
    })

    async def fake_construct(client, bundle, *, output_dir, attempts):
        qid = str(bundle["question_id"])
        if qid == "q1_0":
            return {"status": "variant_unavailable", "source_question_id": qid,
                    "paired_source_level": bundle["level"], "history": []}
        return {
            "status": "verified", "source_question_id": qid,
            "paired_source_level": bundle["level"], "variant_question": "variant",
            "variant_answer": "1", "structure_spec": {"core_skill": "x"},
            "verification": {},
        }

    monkeypatch.setattr(principle_test_module, "_construct_variant", fake_construct)
    monkeypatch.setattr(principle_test_module, "Stage2ModelClient", lambda *args, **kwargs: object())
    config = SimpleNamespace(stage2=SimpleNamespace(principle_test_variant_attempts=3))
    result = asyncio.run(principle_test_module.generate_variants(
        config, test_root=tmp_path, concurrency=2
    ))

    assert result["verified_count"] == 40
    rewritten = json.loads((tmp_path / "source_bundles" / "source40_bundles.json").read_text())
    assert "r1_0" in {item["question_id"] for item in rewritten}
    assert "q1_0" not in {item["question_id"] for item in rewritten}
    manifest = json.loads((tmp_path / "manifests" / "source40.json").read_text())
    assert manifest["manifest_hash"] != "old"
