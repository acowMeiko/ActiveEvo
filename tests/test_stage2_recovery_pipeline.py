from __future__ import annotations

import json

from core.config import Config
from modules.stage2.model_client import parse_json_object
from modules.stage2.recovery_pipeline import (
    EXPECTED_RECOVERY_BY_LEVEL,
    _explicit_no_reliable,
    _fit_qwen_context,
    _qwen_candidate,
    _round_robin_recovery_order,
    _wilson_interval,
)
from pipeline import build_parser


def test_only_explicit_empty_final_json_is_no_reliable() -> None:
    valid = {
        "status": "no_reliable_principle",
        "synthesis": {
            "parse_success": True,
            "finish_reason": "stop",
            "raw_response": json.dumps({"failure_families": []}),
        },
    }
    malformed = {
        "status": "no_reliable_principle",
        "synthesis": {
            "parse_success": False,
            "finish_reason": "stop",
            "raw_response": "not-json",
        },
    }
    invalid_families = {
        "status": "no_reliable_principle",
        "synthesis": {
            "parse_success": True,
            "finish_reason": "stop",
            "raw_response": json.dumps({"failure_families": [{"bad": True}]}),
        },
    }
    assert _explicit_no_reliable(valid)
    assert not _explicit_no_reliable(malformed)
    assert not _explicit_no_reliable(invalid_families)


def test_recovery_prefix8_contains_two_questions_per_level() -> None:
    ids = {
        level: [f"{level}-{index}" for index in range(count)]
        for level, count in EXPECTED_RECOVERY_BY_LEVEL.items()
    }
    ordered = _round_robin_recovery_order(ids)
    assert len(ordered) == sum(EXPECTED_RECOVERY_BY_LEVEL.values())
    assert {
        level: sum(qid.startswith(f"{level}-") for qid in ordered[:8])
        for level in ids
    } == {level: 2 for level in ids}


def test_oversized_pairing_hints_are_compacted_without_dropping_rollouts() -> None:
    long_hint = "same duplicated divergence " * 1000
    bundle = {
        "question_id": "q-long",
        "question_text": "Solve symbolically.",
        "rollouts": [
            {"rollout_id": f"r{i}", "correct": i == 0, "response": "full trace"}
            for i in range(10)
        ],
        "trajectory_pairs": [
            {"wrong_rollout_id": f"r{i}", "closest_correct_rollout_id": "r0",
             "divergence_step": 1, "divergence_step_text": long_hint}
            for i in range(1, 10)
        ],
    }
    compact, metadata = _fit_qwen_context(bundle, 3)
    assert metadata["applied"] is True
    assert len(compact["rollouts"]) == 10
    assert compact["rollouts"] == bundle["rollouts"]
    assert all(len(row["divergence_step_text"]) <= 600 for row in compact["trajectory_pairs"])


def test_qwen_candidate_id_and_provenance_are_generator_specific() -> None:
    record = {"question_id": "q1", "level": "level2"}
    family = {
        "family_id": "family_1",
        "root_cause": "cause",
        "supporting_wrong_rollout_ids": ["w1"],
        "contrast_correct_rollout_ids": ["c1"],
        "content": {"trigger_condition": "condition"},
    }
    candidate = _qwen_candidate(record, family)
    assert candidate["candidate_id"].startswith("qwen_")
    assert candidate["generator_id"] == "qwen"


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = _wilson_interval(2, 10)
    assert lower < 0.2 < upper


def test_incremental_recovery_cli_contract() -> None:
    parser = build_parser(Config.from_core_file())
    args = parser.parse_args([
        "stage2", "--mode", "incremental",
        "--incremental-phase", "recover-generate",
        "--target-size", "1200", "--generator", "qwen",
        "--recovery-limit", "30",
    ])
    assert args.incremental_phase == "recover-generate"
    assert args.generator == "qwen"
    assert args.recovery_limit == 30


def test_json_parser_never_accepts_top_level_list() -> None:
    assert parse_json_object('[{"failure_families": []}]') == {}
