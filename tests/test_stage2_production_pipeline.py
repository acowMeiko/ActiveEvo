from __future__ import annotations

import asyncio
import json

from core.config import Config
from modules.stage2.principle_generation import build_generation_prompt
from modules.stage2.principle_test import _generation_record_reusable
from modules.stage2.production_pipeline import (
    BATCH300_QUOTAS,
    PRODUCTION_QUOTAS,
    _candidate_effect,
    _resolve_kimi_audit_decision,
    _reuse_generation_records,
    _select_kimi_initial_candidates,
    _revision_transaction,
    prepare_production_target,
)


def _candidate() -> dict:
    return {"candidate_id": "c1", "question_id": "q1", "level": "level2"}


def _rows(values: list[bool]) -> list[dict]:
    return [{"seed": 42 + index, "correct": value, "status": "complete"} for index, value in enumerate(values)]


def test_production_quota_contract() -> None:
    assert sum(PRODUCTION_QUOTAS.values()) == 1200
    assert sum(BATCH300_QUOTAS.values()) == 300
    assert all(BATCH300_QUOTAS[key] <= PRODUCTION_QUOTAS[key] for key in PRODUCTION_QUOTAS)


def test_prepare_freezes_nested_300_without_protocol40(tmp_path) -> None:
    result = prepare_production_target(
        Config.from_core_file(), production_root=tmp_path,
        target_size=300, backbone="llama",
    )
    production = json.loads((tmp_path / "manifests" / "production1200.json").read_text())
    batch = json.loads((tmp_path / "manifests" / "batch300.json").read_text())
    reserve = json.loads((tmp_path / "manifests" / "reserve18.json").read_text())
    assert result["target_size"] == 300
    assert len(production["source_question_ids"]) == 1200
    assert len(batch["source_question_ids"]) == 300
    assert len(reserve["source_question_ids"]) == 18
    assert set(batch["source_question_ids"]).issubset(production["source_question_ids"])
    assert not set(production["excluded_protocol_question_ids"]) & set(production["source_question_ids"])


def test_source_helpful_requires_both_controls() -> None:
    baseline = {("q1", 42 + i): row for i, row in enumerate(_rows([True] * 4 + [False] * 6))}
    generic = _rows([True] * 3 + [False] * 7)
    guided = _rows([True] * 5 + [False] * 5)
    effect = _candidate_effect(_candidate(), baseline, generic, guided)
    assert effect["delta_vs_generic"] == 2
    assert effect["delta_vs_original"] == 1
    assert effect["effect_label"] == "source_helpful"


def test_improvement_below_original_is_not_admitted() -> None:
    baseline = {("q1", 42 + i): row for i, row in enumerate(_rows([True] * 7 + [False] * 3))}
    generic = _rows([True] * 2 + [False] * 8)
    guided = _rows([True] * 5 + [False] * 5)
    effect = _candidate_effect(_candidate(), baseline, generic, guided)
    assert effect["delta_vs_generic"] == 3
    assert effect["delta_vs_original"] == -2
    assert effect["effect_label"] == "source_harmful"


def test_generation_prompt_is_post_draft_and_preservation_aware() -> None:
    bundle = {
        "question_text": "A symbolic question",
        "rollouts": [
            {"rollout_id": "w", "correct": False, "response": "wrong"},
            {"rollout_id": "c", "correct": True, "response": "correct"},
        ],
        "trajectory_pairs": [],
    }
    prompt = build_generation_prompt(bundle, 3)
    assert "AFTER a solver has produced a complete draft" in prompt
    assert "preserve every valid part" in prompt
    assert "source-specific numbers" in prompt


def test_cross_batch_generation_reuse_is_hash_gated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.stage2.production_pipeline.generation_input_hash",
        lambda _config, bundle: f"hash-{bundle['question_id']}",
    )
    reuse_root = tmp_path / "batch_0300"
    source_dir = reuse_root / "raw_candidates" / "questions"
    source_dir.mkdir(parents=True)
    compatible = {
        "question_id": "q1",
        "status": "complete",
        "generation_input_hash": "hash-q1",
    }
    incompatible = {
        "question_id": "q2",
        "status": "complete",
        "generation_input_hash": "old-hash",
    }
    (source_dir / "q1.json").write_text(json.dumps(compatible), encoding="utf-8")
    (source_dir / "q2.json").write_text(json.dumps(incompatible), encoding="utf-8")

    target_dir = tmp_path / "batch_1200" / "raw_candidates" / "questions"
    summary = _reuse_generation_records(
        object(), bundles={"q1": {"question_id": "q1"}, "q2": {"question_id": "q2"}},
        result_dir=target_dir, reuse_roots=(reuse_root,),
    )

    assert summary == {
        "reused_question_count": 1,
        "preexisting_question_count": 0,
        "incompatible_target_question_count": 0,
        "pending_generation_count": 1,
    }
    assert json.loads((target_dir / "q1.json").read_text()) == compatible
    assert not (target_dir / "q2.json").exists()

    resumed = _reuse_generation_records(
        object(), bundles={"q1": {"question_id": "q1"}, "q2": {"question_id": "q2"}},
        result_dir=target_dir, reuse_roots=(reuse_root,),
    )
    assert resumed["reused_question_count"] == 0
    assert resumed["preexisting_question_count"] == 1
    assert resumed["pending_generation_count"] == 1


def test_kimi_sample_is_level_balanced_and_one_per_question() -> None:
    candidates = {}
    effects = []
    for level in range(1, 5):
        for question in range(3):
            qid = f"l{level}q{question}"
            for candidate_index, delta in enumerate((2, 5)):
                cid = f"{qid}c{candidate_index}"
                candidates[cid] = {
                    "candidate_id": cid, "question_id": qid, "level": f"level{level}",
                }
                effects.append({
                    "candidate_id": cid, "question_id": qid,
                    "effect_label": "source_helpful", "delta_vs_generic": delta,
                    "paired_vs_generic": {"gain": delta, "loss": 0},
                })

    selected = _select_kimi_initial_candidates(candidates, effects, limit=8)
    assert len(selected) == 8
    assert len({row["question_id"] for row in selected}) == 8
    assert all(row["candidate_id"].endswith("c1") for row in selected)
    counts = {}
    for row in selected:
        counts[row["level"]] = counts.get(row["level"], 0) + 1
    assert counts == {"level1": 2, "level2": 2, "level3": 2, "level4": 2}


def test_kimi_decision_dimension_conflict_is_not_rejected() -> None:
    audit = {
        "decision": "pass",
        "dimensions": {
            "source_fidelity": {"pass": True},
            "mathematical_validity": {"pass": False},
            "instance_leakage": {"pass": True},
            "condition_decidability": {"pass": True},
            "action_executability": {"pass": True},
            "verification_executability": {"pass": True},
        },
    }
    decision, errors = _resolve_kimi_audit_decision(audit)
    assert decision == "audit_inconsistent"
    assert errors and "reported=pass:derived=reject" in errors[0]


def test_transport_terminal_failure_is_retryable() -> None:
    record = {
        "status": "terminal_failed",
        "generation_input_hash": "same",
        "diagnosis": {
            "response": {
                "content": "",
                "reasoning": "",
                "attempts": [{
                    "transport_attempts": [{"error": "network_exception"}],
                }],
            },
        },
    }
    assert _generation_record_reusable(record, "same") is False


def test_non_transport_terminal_failure_remains_terminal() -> None:
    record = {
        "status": "terminal_failed",
        "generation_input_hash": "same",
        "diagnosis": {"response": {"content": "", "reasoning": "", "attempts": []}},
    }
    assert _generation_record_reusable(record, "same") is True


def test_corrupt_rollout_checkpoint_is_archived_and_recomputed(tmp_path) -> None:
    class Execute:
        async def _call_with_router_metadata(self, **kwargs):
            text = "Final Answer: \\boxed{1}" if "finalizer" in kwargs["prompt"].lower() else "Reasoning: 1"
            return {
                "text": text,
                "raw": {"choices": [{"finish_reason": "stop"}]},
                "usage": {"completion_tokens": 4},
            }, kwargs["url_override"]

    class Runner:
        urls = ["http://actor.test/v1/chat/completions"]
        model = "llama"
        timeout = 30
        semaphore = asyncio.Semaphore(1)
        execute = Execute()

    path = tmp_path / "candidate_s42.json"
    path.write_text("", encoding="utf-8")
    result = asyncio.run(_revision_transaction(
        Runner(), path=path, question_id="q1", candidate_id="c1",
        question="What is 1?", expected_answer="1", seed=42,
        prompt="Solve the problem.", stage="forced_c/raw",
    ))
    assert result["status"] == "complete"
    assert json.loads(path.read_text())["status"] == "complete"
    archived = list((tmp_path / "corrupt_checkpoints").glob("*.corrupt"))
    assert len(archived) == 1
    assert archived[0].read_text() == ""
