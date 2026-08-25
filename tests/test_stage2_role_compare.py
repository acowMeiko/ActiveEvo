from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import modules.stage2.model_client as model_client_module
import modules.stage2.role_compare as role
from core.config import Config
from modules.stage2.model_client import Stage2ModelClient


def _bundle(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "level": "level2",
        "question_text": "Find a general method for a constrained integer problem.",
        "expected_answer": "17",
        "rollouts": [
            {"rollout_id": "w1", "correct": False, "response": "wrong"},
            {"rollout_id": "c1", "correct": True, "response": "correct"},
        ],
        "trajectory_pairs": [],
    }


def _content() -> dict:
    return {
        "introspection": "A local condition was treated as sufficient.",
        "trigger_condition": "The problem asks for an integer satisfying two constraints.",
        "required_evidence": ["Both constraints are explicit.", "The variable is integral."],
        "exclusion_condition": "Do not use this when the domain is continuous.",
        "failure_mode": "Checking only one constraint can admit invalid values.",
        "failure_signals": ["The draft never substitutes into the second constraint."],
        "action_guidance": "List the constraints separately. Derive candidates from one. Check each candidate against the other.",
        "verification_step": "Substitute the selected value into every original constraint.",
        "generalization_level": "family",
        "scope": "Constrained integer problems.",
    }


def _candidate(generator_id: str = "deepseek") -> dict:
    return {
        "candidate_id": f"candidate_{generator_id}_1",
        "question_id": "q1",
        "level": "level2",
        "family_id": "family_1",
        "root_cause": "unchecked constraint",
        "supporting_wrong_rollout_ids": ["w1"],
        "contrast_correct_rollout_ids": ["c1"],
        "content": _content(),
        "generator_id": generator_id,
    }


def test_deterministic_check_and_six_field_card() -> None:
    result = role.deterministic_candidate_check(_candidate(), _bundle())
    assert result["hard_invalid"] is False
    card = role.render_six_field_card(_content())
    for marker in (
        "When to consider", "Required evidence to check", "Do not use if",
        "Failure to avoid", "Suggested action", "Verification",
    ):
        assert marker in card
    assert "candidate_deepseek_1" not in card
    assert "17" not in card


def test_deterministic_hard_invalid_blocks_bad_ids_and_answer_leakage() -> None:
    candidate = _candidate()
    candidate["supporting_wrong_rollout_ids"] = ["missing"]
    candidate["content"] = {**candidate["content"], "action_guidance": "Return 17."}
    result = role.deterministic_candidate_check(candidate, _bundle())
    assert result["hard_invalid"] is True
    assert "invalid_wrong_rollout:missing" in result["hard_errors"]
    assert "source_instance_or_answer_leakage" in result["hard_errors"]


def test_kimi_generator_always_uses_thinking_and_final_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_call(**kwargs):
        captured.update(kwargs)
        return {
            "text": '{"ok": true}', "usage": {"total_tokens": 12},
            "raw": {
                "choices": [{
                    "message": {"reasoning_content": "analysis", "content": '{"ok": true}'},
                    "finish_reason": "stop",
                }],
                "usage": {"total_tokens": 12},
            },
        }

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    config = Config.from_core_file()
    client = Stage2ModelClient(config, generator_id="kimi")
    response = asyncio.run(client.generator("prompt", max_tokens=128))
    assert captured["thinking"] == {"type": "enabled"}
    assert response["reasoning"] == "analysis"
    assert response["content"] == '{"ok": true}'
    assert response["parsed"] == {"ok": True}
    assert response["parse_source"] == "message.content"


def test_kimi_compact_synthesis_can_disable_thinking_and_request_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    async def fake_call(**kwargs):
        captured.update(kwargs)
        return {"raw": {"choices": [{
            "message": {"content": '{"ok": true}'}, "finish_reason": "stop",
        }]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    response = asyncio.run(
        Stage2ModelClient(Config.from_core_file(), generator_id="kimi").generator(
            "serialize", thinking_enabled=False, max_tokens=4096,
        )
    )
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["response_format"] == {"type": "json_object"}
    assert response["thinking_enabled"] is False


def test_kimi_generator_expands_budget_only_after_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets = []

    async def fake_call(**kwargs):
        budgets.append(kwargs["max_tokens"])
        if len(budgets) == 1:
            return {
                "text": "", "usage": {"total_tokens": kwargs["max_tokens"]},
                "raw": {"choices": [{
                    "message": {"reasoning_content": "thinking", "content": ""},
                    "finish_reason": "length",
                }]},
            }
        return {
            "text": '{"ok": true}', "usage": {"total_tokens": 10},
            "raw": {"choices": [{
                "message": {"reasoning_content": "thinking", "content": '{"ok": true}'},
                "finish_reason": "stop",
            }]},
        }

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    config = Config.from_core_file()
    client = Stage2ModelClient(config, generator_id="kimi")
    response = asyncio.run(client.generator("prompt", max_tokens=4096))
    assert budgets == [4096, 8192]
    assert response["parsed"] == {"ok": True}
    assert len(response["attempts"]) == 2


def test_saved_truncation_is_recoverable_only_below_large_budget() -> None:
    config = Config.from_core_file()
    synthesis = {
        "finish_reason": "length", "failure_reason": "thinking_truncated",
        "generation_attempts": [{"max_tokens": 4096}],
    }
    assert role._needs_kimi_length_recovery(config, synthesis)
    synthesis["generation_attempts"].append({"max_tokens": 8192})
    assert not role._needs_kimi_length_recovery(config, synthesis)


def test_cross_audit_is_blind_and_runs_once_on_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_review(client, bundle, candidate, *, reviewer_id):
        calls.append((candidate["generator_id"], reviewer_id))
        return {
            "status": "complete", "reviewer_id": reviewer_id,
            "critic_response": {}, "formatter_response": {}, "protocol_errors": [],
            "audit": {
                "dimensions": {
                    name: {"pass": True, "reason": "ok", "evidence_refs": []}
                    for name in role.AUDIT_DIMENSIONS
                },
                "decision": "pass",
            },
        }

    monkeypatch.setattr(role, "_semantic_review", fake_review)
    config = Config.from_core_file()

    async def run(generator_id):
        bank = {
            "generator_id": generator_id,
            "candidates": [_candidate(generator_id)],
        }
        return await role._cross_audit_bank(
            config, bank=bank, bundle_by_id={"q1": _bundle()},
            output_root=tmp_path / "cross", initial_concurrency=1,
            maximum_concurrency=2,
        )

    ds = asyncio.run(run("deepseek"))
    km = asyncio.run(run("kimi"))
    assert calls == [("deepseek", "kimi"), ("kimi", "deepseek")]
    assert ds[0]["model_calls"] == 1
    assert km[0]["model_calls"] == 1
    assert ds[0]["repair"] is None and ds[0]["reaudit"] is None


def test_role_compare_reuses_deepseek_bank_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "smoke8"
    bundles = [{**_bundle(f"q{i}"), "question_id": f"q{i}"} for i in range(8)]
    source_path = root / "source_bundles" / "source40_bundles.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(bundles), encoding="utf-8")
    candidates = []
    for index in range(17):
        item = _candidate("deepseek")
        item.update({
            "candidate_id": f"candidate_ds_{index}",
            "question_id": f"q{index % 8}",
        })
        candidates.append(item)
    bank = {
        "candidate_count": 17, "candidate_bank_hash": "frozen-hash",
        "candidates": candidates,
    }
    bank_path = root / "raw_candidates" / "candidate_bank.json"
    bank_path.parent.mkdir(parents=True)
    original = json.dumps(bank, sort_keys=True)
    bank_path.write_text(original, encoding="utf-8")

    kimi_bank = {
        "generator_id": "kimi", "question_count": 8, "candidate_count": 1,
        "candidate_bank_hash": "kimi-hash", "status_counts": {"complete": 8},
        "candidates": [_candidate("kimi")],
    }

    async def fake_generate(*args, **kwargs):
        return kimi_bank

    async def fake_audit(config, *, bank, **kwargs):
        return [{
            "candidate_id": item["candidate_id"], "generator_id": bank["generator_id"],
            "final_decision": "pass", "deterministic": {"hard_invalid": False},
            "final_candidate": item,
        } for item in bank["candidates"]]

    async def fake_eval(*args, **kwargs):
        effects = []
        for candidate_bank in kwargs["banks"]:
            for item in candidate_bank["candidates"]:
                effects.append({
                    "candidate_id": item["candidate_id"], "generator_id": item["generator_id"],
                    "version": "raw", "question_id": item["question_id"],
                    "delta_correct": 0, "paired": {"gain": 0, "loss": 0},
                })
        return {"effects": effects, "rollout_count": 0}

    monkeypatch.setattr(role, "_generate_kimi_bank", fake_generate)
    monkeypatch.setattr(role, "_cross_audit_bank", fake_audit)
    monkeypatch.setattr(role, "_evaluate_role_candidates", fake_eval)
    monkeypatch.setattr(role, "_healthy_actor_urls", lambda ports: [f"http://localhost:{p}" for p in ports])
    monkeypatch.setenv("ACTIVEEVO_GENERATOR_API_KEY", "test")
    monkeypatch.setenv("ACTIVEEVO_AUDITOR_API_KEY", "test")
    config = Config.from_core_file()
    result = asyncio.run(role.run_role_compare(
        config, test_root=root, ports=(8101, 8103), model="llama",
    ))
    assert bank_path.read_text(encoding="utf-8") == original
    assert result["formal_claim_allowed"] is False
    assert (root / "role_compare_v1" / "summary.json").exists()
