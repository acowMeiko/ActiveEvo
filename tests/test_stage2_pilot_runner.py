import asyncio
from pathlib import Path
from types import SimpleNamespace

import modules.stage2.pilot_runner as runner
from utils.io_utils import atomic_save_json, load_json


def _packet(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "question": f"Question {question_id}",
        "answer": "7",
        "evidence_complete": True,
        "rollouts": [
            {"rollout_id": f"{question_id}_correct", "response": "valid", "correct": True},
            {"rollout_id": f"{question_id}_wrong", "response": "invalid", "correct": False},
        ],
    }


def _content() -> dict:
    return {
        "introspection": "The failed draft applied an operation before checking its premise.",
        "trigger_condition": "Consider this when the requested quantity depends on a constrained relation.",
        "required_evidence": ["The relation is explicitly given.", "The target depends on that relation."],
        "exclusion_condition": "Do not use it when the relation is absent.",
        "failure_mode": "Applying a transformation outside its valid conditions.",
        "failure_signals": ["A draft applies the transformation without checking its premise."],
        "action_guidance": "First identify the stated relation. Then verify its premises. Apply the transformation only after that check.",
        "verification_step": "Substitute the result back into the original relation and check every constraint.",
        "generalization_level": "family",
        "scope": "Problems with the same constrained relation.",
    }


def test_pilot_manifest_keeps_two_dev_and_adds_twenty_three(monkeypatch, tmp_path: Path) -> None:
    packets = {
        f"level{level}": [_packet(str(level * 1000 + index)) for index in range(40)]
        for level in range(1, 5)
    }
    monkeypatch.setattr(runner, "_load_packets_by_level", lambda config, backbone: packets)
    dev_by_level = {
        f"level{level}": [str(level * 1000), str(level * 1000 + 1)]
        for level in range(1, 5)
    }
    dev_path = tmp_path / "dev8.json"
    atomic_save_json(dev_path, {"question_ids_by_level": dev_by_level})
    manifest = runner.prepare_pilot_manifest(
        SimpleNamespace(), output_root=tmp_path / "pilot", dev_manifest_path=dev_path
    )

    assert manifest["actual_count"] == 100
    for level in runner.LEVELS:
        assert len(manifest["question_ids_by_level"][level]) == 25
        assert manifest["question_ids_by_level"][level][:2] == dev_by_level[level]
    assert len(manifest["bundle_hashes"]) == 100


def test_checkpoint_protocol_hash_mismatch_is_not_reused(tmp_path: Path) -> None:
    path = tmp_path / "stage.json"
    runner._save_stage(path, "old", {"success": True})
    assert runner._load_stage(path, "old")["success"] is True
    assert runner._load_stage(path, "new") is None


def test_dual_audits_are_independently_checkpointed(monkeypatch, tmp_path: Path) -> None:
    calls = []

    async def fake_audit(client, bundle, candidate, *, auditor_id):
        calls.append(auditor_id)
        return {"auditor_id": auditor_id, "decision": "pass", "protocol_compliant": True}

    monkeypatch.setattr(runner, "audit_principle", fake_audit)
    kwargs = dict(
        client=object(), bundle={}, candidate={}, semaphores={
            "deepseek": asyncio.Semaphore(1), "qwen": asyncio.Semaphore(1),
        },
        candidate_root=tmp_path, prefix="initial", run_hash="same",
    )
    first = asyncio.run(runner._persist_dual_audit(**kwargs))
    second = asyncio.run(runner._persist_dual_audit(**kwargs))

    assert set(first) == {"deepseek", "qwen"}
    assert second == first
    assert sorted(calls) == ["deepseek", "qwen"]
    assert (tmp_path / "initial_deepseek_audit.json").exists()
    assert (tmp_path / "initial_qwen_audit.json").exists()


def test_one_failed_auditor_does_not_cancel_other_checkpoint(monkeypatch, tmp_path: Path) -> None:
    async def fake_audit(client, bundle, candidate, *, auditor_id):
        if auditor_id == "deepseek":
            return {"decision": "reject", "protocol_compliant": False, "protocol_errors": ["empty"]}
        await asyncio.sleep(0.01)
        return {"decision": "pass", "protocol_compliant": True}

    monkeypatch.setattr(runner, "audit_principle", fake_audit)
    try:
        asyncio.run(runner._persist_dual_audit(
            object(), {}, {"candidate_id": "c1"}, semaphores={
                "deepseek": asyncio.Semaphore(1), "qwen": asyncio.Semaphore(1),
            },
            candidate_root=tmp_path, prefix="initial", run_hash="same",
        ))
    except RuntimeError:
        pass
    else:
        raise AssertionError("A non-compliant auditor must fail the stage")

    assert (tmp_path / "initial_deepseek_audit.json").exists()
    qwen = load_json(str(tmp_path / "initial_qwen_audit.json"))
    assert qwen["audit"]["protocol_compliant"] is True


def test_resume_reuses_diagnosis_and_restarts_at_synthesis(monkeypatch, tmp_path: Path) -> None:
    calls = {"diagnosis": 0, "synthesis": 0}

    async def fake_diagnosis(client, bundle, maximum):
        calls["diagnosis"] += 1
        return {"success": True, "diagnosis_text": "persisted", "response": {}}

    async def fake_synthesis(client, bundle, maximum, diagnosis, correction_feedback=""):
        calls["synthesis"] += 1
        if calls["synthesis"] == 1:
            return {"parse_success": False, "failure_reason": "temporary"}
        return {
            "parse_success": True,
            "schema_failures": [],
            "routing_semantics": {},
            "families": [{
                "family_id": "family_1",
                "root_cause": "premise not checked",
                "supporting_wrong_rollout_ids": ["w"],
                "contrast_correct_rollout_ids": ["c"],
                "content": _content(),
            }],
        }

    async def fake_dual(*args, **kwargs):
        return {
            name: {"decision": "pass", "protocol_compliant": True}
            for name in ("deepseek", "qwen")
        }

    monkeypatch.setattr(runner, "diagnose_principles", fake_diagnosis)
    monkeypatch.setattr(runner, "synthesize_principles", fake_synthesis)
    monkeypatch.setattr(runner, "_persist_dual_audit", fake_dual)
    config = SimpleNamespace(stage2=SimpleNamespace(max_principles_per_question=3))
    bundle = {
        "question_id": "q1", "bundle_id": "b1", "question_text": "A symbolic problem",
        "expected_answer": "7", "level": "level2", "rollouts": [],
    }
    kwargs = dict(
        config=config, generator_id="deepseek", bundle=bundle, branch_root=tmp_path,
        run_hash="protocol", client=object(), generation_semaphore=asyncio.Semaphore(1),
        audit_semaphores={"deepseek": asyncio.Semaphore(1), "qwen": asyncio.Semaphore(1)},
        retry_terminal=False,
    )
    first = asyncio.run(runner._process_question(**kwargs))
    second = asyncio.run(runner._process_question(**kwargs))

    assert first["status"] == "retryable_failed"
    assert second["status"] == "completed"
    assert second["failure_attempts"] == 1
    assert second["resume_count"] == 2
    assert calls == {"diagnosis": 1, "synthesis": 2}
    synthesis = load_json(str(tmp_path / "questions" / "q1" / "synthesis.json"))
    assert len(synthesis["attempt_history"]) == 1
