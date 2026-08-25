import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import modules.stage2.incremental_registry as registry
from utils.io_utils import atomic_save_json


def _pools(per_level: int = 220) -> dict[str, list[dict]]:
    return {
        level: [
            {
                "question_id": f"{level}_q{index:04d}",
                "level": level,
                "question_text": f"Question {level} {index}",
                "expected_answer": str(index),
            }
            for index in range(per_level)
        ]
        for level in registry.LEVELS
    }


def _config() -> SimpleNamespace:
    return SimpleNamespace(stage2=SimpleNamespace(principle_test_seed=42))


def _protocol() -> dict:
    return {
        "protocol_hash": "protocol-v1",
        "generation_protocol_hash": "generation-v1",
        "candidate_protocol_hash": "candidate-v1",
        "auditor": {"protocol": "kimi_single"},
        "intervention_arm": "direct_short_card",
    }


def test_incremental_targets_are_nested_and_keep_explicit_quotas(
    tmp_path: Path, monkeypatch
) -> None:
    pools = _pools()
    monkeypatch.setattr(registry, "_load_bundles_by_level", lambda *_: (pools, {"x": "y"}))
    monkeypatch.setattr(registry, "generation_input_hash", lambda *_: "bundle-v1")
    workspace = tmp_path / "workspace"
    registry_root = tmp_path / "registry"

    results = {}
    for target in (40, 100, 600):
        registry.prepare_incremental_target(
            _config(), target=target, workspace_root=workspace,
            registry_root=registry_root, protocol=_protocol(),
        )
        results[target] = json.loads(
            (workspace / f"target_{target:04d}" / "manifests" / "source40.json")
            .read_text(encoding="utf-8")
        )

    assert set(results[40]["source_question_ids"]).issubset(results[100]["source_question_ids"])
    assert set(results[100]["source_question_ids"]).issubset(results[600]["source_question_ids"])
    assert results[40]["level_quotas"] == {level: 10 for level in registry.LEVELS}
    assert results[100]["level_quotas"] == {level: 25 for level in registry.LEVELS}
    assert results[600]["level_quotas"] == {
        "level1": 100, "level2": 200, "level3": 200, "level4": 100,
    }


def test_registry_reuse_requires_matching_generation_input_hash(
    tmp_path: Path, monkeypatch
) -> None:
    pools = _pools(12)
    monkeypatch.setattr(registry, "_load_bundles_by_level", lambda *_: (pools, {}))
    monkeypatch.setattr(registry, "generation_input_hash", lambda *_: "expected")
    initial = registry.prepare_incremental_target(
        _config(), target=40, workspace_root=tmp_path / "workspace",
        registry_root=tmp_path / "registry", protocol=_protocol(),
    )
    qid = next(qid for qid in initial["questions"] if qid.startswith("level1_"))
    generation_path = registry._registry_generation_path(
        tmp_path / "registry", qid, "generation-v1"
    )
    atomic_save_json(generation_path, {
        "question_id": qid, "level": "level1",
        "status": "no_reliable_principle", "generation_input_hash": "stale",
    })
    result = registry.prepare_incremental_target(
        _config(), target=40, workspace_root=tmp_path / "workspace",
        registry_root=tmp_path / "registry", protocol=_protocol(),
    )
    assert result["questions"][qid]["generation"] == "missing"

    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["generation_input_hash"] = "expected"
    atomic_save_json(generation_path, generation)
    result = registry.prepare_incremental_target(
        _config(), target=40, workspace_root=tmp_path / "workspace",
        registry_root=tmp_path / "registry", protocol=_protocol(),
    )
    assert result["questions"][qid]["generation"] == "reusable"
    assert result["questions"][qid]["audit"] == "reusable"


def test_freeze_is_immutable_and_records_terminal_questions(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = _protocol()
    monkeypatch.setattr(registry, "current_stage2_protocol", lambda *_, **__: protocol)
    root = tmp_path / "round"
    qid = "q1"
    manifest = {
        "target_count": 1,
        "source_question_ids": [qid],
        "source_bundle_hashes": {qid: "bundle-hash"},
        "manifest_hash": "manifest-hash",
    }
    generation = {
        "question_id": qid, "level": "level1",
        "status": "no_reliable_principle", "generation_input_hash": "generation-input",
    }
    bank = {
        "candidate_count": 0, "candidates": [], "candidate_bank_hash": "bank-hash",
    }
    atomic_save_json(root / "manifests" / "source40.json", manifest)
    atomic_save_json(root / "raw_candidates" / "candidate_bank.json", bank)
    atomic_save_json(root / "raw_candidates" / "questions" / f"{qid}.json", generation)
    atomic_save_json(root / "source_evaluation" / "frozen_protocol.json", {
        "audit_protocol": "kimi_single", "intervention_arm": "direct_short_card",
    })
    quality_path = tmp_path / "quality.json"
    atomic_save_json(quality_path, {
        "approved": True, "reviewer": "test", "rationale": "checked",
        "target_count": 1, "manifest_hash": "manifest-hash",
        "candidate_bank_hash": "bank-hash", "protocol_hash": "protocol-v1",
    })

    release = registry.freeze_stage2_round(
        _config(), target_root=root, registry_root=tmp_path / "registry",
        state_root=tmp_path / "state", target_count=1,
        quality_decision_path=quality_path,
    )
    assert release["question_terminal_counts"] == {"no_reliable_principle": 1}
    assert (tmp_path / "state" / "releases" / "round_0001" / "release.json").exists()
    assert registry.freeze_stage2_round(
        _config(), target_root=root, registry_root=tmp_path / "registry",
        state_root=tmp_path / "state", target_count=1,
        quality_decision_path=quality_path,
    )["release_hash"] == release["release_hash"]

    changed = dict(generation)
    changed["status"] = "terminal_failed"
    atomic_save_json(root / "raw_candidates" / "questions" / f"{qid}.json", changed)
    with pytest.raises(RuntimeError, match="immutable"):
        registry.freeze_stage2_round(
            _config(), target_root=root, registry_root=tmp_path / "registry",
            state_root=tmp_path / "state", target_count=1,
            quality_decision_path=quality_path,
        )


def test_unapproved_round_cannot_be_frozen(tmp_path: Path) -> None:
    quality_path = tmp_path / "quality.json"
    atomic_save_json(quality_path, {"approved": False})
    with pytest.raises(RuntimeError, match="not been approved"):
        registry.freeze_stage2_round(
            _config(), target_root=tmp_path / "round",
            registry_root=tmp_path / "registry", state_root=tmp_path / "state",
            target_count=40, quality_decision_path=quality_path,
        )


def test_quality_decision_must_be_bound_to_exact_round(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(registry, "current_stage2_protocol", lambda *_, **__: _protocol())
    root = tmp_path / "round"
    atomic_save_json(root / "manifests" / "source40.json", {
        "target_count": 1, "source_question_ids": [],
        "source_bundle_hashes": {}, "manifest_hash": "manifest-hash",
    })
    atomic_save_json(root / "raw_candidates" / "candidate_bank.json", {
        "candidate_count": 0, "candidates": [], "candidate_bank_hash": "bank-hash",
    })
    atomic_save_json(root / "source_evaluation" / "frozen_protocol.json", {
        "audit_protocol": "kimi_single", "intervention_arm": "direct_short",
    })
    quality_path = tmp_path / "quality.json"
    atomic_save_json(quality_path, {
        "approved": True, "reviewer": "test", "rationale": "checked",
        "target_count": 1, "manifest_hash": "wrong-manifest",
        "candidate_bank_hash": "bank-hash", "protocol_hash": "protocol-v1",
    })
    with pytest.raises(RuntimeError, match="not bound"):
        registry.freeze_stage2_round(
            _config(), target_root=root, registry_root=tmp_path / "registry",
            state_root=tmp_path / "state", target_count=1,
            quality_decision_path=quality_path,
        )
