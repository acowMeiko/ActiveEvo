import json
from pathlib import Path

import pytest

from modules.stage3_snapshot import prepare_transfer_snapshot, select_safe_inconclusive


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prepare_transfer_snapshot_isolated_status_override(tmp_path: Path) -> None:
    pending = tmp_path / "pending.jsonl"
    source = {
        "memory_id": "candidate_1",
        "question_id": "q1",
        "status": "pending_transfer_validation",
        "content": {"action_guidance": "check the invariant"},
    }
    pending.write_text(json.dumps(source) + "\n", encoding="utf-8")
    generation = tmp_path / "questions"
    _write_json(generation / "q1.json", {
        "question_id": "q1",
        "synthesis": {"routing_semantics": {
            "core_skill": "invariant",
            "context_constraints": "integer sequence",
            "task_objective": "derive the final value",
        }},
    })

    manifest = prepare_transfer_snapshot(
        pending_store=pending,
        generation_dir=generation,
        snapshot_dir=tmp_path / "snapshot",
    )
    result = json.loads(
        (tmp_path / "snapshot/principles/principles.json").read_text(encoding="utf-8")
    )
    assert manifest["memory_count"] == 1
    assert result[0]["status"] == "active"
    assert result[0]["source_status"] == "pending_transfer_validation"
    assert result[0]["evaluation_only_status_override"] is True
    assert result[0]["core_skill"] == "invariant"
    assert json.loads(pending.read_text(encoding="utf-8"))["status"] == "pending_transfer_validation"


def test_prepare_transfer_snapshot_fails_closed_without_routing(tmp_path: Path) -> None:
    pending = tmp_path / "pending.jsonl"
    pending.write_text(json.dumps({
        "memory_id": "candidate_1", "question_id": "q1",
        "status": "pending_transfer_validation", "content": {},
    }) + "\n", encoding="utf-8")
    generation = tmp_path / "questions"
    generation.mkdir()
    with pytest.raises(ValueError, match="Missing three-facet routing semantics"):
        prepare_transfer_snapshot(
            pending_store=pending,
            generation_dir=generation,
            snapshot_dir=tmp_path / "snapshot",
        )


def test_safe_inconclusive_selection_and_expanded_snapshot(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    candidates = [
        {"candidate_id": "safe", "question_id": "q2", "level": "level3", "content": {"action_guidance": "check"}},
        {"candidate_id": "has_loss", "question_id": "q3", "level": "level3", "content": {"action_guidance": "avoid"}},
        {"candidate_id": "regressed", "question_id": "q4", "level": "level3", "content": {"action_guidance": "verify"}},
    ]
    _write_json(batch / "raw_candidates/candidate_bank.json", {"candidates": candidates})
    _write_json(batch / "local_validation/candidate_effects_raw.json", [
        {"candidate_id": "safe", "effect_label": "inconclusive", "delta_vs_original": 0, "paired_vs_generic": {"gain": 1, "loss": 0}},
        {"candidate_id": "has_loss", "effect_label": "inconclusive", "delta_vs_original": 0, "paired_vs_generic": {"gain": 1, "loss": 1}},
        {"candidate_id": "regressed", "effect_label": "inconclusive", "delta_vs_original": -1, "paired_vs_generic": {"gain": 0, "loss": 0}},
    ])
    for candidate in candidates:
        _write_json(batch / "format_checks" / f"{candidate['candidate_id']}.json", {"hard_invalid": False})

    selected = select_safe_inconclusive(batch)
    assert [row["memory_id"] for row in selected] == ["safe"]
    assert selected[0]["content_audit_status"] == "not_run"

    pending = tmp_path / "pending.jsonl"
    pending.write_text(json.dumps({
        "memory_id": "validated", "question_id": "q1",
        "status": "pending_transfer_validation", "content": {"action_guidance": "validated"},
    }) + "\n", encoding="utf-8")
    generation = tmp_path / "questions"
    for qid in ("q1", "q2"):
        _write_json(generation / f"{qid}.json", {
            "question_id": qid,
            "synthesis": {"routing_semantics": {
                "core_skill": f"skill-{qid}",
                "context_constraints": f"constraints-{qid}",
                "task_objective": f"objective-{qid}",
            }},
        })
    manifest = prepare_transfer_snapshot(
        pending_store=pending,
        generation_dir=generation,
        snapshot_dir=tmp_path / "expanded",
        exploratory_batch_root=batch,
        expected_exploratory_count=1,
    )
    records = json.loads((tmp_path / "expanded/principles/principles.json").read_text())
    assert manifest["memory_count"] == 2
    assert manifest["validated_memory_count"] == 1
    assert manifest["exploratory_memory_count"] == 1
    by_id = {row["memory_id"]: row for row in records}
    assert by_id["safe"]["status"] == "active"
    assert by_id["safe"]["source_status"] == "source_inconclusive_no_loss"
    assert "unaudited" in by_id["safe"]["evidence_tier"]


def test_expanded_snapshot_fails_on_unexpected_exploratory_count(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    _write_json(batch / "raw_candidates/candidate_bank.json", {"candidates": []})
    _write_json(batch / "local_validation/candidate_effects_raw.json", [])
    (batch / "format_checks").mkdir(parents=True)
    pending = tmp_path / "pending.jsonl"
    pending.write_text("", encoding="utf-8")
    generation = tmp_path / "questions"
    generation.mkdir()
    with pytest.raises(ValueError, match="count mismatch"):
        prepare_transfer_snapshot(
            pending_store=pending,
            generation_dir=generation,
            snapshot_dir=tmp_path / "snapshot",
            exploratory_batch_root=batch,
            expected_exploratory_count=92,
        )
