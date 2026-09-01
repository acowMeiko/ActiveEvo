import json
from pathlib import Path

import pytest

from modules.stage3_snapshot import prepare_transfer_snapshot


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
