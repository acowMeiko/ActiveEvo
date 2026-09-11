from __future__ import annotations

import numpy as np

from modules.stage3_postdraft import (
    extract_json_object,
    validate_generic_locator,
    validate_memory_judgment,
)
from modules.stage3_retrieval import retrieve_three_facet_joint
from template.prompt_template import POSTDRAFT_MEMORY_REVISION_PROMPT


class _FakeMemory:
    def __init__(self) -> None:
        self.principles = [
            {"memory_id": "balanced", "question_id": "q1", "status": "active"},
            {"memory_id": "weak", "question_id": "q2", "status": "active"},
            {"memory_id": "duplicate", "question_id": "q1", "status": "active"},
        ]
        self._core_embeddings = np.asarray([[0.8], [1.0], [0.7]], dtype=np.float32)
        self._condition_embeddings = np.asarray([[0.8], [1.0], [0.7]], dtype=np.float32)
        self._target_embeddings = np.asarray([[0.8], [0.1], [0.7]], dtype=np.float32)

    def _ensure_index_alignment(self):
        return True

    def _encode_features(self, features):
        unit = np.asarray([1.0], dtype=np.float32)
        return unit, unit, unit


def test_joint_score_penalizes_one_weak_facet_and_deduplicates_source():
    ranked, removed = retrieve_three_facet_joint(_FakeMemory(), {}, top_k=24)
    assert [row["memory_id"] for row in ranked] == ["balanced", "weak"]
    assert ranked[0]["joint_score"] > ranked[1]["joint_score"]
    assert removed[0]["memory_id"] == "duplicate"


def test_json_parser_supports_fence_and_embedded_object():
    assert extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    assert extract_json_object('answer follows: {"ok": true} trailing') == {"ok": True}


def test_memory_judgment_requires_input_id_exact_signal_and_real_quote():
    draft = "First compute x = 3. Then divide by zero, which gives 8."
    candidate = {
        "memory_id": "candidate_a",
        "content": {"failure_signals": ["The draft divides by zero."]},
    }
    valid = {
        "selected_memory_id": "candidate_a",
        "problem_structure_match": True,
        "all_required_evidence_present": True,
        "exclusion_applies": False,
        "matched_failure_signal": "The draft divides by zero.",
        "draft_error_quote": "Then divide by zero, which gives 8.",
        "action_applicable": True,
    }
    assert validate_memory_judgment(valid, [candidate], draft)["accepted"] is True
    invalid_signal = dict(valid, matched_failure_signal="It divides incorrectly.")
    assert validate_memory_judgment(invalid_signal, [candidate], draft)["accepted"] is False
    invalid_quote = dict(valid, draft_error_quote="A quote not present in the draft")
    assert validate_memory_judgment(invalid_quote, [candidate], draft)["accepted"] is False
    invalid_id = dict(valid, selected_memory_id="candidate_b")
    assert validate_memory_judgment(invalid_id, [candidate], draft)["accepted"] is False


def test_rejection_is_fail_closed():
    result = validate_memory_judgment(
        {"selected_memory_id": None},
        [{"memory_id": "candidate_a", "content": {"failure_signals": ["x"]}}],
        "draft",
    )
    assert result == {"accepted": False, "reason": "model_rejected", "selected_memory_id": None}


def test_generic_locator_requires_exact_quote():
    assert validate_generic_locator(
        {"issue_found": True, "draft_error_quote": "x = 4", "issue_description": "arithmetic"},
        "Therefore x = 4 and the result follows.",
    )["valid"] is True
    assert validate_generic_locator(
        {"issue_found": True, "draft_error_quote": "x = 5", "issue_description": "arithmetic"},
        "Therefore x = 4 and the result follows.",
    )["valid"] is False


def test_revision_prompt_contains_only_local_correction_fields():
    prompt = POSTDRAFT_MEMORY_REVISION_PROMPT.format(
        question="current question", draft="current draft", draft_error_quote="bad step",
        matched_failure_signal="bad pattern", action_guidance="fix locally",
        verification_step="substitute back",
    )
    assert "current question" in prompt
    assert "fix locally" in prompt
    for forbidden in ("memory_id", "retrieval score", "source answer", "required_evidence", "trigger_condition"):
        assert forbidden not in prompt.lower()
