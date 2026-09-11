import json

import numpy as np

from modules.plan_module import PlanModule
from modules.stage3_retrieval import (
    DenseTextIndex,
    deduplicate_source_groups,
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_union,
    retrieve_hierarchical,
    retrieve_three_facet_union,
    serializable_entries,
    soft_draft_rerank,
)
from scripts.stage3.retrieval_compare import _render_draft_signature_prompt, parse_draft_judgments


class FakeMemory:
    def __init__(self):
        self.principles = [
            {"memory_id": "a", "status": "active"},
            {"memory_id": "b", "status": "active"},
            {"memory_id": "c", "status": "active"},
        ]
        self._core_embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        self._condition_embeddings = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]], dtype=np.float32)
        self._target_embeddings = np.asarray([[0.4, 0.6], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32)

    def _ensure_index_alignment(self):
        return True

    def _encode_features(self, _features):
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        return query, query, query


class FakePlan:
    @staticmethod
    def _extract_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_gate_json(text):
        parsed = FakePlan._extract_json(text)
        if isinstance(parsed, list):
            return {"judgments": parsed}, "direct_list_normalized"
        if isinstance(parsed, dict):
            return parsed, "wrapped_object"
        return None, "parse_failed"


def test_hierarchical_retrieval_uses_core_and_condition_as_required_facets():
    rows = retrieve_hierarchical(
        FakeMemory(),
        {"core_skill": "x", "context_constraints": "y", "task_objective": "z"},
        top_k=3,
        pool_per_facet=3,
        core_threshold=0.55,
        condition_threshold=0.55,
    )
    assert [row["memory_id"] for row in rows] == ["c", "a"]
    assert "b" not in {row["memory_id"] for row in rows}


def test_rank_fusion_requires_presence_in_both_rankings():
    structure = [
        {"memory_id": "a", "item": {"memory_id": "a"}, "score": 0.9},
        {"memory_id": "b", "item": {"memory_id": "b"}, "score": 0.8},
    ]
    correction = [
        {"memory_id": "b", "score": 0.7},
        {"memory_id": "c", "score": 0.6},
    ]
    result = reciprocal_rank_fusion(structure, correction)
    assert [row["memory_id"] for row in result] == ["b"]
    assert result[0]["retrieval_method"] == "r5_dual_retrieval"


def test_rrf_union_keeps_candidates_present_in_only_one_ranking():
    result = reciprocal_rank_fusion_union(
        [[{"memory_id": "a", "score": 0.9}], [{"memory_id": "b", "score": 0.8}]],
        top_k=2,
    )
    assert {row["memory_id"] for row in result} == {"a", "b"}


def test_source_group_dedup_keeps_first_ranked_candidate():
    rows = [
        {"memory_id": "a", "item": {"question_id": "q1"}, "score": 0.9},
        {"memory_id": "b", "item": {"question_id": "q1"}, "score": 0.8},
        {"memory_id": "c", "item": {"question_id": "q2"}, "score": 0.7},
    ]
    kept, removed = deduplicate_source_groups(rows, top_k=2)
    assert [row["memory_id"] for row in kept] == ["a", "c"]
    assert removed == [{"memory_id": "b", "source_group": "q1", "original_rank": 2, "kept_memory_id": "a"}]


def test_three_facet_union_has_no_hard_condition_veto():
    rows = retrieve_three_facet_union(
        FakeMemory(),
        {"core_skill": "x", "context_constraints": "y", "task_objective": "z"},
        top_k=3,
        pool_per_facet=3,
    )
    assert {row["memory_id"] for row in rows} == {"a", "b", "c"}
    assert all("facet_ranks" in row for row in rows)


def test_soft_draft_rerank_is_union_not_intersection():
    rows = soft_draft_rerank(
        [
            {"memory_id": "structure", "score": 0.9},
            {"memory_id": "error", "score": 0.4},
            {"memory_id": "no_error_match", "score": 0.7},
        ],
        {"error": 0.95},
        structure_weight=0.7,
        error_weight=0.3,
        top_k=3,
    )
    assert {row["memory_id"] for row in rows} == {"structure", "error", "no_error_match"}
    assert all(row["retrieval_method"] == "draft_error_soft_rerank" for row in rows)


def test_serializable_trace_does_not_embed_full_memory_item():
    trace = serializable_entries([{"memory_id": "a", "item": {"secret": 1}, "score": 0.5}])
    assert trace == [{"memory_id": "a", "score": 0.5}]


def test_draft_gate_is_fail_closed_on_missing_candidate_id():
    raw = json.dumps(
        {
            "judgments": [
                {
                    "memory_id": "a",
                    "problem_structure_match": True,
                    "required_evidence_met": True,
                    "exclusion_triggered": False,
                    "failure_signal_observed": True,
                    "repair_action_relevant": True,
                    "confidence": 0.95,
                    "draft_evidence": "visible error",
                    "reason": "match",
                }
            ]
        }
    )
    result = parse_draft_judgments(raw, ["a", "b"], FakePlan())
    assert result["protocol_compliant"] is False
    assert result["selected_ids"] == []


def test_draft_gate_accepts_only_complete_high_confidence_match():
    judgment = {
        "memory_id": "a",
        "problem_structure_match": True,
        "required_evidence_met": True,
        "exclusion_triggered": False,
        "failure_signal_observed": True,
        "repair_action_relevant": True,
        "confidence": 0.8,
        "draft_evidence": "visible error",
        "reason": "match",
    }
    result = parse_draft_judgments(json.dumps({"judgments": [judgment]}), ["a"], FakePlan())
    assert result["protocol_compliant"] is True
    assert result["selected_ids"] == ["a"]


def test_draft_gate_accepts_direct_list_with_complete_ids():
    judgment = {
        "memory_id": "a",
        "problem_structure_match": True,
        "required_evidence_met": True,
        "exclusion_triggered": False,
        "failure_signal_observed": True,
        "repair_action_relevant": True,
        "confidence": 0.9,
        "draft_evidence": "visible error",
        "reason": "match",
    }
    result = parse_draft_judgments(json.dumps([judgment]), ["a"], FakePlan())
    assert result["protocol_compliant"] is True
    assert result["parse_stage"] == "direct_list_normalized"
    assert result["selected_ids"] == ["a"]


def test_draft_signature_prompt_preserves_json_braces():
    prompt = _render_draft_signature_prompt("Q {x}", "draft {y}")
    assert '"attempted_method"' in prompt
    assert "Q {x}" in prompt
    assert "draft {y}" in prompt
    assert "{question}" not in prompt
    assert "{draft}" not in prompt


def test_real_plan_parser_preserves_top_level_gate_list():
    plan = PlanModule.__new__(PlanModule)
    parsed, stage = plan._extract_gate_json('```json\n[{"memory_id":"a"}]\n```')
    assert parsed == {"judgments": [{"memory_id": "a"}]}
    assert stage == "direct_list_normalized"
