import json
import asyncio
import tempfile
from pathlib import Path
from types import MethodType

import numpy as np

from modules.dsd_module import DSDRegistry, canonicalize_stage1_packets
from modules.memory import Memory
from modules.plan_module import PlanModule
from modules.principle_schema import build_memory_views
from modules.stage2_validation import (
    Stage2Checkpoint,
    admission_decision,
    classify_pair,
    deterministic_leakage_check,
    render_solver_view,
    stable_dev_holdout_split,
    summarize_pairs,
)
from modules.stage2_v3 import _bounded_as_completed, _process_candidate, _quality_check


CONTENT = {
    "introspection": "The failed path applied an operation before establishing its precondition.",
    "trigger_condition": "A reversible algebraic transformation is proposed under a stated domain restriction.",
    "required_evidence": ["The current expression is explicit.", "Its domain restriction is explicit."],
    "exclusion_condition": "Reject when the transformation is not reversible on the stated domain.",
    "action_guidance": "Record the domain restriction before applying the transformation.",
    "verification_step": "Substitute each candidate into the original constrained expression.",
    "generalization_level": "family",
    "scope": "Constrained algebraic equations using reversible transformations.",
    "failure_mode": "Invalid or extraneous candidates survive an unrestricted transformation.",
}


def test_stage1_packets_are_canonicalized_to_one_candidate_per_question():
    packets = [
        {"idx": 7, "question": "q", "expected_answer": "a", "chosen_path": "", "rejected_path": "bad"},
        {"idx": 7, "question": "q", "expected_answer": "a", "chosen_path": "good", "rejected_path": "bad", "diff_success": True},
        {"idx": 8, "question": "q2", "expected_answer": "b", "chosen_path": "good2", "rejected_path": "bad2"},
    ]
    canonical = canonicalize_stage1_packets(packets)
    assert [item["question_id"] for item in canonical] == ["7", "8"]
    assert canonical[0]["chosen_path"] == "good"
    selected = canonicalize_stage1_packets(packets, ["8"])
    assert [item["question_id"] for item in selected] == ["8"]


def test_four_views_keep_solver_isolated():
    views = build_memory_views(
        CONTENT,
        source_dsd={"core_skill": "algebra", "context_constraints": "x > 0", "task_objective": "solve x"},
        validation={"retrieval_score": 0.99, "holdout_net_gain": 1},
        audit={"source_question": "secret source"},
    )
    assert set(views) == {"routing_view", "solver_view", "validation_view", "audit_view"}
    solver = views["solver_view"]
    assert set(solver) == {"lesson", "action_guidance", "verification_step"}
    rendered = render_solver_view(solver)
    assert "retrieval_score" not in rendered
    assert "secret source" not in rendered
    assert "x > 0" not in rendered


def test_pair_outcomes_and_summary():
    assert classify_pair(False, True) == "gain"
    assert classify_pair(True, False) == "loss"
    assert classify_pair(True, True) == "both_correct"
    assert classify_pair(False, False) == "both_wrong"
    summary = summarize_pairs([
        {"outcome": "gain"}, {"outcome": "loss"}, {"outcome": "both_correct"}, {"outcome": "both_wrong"}
    ])
    assert summary["net_gain"] == 0
    assert summary["historical_gain_rate"] == 0.25
    assert summary["harm_rate"] == 0.25
    with_failure = summarize_pairs([{"outcome": "gain"}, {"outcome": "infrastructure_failed"}])
    assert with_failure["pair_count"] == 1
    assert with_failure["attempted_pair_count"] == 2
    assert with_failure["infrastructure_failed_count"] == 1


def test_strict_admission_contract():
    active = admission_decision(
        schema_valid=True,
        source_fidelity_pass=True,
        answer_leakage_detected=False,
        accepted_neighbors=4,
        holdout_summary={"gain_count": 1, "loss_count": 0, "net_gain": 1},
    )
    assert active["status"] == "active"
    no_gain = admission_decision(
        schema_valid=True,
        source_fidelity_pass=True,
        answer_leakage_detected=False,
        accepted_neighbors=6,
        holdout_summary={"gain_count": 0, "loss_count": 0, "net_gain": 0},
    )
    assert no_gain["status"] == "provisional"
    harmful = admission_decision(
        schema_valid=True,
        source_fidelity_pass=True,
        answer_leakage_detected=False,
        accepted_neighbors=6,
        holdout_summary={"gain_count": 2, "loss_count": 1, "net_gain": 1},
    )
    assert harmful["status"] == "rejected"
    incomplete = admission_decision(
        schema_valid=True, source_fidelity_pass=True, answer_leakage_detected=False,
        accepted_neighbors=4,
        holdout_summary={"gain_count": 1, "loss_count": 0, "net_gain": 1, "infrastructure_failed_count": 1},
    )
    assert incomplete["status"] == "provisional"


def test_stable_split_sizes_and_disjointness():
    neighbors = [{"question_id": f"q{i}"} for i in range(6)]
    first = stable_dev_holdout_split(neighbors, candidate_id="c1", seed=42)
    second = stable_dev_holdout_split(neighbors, candidate_id="c1", seed=42)
    assert first == second
    dev, holdout = first
    assert len(dev) == len(holdout) == 3
    assert {x["question_id"] for x in dev}.isdisjoint({x["question_id"] for x in holdout})


def test_deterministic_source_constant_leakage():
    leaked = dict(CONTENT)
    leaked["action_guidance"] = "Always multiply the intermediate value by 37."
    result = deterministic_leakage_check(leaked, source_question="There are 37 objects.", expected_answer="74")
    assert result["detected"] is True
    assert "37" in result["copied_source_constants"]


def test_reusable_method_is_not_hard_rejected_as_answer_leakage():
    class FakePlan:
        async def audit_stage2_principle(self, **kwargs):
            return {
                "protocol_compliant": True,
                "quality_pass": False,
                "source_fidelity_pass": True,
                "answer_leakage_detected": True,
                "trigger_decidable": True,
                "required_evidence_decidable": True,
                "exclusion_decidable": True,
                "reasons": ["The action resembles the correct trajectory."],
            }

    class FakePipeline:
        plan = FakePlan()

    principle = dict(CONTENT)
    principle["action_guidance"] = "Find a common denominator before comparing the terms."
    result = asyncio.run(_quality_check(
        FakePipeline(),
        {
            "question_text": "Which of the following fractions is largest?",
            "expected_answer": "B",
            "chosen_trajectory": "Find a common denominator and compare.",
            "hard_negative_trajectory": "Compare denominators directly.",
        },
        [{"CausalError": "Compared denominators without normalizing."}],
        principle,
    ))
    assert result["llm_leakage_suspected"] is True
    assert result["answer_leakage_detected"] is False
    assert result["quality_pass"] is True


def test_checkpoint_resume_and_parent(tmp_path):
    manager = Stage2Checkpoint(tmp_path, "run1")
    manager.save({"run_id": "run1", "current_step": "candidate_processing", "completed_candidate_ids": ["q1"]})
    second = manager.save({"run_id": "run1", "current_step": "complete", "completed_candidate_ids": ["q1", "q2"]})
    payload = json.loads(second.read_text(encoding="utf-8"))
    assert payload["previous_checkpoint_id"] == "run1:candidate_processing"
    assert manager.load()["current_step"] == "complete"


def test_memory_replacement_keeps_version_history(tmp_path):
    memory = Memory(str(tmp_path))
    memory.add({"memory_id": "mem1", "question": "q", "principle": "old", "status": "active"})
    memory.replace_version("mem1", {"question": "q", "principle": "new", "status": "active"}, reason="better evidence")
    item = memory.principles[0]
    assert item["version_id"] == "mem1_v2"
    assert item["parent_version_id"] == "mem1_v1"
    assert item["version_history"][0]["principle"] == "old"
    assert item["version_history"][0]["status"] == "deprecated"


def test_memory_additions_append_indices_without_full_rebuild(tmp_path):
    class FakeEmbeddingModel:
        def __init__(self):
            self.encoded_item_counts = []

        def encode(self, texts, show_progress_bar=False):
            self.encoded_item_counts.append(len(texts))
            rows = []
            for index, text in enumerate(texts):
                base = float(len(str(text)) + index + 1)
                rows.append([base, base + 1.0, base + 2.0, base + 3.0])
            return np.asarray(rows, dtype=np.float32)

    memory = Memory(str(tmp_path))
    model = FakeEmbeddingModel()
    memory._embedding_model = model

    first = {
        "memory_id": "mem1", "status": "active", "principle": "first lesson",
        "core_skill": "algebra", "context_constraints": "positive domain",
        "task_objective": "solve equation",
    }
    memory.add(first)
    memory.index_added_principle(first)
    assert memory._core_embeddings.shape[0] == 1
    assert memory._faiss_core.ntotal == 1
    calls_after_first = len(model.encoded_item_counts)

    second = {
        "memory_id": "mem2", "status": "active", "principle": "second lesson",
        "core_skill": "geometry", "context_constraints": "nondegenerate triangle",
        "task_objective": "find an angle",
    }
    memory.add(second)
    memory.index_added_principle(second)

    assert memory._core_embeddings.shape[0] == 2
    assert memory._single_embeddings.shape[0] == 2
    assert memory._faiss_core.ntotal == 2
    assert memory._faiss_single.ntotal == 2
    # Appending encodes only the new item's three facets plus one solver text.
    assert model.encoded_item_counts[calls_after_first:] == [3, 1]


def test_dsd_validation_neighbors_never_relax_thresholds():
    registry = DSDRegistry.__new__(DSDRegistry)
    registry.records = [
        {"question_id": "source", "question_text": "source", "source_dsd": {}},
        {"question_id": "weak", "question_text": "weak", "source_dsd": {}},
    ]
    registry._by_question_id = {item["question_id"]: item for item in registry.records}
    registry._neighbor_cache = {}
    registry._core_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    registry._context_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    registry._target_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert registry.get_neighbors("source", top_k=12) == []


def test_dsd_cache_round_trip_restores_all_three_routing_facets():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        registry = DSDRegistry("dpo_level1", 1, root, cache_dir=root / "cache")
        registry.records = [{"question_id": "q1", "question_text": "q", "source_dsd": {}, "stage2_candidate": True}]
        registry._by_question_id = {"q1": registry.records[0]}
        registry._neighbor_cache = {"q1": []}
        registry._core_embeddings = np.ones((1, 3), dtype=np.float32)
        registry._context_embeddings = np.ones((1, 3), dtype=np.float32) * 2
        registry._target_embeddings = np.ones((1, 3), dtype=np.float32) * 3
        registry._save_cache("fixed-key")

        restored = DSDRegistry("dpo_level1", 1, root, cache_dir=root / "cache")
        assert restored.encoder is None
        assert restored._load_cache("fixed-key") is True
        assert restored.encoder is None
        assert restored._core_embeddings.shape == (1, 3)
        assert float(restored._context_embeddings[0, 0]) == 2.0
        assert float(restored._target_embeddings[0, 0]) == 3.0


def test_bounded_candidate_scheduler_limits_inflight_work():
    async def scenario():
        running = 0
        peak = 0

        async def process(item):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.002)
            running -= 1
            return item

        outputs = []
        items = [{"question_id": str(i)} for i in range(25)]
        async for _, task in _bounded_as_completed(items, 4, process):
            outputs.append(await task)
        return peak, outputs

    peak, outputs = asyncio.run(scenario())
    assert peak <= 4
    assert len(outputs) == 25


def test_self_cpe_accepts_direct_exact_schema_without_repair():
    plan = PlanModule.__new__(PlanModule)
    calls = []

    async def fake_call(self, prompt, **kwargs):
        calls.append(kwargs)
        return json.dumps(CONTENT)

    plan._call_llm = MethodType(fake_call, plan)
    result = asyncio.run(plan.self_cpe_extract(
        question="q", diff_list=[{"CausalError": "e"}],
        source_dsd={"core_skill": "algebra", "context_constraints": "domain", "task_objective": "solve"},
    ))
    assert result["status"] == "success"
    assert result["parse_stage"] == "direct_object_normalized"
    assert result["generation_trace"]["repair_used"] is False
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0


def test_self_cpe_uses_one_exact_schema_repair_for_plain_text():
    plan = PlanModule.__new__(PlanModule)
    responses = iter(["Step 1: unsupported prose", json.dumps({"principle": CONTENT})])
    calls = []

    async def fake_call(self, prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    plan._call_llm = MethodType(fake_call, plan)
    result = asyncio.run(plan.self_cpe_extract(
        question="q", diff_list=[{"CausalError": "e"}],
        source_dsd={"core_skill": "algebra", "context_constraints": "domain", "task_objective": "solve"},
    ))
    assert result["status"] == "success"
    assert result["parse_stage"] == "wrapped_object"
    assert result["generation_trace"]["repair_used"] is True
    assert len(calls) == 2


def test_stage2_neighbor_gate_fails_closed_on_missing_id():
    plan = PlanModule.__new__(PlanModule)

    async def fake_call(self, prompt, **kwargs):
        return json.dumps({"judgments": []})

    plan._call_llm = MethodType(fake_call, plan)
    result = asyncio.run(plan.judge_stage2_neighbors(
        source_question="source",
        source_dsd={"core_skill": "algebra"},
        routing_view={"trigger_condition": "equation"},
        neighbors=[{"question_id": "n1", "question_text": "neighbor", "source_dsd": {}}],
    ))
    assert not result["protocol_compliant"]
    assert result["selected_ids"] == []
    assert result["judgments"][0]["accepted"] is False


def test_stage2_neighbor_gate_accepts_fenced_object_without_format_repair():
    plan = PlanModule.__new__(PlanModule)

    async def fake_call(self, prompt, **kwargs):
        return "```json\n" + json.dumps({"judgments": [{
            "neighbor_id": "n1",
            "trigger_match": True,
            "required_evidence_match": True,
            "structural_match": True,
            "scope_match": True,
            "contraindicated": False,
            "confidence": 0.9,
            "reason": "All required mathematical structure is present.",
        }]}) + "\n```"

    plan._call_llm = MethodType(fake_call, plan)
    result = asyncio.run(plan.judge_stage2_neighbors(
        source_question="source",
        source_dsd={"core_skill": "algebra"},
        routing_view={"trigger_condition": "equation"},
        neighbors=[{"question_id": "n1", "question_text": "neighbor", "source_dsd": {}}],
    ))
    assert result["protocol_compliant"]
    assert result["parse_stage"] == "wrapped_object"
    assert result["repair_used"] is False
    assert result["selected_ids"] == ["n1"]


def test_stage2_neighbor_gate_accepts_direct_list_and_records_parse_stage():
    plan = PlanModule.__new__(PlanModule)

    async def fake_call(self, prompt, **kwargs):
        return json.dumps([{
            "neighbor_id": "n1",
            "trigger_match": True,
            "required_evidence_match": True,
            "structural_match": True,
            "scope_match": True,
            "contraindicated": False,
            "confidence": 0.9,
            "reason": "All required mathematical structure is present.",
        }])

    plan._call_llm = MethodType(fake_call, plan)
    result = asyncio.run(plan.judge_stage2_neighbors(
        source_question="source",
        source_dsd={"core_skill": "algebra"},
        routing_view={"trigger_condition": "equation"},
        neighbors=[{"question_id": "n1", "question_text": "neighbor", "source_dsd": {}}],
    ))
    assert result["protocol_compliant"]
    assert result["parse_stage"] == "direct_list_normalized"
    assert result["repair_used"] is False


def test_candidate_flow_requires_paired_holdout_gain():
    class FakePlan:
        async def self_cpe_extract(self, **kwargs):
            return {"status": "success", "principle": dict(CONTENT)}

        async def audit_stage2_principle(self, **kwargs):
            return {
                "status": "success",
                "protocol_compliant": True,
                "quality_pass": True,
                "source_fidelity_pass": True,
                "answer_leakage_detected": False,
                "trigger_decidable": True,
                "required_evidence_decidable": True,
                "exclusion_decidable": True,
            }

        async def judge_stage2_neighbors(self, neighbors, **kwargs):
            return {
                "status": "success",
                "protocol_compliant": True,
                "protocol_errors": [],
                "selected_ids": [item["question_id"] for item in neighbors],
                "judgments": [
                    {"neighbor_id": item["question_id"], "accepted": True, "confidence": 0.99}
                    for item in neighbors
                ],
            }

    class FakeExecute:
        model = "fake"
        urls = ["http://fake"]

        async def solve_v1(self, question, principle_text="", **kwargs):
            answer = "correct" if principle_text else "wrong"
            return {
                "protocol_version": "solver_v1",
                "solver_url": "http://fake",
                "infrastructure_success": True,
                "composed_output": answer,
                "reasoner": {},
                "finalizer": {},
            }

        @staticmethod
        def check_answer(generated, expected):
            return generated == expected

    class FakeRegistry:
        def get_neighbors(self, question_id, top_k):
            return [
                {
                    "question_id": f"n{i}",
                    "question_text": f"neighbor {i}",
                    "expected_answer": "correct",
                    "source_dsd": {},
                    "dimension_similarities": {},
                }
                for i in range(6)
            ]

    pipeline = type("Pipeline", (), {"plan": FakePlan(), "execute": FakeExecute()})()
    record = {
        "question_id": "source",
        "question_text": "A symbolic constrained equation.",
        "expected_answer": "correct",
        "chosen_trajectory": "valid reasoning",
        "hard_negative_trajectory": "invalid unrestricted operation",
        "source_dsd": {
            "core_skill": "algebra",
            "context_constraints": "stated domain restriction",
            "task_objective": "solve the equation",
        },
        "contrast": {"differences": [{"CausalError": "ignored the domain"}]},
    }
    unit = asyncio.run(_process_candidate(
        pipeline,
        FakeRegistry(),
        record,
        dataset="dpo_level2",
        round_id=2,
        seed=42,
        generation_semaphore=asyncio.Semaphore(2),
        solver_semaphore=asyncio.Semaphore(2),
    ))
    assert unit["status"] == "active"
    assert unit["holdout_summary"]["gain_count"] == 3
    assert unit["holdout_summary"]["loss_count"] == 0
    assert set(unit["validation_view"]["dev_neighbor_ids"]).isdisjoint(
        unit["validation_view"]["holdout_neighbor_ids"]
    )


def test_nv_off_uses_frozen_quality_candidate_without_neighbor_calls():
    class NoCallsPlan:
        def __getattr__(self, name):
            raise AssertionError(f"frozen NV-off path called Plan.{name}")

    class NoNeighbors:
        def get_neighbors(self, *args, **kwargs):
            raise AssertionError("NV-off must not retrieve validation neighbors")

    record = {
        "question_id": "q1", "question_text": "Solve a constrained equation.",
        "source_dsd": {"core_skill": "algebra", "context_constraints": "domain", "task_objective": "solve"},
    }
    prepared = {
        "candidate_id": "candidate_test", "question_id": "q1",
        "source_question_text": record["question_text"], "source_dsd": record["source_dsd"],
        "source_round": 1, "source_dataset": "dpo_level1", "schema_version": "3.0",
        "status": "candidate_ready", "content": dict(CONTENT),
        "diff": [{"CausalError": "ignored domain"}],
        "quality_check": {"quality_pass": True, "source_fidelity_pass": True, "answer_leakage_detected": False},
        "schema_check": {"valid": True, "errors": []},
    }
    pipeline = type("Pipeline", (), {"plan": NoCallsPlan(), "execute": object()})()
    unit = asyncio.run(_process_candidate(
        pipeline, NoNeighbors(), record, dataset="dpo_level1", round_id=1, seed=42,
        generation_semaphore=asyncio.Semaphore(1), solver_semaphore=asyncio.Semaphore(1),
        prepared_candidate=prepared, neighbor_validation=False,
    ))
    assert unit["status"] == "active"
    assert unit["qualified_for_memory"] is True
    assert unit["content"] == CONTENT
    assert unit["neighbor_validation"] is False
