import asyncio
import json
import math
import tempfile
from pathlib import Path

from core.api_call import _logprob_entropy_summary
from modules.stage1_miner import Stage1Miner, save_rankings_and_manifests, write_analysis_csv
from modules.stage1_sampler import Stage1Sampler, normalized_answer_entropy
from modules.stage2.contracts import build_learning_bundles
from modules.stage2.principle_generation import build_generation_prompt


def _api_payload(logps):
    return {"choices": [{"logprobs": {"content": [
        {"token": "x", "logprob": logps[0], "top_logprobs": [
            {"token": str(i), "logprob": value} for i, value in enumerate(logps)
        ]}
    ]}}]}


def test_entropy_known_distributions_and_missing() -> None:
    peaked = _logprob_entropy_summary(_api_payload([0.0, -20.0, -20.0, -20.0, -20.0]), 20)
    uniform = _logprob_entropy_summary(_api_payload([-1.0] * 5), 20)
    missing = _logprob_entropy_summary({}, 20)
    assert peaked["normalized_entropy"] < 0.01
    assert math.isclose(uniform["normalized_entropy"], 1.0, abs_tol=1e-9)
    assert sum(uniform["histogram"]["counts"]) == uniform["entropy_token_count"]
    assert missing["available"] is False
    assert missing["normalized_entropy"] is None


def test_answer_entropy() -> None:
    same = [r"Final Answer: \boxed{1}"] * 10
    split = [r"Final Answer: \boxed{1}"] * 5 + [r"Final Answer: \boxed{2}"] * 5
    assert normalized_answer_entropy(same, 10) == 0.0
    assert normalized_answer_entropy(split, 10) > 0.0


def test_three_zones_are_primary_and_entropy_cannot_change_them() -> None:
    sampler = Stage1Sampler(["mock"], "mock")
    for correct_count, zone, level in (
        (10, "master", 0), (0, "panic", 0), (9, "learning", 1),
        (7, "learning", 2), (4, "learning", 3), (2, "learning", 4),
    ):
        records = []
        for index in range(10):
            records.append({
                "response": r"Final Answer: \boxed{1}" if index < correct_count else r"Final Answer: \boxed{2}",
                "is_correct": index < correct_count,
                "transport_status": "success",
                "entropy": {"available": index % 2 == 0, "normalized_entropy": 0.99 if index % 2 == 0 else None},
            })
        result = sampler.finalize_result({"planned_trials": 10, "trial_records": records})
        assert result["zone"] == zone
        assert result["assigned_level"] == level


def test_invalid_input_is_not_misclassified_as_panic() -> None:
    async def run():
        sampler = MockSampler()
        result = await sampler.sample_one_task(
            {"id": "bad", "question": "mock correct 0", "answer": "", "difficulty_level": "Level 1"},
            0,
        )
        assert result["zone"] == "invalid_input"
        assert result["zone"] not in {"master", "learning", "panic"}
        assert result["trial_records"] == []
    asyncio.run(run())


def test_strict_final_answer_matcher() -> None:
    from benchmarks.matcher import answers_equivalent, check_answer, check_answer_detailed
    assert not check_answer("Reasoning: intermediate value is 7, but unfinished", "7")
    assert check_answer(r"Reasoning: done.\nFinal Answer: \boxed{7}", "7")
    assert answers_equivalent("7.5", r"\frac{15}{2}")
    assert not answers_equivalent("x" * 1000, "1")
    assert not answers_equivalent("(x+1)^1000000000", "1")
    skipped = check_answer_detailed(
        "Final Answer: \\boxed{" + ("x" * 300) + "}", "1"
    )
    assert skipped["status"] == "symbolic_skipped"
    assert skipped["correct"] is False


class MockSampler(Stage1Sampler):
    def __init__(self):
        super().__init__(["mock://0", "mock://1", "mock://2", "mock://3"], "mock", k_trials=10)
        self.current_task = -1

    async def sample_one_task(self, task, task_idx, existing_records=None, checkpoint_callback=None):
        self.current_task = task_idx
        return await super().sample_one_task(task, task_idx, existing_records, checkpoint_callback)

    async def _single_inference_with_metadata(self, question, seed):
        correct_limit = int(question.split()[-1])
        answer = 1 if seed - 42 < correct_limit else 2
        entropy = 0.1 + 0.01 * (seed - 42)
        return {
            "text": f"Reasoning: mock.\nCheck: mock.\nFinal Answer: \\boxed{{{answer}}}",
            "endpoint": f"mock://{seed % 4}", "transport_status": "success", "duration_sec": 0.01,
            "entropy": {
                "available": True, "normalized_entropy": entropy, "trajectory_entropy": entropy,
                "std": 0.0, "p25": entropy, "p50": entropy, "p75": entropy, "p90": entropy,
                "histogram": {"bins": 20, "range": [0.0, 1.0], "counts": [0] * 20},
                "token_count": 1, "entropy_token_count": 1, "coverage": 1.0,
                "avg_neg_logprob": 0.1, "chosen_token_mean_surprisal": 0.1,
                "avg_top_k_probability_mass": 0.9, "method": "top_logprobs_normalized_entropy",
            },
        }


def test_mock_ten_question_end_to_end_and_stage2_contract() -> None:
    async def run():
        sampler = MockSampler()
        correct_limits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        tasks = [
            {"id": f"q{i}", "question": f"mock correct {limit}", "answer": "1", "difficulty_level": "Level 2"}
            for i, limit in enumerate(correct_limits)
        ]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "checkpoint.jsonl")
            sampler.task_batch_size = 3
            sampled = await sampler.sample_all(tasks, concurrency=4, checkpoint_path=checkpoint)
            assert len(sampled["results"]) == 10
            for item in sampled["results"]:
                assert [record["seed"] for record in item["trial_records"]] == list(range(42, 52))
            assert sampled["summary"]["master_zone"] == 1
            assert sampled["summary"]["panic_zone"] == 1
            assert sampled["summary"]["learning_zone"] == 8
            assert sampled["config"]["task_batch_size"] == 3

            miner = Stage1Miner(experiment_seed=42, dataset_sha256="testhash")
            mined = await miner.run(sampled["level_data"])
            bundles = [item for values in mined["packets"].values() for item in values]
            assert len(bundles) == 8
            assert len({item["question_id"] for item in bundles}) == 8
            assert all(len(item["rollouts"]) == 10 for item in bundles)

            output = Path(directory) / "stage1"
            packet_dir = output / "02_learning_packets" / "mock"
            Stage1Miner.save(mined["packets"], packet_dir)
            save_rankings_and_manifests(mined["rankings"], output)
            write_analysis_csv(mined["packets"], output / "analysis" / "questions.csv")
            assert "stage3_outcome" in (output / "analysis" / "questions.csv").read_text()
            for budget in (100, 200, 400, 600):
                manifests = [json.loads((output / "selection_manifests" / f"budget_{budget}_{policy}.json").read_text())
                             for policy in ("random", "high_token_entropy", "low_token_entropy")]
                assert manifests[0]["level_quotas"] == manifests[1]["level_quotas"] == manifests[2]["level_quotas"]

            normalized = build_learning_bundles([bundles[0]], bundles[0]["level"])[0]
            assert normalized["capability_signal"] == bundles[0]["capability_signal"]
            assert normalized["rollouts"][0]["entropy"] == bundles[0]["rollouts"][0]["entropy"]
            prompt = build_generation_prompt(normalized, 3)
            for forbidden in ("h_token", "h_answer", "h_gap", "within_level_rankings", "selection"):
                assert forbidden not in prompt

            try:
                from scripts.stage1.analyze import generate_figures
                figure_paths = generate_figures(output)
                assert len(figure_paths) == 4
                assert all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
            except RuntimeError as exc:
                if "matplotlib is required" not in str(exc):
                    raise
    asyncio.run(run())


def test_checkpoint_resumes_by_question_and_seed() -> None:
    async def run():
        sampler = MockSampler()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            existing = []
            for seed in (42, 43):
                existing.append({
                    "rollout_id": f"existing-{seed}", "trial": seed - 41, "seed": seed,
                    "response": "Reasoning: mock.\nCheck: mock.\nFinal Answer: \\boxed{1}",
                    "final_answer": "1", "is_correct": True, "model": "mock",
                    "prompt_version": "stage1_direct_v1", "prompt_hash": "hash",
                    "endpoint": "mock://0", "transport_status": "success",
                    "entropy": {"available": True, "normalized_entropy": 0.2},
                })
            for record in existing:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"task_index": 0, "record": record}) + "\n")
            loaded = sampler._load_checkpoint(str(path))
            assert [record["seed"] for record in loaded["0"]] == [42, 43]
            result = await sampler.sample_one_task(
                {"id": "q0", "question": "mock correct 10", "answer": "1", "difficulty_level": "Level 1"},
                0, loaded["0"], None,
            )
            assert [record["seed"] for record in result["trial_records"]] == list(range(42, 52))
            assert len({record["seed"] for record in result["trial_records"]}) == 10
            assert result["zone"] == "master"
    asyncio.run(run())
