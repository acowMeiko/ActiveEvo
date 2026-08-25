"""Stage 1 capability-frontier sampling.

Accuracy is the only signal used to assign levels. Token uncertainty is kept
as an orthogonal, auditable signal for ranking questions *within* a level.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from tqdm.asyncio import tqdm as async_tqdm

from benchmarks.matcher import check_answer_detailed, extract_final_answer
from core.api_call import async_llm_call_with_metadata
from modules.execute_module import URLRouter
from template.prompt_template import (
    SOLVER_V1_FINALIZER_PROMPT,
    SOLVER_V1_REASONER_PROMPT,
)

logger = logging.getLogger(__name__)


STAGE1_SOLVER_PROTOCOL = "solver_v1.1"


def _template_hash(template: str) -> str:
    return hashlib.sha256(template.strip().encode("utf-8")).hexdigest()


def prompt_hashes() -> Dict[str, str]:
    return {
        "reasoner": _template_hash(SOLVER_V1_REASONER_PROMPT),
        "finalizer": _template_hash(SOLVER_V1_FINALIZER_PROMPT),
    }


def prompt_hash() -> str:
    payload = json.dumps(prompt_hashes(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_final_answer(text: str) -> str:
    value = (extract_final_answer(text) or "").strip().lower()
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\displaystyle", "").replace(r"\,", "")
    value = value.replace("−", "-").replace(" ", "").strip("$.,;:")
    return value


def normalized_answer_entropy(answers: Sequence[str], planned_trials: int) -> Optional[float]:
    """Entropy of final-answer frequencies, normalized by log(planned_trials)."""
    values = [normalize_final_answer(answer) or "__missing_final_answer__" for answer in answers]
    if not values or planned_trials <= 1:
        return None
    counts = Counter(values)
    total = len(values)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return max(0.0, min(1.0, entropy / math.log(planned_trials)))


def mean_available(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(valid) / len(valid) if valid else None


class Stage1Sampler:
    """Run exactly K reproducible rollouts and preserve complete evidence."""

    def __init__(
        self,
        urls: List[str],
        model: str,
        api_key: str = "EMPTY",
        k_trials: int = 10,
        panic_max_correct: int = 1,
        master_min_correct: int = 9,
        experiment_seed: int = 42,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        finalizer_max_tokens: int = 128,
        finalizer_timeout: int = 60,
        timeout: int = 600,
        request_logprobs: bool = True,
        top_logprobs: int = 5,
        entropy_histogram_bins: int = 20,
        task_batch_size: int = 1000,
        disable_early_stop: bool = True,
        **_legacy: Any,
    ):
        self.urls = list(urls)
        self.model = model
        self.api_key = api_key
        self.k_trials = int(k_trials)
        self.panic_max_correct = int(panic_max_correct)
        self.master_min_correct = int(master_min_correct)
        if not 0 <= self.panic_max_correct < self.master_min_correct <= self.k_trials:
            raise ValueError(
                "zone thresholds must satisfy "
                "0 <= panic_max_correct < master_min_correct <= k_trials"
            )
        self.experiment_seed = int(experiment_seed)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.finalizer_max_tokens = int(finalizer_max_tokens)
        self.finalizer_timeout = int(finalizer_timeout)
        self.timeout = int(timeout)
        self.request_logprobs = bool(request_logprobs)
        self.top_logprobs = int(top_logprobs)
        self.entropy_histogram_bins = int(entropy_histogram_bins)
        self.task_batch_size = max(1, int(task_batch_size))
        self.disable_early_stop = bool(disable_early_stop)
        self.router = URLRouter(self.urls)

    @property
    def rollout_seeds(self) -> List[int]:
        return [self.experiment_seed + index for index in range(self.k_trials)]

    @staticmethod
    def level_from_correct_count(
        correct_count: int,
        planned_trials: int = 10,
        panic_max_correct: int = 1,
        master_min_correct: int = 9,
    ) -> int:
        if (
            planned_trials <= 0
            or correct_count <= panic_max_correct
            or correct_count >= master_min_correct
        ):
            return 0
        rate = correct_count / planned_trials
        if rate >= 0.75:
            return 1
        if rate >= 0.50:
            return 2
        if rate >= 0.25:
            return 3
        return 4

    @classmethod
    def level_from_result(cls, result: Dict[str, Any]) -> int:
        return cls.level_from_correct_count(
            int(result.get("correct_count") or 0),
            int(result.get("planned_trials") or 10),
            int(result.get("panic_max_correct") or 1),
            int(result.get("master_min_correct") or 9),
        )

    @staticmethod
    def zone_from_correct_count(
        correct_count: int,
        planned_trials: int,
        panic_max_correct: int = 1,
        master_min_correct: int = 9,
    ) -> str:
        if correct_count <= panic_max_correct:
            return "panic"
        if correct_count >= master_min_correct:
            return "master"
        return "learning"

    async def _single_inference_with_metadata(self, question: str, seed: int) -> Dict[str, Any]:
        reasoner_prompt = SOLVER_V1_REASONER_PROMPT.format(question=question)
        attempted: List[str] = []
        for _ in range(max(1, len(self.urls))):
            url = self.router.next()
            if url in attempted and len(attempted) < len(self.urls):
                continue
            attempted.append(url)
            try:
                reasoner = await async_llm_call_with_metadata(
                    prompt=reasoner_prompt,
                    model=self.model,
                    url=url,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    retries=1,
                    timeout=self.timeout,
                    request_logprobs=self.request_logprobs,
                    top_logprobs=self.top_logprobs,
                    entropy_histogram_bins=self.entropy_histogram_bins,
                    include_raw=False,
                    seed=seed,
                )
                reasoner_text = str(reasoner.get("text") or "").strip()
                if not reasoner_text:
                    self.router.report_failure(url, "empty reasoner response")
                    continue
                finalizer_prompt = SOLVER_V1_FINALIZER_PROMPT.format(
                    question=question,
                    reasoner_output=reasoner_text,
                )
                finalizer = await async_llm_call_with_metadata(
                    prompt=finalizer_prompt,
                    model=self.model,
                    url=url,
                    api_key=self.api_key,
                    temperature=0.0,
                    top_p=self.top_p,
                    max_tokens=self.finalizer_max_tokens,
                    retries=1,
                    timeout=self.finalizer_timeout,
                    request_logprobs=False,
                    include_raw=False,
                    seed=seed,
                )
                finalizer_text = str(finalizer.get("text") or "").strip()
                if not finalizer_text:
                    self.router.report_failure(url, "empty finalizer response")
                    continue
                self.router.report_success(url)
                return {
                    "text": f"{reasoner_text}\n\n{finalizer_text}",
                    "endpoint": url,
                    "transport_status": "success",
                    # Stage 1 uncertainty describes mathematical reasoning,
                    # not the deterministic formatting call.
                    "entropy": reasoner.get("entropy") or {
                        "available": False,
                        "normalized_entropy": None,
                        "method": "unavailable",
                    },
                    "duration_sec": (
                        float(reasoner.get("duration_sec") or 0.0)
                        + float(finalizer.get("duration_sec") or 0.0)
                    ),
                    "solver_protocol": STAGE1_SOLVER_PROTOCOL,
                    "prompt_hashes": prompt_hashes(),
                    "reasoner_output": reasoner_text,
                    "finalizer_output": finalizer_text,
                    "reasoner_usage": reasoner.get("usage"),
                    "finalizer_usage": finalizer.get("usage"),
                }
            except Exception as exc:
                self.router.report_failure(url, str(exc))
        return {
            "text": "",
            "endpoint": attempted[-1] if attempted else "",
            "transport_status": "infrastructure_failed",
            "entropy": {"available": False, "normalized_entropy": None, "method": "unavailable"},
        }

    def finalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        records = list(result.get("trial_records") or [])
        correct_count = sum(record.get("is_correct") is True for record in records)
        wrong_count = sum(record.get("is_correct") is False for record in records)
        successful = [record for record in records if record.get("transport_status") == "success"]
        planned = int(result.get("planned_trials") or self.k_trials)
        rate = correct_count / planned if planned else 0.0
        token_values = [record.get("entropy", {}).get("normalized_entropy") for record in successful]
        correct_values = [
            record.get("entropy", {}).get("normalized_entropy")
            for record in successful if record.get("is_correct") is True
        ]
        wrong_values = [
            record.get("entropy", {}).get("normalized_entropy")
            for record in successful if record.get("is_correct") is False
        ]
        token_mean = mean_available(token_values)
        correct_mean = mean_available(correct_values)
        wrong_mean = mean_available(wrong_values)
        available = sum(isinstance(value, (int, float)) for value in token_values)
        level = self.level_from_correct_count(
            correct_count, planned, self.panic_max_correct, self.master_min_correct
        )
        complete = len(successful) == planned
        zone = (
            "incomplete" if not complete else
            self.zone_from_correct_count(
                correct_count, planned, self.panic_max_correct, self.master_min_correct
            )
        )
        if not complete:
            level = 0
        result.update({
            "actual_trials": len(records),
            "successful_trials": len(successful),
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "success_rate": rate,
            "sr": rate,
            "zone": zone,
            "assigned_level": level,
            "panic_max_correct": self.panic_max_correct,
            "master_min_correct": self.master_min_correct,
            "correct_trajectories": [r["response"] for r in successful if r.get("is_correct")],
            "wrong_trajectories": [r["response"] for r in successful if r.get("is_correct") is False],
            "capability_signal": {
                "primary_metric": "success_rate",
                "correct_count": correct_count,
                "wrong_count": wrong_count,
                "success_rate": rate,
                "assigned_level": f"level{level}" if level else zone,
                "entropy": {
                    "available": available > 0,
                    "coverage": available / len(successful) if successful else 0.0,
                    "h_token": token_mean,
                    "h_answer": normalized_answer_entropy(
                        [record.get("response", "") for record in successful], planned
                    ),
                    "h_gap": (
                        wrong_mean - correct_mean
                        if wrong_mean is not None and correct_mean is not None else None
                    ),
                    "correct_mean": correct_mean,
                    "wrong_mean": wrong_mean,
                    "method": "top_logprobs_normalized_entropy",
                },
            },
        })
        return result

    async def sample_one_task(
        self,
        task: Dict[str, Any],
        task_idx: int,
        existing_records: Optional[Sequence[Dict[str, Any]]] = None,
        checkpoint_callback: Optional[Callable[[int, Dict[str, Any], List[Dict[str, Any]]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        question = str(task.get("question") or "").strip()
        expected = str(task.get("answer") or "").strip()
        question_id = str(task.get("id") or task.get("question_id") or task_idx)
        if not question or not expected or not str(task.get("difficulty_level") or "").strip():
            return {
                "schema_version": "stage1.1", "idx": task_idx, "question_id": question_id,
                "question": question, "expected_answer": expected,
                "orig_level": task.get("difficulty_level", ""), "planned_trials": self.k_trials,
                "trial_records": [], "actual_trials": 0, "successful_trials": 0,
                "correct_count": 0, "wrong_count": 0, "success_rate": 0.0, "sr": 0.0,
                "zone": "invalid_input", "assigned_level": 0,
                "correct_trajectories": [], "wrong_trajectories": [],
                "sampling_config": {},
                "capability_signal": {
                    "primary_metric": "success_rate", "correct_count": 0, "wrong_count": 0,
                    "success_rate": 0.0, "assigned_level": "invalid_input",
                    "entropy": {"available": False, "coverage": 0.0, "h_token": None,
                                "h_answer": None, "h_gap": None, "correct_mean": None,
                                "wrong_mean": None, "method": "unavailable"},
                },
            }
        records = [dict(item) for item in (existing_records or []) if isinstance(item, dict)]
        completed_seeds = {
            int(item["seed"]) for item in records
            if (
                item.get("seed") is not None
                and item.get("transport_status") == "success"
                and item.get("prompt_version") == STAGE1_SOLVER_PROTOCOL
                and item.get("prompt_hash") == prompt_hash()
            )
        }
        for trial, seed in enumerate(self.rollout_seeds, start=1):
            if seed in completed_seeds:
                continue
            metadata = await self._single_inference_with_metadata(question, seed)
            response = str(metadata.get("text") or "")
            extracted = extract_final_answer(response)
            transport_ok = metadata.get("transport_status") == "success"
            match = (
                check_answer_detailed(response, expected)
                if transport_ok else
                {"correct": False, "status": "not_run", "method": "none", "reason": "transport_failed"}
            )
            record = {
                "rollout_id": hashlib.sha256(f"{question_id}:{seed}".encode()).hexdigest()[:20],
                "trial": trial,
                "seed": seed,
                "response": response,
                "final_answer": extracted,
                "is_correct": bool(match["correct"]) if transport_ok else None,
                "match_status": match.get("status", "not_matched"),
                "match_method": match.get("method", "none"),
                "match_reason": match.get("reason", ""),
                "model": self.model,
                "prompt_version": STAGE1_SOLVER_PROTOCOL,
                "prompt_hash": prompt_hash(),
                "prompt_hashes": metadata.get("prompt_hashes") or prompt_hashes(),
                "reasoner_output": metadata.get("reasoner_output", ""),
                "finalizer_output": metadata.get("finalizer_output", ""),
                "reasoner_usage": metadata.get("reasoner_usage"),
                "finalizer_usage": metadata.get("finalizer_usage"),
                "endpoint": metadata.get("endpoint", ""),
                "transport_status": metadata.get("transport_status", "infrastructure_failed"),
                "duration_sec": metadata.get("duration_sec"),
                "entropy": metadata.get("entropy") or {"available": False, "normalized_entropy": None},
            }
            records = [item for item in records if int(item.get("seed", -1)) != seed]
            records.append(record)
            records.sort(key=lambda item: int(item.get("seed", -1)))
            if checkpoint_callback is not None:
                await checkpoint_callback(task_idx, record, records)
        result = {
            "schema_version": "stage1.1",
            "idx": task_idx,
            "question_id": question_id,
            "question": question,
            "expected_answer": expected,
            "orig_level": task.get("difficulty_level", ""),
            "planned_trials": self.k_trials,
            "trial_records": records,
            "sampling_config": {
                "model": self.model,
                "prompt_version": STAGE1_SOLVER_PROTOCOL,
                "prompt_hash": prompt_hash(),
                "prompt_hashes": prompt_hashes(),
                "k_trials": self.k_trials,
                "panic_max_correct": self.panic_max_correct,
                "master_min_correct": self.master_min_correct,
                "rollout_seeds": self.rollout_seeds,
                "experiment_seed": self.experiment_seed,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_logprobs": self.top_logprobs,
                "entropy_histogram_bins": self.entropy_histogram_bins,
                "max_tokens": self.max_tokens,
                "reasoner_max_tokens": self.max_tokens,
                "finalizer_max_tokens": self.finalizer_max_tokens,
                "finalizer_temperature": 0.0,
                "finalizer_timeout": self.finalizer_timeout,
                "early_stop": False,
            },
        }
        return self.finalize_result(result)

    @classmethod
    def reclassify_sampling_data(
        cls,
        data: Dict[str, Any],
        *,
        panic_max_correct: int = 1,
        master_min_correct: int = 9,
    ) -> Dict[str, Any]:
        """Rebuild zones and level_data from saved rollouts without LLM calls."""
        results = list(data.get("results") or [])
        levels = {str(level): [] for level in range(1, 5)}
        for result in results:
            if result.get("zone") == "invalid_input":
                continue
            planned = int(result.get("planned_trials") or 10)
            successful = int(result.get("successful_trials") or 0)
            if successful != planned:
                result["zone"] = "incomplete"
                result["assigned_level"] = 0
                continue
            correct_count = int(result.get("correct_count") or 0)
            zone = cls.zone_from_correct_count(
                correct_count, planned, panic_max_correct, master_min_correct
            )
            level = cls.level_from_correct_count(
                correct_count, planned, panic_max_correct, master_min_correct
            )
            result["zone"] = zone
            result["assigned_level"] = level
            result["panic_max_correct"] = panic_max_correct
            result["master_min_correct"] = master_min_correct
            signal = result.setdefault("capability_signal", {})
            signal["assigned_level"] = f"level{level}" if level else zone
            if level:
                levels[str(level)].append(result)

        config = dict(data.get("config") or {})
        config.update({
            "panic_max_correct": panic_max_correct,
            "master_min_correct": master_min_correct,
            "zone_policy": "panic_0_1_learning_2_8_master_9_10",
        })
        data["config"] = config
        data["level_data"] = levels
        data["summary"] = {
            "total": len(results),
            "master_zone": sum(item.get("zone") == "master" for item in results),
            "learning_zone": sum(item.get("zone") == "learning" for item in results),
            "panic_zone": sum(item.get("zone") == "panic" for item in results),
            "incomplete": sum(item.get("zone") == "incomplete" for item in results),
            "invalid_input": sum(item.get("zone") == "invalid_input" for item in results),
            "zone_policy": {
                "master": f"correct_count >= {master_min_correct}",
                "learning": (
                    f"{panic_max_correct} < correct_count < {master_min_correct}"
                ),
                "panic": f"correct_count <= {panic_max_correct}",
                "incomplete": "successful_trials < planned_trials",
            },
            "levels": {key: len(value) for key, value in levels.items()},
        }
        return data

    @staticmethod
    def _load_checkpoint(path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not path or not Path(path).exists():
            return {}
        try:
            grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                index = str(payload["task_index"])
                record = payload["record"]
                grouped.setdefault(index, {})[int(record["seed"])] = record
            return {
                index: [records[seed] for seed in sorted(records)]
                for index, records in grouped.items()
            }
        except Exception:
            return {}

    async def sample_all(
        self,
        tasks: Sequence[Dict[str, Any]],
        concurrency: int = 16,
        checkpoint_path: Optional[str] = None,
        collect_all_samples: bool = False,
    ) -> Dict[str, Any]:
        del collect_all_samples  # all zones are always retained in stage1.1
        partial = self._load_checkpoint(checkpoint_path)
        results: Dict[int, Dict[str, Any]] = {}
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def save_partial(index: int, latest: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
            if not checkpoint_path:
                return
            async with lock:
                target = Path(checkpoint_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                partial[str(index)] = records
                with open(target, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"task_index": index, "record": latest}, ensure_ascii=False) + "\n")
                if latest.get("match_status") in {"symbolic_skipped", "symbolic_timeout"}:
                    audit_dir = target.parent.parent / "analysis"
                    audit_dir.mkdir(parents=True, exist_ok=True)
                    prefix = target.name.split("_unified_sampling_results", 1)[0]
                    audit_path = audit_dir / f"{prefix}_matcher_skipped.jsonl"
                    with open(audit_path, "a", encoding="utf-8") as audit_handle:
                        audit_handle.write(json.dumps({
                            "task_index": index,
                            "seed": latest.get("seed"),
                            "final_answer": latest.get("final_answer"),
                            "match_status": latest.get("match_status"),
                            "match_method": latest.get("match_method"),
                            "match_reason": latest.get("match_reason"),
                        }, ensure_ascii=False) + "\n")

        async def sample(index: int, task: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.sample_one_task(task, index, partial.get(str(index), []), save_partial)

        # Process bounded windows with a fixed worker pool.  Creating one
        # coroutine per dataset row caused needless scheduler/memory pressure
        # on full 6k-question runs.
        progress = async_tqdm(total=len(tasks), desc="Stage 1 sampling")
        try:
            for batch_start in range(0, len(tasks), self.task_batch_size):
                batch_end = min(len(tasks), batch_start + self.task_batch_size)
                queue: asyncio.Queue[tuple[int, Dict[str, Any]]] = asyncio.Queue()
                for index in range(batch_start, batch_end):
                    queue.put_nowait((index, tasks[index]))

                async def worker() -> None:
                    while True:
                        try:
                            index, task = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        try:
                            result = await sample(index, task)
                            results[int(result["idx"])] = result
                            progress.update(1)
                        finally:
                            queue.task_done()

                workers = [
                    asyncio.create_task(worker())
                    for _ in range(min(max(1, int(concurrency)), batch_end - batch_start))
                ]
                await asyncio.gather(*workers)
        finally:
            progress.close()
        ordered = [results[index] for index in sorted(results)]
        levels = {str(level): [] for level in range(1, 5)}
        for result in ordered:
            level = int(result.get("assigned_level") or 0)
            if level:
                levels[str(level)].append(result)
        return {
            "schema_version": "stage1.1",
            "config": {
                "k_trials": self.k_trials,
                "panic_max_correct": self.panic_max_correct,
                "master_min_correct": self.master_min_correct,
                "experiment_seed": self.experiment_seed,
                "rollout_seeds": self.rollout_seeds,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_logprobs": self.top_logprobs,
                "task_batch_size": self.task_batch_size,
                "early_stop": False,
                "level_policy": "accuracy_only",
                "entropy_role": "within_level_ranking_only",
            },
            "results": ordered,
            "level_data": levels,
            "summary": {
                "total": len(ordered),
                "master_zone": sum(item["zone"] == "master" for item in ordered),
                "learning_zone": sum(item["zone"] == "learning" for item in ordered),
                "panic_zone": sum(item["zone"] == "panic" for item in ordered),
                "incomplete": sum(
                    item["zone"] != "invalid_input" and item["successful_trials"] != self.k_trials
                    for item in ordered
                ),
                "invalid_input": sum(item["zone"] == "invalid_input" for item in ordered),
                "zone_policy": {
                    "master": f"correct_count >= {self.master_min_correct}",
                    "learning": (
                        f"{self.panic_max_correct} < correct_count < "
                        f"{self.master_min_correct}"
                    ),
                    "panic": f"correct_count <= {self.panic_max_correct}",
                    "incomplete": "successful_trials < planned_trials",
                },
                "levels": {key: len(value) for key, value in levels.items()},
            },
        }

    @staticmethod
    def save(data: Dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_path)
        return output_path
