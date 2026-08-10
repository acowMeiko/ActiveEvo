"""Benchmark 评估器 — 独立于 pipeline，支持 guided/direct 两种模式。

用法 (单次):
  python benchmarks/evaluator.py \
    --test-file data/test/benchmarks/math500_stage3.json \
    --memory-dir memory/llama \
    --ports 8000 8001 8002 8003 \
    --model llama \
    --mode guided \
    --output output/benchmarks/math500/llama \
    --concurrency 64
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm.asyncio import tqdm as async_tqdm

from core.config import Config
from core.runtime_metrics import collect_runtime_metrics
from core.thread_checkpoint import ThreadCheckpointManager
from modules.plan_module import PlanModule
from modules.memory_module import MemoryModule
from modules.execute_module import ExecuteModule, filter_healthy_urls
from modules.principle_schema import principle_content_from_item, render_solver_memory
from modules.reranker_module import QwenApplicabilityReranker
from benchmarks.matcher import check_answer_for_item
from utils.io_utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("benchmarks.evaluator")


def _build_retry_prompt(question: str, principle_text: str, previous_response: str) -> str:
    # Keep only the last 600 chars of the previous response to save context
    prev_snippet = previous_response[-600:] if len(previous_response) > 600 else previous_response
    sections = ["[Retry Instruction]\nYour previous answer was incorrect. Give ONLY the final answer with \\boxed{answer} notation. No step-by-step reasoning needed."]
    if principle_text:
        sections.append(f"[Experience]\n{principle_text}")
    sections.append(f"[Previous Attempt (last part)]\n{prev_snippet}")
    sections.append(f"[Problem]\n{question}")
    return "\n\n".join(sections)


async def evaluate_single(
    config: Config,
    ports: List[int],
    model_name: str,
    test_file: str,
    mode: str = "guided",  # guided | baseline | direct | base
    threshold: float = 0.55,
    top_k: int = 5,
    concurrency: int = 64,
    retry_on_wrong: int = 2,
    output_dir: Path | None = None,
    memory_dir: str | None = None,
    answer_max_tokens: int = 1024,
    timeout: int = 600,
    retrieval_method: str = "dsd",  # "dsd" | "single"
    resume: bool = False,
    seed: int | None = None,
    applicability_gate: bool = False,
    candidate_top_k: int = 10,
    gate_top_k: int = 5,
    applicability_confidence: float = 0.70,
    deterministic_endpoint_routing: bool = False,
    reranker_model: str | None = None,
    reranker_device: str = "cpu",
    reranker_threshold: float = 0.50,
    reranker_batch_size: int = 4,
    max_injected_memories: int = 1,
):
    """运行单个 benchmark 评估。

    resume=True: 加载已有 eval_results.json, 只重测上次做错的题,
    最终结果 = 旧正确 + 新结果, 覆盖写入 output_dir.
    """
    test_data = load_json(test_file)
    if isinstance(test_data, dict):
        test_data = test_data.get("items", test_data.get("results", test_data.get(list(test_data.keys())[0], [])))
    total_items = len(test_data)
    logger.info("已加载 %d 道题: %s (mode=%s)", total_items, test_file, mode)

    # ── Resume: 加载旧结果, 筛出做错的题 ──────────────────
    old_correct_map: Dict[int, Dict] = {}
    retry_indices: set = set()
    if resume and output_dir:
        old_path = Path(output_dir) / "eval_results.json"
        if old_path.exists():
            old_results = load_json(str(old_path))
            for r in old_results:
                if r.get("is_correct"):
                    old_correct_map[r["idx"]] = r
                else:
                    retry_indices.add(r["idx"])
            if retry_indices:
                logger.info("Resume: %d 已正确跳过, 重测 %d 题",
                            len(old_correct_map), len(retry_indices))
            else:
                logger.info("Resume: 全部 %d 题已正确, 无需重测", total_items)
                old_summary = {}
                old_summary_path = Path(output_dir) / "eval_summary.json"
                if old_summary_path.exists():
                    old_summary = load_json(str(old_summary_path))
                return {"results": old_results, "summary": old_summary}

    # 构建待评测列表 (resume 模式只包含错题, 保留原始 idx)
    if retry_indices:
        indexed_items = [(i, item) for i, item in enumerate(test_data) if i in retry_indices]
    else:
        indexed_items = list(enumerate(test_data))

    metrics_context = collect_runtime_metrics("benchmark_eval", {"test_file": str(test_file), "mode": mode, "model": model_name})
    runtime = metrics_context.__enter__()

    # 构建 URL 列表 + 健康检查
    urls = [f"http://localhost:{p}/v1/chat/completions" for p in ports]
    urls = filter_healthy_urls(urls)
    if not urls:
        logger.error("No healthy model endpoints found on ports %s", ports)
        return {"results": [], "summary": {}}

    base_mode = mode == "base"

    if not base_mode:
        plan = PlanModule(
            config, urls=urls, model_override=model_name, timeout=timeout, seed=seed,
        )
        mem_root = Path(memory_dir) if memory_dir else Path("memory")
        mem_root.mkdir(parents=True, exist_ok=True)
        memory = MemoryModule(config, memory_dir=str(mem_root))
        memory.load()
        if not memory.is_indexed:
            memory.build_indices()
            memory.save_indices()
    reranker = None
    if not base_mode and reranker_model:
        reranker = QwenApplicabilityReranker(
            reranker_model,
            device=reranker_device,
            batch_size=reranker_batch_size,
        )
    else:
        plan = None
        memory = None

    execute = ExecuteModule(config, ports=ports, model_override=model_name, seed=seed)
    checkpoint_manager = (
        ThreadCheckpointManager(Path(output_dir) / "checkpoints") if output_dir else None
    )
    store_version = memory.memory.store_version if memory else 0

    semaphore = asyncio.Semaphore(concurrency)
    logger.info("开始评估: %d 题, mode=%s, concurrency=%d", len(test_data), mode, concurrency)

    async def solve_one(item: Dict, idx: int) -> Dict:
        question = item.get("question") or item.get("problem") or ""
        expected = item.get("answer", "")
        solver_url = urls[idx % len(urls)] if deterministic_endpoint_routing else None
        if not question or not expected:
            return {"idx": idx, "status": "skipped", "is_correct": False, "question": question}

        # Base mode: single rollout, no plan/memory/retry
        if base_mode:
            async with semaphore:
                response = await execute.answer_direct(
                    question, max_tokens=answer_max_tokens, seed=seed,
                    url_override=solver_url,
                )
            is_correct = check_answer_for_item(response, item)
            return {
                "idx": idx,
                "status": "success",
                "question": question,
                "mode": mode,
                "retrieval_used": False,
                "retrieval_count": 0,
                "retrieved_principles": [],
                "model_answer": response,
                "expected_answer": expected,
                "is_correct": is_correct,
                "first_attempt_correct": is_correct,
                "attempt_count": 1,
                "retry_used": False,
                "rescued_by_retry": False,
                "attempts": [{"attempt": 1, "is_correct": is_correct, "response_preview": (response or "")[:300]}],
                "seed": seed,
                "solver_url": solver_url,
            }

        use_principle = False
        retrieval_count = 0
        retrieved_items: List[Dict] = []
        retrieval_trace: Dict[str, Any] = {
            "candidate_ids": [],
            "candidate_schema_sources": {},
            "applicability_gate_enabled": applicability_gate,
            "applicability": None,
            "reranker_enabled": reranker is not None,
            "reranked_candidates": [],
            "selected_ids": [],
            "injected_memory": [],
        }

        if mode == "guided":
            if retrieval_method == "single":
                # Single-vector: LLM generates holistic description → embed → KNN
                async with semaphore:
                    description_text = await plan.extract_holistic_description(question)
                retrieved = memory.retrieve_single_vector(
                    description_text,
                    top_k=max(top_k, candidate_top_k if applicability_gate else top_k),
                    threshold=threshold,
                )
            else:
                # DSD: 3-facet decomposition → gated retrieval
                async with semaphore:
                    features = await plan.extract_semantics(question)
                if reranker is not None:
                    retrieved = memory.retrieve_high_recall(features, top_k=candidate_top_k)
                else:
                    retrieved = memory.retrieve(
                        features,
                        top_k=max(top_k, candidate_top_k if applicability_gate else top_k),
                        threshold=threshold,
                    )
            passing = [r for r in retrieved if r.get("item", {}).get("principle")]
            if passing:
                candidate_records = []
                for result in passing[:candidate_top_k]:
                    content, schema_source = principle_content_from_item(result["item"])
                    if not content:
                        continue
                    memory_id = result.get("memory_id") or result["item"].get("memory_id", "")
                    candidate_records.append({
                        "memory_id": memory_id,
                        "content": content,
                        "schema_source": schema_source,
                        "retrieval": result,
                    })
                    retrieval_trace["candidate_ids"].append(memory_id)
                    retrieval_trace["candidate_schema_sources"][memory_id] = schema_source

                if reranker is not None and candidate_records:
                    ranked = reranker.rank(question, candidate_records)
                    gate_batch_ids = {
                        candidate["memory_id"] for candidate in ranked[:gate_top_k]
                    } if applicability_gate else set()
                    retrieval_trace["reranked_candidates"] = [
                        {
                            "memory_id": candidate["memory_id"],
                            "rerank_score": candidate["rerank_score"],
                            "retrieval_score": float(candidate["retrieval"].get("score", 0.0)),
                            "sent_to_gate": candidate["memory_id"] in gate_batch_ids,
                            "selected": False,
                        }
                        for candidate in ranked
                    ]
                    if applicability_gate:
                        candidate_records = ranked[:gate_top_k]
                    else:
                        candidate_records = [
                            candidate for candidate in ranked
                            if candidate["rerank_score"] >= reranker_threshold
                        ][:max_injected_memories]

                if applicability_gate and candidate_records:
                    async with semaphore:
                        gate_result = await plan.judge_memory_applicability(
                            question,
                            candidate_records,
                            confidence_threshold=applicability_confidence,
                        )
                    retrieval_trace["applicability"] = gate_result
                    selected_id_set = set(gate_result.get("selected_ids", []))
                    candidate_records = [
                        candidate for candidate in candidate_records
                        if candidate["memory_id"] in selected_id_set
                    ][:max_injected_memories]

                final_selected_ids = {candidate["memory_id"] for candidate in candidate_records}
                for trace_candidate in retrieval_trace["reranked_candidates"]:
                    trace_candidate["selected"] = trace_candidate["memory_id"] in final_selected_ids

                final_limit = max_injected_memories if reranker is not None else top_k
                candidate_records = candidate_records[:final_limit]
                retrieved_items = [candidate["retrieval"] for candidate in candidate_records]
                retrieval_trace["selected_ids"] = [candidate["memory_id"] for candidate in candidate_records]
                retrieval_trace["injected_memory"] = [
                    {
                        "memory_id": candidate["memory_id"],
                        "schema_source": candidate["schema_source"],
                        "content": candidate["content"],
                    }
                    for candidate in candidate_records
                ]
                use_principle = bool(candidate_records)
                retrieval_count = len(candidate_records)

        principle_text = ""
        if use_principle and retrieved_items:
            blocks = []
            selected_by_id = {
                entry["memory_id"]: entry for entry in retrieval_trace["injected_memory"]
            }
            for i, r in enumerate(retrieved_items, 1):
                memory_id = r.get("memory_id") or r["item"].get("memory_id", "")
                content = selected_by_id[memory_id]["content"]
                blocks.append(render_solver_memory(content, i))
            principle_text = "\n\n".join(blocks)

        thread_id = f"item_{idx:05d}"
        parent_checkpoint = None
        if checkpoint_manager:
            ready_path = checkpoint_manager.save(
                thread_id=thread_id,
                checkpoint_name="ready_to_infer",
                state={
                    "plan": ["retrieve memory", "gate applicability", "solve complete problem", "verify answer"],
                    "current_step": "memory_gated",
                    "recent_observation": {"question": question, "retrieval_trace": retrieval_trace},
                    "recent_action": "selected applicable memories for one complete solver call",
                    "budget": {"answer_max_tokens": answer_max_tokens, "max_attempts": min(3, 1 + max(0, retry_on_wrong))},
                    "selected_memory_ids": retrieval_trace["selected_ids"],
                    "store_version": store_version,
                    "rng": {"request_seed": seed},
                    "config": {"mode": mode, "threshold": threshold, "top_k": top_k, "ports": ports},
                },
            )
            parent_checkpoint = f"{thread_id}:ready_to_infer"

        attempts = []
        total_attempts = min(3, 1 + max(0, retry_on_wrong))
        is_correct = False
        final_response = ""

        for attempt_idx in range(total_attempts):
            async with semaphore:
                current_question = question
                if attempt_idx > 0 and attempts:
                    current_question = _build_retry_prompt(question, principle_text, final_response)
                if use_principle and principle_text:
                    response = await execute.answer_guided(
                        current_question, principle_text,
                        max_tokens=answer_max_tokens, seed=seed,
                        url_override=solver_url,
                    )
                elif mode == "baseline":
                    response = await execute.answer_baseline(
                        current_question, max_tokens=answer_max_tokens, seed=seed,
                        url_override=solver_url,
                    )
                else:
                    response = await execute.answer_direct(
                        current_question, max_tokens=answer_max_tokens, seed=seed,
                        url_override=solver_url,
                    )

            current_correct = check_answer_for_item(response, item)
            attempts.append({
                "attempt": attempt_idx + 1,
                "is_correct": current_correct,
                "response_preview": (response or "")[:300],
            })
            final_response = response
            if current_correct:
                is_correct = True
                break

        if checkpoint_manager:
            checkpoint_manager.save(
                thread_id=thread_id,
                checkpoint_name="inference_completed",
                parent_checkpoint=parent_checkpoint,
                state={
                    "current_step": "inference_completed",
                    "recent_action": "completed full-problem solver inference",
                    "recent_tool_result": {"response_preview": final_response[:1000], "attempt_count": len(attempts)},
                    "selected_memory_ids": retrieval_trace["selected_ids"],
                    "store_version": store_version,
                    "rng": {"request_seed": seed},
                    "evaluation_state": {"answer_verified": is_correct},
                },
            )
            parent_checkpoint = f"{thread_id}:inference_completed"

        result = {
            "idx": idx,
            "status": "success",
            "question": question,
            "mode": mode,
            "retrieval_used": use_principle,
            "retrieval_count": retrieval_count,
            "retrieved_principles": retrieved_items,
            "retrieval_trace": retrieval_trace,
            "injected_principle_text": principle_text,
            "model_answer": final_response,
            "expected_answer": expected,
            "is_correct": is_correct,
            "first_attempt_correct": attempts[0]["is_correct"] if attempts else False,
            "attempt_count": len(attempts),
            "retry_used": len(attempts) > 1,
            "rescued_by_retry": is_correct and any(not a["is_correct"] for a in attempts[:-1]),
            "attempts": attempts,
            "seed": seed,
            "solver_url": solver_url,
        }
        if checkpoint_manager:
            checkpoint_manager.save(
                thread_id=thread_id,
                checkpoint_name="result_committed",
                parent_checkpoint=parent_checkpoint,
                state={
                    "current_step": "result_committed",
                    "recent_action": "verified and committed evaluation result",
                    "thread_status": "completed",
                    "selected_memory_ids": retrieval_trace["selected_ids"],
                    "store_version": store_version,
                    "rng": {"request_seed": seed},
                    "evaluation_state": {
                        "is_correct": is_correct,
                        "first_attempt_correct": result["first_attempt_correct"],
                        "attempt_count": len(attempts),
                    },
                },
            )
        return result

    tasks = [solve_one(item, orig_idx) for orig_idx, item in indexed_items]
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Eval {Path(test_file).stem}"):
        results.append(await coro)

    # ── Resume 合并: 旧正确 + 新结果 ──────────────────────
    if old_correct_map:
        merged = []
        for i in range(total_items):
            if i in old_correct_map:
                merged.append(old_correct_map[i])
            else:
                new_r = next((r for r in results if r["idx"] == i), None)
                if new_r is not None:
                    merged.append(new_r)
                else:
                    merged.append({"idx": i, "status": "skipped", "is_correct": False, "question": ""})
        results = merged

    results.sort(key=lambda x: x.get("idx", 0))
    correct_count = sum(1 for r in results if r.get("is_correct"))
    total_valid = sum(1 for r in results if r.get("status") != "skipped")

    summary = {
        "test_file": str(test_file),
        "mode": mode,
        "model": model_name,
        "seed": seed,
        "applicability_gate": {
            "enabled": applicability_gate,
            "candidate_top_k": candidate_top_k,
            "gate_top_k": gate_top_k,
            "confidence_threshold": applicability_confidence,
            "deterministic_endpoint_routing": deterministic_endpoint_routing,
        },
        "reranker": {
            "enabled": reranker is not None,
            "model": reranker_model,
            "device": reranker_device if reranker is not None else None,
            "threshold": reranker_threshold,
            "candidate_top_k": candidate_top_k,
            "max_injected_memories": max_injected_memories,
        },
        "performance": {
            "correct": correct_count,
            "total_valid": total_valid,
            "scope_total": len(test_data),
            "accuracy": correct_count / len(test_data) if len(test_data) else 0,
        },
        "retrieval": {
            "samples_with_retrieval": sum(1 for r in results if r.get("retrieval_used")),
            "retrieval_used_ratio": sum(1 for r in results if r.get("retrieval_used")) / max(1, len(results)),
        },
        "retry_stats": {
            "first_attempt_correct": sum(1 for r in results if r.get("first_attempt_correct")),
            "retry_used_count": sum(1 for r in results if r.get("retry_used")),
            "rescued_by_retry": sum(1 for r in results if r.get("rescued_by_retry")),
        },
        "memory_experience_count": memory.principle_count if memory else 0,
        "runtime_metrics": runtime.finish(),
    }

    metrics_context.__exit__(None, None, None)

    perf = summary["performance"]
    logger.info("完成: %d/%d correct (%.4f)", perf["correct"], perf["scope_total"], perf["accuracy"])

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(results, str(out / "eval_results.json"))
        save_json(summary, str(out / "eval_summary.json"))
        logger.info("结果已保存: %s", out)

    return {"results": results, "summary": summary}


def main():
    p = argparse.ArgumentParser(description="Benchmark 独立评估器")
    p.add_argument("--test-file", required=True, help="测试集 JSON 路径")
    p.add_argument("--memory-dir", default="memory/v3", help="Memory 根目录")
    p.add_argument("--ports", type=int, nargs="+", required=True, help="API 端口列表")
    p.add_argument("--model", default="llama", help="模型名 (llama/qwen)")
    p.add_argument("--mode", default="guided", choices=["guided", "baseline", "direct", "base"])
    p.add_argument("--threshold", type=float, default=0.55, help="检索相似度阈值")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--retry-on-wrong", type=int, default=2)
    p.add_argument("--answer-max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--retrieval-method", default="dsd", choices=["dsd", "single"], help="检索方法")
    p.add_argument("--output", help="输出目录")
    p.add_argument("--seed", type=int, default=None, help="OpenAI-compatible request seed")
    p.add_argument("--applicability-gate", action="store_true", help="Use trigger/scope/failure LLM routing before solving")
    p.add_argument("--candidate-top-k", type=int, default=10, help="Candidates sent to the applicability gate")
    p.add_argument("--gate-top-k", type=int, default=5, help="Top reranked candidates judged in one Gate call")
    p.add_argument("--applicability-confidence", type=float, default=0.70)
    p.add_argument("--deterministic-endpoint-routing", action="store_true", help="Map item idx to a stable endpoint for paired runs")
    p.add_argument("--reranker-model", help="Local Qwen3-Reranker model path")
    p.add_argument("--reranker-device", default="cpu", help="Reranker torch device, e.g. cpu or cuda:2")
    p.add_argument("--reranker-threshold", type=float, default=0.50, help="Absolute applicability probability required for injection")
    p.add_argument("--reranker-batch-size", type=int, default=4)
    p.add_argument("--max-injected-memories", type=int, default=1)
    args = p.parse_args()

    config = Config.from_core_file()
    output_dir = Path(args.output) if args.output else None

    asyncio.run(evaluate_single(
        config=config,
        ports=args.ports,
        model_name=args.model,
        test_file=args.test_file,
        mode=args.mode,
        threshold=args.threshold,
        top_k=args.top_k,
        concurrency=args.concurrency,
        retry_on_wrong=args.retry_on_wrong,
        output_dir=output_dir,
        memory_dir=args.memory_dir,
        answer_max_tokens=args.answer_max_tokens,
        timeout=args.timeout,
        retrieval_method=args.retrieval_method,
        seed=args.seed,
        applicability_gate=args.applicability_gate,
        candidate_top_k=args.candidate_top_k,
        gate_top_k=args.gate_top_k,
        applicability_confidence=args.applicability_confidence,
        deterministic_endpoint_routing=args.deterministic_endpoint_routing,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_threshold=args.reranker_threshold,
        reranker_batch_size=args.reranker_batch_size,
        max_injected_memories=args.max_injected_memories,
    ))


if __name__ == "__main__":
    main()
