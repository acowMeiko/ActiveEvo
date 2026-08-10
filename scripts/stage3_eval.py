#!/usr/bin/env python
"""
scripts/stage3_eval.py — 独立的 Stage 3 测试集评估脚本

从 AgentPipeline 中抽离, 不依赖 pipeline 的其他 stage。
用法:
  python scripts/stage3_eval.py \
    --test-input data/test/deepscaler.json \
    --memory-dir memory \
    --ports 8000 8001 8002 8003 \
    --model qwen \
    --mode guided \
    --output output/stage3_eval/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from tqdm.asyncio import tqdm as async_tqdm

from core.config import Config
from core.runtime_metrics import collect_runtime_metrics
from modules.plan_module import PlanModule
from modules.memory_module import MemoryModule
from modules.execute_module import (
    ExecuteModule,
    URLRouter,
    check_answer_match,
    extract_final_answer,
    filter_healthy_urls,
)
from utils.io_utils import load_json, save_json
from utils.official_dataset_utils import evaluate_general365_prediction, evaluate_svamp_prediction
from utils.summary_utils import build_stage3_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("stage3_eval")


def _normalize_level_tag(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(\d+)", text)
    return f"Level {match.group(1)}" if match else text


def _build_retry_prompt(question: str, principle_text: str, previous_response: str) -> str:
    sections = [f"[Question]\n{question}"]
    if principle_text:
        sections.append(f"[Relevant Experiences]\n{principle_text}")
    sections.append(
        "[Retry Instruction]\n"
        "Your previous answer was likely incorrect.\n"
        "Re-solve the problem from scratch, re-check all constraints, and correct the final boxed answer if needed."
    )
    sections.append(f"[Previous Attempt]\n{previous_response}")
    return "\n\n".join(sections)


async def run_eval(
    test_data: List[Dict[str, Any]],
    *,
    plan: PlanModule,
    memory: MemoryModule,
    execute: ExecuteModule,
    mode: str = "guided",
    threshold: float = 0.55,
    top_k: int = 5,
    retry_on_wrong: int = 2,
    concurrency: int = 10,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """独立评估: 对每条 test item 检索记忆并答题。"""
    metrics_context = collect_runtime_metrics("stage3_eval", {"dataset": dataset, "mode": mode})
    runtime = metrics_context.__enter__()

    if not memory.is_indexed:
        memory.load()
        if not memory.is_indexed:
            memory.build_indices()
            memory.save_indices()

    semaphore = asyncio.Semaphore(concurrency)
    logger.info(
        "Stage 3 Eval: %d 题, mode=%s, threshold=%.2f, top_k=%d",
        len(test_data), mode, threshold, top_k,
    )

    async def solve_one(item: Dict, idx: int) -> Dict:
        question = item.get("question") or item.get("problem") or ""
        expected = item.get("answer", "")
        if not question or not expected:
            return {"idx": idx, "status": "skipped", "is_correct": False}

        async with semaphore:
            semantics = await plan.extract_semantics(question)

        use_principle = False
        principle_text = ""
        retrieval_count = 0
        retrieved_items: List[Dict] = []

        if mode == "guided":
            retrieved = memory.retrieve(semantics, top_k=top_k, threshold=threshold)
            passing = [r for r in retrieved if r.get("item", {}).get("principle")]
            if passing:
                use_principle = True
                retrieval_count = len(passing)
                retrieved_items = passing
                blocks = []
                for i, r in enumerate(passing, 1):
                    p = r["item"].get("principle", "")
                    score = r.get("score", 0.0)
                    blocks.append(
                        f"[Experience {i}]\n"
                        f"- association_score: {score:.2f}\n"
                        f"- guidance: {p}"
                    )
                principle_text = "\n\n".join(blocks)

        # 答题 + 重试
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
                    response = await execute.answer_guided(current_question, principle_text)
                elif mode == "baseline":
                    response = await execute.answer_baseline(current_question)
                else:
                    response = await execute.answer_direct(current_question)

            current_correct = execute.check_answer(response, expected)
            attempts.append({
                "attempt": attempt_idx + 1,
                "is_correct": current_correct,
                "response_preview": (response or "")[:300],
            })
            final_response = response
            if current_correct:
                is_correct = True
                break

        return {
            "idx": idx,
            "status": "success",
            "difficulty_level": _normalize_level_tag(item.get("difficulty_level") or item.get("level")),
            "question": question,
            "mode": mode,
            "retrieval_used": use_principle,
            "retrieval_count": retrieval_count,
            "retrieved_principles": retrieved_items,
            "model_answer": final_response,
            "expected_answer": expected,
            "is_correct": is_correct,
            "first_attempt_correct": attempts[0]["is_correct"] if attempts else False,
            "attempt_count": len(attempts),
            "retry_used": len(attempts) > 1,
            "rescued_by_retry": is_correct and any(not a["is_correct"] for a in attempts[:-1]),
            "attempts": attempts,
        }

    tasks = [solve_one(item, idx) for idx, item in enumerate(test_data)]
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Stage 3 Eval"):
        results.append(await coro)

    results.sort(key=lambda x: x.get("idx", 0))
    correct_count = sum(1 for r in results if r.get("is_correct"))
    summary = build_stage3_summary(
        results,
        scope_total=len(test_data),
        correct=correct_count,
        memory_experience_count=memory.principle_count,
        dataset=dataset,
        mode=mode,
        runtime_metrics=runtime.finish(),
    )
    metrics_context.__exit__(None, None, None)

    perf = summary["performance"]
    logger.info(
        "Stage 3 Eval 完成: %d/%d correct (%.4f)",
        perf["correct"], perf["scope_total"], perf["accuracy"],
    )
    return {"results": results, "summary": summary}


def main():
    p = argparse.ArgumentParser(description="独立 Stage 3 测试集评估")
    p.add_argument("--test-input", type=Path, required=True, help="测试集 JSON 路径")
    p.add_argument("--backbone", type=str, default="", help="模型标识 (qwen/llama), Memory 命名空间隔离")
    p.add_argument("--memory-dir", type=str, default="memory/v3", help="Memory 根目录")
    p.add_argument("--model", "-m", type=str, default=None, help="模型名, 默认使用 config 中的 weak_model_name")
    p.add_argument("--ports", type=int, nargs="+", default=[8000],
                   help="vLLM 端口列表")
    p.add_argument("--host", default="localhost")
    p.add_argument("--mode", type=str, default="guided",
                   choices=["guided", "baseline", "direct"])
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--retry-on-wrong", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--output", type=Path, default=None, help="输出目录")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    if not args.test_input.exists():
        logger.error("测试集不存在: %s", args.test_input)
        sys.exit(1)

    config = Config.from_core_file()
    config.ensure_dirs()

    urls = [f"http://{args.host}:{port}/v1/chat/completions" for port in args.ports]
    urls = filter_healthy_urls(urls)
    if not urls:
        logger.error("No available vLLM endpoints")
        sys.exit(1)

    memory_dir = str(Path(args.memory_dir) / args.backbone if args.backbone else Path(args.memory_dir))
    plan = PlanModule(config, urls=urls, model_override=args.model, timeout=args.timeout)
    memory = MemoryModule(config, memory_dir=memory_dir)
    execute = ExecuteModule(config, ports=args.ports, urls=urls, model_override=args.model)

    test_data = load_json(str(args.test_input))
    if not isinstance(test_data, list):
        logger.error("测试集不是 JSON list")
        sys.exit(1)

    result = asyncio.run(run_eval(
        test_data=test_data,
        plan=plan,
        memory=memory,
        execute=execute,
        mode=args.mode,
        threshold=args.threshold,
        top_k=args.top_k,
        retry_on_wrong=args.retry_on_wrong,
        concurrency=args.concurrency,
        dataset=args.test_input.stem,
    ))

    output_dir = args.output or Path("output/stage3_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result["results"], str(output_dir / "eval_results.json"))
    save_json(result["summary"], str(output_dir / "eval_summary.json"))
    logger.info("结果已保存: %s", output_dir)


if __name__ == "__main__":
    main()
