#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm.asyncio import tqdm as async_tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from core.api_call import async_llm_call
from core.config import Config
from utils.embedding_utils import DEFAULT_EMBEDDING_MODEL, load_sentence_transformer_with_fallback
from evaluate_principle_guided_test import (
    check_answer_match_with_item,
    parse_level,
)
from template.prompt_template import DIRECT_ANSWER_PROMPT, GUIDED_ANSWER_PROMPT
from utils.summary_utils import build_stage3_summary


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_urls(base_url: str, ports: Optional[Sequence[int]] = None) -> List[str]:
    if not ports:
        return [base_url]
    urls = []
    for port in ports:
        if re.search(r":\d+(?=/|$)", base_url):
            urls.append(re.sub(r":\d+(?=/|$)", f":{port}", base_url))
        else:
            urls.append(re.sub(r"^(https?://[^/]+)", rf"\1:{port}", base_url))
    return urls


class DynamicRouter:
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.counter = 0

    def get_url(self) -> str:
        url = self.urls[self.counter % len(self.urls)]
        self.counter += 1
        return url


def normalize_matrix(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return arr / norms


class SingleVectorPrincipleRetriever:
    def __init__(self, memory_dir: Path, model_name: str, device: str):
        self.memory_dir = memory_dir
        self.records: List[Dict[str, Any]] = load_json(memory_dir / "single_vector_principles.json")
        payload = np.load(memory_dir / "single_vector_principles.npz")
        self.embeddings = normalize_matrix(payload["embeddings"])
        self.model, self.loaded_from = load_sentence_transformer_with_fallback(
            model_name=model_name,
            device=device,
            local_files_only=True,
        )

    def encode_queries(self, questions: List[str], batch_size: int) -> np.ndarray:
        encoded = self.model.encode(questions, batch_size=batch_size, show_progress_bar=True)
        return normalize_matrix(encoded)

    def retrieve_from_vector(self, query_vector: np.ndarray, top_k: int, score_threshold: float) -> List[Dict[str, Any]]:
        scores = self.embeddings @ query_vector.astype(np.float32)
        if scores.size == 0:
            return []
        order = np.argsort(-scores)[:top_k]
        results: List[Dict[str, Any]] = []
        for idx in order:
            score = float(scores[idx])
            if score < score_threshold:
                continue
            record = self.records[int(idx)]
            results.append(
                {
                    "score": score,
                    "memory_id": record.get("memory_id", ""),
                    "question": record.get("question", ""),
                    "principle": record.get("principle", ""),
                    "source_round": record.get("source_round"),
                    "source_dataset": record.get("source_dataset", ""),
                }
            )
        return results


def format_principles(retrieved: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, item in enumerate(retrieved, 1):
        blocks.append(f"[Experience {i} (single-vector relevance: {item['score']:.2f})]:\n{item['principle']}")
    return "\n\n".join(blocks)


def build_retry_prompt(question: str, principle_text: str, previous_response: str) -> str:
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


async def solve_one(
    item: Dict[str, Any],
    idx: int,
    retrievals: List[Dict[str, Any]],
    model: str,
    router: DynamicRouter,
    api_key: str,
    timeout: int,
    retry_on_wrong: int,
    answer_max_tokens: int,
) -> Dict[str, Any]:
    question = item.get("question") or item.get("problem") or ""
    difficulty = parse_level(item.get("difficulty_level", "Unknown"))
    principle_block = format_principles(retrievals)
    retrieval_used = bool(principle_block)
    prompt = (
        GUIDED_ANSWER_PROMPT.format(principle_text=principle_block, question=question)
        if retrieval_used
        else DIRECT_ANSWER_PROMPT.format(question=question)
    )

    attempts: List[Dict[str, Any]] = []
    final_response = ""
    is_correct = False
    total_attempts = min(3, 1 + max(0, retry_on_wrong))
    for attempt_idx in range(total_attempts):
        current_prompt = prompt
        if attempt_idx > 0 and attempts:
            current_prompt = build_retry_prompt(question, principle_block if retrieval_used else "", final_response)
        response = await async_llm_call(
            prompt=current_prompt,
            model=model,
            url=router.get_url(),
            api_key=api_key,
            timeout=timeout,
            temperature=0.0,
            max_tokens=answer_max_tokens,
        )
        current_correct = check_answer_match_with_item(response, item)
        attempts.append({"attempt": attempt_idx + 1, "is_correct": current_correct, "model_response": response})
        final_response = response
        if current_correct:
            is_correct = True
            break

    return {
        "idx": idx,
        "status": "success",
        "evaluation_key": item.get("id", f"sample_{idx}"),
        "source_item": item,
        "difficulty_level": difficulty,
        "question": question,
        "mode": "single_vector_principle_guided",
        "retrieval_used": retrieval_used,
        "retrieval_count": len(retrievals),
        "retrieved_principles": retrievals,
        "model_answer": final_response,
        "expected_answer": item.get("answer", ""),
        "is_correct": is_correct,
        "first_attempt_correct": attempts[0]["is_correct"] if attempts else False,
        "attempt_count": len(attempts),
        "retry_used": len(attempts) > 1,
        "rescued_by_retry": is_correct and any(not attempt["is_correct"] for attempt in attempts[:-1]),
        "attempts": attempts,
        "carried_forward": False,
    }


def summarize(results: List[Dict[str, Any]], *, current_round_total: int, carried_forward_count: int, test_scope_key: str) -> Dict[str, Any]:
    valid = [r for r in results if r.get("status") in {"success", "carried_forward"}]
    correct = sum(1 for r in valid if r.get("is_correct"))
    summary = build_stage3_summary(
        results,
        scope_total=len(valid),
        correct=correct,
        memory_experience_count=0,
        dataset="single_vector_ablation",
        mode="single_vector_principle_guided",
        test_scope_key=test_scope_key,
        runtime_metrics=None,
    )
    summary["meta"]["current_round_total"] = current_round_total
    summary["meta"]["carried_forward_count"] = carried_forward_count
    return summary


async def run_level(
    *,
    data: List[Dict[str, Any]],
    previous_results: Optional[List[Dict[str, Any]]],
    retriever: SingleVectorPrincipleRetriever,
    memory_dir: Path,
    output_dir: Path,
    level: int,
    args: argparse.Namespace,
    router: DynamicRouter,
) -> List[Dict[str, Any]]:
    unresolved = []
    results: List[Optional[Dict[str, Any]]] = [None] * len(data)
    carried_forward_count = 0
    if previous_results:
        for idx, previous in enumerate(previous_results):
            if previous.get("is_correct"):
                carried = dict(previous)
                carried["status"] = "carried_forward"
                carried["carried_forward"] = True
                results[idx] = carried
                carried_forward_count += 1
            else:
                unresolved.append(idx)
    else:
        unresolved = list(range(len(data)))

    questions = [data[idx].get("question") or data[idx].get("problem") or "" for idx in unresolved]
    query_embeddings = retriever.encode_queries(questions, batch_size=args.embedding_batch_size) if questions else np.zeros((0, 1), dtype=np.float32)
    retrieval_map = {
        idx: retriever.retrieve_from_vector(query_embeddings[pos], top_k=args.top_k, score_threshold=args.score_threshold)
        for pos, idx in enumerate(unresolved)
    }

    semaphore = asyncio.Semaphore(args.concurrency)

    async def sem_task(idx: int) -> Dict[str, Any]:
        async with semaphore:
            return await solve_one(
                item=data[idx],
                idx=idx,
                retrievals=retrieval_map[idx],
                model=args.model,
                router=router,
                api_key=args.api_key,
                timeout=args.timeout,
                retry_on_wrong=args.retry_on_wrong,
                answer_max_tokens=args.answer_max_tokens,
            )

    tasks = [sem_task(idx) for idx in unresolved]
    solved = await async_tqdm.gather(*tasks, desc=f"single-vector level {level}", leave=True) if tasks else []
    for item in solved:
        results[item["idx"]] = item

    finalized = [r for r in results if r is not None]
    summary = summarize(
        finalized,
        current_round_total=len(unresolved),
        carried_forward_count=carried_forward_count,
        test_scope_key=args.test_scope_key,
    )
    summary["meta"].update(
        {
            "ablation": "single_vector_principle",
            "level": level,
            "memory_dir": str(memory_dir),
            "score_threshold": args.score_threshold,
            "top_k": args.top_k,
            "embedding_model_loaded_from": retriever.loaded_from,
        }
    )

    save_json(output_dir / f"cumulative_level{level}_single_vector_results.json", finalized)
    save_json(output_dir / f"cumulative_level{level}_single_vector_summary.json", summary)
    return finalized


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run stage3 with direct single-vector principle retrieval.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--model", default="base")
    parser.add_argument("--url", default=None)
    parser.add_argument("--ports", type=int, nargs="+", default=[8000, 8001, 8002, 8003])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retry-on-wrong", type=int, default=2)
    parser.add_argument("--answer-max-tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--embedding-model", default=os.environ.get("MEMUA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--embedding-device", default=os.environ.get("MEMUA_EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--test-scope-key", default="test_deepscaler_single_vector")
    args = parser.parse_args()

    config = Config.from_core_file()
    llm_url = args.url or config.hyperparams.weak_model_url
    router = DynamicRouter(build_urls(llm_url, args.ports))
    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("Input test set must be a JSON list.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_results: Optional[List[Dict[str, Any]]] = None
    run_summaries: Dict[str, Any] = {}
    for level in args.levels:
        memory_dir = args.memory_root / f"cumulative_level{level}"
        retriever = SingleVectorPrincipleRetriever(memory_dir, args.embedding_model, args.embedding_device)
        previous_results = await run_level(
            data=data,
            previous_results=previous_results,
            retriever=retriever,
            memory_dir=memory_dir,
            output_dir=args.output_dir,
            level=level,
            args=args,
            router=router,
        )
        summary_path = args.output_dir / f"cumulative_level{level}_single_vector_summary.json"
        run_summaries[f"cumulative_level{level}"] = load_json(summary_path)
    save_json(args.output_dir / "single_vector_rerun_summaries.json", run_summaries)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
