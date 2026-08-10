"""Replay a frozen retrieval run while evaluating a new solver prompt.

This tool intentionally performs no recall or reranking.  In guided mode it
reuses the exact selected memory and solver-visible text from a reference
``eval_results.json`` so that the only experimental change is the solver
prompt (and explicitly supplied generation budget).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.matcher import check_answer_for_item
from core.config import Config
from modules.execute_module import ExecuteModule


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("items", "results", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"No item list found in {path}")


async def _evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = _load_items(args.test_file)
    reference = _load_items(args.reference_results)
    reference_by_idx = {int(record["idx"]): record for record in reference}
    missing = sorted(set(range(len(items))) - set(reference_by_idx))
    if missing:
        raise ValueError(f"Reference results are missing indices: {missing[:10]}")

    executor = ExecuteModule(
        Config.from_core_file(),
        ports=args.ports,
        model_override=args.model,
        seed=args.seed,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    urls = [f"http://localhost:{port}/v1/chat/completions" for port in args.ports]

    async def solve(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        question = item.get("question") or item.get("problem") or ""
        expected = item.get("answer", "")
        frozen = reference_by_idx[idx]
        injected_text = str(frozen.get("injected_principle_text") or "")
        use_memory = bool(
            args.mode == "guided"
            and frozen.get("retrieval_used")
            and injected_text.strip()
        )
        url = urls[idx % len(urls)]
        async with semaphore:
            if use_memory:
                response = await executor.answer_guided(
                    question,
                    injected_text,
                    max_tokens=args.answer_max_tokens,
                    seed=args.seed,
                    url_override=url,
                )
            else:
                response = await executor.answer_direct(
                    question,
                    max_tokens=args.answer_max_tokens,
                    seed=args.seed,
                    url_override=url,
                )

        correct = check_answer_for_item(response, item)
        return {
            "idx": idx,
            "status": "success",
            "question": question,
            "mode": args.mode,
            "retrieval_used": use_memory,
            "retrieval_count": int(frozen.get("retrieval_count", 0)) if use_memory else 0,
            "retrieved_principles": frozen.get("retrieved_principles", []) if use_memory else [],
            "retrieval_trace": frozen.get("retrieval_trace", {}) if args.mode == "guided" else {},
            "injected_principle_text": injected_text if use_memory else "",
            "model_answer": response,
            "expected_answer": expected,
            "is_correct": correct,
            "first_attempt_correct": correct,
            "attempt_count": 1,
            "retry_used": False,
            "rescued_by_retry": False,
            "attempts": [{
                "attempt": 1,
                "is_correct": correct,
                "response_preview": (response or "")[:300],
            }],
            "seed": args.seed,
            "solver_url": url,
            "retrieval_replayed_from": str(args.reference_results),
        }

    tasks = [solve(idx, item) for idx, item in enumerate(items)]
    results: list[dict[str, Any]] = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)
    results.sort(key=lambda record: record["idx"])

    correct = sum(bool(record["is_correct"]) for record in results)
    summary = {
        "test_file": str(args.test_file),
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "answer_max_tokens": args.answer_max_tokens,
        "retrieval_replay": True,
        "reference_results": str(args.reference_results),
        "performance": {
            "correct": correct,
            "scope_total": len(results),
            "accuracy": correct / len(results) if results else 0.0,
        },
        "retrieval": {
            "samples_with_retrieval": sum(bool(record["retrieval_used"]) for record in results),
        },
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--mode", choices=("base", "guided"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    args = parser.parse_args()

    results, summary = asyncio.run(_evaluate(args))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
