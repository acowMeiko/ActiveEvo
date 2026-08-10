"""Serial-per-endpoint counterfactual replay for samples that received memory."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.matcher import check_answer_for_item
from core.config import Config
from modules.execute_module import ExecuteModule
from modules.execute_module import extract_final_answer


async def run(args: argparse.Namespace) -> None:
    guided = json.loads(Path(args.guided).read_text(encoding="utf-8"))
    test_items = json.loads(Path(args.test_file).read_text(encoding="utf-8"))
    selected = [row for row in guided if row.get("retrieval_used") and row.get("injected_principle_text")]
    config = Config.from_core_file()
    ports = list(args.ports)
    urls = {port: f"http://127.0.0.1:{port}/v1/chat/completions" for port in ports}
    locks = {port: asyncio.Lock() for port in ports}
    execute = ExecuteModule(config, ports=ports, model_override=args.model, seed=args.seed)

    async def replay(row: dict) -> dict:
        idx = int(row["idx"])
        item = test_items[idx]
        question = item.get("question") or item.get("problem") or row.get("question", "")
        port = ports[idx % len(ports)]
        async with locks[port]:
            base_answer = await execute.answer_direct(
                question,
                max_tokens=args.answer_max_tokens,
                seed=args.seed,
                url_override=urls[port],
            )
            guided_answer = await execute.answer_guided(
                question,
                row["injected_principle_text"],
                max_tokens=args.answer_max_tokens,
                seed=args.seed,
                url_override=urls[port],
            )
        base_correct = check_answer_for_item(base_answer, item)
        guided_correct = check_answer_for_item(guided_answer, item)
        return {
            "idx": idx,
            "level": (item.get("_sampling", {}) or {}).get("stratified_level"),
            "port": port,
            "seed": args.seed,
            "selected_ids": (row.get("retrieval_trace", {}) or {}).get("selected_ids", []),
            "injected_principle_text": row["injected_principle_text"],
            "base_answer": base_answer,
            "guided_answer": guided_answer,
            "base_final_answer": extract_final_answer(base_answer or ""),
            "guided_final_answer": extract_final_answer(guided_answer or ""),
            "base_correct": base_correct,
            "guided_correct": guided_correct,
            "full_response_changed": (base_answer or "").strip() != (guided_answer or "").strip(),
        }

    results = []
    for task in asyncio.as_completed([replay(row) for row in selected]):
        results.append(await task)
        print(f"completed {len(results)}/{len(selected)}", flush=True)
    results.sort(key=lambda row: row["idx"])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    improved = sum(not row["base_correct"] and row["guided_correct"] for row in results)
    degraded = sum(row["base_correct"] and not row["guided_correct"] for row in results)
    print(json.dumps({
        "n": len(results),
        "base_correct": sum(row["base_correct"] for row in results),
        "guided_correct": sum(row["guided_correct"] for row in results),
        "improved": improved,
        "degraded": degraded,
        "full_response_changed": sum(row["full_response_changed"] for row in results),
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guided", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-max-tokens", type=int, default=2048)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
