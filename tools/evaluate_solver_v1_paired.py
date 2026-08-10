#!/usr/bin/env python3
"""Run frozen Solver-v1 Base/Guided pairs on fixed GPU endpoints.

Retrieval decisions and solver-visible memory are replayed from a reference
guided run.  This isolates the new two-stage solver before live retrieval is
tested.  Each endpoint handles one complete pair at a time.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.matcher import check_answer_for_item
from core.config import Config
from modules.execute_module import (
    SOLVER_V1_FINALIZER_MAX_TOKENS,
    SOLVER_V1_FINALIZER_TEMPERATURE,
    SOLVER_V1_PROMPT_HASHES,
    SOLVER_V1_PROTOCOL,
    SOLVER_V1_REASONER_MAX_TOKENS,
    SOLVER_V1_REASONER_REPETITION_PENALTY,
    SOLVER_V1_REASONER_TEMPERATURE,
    SOLVER_V1_TOP_P,
    ExecuteModule,
)
from template.prompt_template import (
    SOLVER_V1_FINALIZER_PROMPT,
    SOLVER_V1_GUIDED_REASONER_PROMPT,
    SOLVER_V1_REASONER_PROMPT,
)


def load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rescues = sum(not row["base_correct"] and row["guided_correct"] for row in rows)
    harms = sum(row["base_correct"] and not row["guided_correct"] for row in rows)
    injected = [row for row in rows if row["memory_injected"]]
    by_seed = {}
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        seed_rescues = sum(not row["base_correct"] and row["guided_correct"] for row in seed_rows)
        seed_harms = sum(row["base_correct"] and not row["guided_correct"] for row in seed_rows)
        by_seed[str(seed)] = {
            "pair_count": len(seed_rows),
            "base_accuracy": sum(row["base_correct"] for row in seed_rows) / len(seed_rows),
            "guided_accuracy": sum(row["guided_correct"] for row in seed_rows) / len(seed_rows),
            "rescues": seed_rescues,
            "harms": seed_harms,
            "net_uplift": (seed_rescues - seed_harms) / len(seed_rows),
        }

    by_memory: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["memory_injected"]:
            continue
        for memory_id in row.get("selected_memory_ids") or ["unknown_memory"]:
            stats = by_memory.setdefault(str(memory_id), {
                "invocations": 0,
                "base_correct": 0,
                "guided_correct": 0,
                "rescues": 0,
                "harms": 0,
            })
            stats["invocations"] += 1
            stats["base_correct"] += int(row["base_correct"])
            stats["guided_correct"] += int(row["guided_correct"])
            stats["rescues"] += int(not row["base_correct"] and row["guided_correct"])
            stats["harms"] += int(row["base_correct"] and not row["guided_correct"])
    for stats in by_memory.values():
        n_memory = stats["invocations"]
        stats["historical_base_accuracy"] = stats["base_correct"] / n_memory
        stats["historical_guided_accuracy"] = stats["guided_correct"] / n_memory
        stats["historical_net_uplift"] = (stats["rescues"] - stats["harms"]) / n_memory

    return {
        "pair_count": n,
        "unique_items": len({row["idx"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "base_correct": sum(row["base_correct"] for row in rows),
        "guided_correct": sum(row["guided_correct"] for row in rows),
        "base_accuracy": sum(row["base_correct"] for row in rows) / n if n else 0.0,
        "guided_accuracy": sum(row["guided_correct"] for row in rows) / n if n else 0.0,
        "rescues": rescues,
        "harms": harms,
        "net_uplift": (rescues - harms) / n if n else 0.0,
        "injected_pair_count": len(injected),
        "injected_rescues": sum(not row["base_correct"] and row["guided_correct"] for row in injected),
        "injected_harms": sum(row["base_correct"] and not row["guided_correct"] for row in injected),
        "base_reasoner_token_limit": sum(row["base"]["reasoner"]["hit_token_limit"] for row in rows),
        "guided_reasoner_token_limit": sum(row["guided"]["reasoner"]["hit_token_limit"] for row in rows),
        "base_likely_loops": sum(row["base"]["reasoner"]["repetition"]["likely_loop"] for row in rows),
        "guided_likely_loops": sum(row["guided"]["reasoner"]["repetition"]["likely_loop"] for row in rows),
        "base_hard_loops": sum(row["base"]["reasoner"]["repetition"].get("hard_loop", False) for row in rows),
        "guided_hard_loops": sum(row["guided"]["reasoner"]["repetition"].get("hard_loop", False) for row in rows),
        "base_reasoner_avg_tokens": (
            sum((row["base"]["reasoner"]["completion_tokens"] or 0) for row in rows) / n if n else 0.0
        ),
        "guided_reasoner_avg_tokens": (
            sum((row["guided"]["reasoner"]["completion_tokens"] or 0) for row in rows) / n if n else 0.0
        ),
        "base_reasoner_format_valid": sum(row["base"]["reasoner"]["format_valid"] for row in rows),
        "guided_reasoner_format_valid": sum(row["guided"]["reasoner"]["format_valid"] for row in rows),
        "base_finalizer_format_valid": sum(row["base"]["finalizer"]["format_valid"] for row in rows),
        "guided_finalizer_format_valid": sum(row["guided"]["finalizer"]["format_valid"] for row in rows),
        "non_injected_final_answer_disagreements": sum(
            row["base"]["final_answer"] != row["guided"]["final_answer"]
            for row in rows if not row["memory_injected"]
        ),
        "by_seed": by_seed,
        "historical_uplift_by_memory": by_memory,
    }


async def run(args: argparse.Namespace) -> None:
    items = load_list(args.test_file)
    reference = load_list(args.reference_guided)
    reference_by_idx = {int(row["idx"]): row for row in reference}
    selected_indices = list(range(len(items)))
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]
    missing = [idx for idx in selected_indices if idx not in reference_by_idx]
    if missing:
        raise ValueError(f"Reference is missing indices: {missing[:10]}")

    config = Config.from_core_file()
    ports = list(args.ports)
    urls = {port: f"http://127.0.0.1:{port}/v1/chat/completions" for port in ports}
    locks = {port: asyncio.Lock() for port in ports}
    executor = ExecuteModule(config, ports=ports, model_override=args.model)

    async def evaluate_pair(idx: int, seed: int) -> dict[str, Any]:
        item = items[idx]
        question = str(item.get("question") or item.get("problem") or "")
        frozen = reference_by_idx[idx]
        injected_text = str(frozen.get("injected_principle_text") or "").strip()
        memory_injected = bool(frozen.get("retrieval_used") and injected_text)
        port = ports[idx % len(ports)]
        url = urls[port]
        # Counterbalance call order without changing either condition.
        guided_first = bool((idx + seed) % 2)
        async with locks[port]:
            if guided_first:
                guided = await executor.solve_v1(
                    question, injected_text if memory_injected else "", seed=seed,
                    url_override=url,
                    reasoner_repetition_penalty=args.reasoner_repetition_penalty,
                )
                base = await executor.solve_v1(
                    question, seed=seed, url_override=url,
                    reasoner_repetition_penalty=args.reasoner_repetition_penalty,
                )
            else:
                base = await executor.solve_v1(
                    question, seed=seed, url_override=url,
                    reasoner_repetition_penalty=args.reasoner_repetition_penalty,
                )
                guided = await executor.solve_v1(
                    question, injected_text if memory_injected else "", seed=seed,
                    url_override=url,
                    reasoner_repetition_penalty=args.reasoner_repetition_penalty,
                )
        base_correct = check_answer_for_item(base["finalizer"]["output"], item)
        guided_correct = check_answer_for_item(guided["finalizer"]["output"], item)
        return {
            "idx": idx,
            "seed": seed,
            "port": port,
            "pair_order": "guided_then_base" if guided_first else "base_then_guided",
            "question": question,
            "expected_answer": item.get("answer", ""),
            "difficulty_level": item.get("difficulty_level", ""),
            "memory_injected": memory_injected,
            "selected_memory_ids": (frozen.get("retrieval_trace") or {}).get("selected_ids", []),
            "injected_principle_text": injected_text if memory_injected else "",
            "base_correct": base_correct,
            "guided_correct": guided_correct,
            "outcome": (
                "rescue" if not base_correct and guided_correct else
                "harm" if base_correct and not guided_correct else
                "both_correct" if base_correct else "both_wrong"
            ),
            "base": base,
            "guided": guided,
        }

    jobs = [(idx, seed) for seed in args.seeds for idx in selected_indices]
    rows: list[dict[str, Any]] = []
    tasks = [asyncio.create_task(evaluate_pair(idx, seed)) for idx, seed in jobs]
    for task in asyncio.as_completed(tasks):
        row = await task
        rows.append(row)
        if len(rows) == 1 or len(rows) == len(jobs) or len(rows) % args.progress_every == 0:
            print(
                f"completed {len(rows)}/{len(jobs)} last_idx={row['idx']} "
                f"seed={row['seed']} outcome={row['outcome']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["seed"], row["idx"]))

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol_version": SOLVER_V1_PROTOCOL,
        "model": args.model,
        "ports": ports,
        "reasoner_max_tokens": SOLVER_V1_REASONER_MAX_TOKENS,
        "finalizer_max_tokens": SOLVER_V1_FINALIZER_MAX_TOKENS,
        "reasoner_temperature": SOLVER_V1_REASONER_TEMPERATURE,
        "finalizer_temperature": SOLVER_V1_FINALIZER_TEMPERATURE,
        "top_p": SOLVER_V1_TOP_P,
        "reasoner_repetition_penalty": args.reasoner_repetition_penalty,
        "finalizer_repetition_penalty": 1.0,
        "prompt_hashes": SOLVER_V1_PROMPT_HASHES,
        "test_file": str(args.test_file),
        "test_file_sha256": file_sha256(args.test_file),
        "reference_guided": str(args.reference_guided),
        "reference_guided_sha256": file_sha256(args.reference_guided),
        "retrieval_replayed": True,
        "answer_dependent_retry": False,
        "one_active_pair_per_endpoint": True,
        "metrics": aggregate(rows),
    }
    (output / "paired_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "prompt_snapshot.json").write_text(
        json.dumps({
            "protocol_version": SOLVER_V1_PROTOCOL,
            "prompt_hashes": SOLVER_V1_PROMPT_HASHES,
            "reasoner_base_template": SOLVER_V1_REASONER_PROMPT,
            "reasoner_guided_template": SOLVER_V1_GUIDED_REASONER_PROMPT,
            "finalizer_template": SOLVER_V1_FINALIZER_PROMPT,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = summary["metrics"]
    print(
        f"done output={output} rp={args.reasoner_repetition_penalty:.2f} "
        f"caps={metrics['base_reasoner_token_limit'] + metrics['guided_reasoner_token_limit']} "
        f"hard_loops={metrics['base_hard_loops'] + metrics['guided_hard_loops']} "
        f"format={metrics['base_reasoner_format_valid'] + metrics['guided_reasoner_format_valid']}/{2 * metrics['pair_count']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--reference-guided", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", default=[8004, 8006, 8007])
    parser.add_argument("--model", default="llama")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reasoner-repetition-penalty",
        type=float,
        default=SOLVER_V1_REASONER_REPETITION_PENALTY,
        choices=(1.00, 1.03, 1.05, 1.08),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
