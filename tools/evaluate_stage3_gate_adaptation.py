#!/usr/bin/env python3
"""Evaluate Base vs accepted raw/adapted memory under one Reasoner temperature."""
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
    SOLVER_V1_TOP_P,
    ExecuteModule,
)


CONDITIONS = ("base", "raw", "adapted")


def load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    values = [row["conditions"][condition] for row in rows if condition in row["conditions"]]
    n = len(values)
    return {
        "count": n,
        "correct": sum(value["correct"] for value in values),
        "accuracy": sum(value["correct"] for value in values) / n if n else 0.0,
        "memory_injected": sum(value["memory_injected"] for value in values),
        "reasoner_token_limit": sum(value["solver"]["reasoner"]["hit_token_limit"] for value in values),
        "hard_loops": sum(value["solver"]["reasoner"]["repetition"].get("hard_loop", False) for value in values),
        "likely_loops": sum(value["solver"]["reasoner"]["repetition"]["likely_loop"] for value in values),
        "reasoner_format_valid": sum(value["solver"]["reasoner"]["format_valid"] for value in values),
        "finalizer_format_valid": sum(value["solver"]["finalizer"]["format_valid"] for value in values),
        "average_reasoner_tokens": (
            sum((value["solver"]["reasoner"]["completion_tokens"] or 0) for value in values) / n if n else 0.0
        ),
    }


def paired_effect(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["experiment_eligible"]
        and left in row["conditions"] and right in row["conditions"]
    ]
    rescues = sum(
        not row["conditions"][left]["correct"] and row["conditions"][right]["correct"]
        for row in eligible
    )
    harms = sum(
        row["conditions"][left]["correct"] and not row["conditions"][right]["correct"]
        for row in eligible
    )
    return {
        "eligible_count": len(eligible),
        "left": left,
        "right": right,
        "rescues": rescues,
        "harms": harms,
        "net": rescues - harms,
        "changed_final_answers": sum(
            row["conditions"][left]["solver"]["final_answer"]
            != row["conditions"][right]["solver"]["final_answer"]
            for row in eligible
        ),
    }


async def run(args: argparse.Namespace) -> None:
    items = load_list(args.test_file)
    gate_records = load_list(args.gate_records)
    gate_by_idx = {int(row["idx"]): row for row in gate_records}
    selected_indices = list(range(min(len(items), args.limit or len(items))))
    missing = [idx for idx in selected_indices if idx not in gate_by_idx]
    if missing:
        raise ValueError(f"Gate records missing indices: {missing[:10]}")
    ports = list(args.ports)
    urls = {port: f"http://127.0.0.1:{port}/v1/chat/completions" for port in ports}
    locks = {port: asyncio.Lock() for port in ports}
    executor = ExecuteModule(Config.from_core_file(), ports=ports, model_override=args.model)

    async def evaluate_item(idx: int) -> dict[str, Any]:
        item = items[idx]
        question = str(item.get("question") or item.get("problem") or "")
        gate = gate_by_idx[idx]
        eligible = bool(gate.get("experiment_eligible"))
        texts = {"base": ""}
        if eligible:
            texts.update({
                "raw": str(gate.get("raw_memory_text_for_experiment", "")),
                "adapted": str(gate.get("adapted_memory_text_for_experiment", "")),
            })
        port = ports[idx % len(ports)]
        url = urls[port]
        order = [condition for condition in CONDITIONS if condition in texts]
        shift = (idx + args.seed) % len(order)
        order = order[shift:] + order[:shift]
        outputs: dict[str, Any] = {}
        async with locks[port]:
            for condition in order:
                solver = await executor.solve_v1(
                    question,
                    texts[condition],
                    seed=args.seed,
                    url_override=url,
                    reasoner_repetition_penalty=args.repetition_penalty,
                    reasoner_temperature=args.temperature,
                )
                outputs[condition] = {
                    "memory_injected": bool(texts[condition]),
                    "injected_memory_text": texts[condition],
                    "correct": check_answer_for_item(solver["finalizer"]["output"], item),
                    "solver": solver,
                }
        return {
            "idx": idx,
            "question": question,
            "expected_answer": item.get("answer", ""),
            "difficulty_level": item.get("difficulty_level", ""),
            "port": port,
            "call_order": order,
            "memory_id": gate.get("memory_id", ""),
            "candidate_present": gate.get("candidate_present", False),
            "gate_accepted": bool((gate.get("gate_judgment") or {}).get("accepted")),
            "adaptation_status": (gate.get("adaptation") or {}).get("status", "reject"),
            "experiment_eligible": eligible,
            "conditions": outputs,
        }

    tasks = [asyncio.create_task(evaluate_item(idx)) for idx in selected_indices]
    rows: list[dict[str, Any]] = []
    for task in asyncio.as_completed(tasks):
        row = await task
        rows.append(row)
        if len(rows) == 1 or len(rows) == len(tasks) or len(rows) % 5 == 0:
            print(
                f"temperature={args.temperature:.1f} completed {len(rows)}/{len(tasks)} idx={row['idx']}",
                flush=True,
            )
    rows.sort(key=lambda row: row["idx"])
    noneligible = [row for row in rows if not row["experiment_eligible"]]
    summary = {
        "stage": 3,
        "protocol_version": SOLVER_V1_PROTOCOL,
        "experiment": "base_raw_adapted_memory",
        "model": args.model,
        "ports": ports,
        "seed": args.seed,
        "reasoner_temperature": args.temperature,
        "reasoner_repetition_penalty": args.repetition_penalty,
        "reasoner_max_tokens": SOLVER_V1_REASONER_MAX_TOKENS,
        "finalizer_temperature": SOLVER_V1_FINALIZER_TEMPERATURE,
        "finalizer_max_tokens": SOLVER_V1_FINALIZER_MAX_TOKENS,
        "top_p": SOLVER_V1_TOP_P,
        "prompt_hashes": SOLVER_V1_PROMPT_HASHES,
        "test_file": str(args.test_file),
        "test_file_sha256": file_sha256(args.test_file),
        "gate_records": str(args.gate_records),
        "gate_records_sha256": file_sha256(args.gate_records),
        "item_count": len(rows),
        "eligible_count": sum(row["experiment_eligible"] for row in rows),
        "conditions": {condition: condition_metrics(rows, condition) for condition in CONDITIONS},
        "paired_raw_to_adapted": paired_effect(rows, "raw", "adapted"),
        "paired_base_to_raw": paired_effect(rows, "base", "raw"),
        "paired_base_to_adapted": paired_effect(rows, "base", "adapted"),
        "noneligible_count": len(noneligible),
        "noneligible_raw_or_adapted_solver_calls": sum(
            int("raw" in row["conditions"]) + int("adapted" in row["conditions"])
            for row in noneligible
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"temperature={args.temperature:.1f} done output={args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--gate-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", default=[8004, 8006, 8007])
    parser.add_argument("--model", default="llama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--temperature", type=float, choices=(0.0, 0.2), required=True)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
