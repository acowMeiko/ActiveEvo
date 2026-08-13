#!/usr/bin/env python3
"""Official Stage-3 pass@1/pass@3 evaluator with no answer-dependent retry."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.matcher import check_answer_for_item
from core.config import Config
from core.request_context import llm_request_context
from modules.execute_module import ExecuteModule, filter_healthy_urls
from modules.execute_module import SOLVER_V1_PROMPT_HASHES, SOLVER_V1_PROTOCOL
from modules.memory_module import MemoryModule
from modules.plan_module import PlanModule
from modules.principle_schema import (
    principle_content_from_item,
    render_solver_condition_card,
)
from utils.io_utils import atomic_save_json
from template.prompt_template import (
    SOLVER_V1_FINALIZER_PROMPT,
    SOLVER_V1_GUIDED_REASONER_PROMPT,
    SOLVER_V1_REASONER_PROMPT,
)
from utils.io_utils import load_json

logger = logging.getLogger("stage3.evaluate")
SEEDS = (42, 43, 44)


def experience_decision(text: str) -> str:
    match = re.search(
        r"Experience Decision:\s*(use|ignore)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else "missing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "guided"], required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-dir", default="")
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--gate-top-k", type=int, default=6)
    parser.add_argument("--max-injected", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test prefix size; official runs leave this unset.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "raw.log", encoding="utf-8")],
    )
    started = time.monotonic()
    test_data = load_json(args.test_file)
    if not isinstance(test_data, list) or len(test_data) != 400:
        raise ValueError(f"Official DeepScaler input must contain exactly 400 items, got {len(test_data) if isinstance(test_data, list) else 'non-list'}")
    if args.limit:
        test_data = test_data[:args.limit]
    total_items = len(test_data)

    urls = filter_healthy_urls([f"http://127.0.0.1:{port}/v1/chat/completions" for port in args.ports])
    if len(urls) != len(args.ports):
        raise RuntimeError(f"Expected {len(args.ports)} healthy endpoints, found {len(urls)}")
    config = Config.from_core_file()
    execute = ExecuteModule(config, urls=urls, model_override=args.model, timeout=300, seed=42)
    atomic_save_json(output_dir / "prompt_snapshot.json", {
        "solver_protocol_version": SOLVER_V1_PROTOCOL,
        "reasoner_base_template": SOLVER_V1_REASONER_PROMPT,
        "reasoner_guided_template": SOLVER_V1_GUIDED_REASONER_PROMPT,
        "finalizer_template": SOLVER_V1_FINALIZER_PROMPT,
        "prompt_hashes": SOLVER_V1_PROMPT_HASHES,
        "reasoner_max_tokens": 8192,
        "finalizer_max_tokens": 128,
        "temperature": args.temperature,
        "seeds": list(SEEDS),
        "answer_dependent_retry": False,
    })
    guided = args.variant != "baseline"
    plan = None
    memory = None
    memory_hash = "none"
    active_memory_count = 0
    if guided:
        if not args.memory_dir:
            raise ValueError("Guided evaluation requires --memory-dir")
        plan = PlanModule(config, urls=urls, model_override=args.model, timeout=120, seed=42)
        memory = MemoryModule(config, memory_dir=args.memory_dir)
        memory.load()
        if not memory.is_indexed:
            memory.build_indices()
            memory.save_indices()
        active_memory_count = sum(1 for item in memory.memory.principles if item.get("status") == "active")
        principle_file = Path(args.memory_dir) / "principles" / "principles.json"
        memory_hash = sha256_file(principle_file) if principle_file.exists() else hashlib.sha256(b"[]").hexdigest()

    result_path = output_dir / "eval_results.json"
    old = load_json(str(result_path)) if result_path.exists() else []
    completed: Dict[int, Dict[str, Any]] = {
        int(item["idx"]): item for item in old
        if isinstance(item, dict) and item.get("status") in {"complete", "infrastructure_failed"}
        and len(item.get("rollouts", [])) == 3
    }
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    transport_path = output_dir / "transport_failures.jsonl"
    transport_failures: List[Dict[str, Any]] = []
    if transport_path.exists():
        for line in transport_path.read_text(encoding="utf-8").splitlines():
            try:
                transport_failures.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    result_lock = asyncio.Lock()

    async def retrieve_memory(question: str, idx: int) -> Dict[str, Any]:
        if not guided or plan is None or memory is None or active_memory_count == 0:
            return {
                "used": False, "candidate_ids": [], "selected_ids": [],
                "solver_text": "", "condition_cards": [], "adapter_results": [],
            }
        with llm_request_context(f"stage3_{idx}", "routing_semantics"):
            semantics = await plan.extract_semantics(question)
        retrieved = memory.retrieve(semantics, top_k=args.candidate_top_k, threshold=args.threshold)
        candidates = []
        for entry in retrieved[:args.candidate_top_k]:
            item = entry.get("item", {})
            content, schema_source = principle_content_from_item(item)
            if not content or item.get("status") != "active":
                continue
            candidates.append({
                "memory_id": entry.get("memory_id") or item.get("memory_id", ""),
                "content": content, "schema_source": schema_source, "retrieval": entry,
            })
        trace = {
            "candidate_ids": [item["memory_id"] for item in candidates],
            "selected_ids": [],
            "used": False,
            "solver_text": "",
            "condition_cards": [],
            "adapter_results": [],
            "raw_principles": [],
        }
        if not candidates:
            return trace
        with llm_request_context(f"stage3_{idx}", "application_gate"):
            gate = await plan.judge_memory_applicability(question, candidates[:args.gate_top_k])
        selected = set(gate.get("selected_ids", [])) if gate.get("protocol_compliant") else set()
        accepted = [item for item in candidates if item["memory_id"] in selected]
        adapted = []
        adapter_results = []
        judgments_by_id = {
            str(item.get("memory_id", "")): item
            for item in gate.get("judgments", [])
            if isinstance(item, dict)
        }
        for item in accepted:
            if len(adapted) >= args.max_injected:
                break
            with llm_request_context(f"stage3_{idx}", "memory_adapter/{item['memory_id']}"):
                adapter = await plan.adapt_memory_for_question(
                    question,
                    item,
                    judgments_by_id.get(item["memory_id"], {"accepted": False}),
                )
            adapter_results.append({
                "memory_id": item["memory_id"],
                "status": adapter.get("status", "reject"),
                "schema_errors": adapter.get("schema_errors", []),
                "reason": adapter.get("reason", ""),
                "raw_response": adapter.get("raw_response", ""),
                "source_principle": adapter.get("source_principle", item["content"]),
                "condition_card": adapter.get("condition_card", {}),
                "field_diff": {
                    "trigger_to_when_to_consider": {
                        "source": item["content"].get("trigger_condition", ""),
                        "card": (adapter.get("condition_card") or {}).get("when_to_consider", ""),
                    },
                    "required_evidence_to_evidence_found": {
                        "source": item["content"].get("required_evidence", []),
                        "card": (adapter.get("condition_card") or {}).get("evidence_found", []),
                    },
                    "exclusion_to_do_not_use_if": {
                        "source": item["content"].get("exclusion_condition", ""),
                        "card": (adapter.get("condition_card") or {}).get("do_not_use_if", ""),
                    },
                    "failure_mode_to_failure_to_avoid": {
                        "source": item["content"].get("failure_mode", ""),
                        "card": (adapter.get("condition_card") or {}).get("failure_to_avoid", ""),
                    },
                    "action_to_suggested_action": {
                        "source": item["content"].get("action_guidance", ""),
                        "card": (adapter.get("condition_card") or {}).get("suggested_action", ""),
                    },
                    "verification_to_verification": {
                        "source": item["content"].get("verification_step", ""),
                        "card": (adapter.get("condition_card") or {}).get("verification", ""),
                    },
                },
            })
            if adapter.get("status") == "use" and adapter.get("condition_card"):
                adapted.append({"memory_id": item["memory_id"], "card": adapter["condition_card"]})
        solver_text = "\n\n".join(
            render_solver_condition_card(item["card"], pos)
            for pos, item in enumerate(adapted, 1)
        )
        trace.update({
            "gate": gate,
            "selected_ids": [item["memory_id"] for item in adapted],
            "used": bool(adapted),
            "adapter_results": adapter_results,
            "condition_cards": adapted,
            "raw_principles": [
                {"memory_id": item["memory_id"], "content": item["content"]}
                for item in accepted
            ],
            # Only the adapter-produced condition card is rendered. Source
            # question, source answer, memory ID, retrieval score, and Gate
            # confidence stay in the trace and never enter the Solver prompt.
            "solver_text": solver_text,
        })
        return trace

    async def solve_rollout(item: Dict[str, Any], idx: int, seed: int, principle_text: str) -> Dict[str, Any]:
        question = str(item.get("question") or item.get("problem") or "")
        seed_pos = SEEDS.index(seed)
        first = (idx + seed_pos) % len(urls)
        attempt_urls = [urls[first], urls[(first + 1) % len(urls)]] if len(urls) > 1 else [urls[first]]
        trace = []
        for endpoint_attempt, endpoint in enumerate(attempt_urls, 1):
            with llm_request_context(f"stage3_{idx}_seed{seed}", f"solver/endpoint_attempt={endpoint_attempt}"):
                response = await execute.solve_v1(
                    question, principle_text=principle_text, seed=seed, url_override=endpoint,
                    reasoner_temperature=args.temperature, reasoner_timeout=300, finalizer_timeout=60,
                )
            trace.append({
                "endpoint_attempt": endpoint_attempt, "endpoint": endpoint,
                "success": bool(response.get("infrastructure_success")),
                "failed_stage": response.get("infrastructure_stage", ""),
            })
            if response.get("infrastructure_success"):
                return {
                    "seed": seed, "temperature": args.temperature, "status": "complete",
                    "correct": check_answer_for_item(str(response.get("composed_output", "")), item),
                    "response": response, "transport_trace": trace,
                    "solver_prompt": response.get("reasoner", {}).get("prompt", ""),
                    "experience_decision": experience_decision(
                        response.get("reasoner", {}).get("output", "")
                    ) if response.get("guided") else "baseline",
                }
        failure = {"idx": idx, "seed": seed, "variant": args.variant, "transport_trace": trace}
        transport_failures.append(failure)
        return {"seed": seed, "temperature": args.temperature, "status": "infrastructure_failed", "correct": False, "transport_trace": trace}

    async def solve_item(idx: int, item: Dict[str, Any]) -> None:
        if idx in completed:
            return
        async with semaphore:
            question = str(item.get("question") or item.get("problem") or "")
            routing = await retrieve_memory(question, idx)
            rollouts = []
            # Do not early-stop after a correct answer: these are three fixed,
            # independent samples of the same original prompt.
            for seed in SEEDS:
                rollouts.append(await solve_rollout(item, idx, seed, routing["solver_text"]))
        status = "complete" if all(r["status"] == "complete" for r in rollouts) else "infrastructure_failed"
        record = {
            "idx": idx, "question": question, "expected_answer": item.get("answer", ""),
            "difficulty_level": item.get("difficulty_level", ""), "variant": args.variant,
            "status": status, "retrieval": {k: v for k, v in routing.items() if k != "solver_text"},
            "rollouts": rollouts,
            "pass1_correct": bool(rollouts[0]["correct"]),
            "pass3_correct": any(bool(r["correct"]) for r in rollouts),
        }
        async with result_lock:
            completed[idx] = record
            atomic_save_json(result_path, [completed[key] for key in sorted(completed)])
            with transport_path.open("w", encoding="utf-8") as handle:
                for failure in transport_failures:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            logger.info("completed %d/%d idx=%d pass1=%s pass3=%s", len(completed), total_items, idx, record["pass1_correct"], record["pass3_correct"])

    await asyncio.gather(*(solve_item(idx, item) for idx, item in enumerate(test_data)))
    results = [completed[key] for key in sorted(completed)]
    seed_accuracy = {}
    for pos, seed in enumerate(SEEDS):
        correct = sum(bool(item["rollouts"][pos]["correct"]) for item in results)
        seed_accuracy[str(seed)] = {"correct": correct, "total": total_items, "accuracy": correct / total_items if total_items else 0.0}
    summary = {
        "variant": args.variant, "solver_protocol_version": SOLVER_V1_PROTOCOL,
        "prompt_hashes": SOLVER_V1_PROMPT_HASHES, "memory_snapshot_hash": memory_hash,
        "active_memory_count": active_memory_count, "total": total_items,
        "pass1_correct": sum(bool(item["pass1_correct"]) for item in results),
        "pass3_correct": sum(bool(item["pass3_correct"]) for item in results),
        "infrastructure_failure_count": sum(item.get("status") != "complete" for item in results),
        "retrieval_used_count": sum(bool((item.get("retrieval") or {}).get("used")) for item in results),
        "retrieval_candidate_count": sum(len((item.get("retrieval") or {}).get("candidate_ids", [])) for item in results),
        "application_gate_selected_count": sum(len((item.get("retrieval") or {}).get("raw_principles", [])) for item in results),
        "adapter_attempt_count": sum(len((item.get("retrieval") or {}).get("adapter_results", [])) for item in results),
        "adapter_accept_count": sum(len((item.get("retrieval") or {}).get("condition_cards", [])) for item in results),
        "solver_use_count": sum(
            rollout.get("experience_decision") == "use"
            for item in results for rollout in item.get("rollouts", [])
        ),
        "solver_ignore_count": sum(
            rollout.get("experience_decision") == "ignore"
            for item in results for rollout in item.get("rollouts", [])
        ),
        "solver_decision_missing_count": sum(
            rollout.get("experience_decision") == "missing"
            for item in results for rollout in item.get("rollouts", [])
        ),
        "seed_accuracy": seed_accuracy, "temperature": args.temperature, "seeds": list(SEEDS),
        "runtime_seconds": time.monotonic() - started,
    }
    summary["pass1_accuracy"] = summary["pass1_correct"] / total_items if total_items else 0.0
    summary["pass3_accuracy"] = summary["pass3_correct"] / total_items if total_items else 0.0
    atomic_save_json(output_dir / "eval_summary.json", summary)
    header = "\n".join([
        f"Variant: {args.variant}", f"Solver protocol: {SOLVER_V1_PROTOCOL}",
        f"Prompt hashes: {json.dumps(SOLVER_V1_PROMPT_HASHES, sort_keys=True)}",
        f"Memory snapshot/hash: {memory_hash}",
        f"Active Memory count: {active_memory_count}",
        f"Pass@1 correct / {total_items}: {summary['pass1_correct']} / {total_items}",
        f"Pass@1 accuracy: {summary['pass1_accuracy']:.6f}",
        f"Pass@3 correct / {total_items}: {summary['pass3_correct']} / {total_items}",
        f"Pass@3 accuracy: {summary['pass3_accuracy']:.6f}",
        f"Infrastructure failure count: {summary['infrastructure_failure_count']}",
        f"Retrieval-used count: {summary['retrieval_used_count']}",
        f"Gate-selected memory count: {summary['application_gate_selected_count']}",
        f"Adapter-accepted card count: {summary['adapter_accept_count']}",
        f"Solver use / ignore / missing: {summary['solver_use_count']} / {summary['solver_ignore_count']} / {summary['solver_decision_missing_count']}",
        f"Runtime: {summary['runtime_seconds']:.1f}s", "",
    ])
    for handler in logging.getLogger().handlers:
        handler.flush()
    raw_text = (output_dir / "raw.log").read_text(encoding="utf-8") if (output_dir / "raw.log").exists() else ""
    (output_dir / "run.log").write_text(header + raw_text, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
