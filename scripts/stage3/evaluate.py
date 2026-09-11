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
from modules.execute_module import (
    SOLVER_V1_FINALIZER_MAX_TOKENS,
    SOLVER_V1_FINALIZER_TEMPERATURE,
    SOLVER_V1_PROMPT_HASHES,
    SOLVER_V1_PROTOCOL,
    SOLVER_V1_REASONER_MAX_TOKENS,
    SOLVER_V1_TOP_P,
    _completion_tokens,
    _finalizer_format_valid,
    _finish_reason,
    _repetition_diagnostics,
)
from modules.memory_module import MemoryModule
from modules.plan_module import PlanModule
from modules.principle_schema import (
    principle_content_from_item,
    render_solver_condition_card,
)
from utils.io_utils import atomic_save_json
from template.prompt_template import (
    FORCED_TIMING_AFTER_DRAFT_PROMPT,
    SOLVER_V1_FINALIZER_PROMPT,
    SOLVER_V1_GUIDED_REASONER_PROMPT,
    SOLVER_V1_REASONER_PROMPT,
)
from utils.io_utils import load_json

logger = logging.getLogger("stage3.evaluate")
DEFAULT_SEEDS = (42, 43, 44)
AFTER_DRAFT_REASONER_MAX_TOKENS = 4096
AFTER_DRAFT_MAX_DRAFT_CHARS = 6000


def experience_decision(text: str) -> str:
    match = re.search(
        r"Experience Decision:\s*(use|ignore)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else "missing"


def after_draft_decision(text: str) -> str:
    match = re.search(
        r"Principle Processing:\s*(corrected_draft|verified_no_change)\b",
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
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Independent rollout seeds. Pass@3 always uses the first three; pass@K uses all supplied seeds.",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--gate-top-k", type=int, default=6)
    parser.add_argument("--max-injected", type=int, default=1)
    parser.add_argument(
        "--guided-protocol",
        choices=["pre_solve", "after_draft"],
        default="pre_solve",
        help="pre_solve is the legacy guided solver; after_draft matches Stage-2 Forced-C validation.",
    )
    parser.add_argument(
        "--baseline-cache-dir",
        default="",
        help="Optional Stage-3 baseline output dir used to reuse same-seed drafts for after_draft guided evaluation.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test prefix size; official runs leave this unset.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline_cache(cache_dir: str) -> Dict[tuple[int, int], Dict[str, Any]]:
    if not cache_dir:
        return {}
    result_path = Path(cache_dir) / "eval_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing baseline cache: {result_path}")
    data = load_json(result_path)
    if not isinstance(data, list):
        raise ValueError(f"Baseline cache must be a list: {result_path}")
    cache: Dict[tuple[int, int], Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("idx", -1))
        for rollout in item.get("rollouts", []):
            if not isinstance(rollout, dict) or rollout.get("status") != "complete":
                continue
            response = rollout.get("response") or {}
            draft = str((response.get("reasoner") or {}).get("output") or "").strip()
            if draft:
                cache[(idx, int(rollout.get("seed")))] = rollout
    return cache


def compact_after_draft_text(text: str, max_chars: int = AFTER_DRAFT_MAX_DRAFT_CHARS) -> str:
    cleaned = "\n".join(line.rstrip() for line in str(text).splitlines()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    half = max_chars // 2
    return "\n".join(
        [
            cleaned[:half].rstrip(),
            "[... middle of draft omitted to fit the model context ...]",
            cleaned[-half:].lstrip(),
        ]
    )


async def main() -> None:
    args = parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain one or more unique integers")
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
    baseline_cache = load_baseline_cache(args.baseline_cache_dir)
    atomic_save_json(output_dir / "prompt_snapshot.json", {
        "solver_protocol_version": SOLVER_V1_PROTOCOL,
        "guided_protocol": args.guided_protocol,
        "reasoner_base_template": SOLVER_V1_REASONER_PROMPT,
        "reasoner_guided_template": SOLVER_V1_GUIDED_REASONER_PROMPT,
        "after_draft_template": FORCED_TIMING_AFTER_DRAFT_PROMPT,
        "finalizer_template": SOLVER_V1_FINALIZER_PROMPT,
        "prompt_hashes": SOLVER_V1_PROMPT_HASHES,
        "after_draft_prompt_hash": hashlib.sha256(
            FORCED_TIMING_AFTER_DRAFT_PROMPT.strip().encode("utf-8")
        ).hexdigest(),
        "reasoner_max_tokens": 8192,
        "after_draft_reasoner_max_tokens": AFTER_DRAFT_REASONER_MAX_TOKENS,
        "after_draft_max_draft_chars": AFTER_DRAFT_MAX_DRAFT_CHARS,
        "finalizer_max_tokens": 128,
        "temperature": args.temperature,
        "seeds": list(seeds),
        "answer_dependent_retry": False,
        "baseline_cache_dir": args.baseline_cache_dir,
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
    expected_record_protocol = args.guided_protocol if guided else "baseline"
    completed: Dict[int, Dict[str, Any]] = {
        int(item["idx"]): item for item in old
        if isinstance(item, dict) and item.get("status") in {"complete", "infrastructure_failed"}
        and len(item.get("rollouts", [])) == len(seeds)
        and [int(row.get("seed", -1)) for row in item.get("rollouts", [])] == list(seeds)
        and item.get("guided_protocol", "baseline" if item.get("variant") == "baseline" else "pre_solve") == expected_record_protocol
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
        seed_pos = seeds.index(seed)
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

    async def baseline_rollout(item: Dict[str, Any], idx: int, seed: int) -> Dict[str, Any]:
        cached = baseline_cache.get((idx, seed))
        if cached:
            return {
                **cached,
                "baseline_cache_used": True,
                "baseline_cache_dir": args.baseline_cache_dir,
            }
        return await solve_rollout(item, idx, seed, "")

    async def solve_after_draft_rollout(
        item: Dict[str, Any],
        idx: int,
        seed: int,
        principle_text: str,
    ) -> Dict[str, Any]:
        baseline = await baseline_rollout(item, idx, seed)
        if baseline.get("status") != "complete":
            return {
                "seed": seed,
                "temperature": args.temperature,
                "status": "infrastructure_failed",
                "correct": False,
                "baseline_rollout": baseline,
                "transport_trace": baseline.get("transport_trace", []),
                "experience_decision": "baseline_failed",
            }
        if not principle_text.strip():
            return {
                **baseline,
                "status": "complete",
                "principle_injected": False,
                "after_draft_protocol": True,
                "experience_decision": "no_memory_reused_baseline",
                "baseline_rollout": baseline,
            }

        question = str(item.get("question") or item.get("problem") or "")
        previous_draft = str(
            ((baseline.get("response") or {}).get("reasoner") or {}).get("output") or ""
        ).strip()
        prompt = FORCED_TIMING_AFTER_DRAFT_PROMPT.format(
            question=question,
            previous_draft=compact_after_draft_text(previous_draft),
            principle_card=principle_text,
        )
        seed_pos = seeds.index(seed)
        first = (idx + seed_pos) % len(urls)
        attempt_urls = [urls[first], urls[(first + 1) % len(urls)]] if len(urls) > 1 else [urls[first]]
        trace = []
        for endpoint_attempt, endpoint in enumerate(attempt_urls, 1):
            with llm_request_context(
                f"stage3_{idx}_seed{seed}",
                f"after_draft_solver/endpoint_attempt={endpoint_attempt}",
            ):
                reasoner_meta, solver_url = await execute._call_with_router_metadata(
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=SOLVER_V1_TOP_P,
                    max_tokens=AFTER_DRAFT_REASONER_MAX_TOKENS,
                    repetition_penalty=1.0,
                    seed=seed,
                    url_override=endpoint,
                    timeout_override=300,
                )
            reasoner_output = str(reasoner_meta.get("text") or "").strip()
            if not reasoner_output:
                trace.append({
                    "endpoint_attempt": endpoint_attempt,
                    "endpoint": endpoint,
                    "stage": "after_draft_reasoner",
                    "success": False,
                })
                continue
            finalizer_prompt = SOLVER_V1_FINALIZER_PROMPT.format(
                question=question,
                reasoner_output=reasoner_output,
            )
            with llm_request_context(
                f"stage3_{idx}_seed{seed}",
                f"after_draft_finalizer/endpoint_attempt={endpoint_attempt}",
            ):
                finalizer_meta, finalizer_url = await execute._call_with_router_metadata(
                    prompt=finalizer_prompt,
                    temperature=SOLVER_V1_FINALIZER_TEMPERATURE,
                    top_p=SOLVER_V1_TOP_P,
                    max_tokens=SOLVER_V1_FINALIZER_MAX_TOKENS,
                    repetition_penalty=1.0,
                    seed=seed,
                    url_override=solver_url or endpoint,
                    timeout_override=60,
                )
            finalizer_output = str(finalizer_meta.get("text") or "").strip()
            success = bool(finalizer_output)
            trace.append({
                "endpoint_attempt": endpoint_attempt,
                "endpoint": endpoint,
                "stage": "after_draft_transaction",
                "reasoner_success": True,
                "finalizer_success": success,
            })
            if success:
                composed = f"{reasoner_output}\n\n{finalizer_output}".strip()
                response = {
                    "protocol_version": SOLVER_V1_PROTOCOL,
                    "guided": True,
                    "guided_protocol": "after_draft",
                    "seed": seed,
                    "solver_url": solver_url,
                    "finalizer_url": finalizer_url,
                    "prompt_hashes": {
                        "reasoner": hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest(),
                        "finalizer": SOLVER_V1_PROMPT_HASHES["finalizer"],
                    },
                    "reasoner_repetition_penalty": 1.0,
                    "reasoner_temperature": args.temperature,
                    "reasoner": {
                        "prompt": prompt,
                        "output": reasoner_output,
                        "finish_reason": _finish_reason(reasoner_meta),
                        "completion_tokens": _completion_tokens(reasoner_meta),
                        "hit_token_limit": _finish_reason(reasoner_meta) == "length",
                        "repetition": _repetition_diagnostics(reasoner_output),
                        "usage": reasoner_meta.get("usage"),
                    },
                    "finalizer": {
                        "prompt": finalizer_prompt,
                        "output": finalizer_output,
                        "finish_reason": _finish_reason(finalizer_meta),
                        "completion_tokens": _completion_tokens(finalizer_meta),
                        "format_valid": _finalizer_format_valid(finalizer_output),
                        "usage": finalizer_meta.get("usage"),
                    },
                    "final_answer": finalizer_output,
                    "composed_output": composed,
                    "infrastructure_success": True,
                }
                return {
                    "seed": seed,
                    "temperature": args.temperature,
                    "status": "complete",
                    "correct": check_answer_for_item(composed, item),
                    "response": response,
                    "transport_trace": trace,
                    "solver_prompt": prompt,
                    "principle_injected": True,
                    "baseline_rollout": baseline,
                    "experience_decision": after_draft_decision(reasoner_output),
                }
        failure = {"idx": idx, "seed": seed, "variant": args.variant, "guided_protocol": "after_draft", "transport_trace": trace}
        transport_failures.append(failure)
        return {
            "seed": seed,
            "temperature": args.temperature,
            "status": "infrastructure_failed",
            "correct": False,
            "transport_trace": trace,
            "baseline_rollout": baseline,
            "experience_decision": "infrastructure_failed",
        }

    async def solve_item(idx: int, item: Dict[str, Any]) -> None:
        if idx in completed:
            return
        async with semaphore:
            question = str(item.get("question") or item.get("problem") or "")
            routing = await retrieve_memory(question, idx)
            rollouts = []
            # Do not early-stop after a correct answer: these are fixed,
            # independent samples of the same original prompt.
            for seed in seeds:
                if guided and args.guided_protocol == "after_draft":
                    rollouts.append(await solve_after_draft_rollout(
                        item, idx, seed, routing["solver_text"],
                    ))
                else:
                    rollouts.append(await solve_rollout(item, idx, seed, routing["solver_text"]))
        status = "complete" if all(r["status"] == "complete" for r in rollouts) else "infrastructure_failed"
        record = {
            "idx": idx, "question": question, "expected_answer": item.get("answer", ""),
            "difficulty_level": item.get("difficulty_level", ""), "variant": args.variant,
            "guided_protocol": expected_record_protocol,
            "status": status, "retrieval": {k: v for k, v in routing.items() if k != "solver_text"},
            "rollouts": rollouts,
            "pass1_correct": bool(rollouts[0]["correct"]),
            "pass3_correct": any(bool(r["correct"]) for r in rollouts[:3]),
            "passk_correct": any(bool(r["correct"]) for r in rollouts),
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
    for pos, seed in enumerate(seeds):
        correct = sum(bool(item["rollouts"][pos]["correct"]) for item in results)
        seed_accuracy[str(seed)] = {"correct": correct, "total": total_items, "accuracy": correct / total_items if total_items else 0.0}
    summary = {
        "variant": args.variant, "solver_protocol_version": SOLVER_V1_PROTOCOL,
        "guided_protocol": args.guided_protocol if guided else "baseline",
        "prompt_hashes": SOLVER_V1_PROMPT_HASHES, "memory_snapshot_hash": memory_hash,
        "active_memory_count": active_memory_count, "total": total_items,
        "pass1_correct": sum(bool(item["pass1_correct"]) for item in results),
        "pass3_correct": sum(bool(item["pass3_correct"]) for item in results),
        "pass_k": len(seeds),
        "passk_correct": sum(bool(item["passk_correct"]) for item in results),
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
        "after_draft_corrected_count": sum(
            rollout.get("experience_decision") == "corrected_draft"
            for item in results for rollout in item.get("rollouts", [])
        ),
        "after_draft_verified_no_change_count": sum(
            rollout.get("experience_decision") == "verified_no_change"
            for item in results for rollout in item.get("rollouts", [])
        ),
        "after_draft_no_memory_reused_baseline_count": sum(
            rollout.get("experience_decision") == "no_memory_reused_baseline"
            for item in results for rollout in item.get("rollouts", [])
        ),
        "seed_accuracy": seed_accuracy, "temperature": args.temperature, "seeds": list(seeds),
        "runtime_seconds": time.monotonic() - started,
    }
    summary["pass1_accuracy"] = summary["pass1_correct"] / total_items if total_items else 0.0
    summary["pass3_accuracy"] = summary["pass3_correct"] / total_items if total_items else 0.0
    summary["passk_accuracy"] = summary["passk_correct"] / total_items if total_items else 0.0
    summary["seed_mean_accuracy"] = (
        sum(row["correct"] for row in seed_accuracy.values()) / (total_items * len(seeds))
        if total_items and seeds else 0.0
    )
    atomic_save_json(output_dir / "eval_summary.json", summary)
    header = "\n".join([
        f"Variant: {args.variant}", f"Solver protocol: {SOLVER_V1_PROTOCOL}",
        f"Guided protocol: {summary['guided_protocol']}",
        f"Prompt hashes: {json.dumps(SOLVER_V1_PROMPT_HASHES, sort_keys=True)}",
        f"Memory snapshot/hash: {memory_hash}",
        f"Active Memory count: {active_memory_count}",
        f"Pass@1 correct / {total_items}: {summary['pass1_correct']} / {total_items}",
        f"Pass@1 accuracy: {summary['pass1_accuracy']:.6f}",
        f"Pass@3 correct / {total_items}: {summary['pass3_correct']} / {total_items}",
        f"Pass@3 accuracy: {summary['pass3_accuracy']:.6f}",
        f"Pass@{summary['pass_k']} correct / {total_items}: {summary['passk_correct']} / {total_items}",
        f"Pass@{summary['pass_k']} accuracy: {summary['passk_accuracy']:.6f}",
        f"Seed-mean accuracy: {summary['seed_mean_accuracy']:.6f}",
        f"Infrastructure failure count: {summary['infrastructure_failure_count']}",
        f"Retrieval-used count: {summary['retrieval_used_count']}",
        f"Gate-selected memory count: {summary['application_gate_selected_count']}",
        f"Adapter-accepted card count: {summary['adapter_accept_count']}",
        f"Solver use / ignore / missing: {summary['solver_use_count']} / {summary['solver_ignore_count']} / {summary['solver_decision_missing_count']}",
        f"After-draft corrected / verified / no-memory-reused-baseline: {summary['after_draft_corrected_count']} / {summary['after_draft_verified_no_change_count']} / {summary['after_draft_no_memory_reused_baseline_count']}",
        f"Runtime: {summary['runtime_seconds']:.1f}s", "",
    ])
    for handler in logging.getLogger().handlers:
        handler.flush()
    raw_text = (output_dir / "raw.log").read_text(encoding="utf-8") if (output_dir / "raw.log").exists() else ""
    (output_dir / "run.log").write_text(header + raw_text, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
