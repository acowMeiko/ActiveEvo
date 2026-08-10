"""Stage 2 v3: causal validation and versioned memory construction."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from core.field_names import CONDITION_FIELD, CORE_FIELD, TARGET_FIELD
from core.request_context import llm_request_context
from modules.dsd_module import DSDRegistry, canonicalize_stage1_packets, stage1_packet_is_candidate
from modules.principle_schema import SCHEMA_VERSION, build_memory_views, render_solver_memory, validate_principle_content
from modules.stage2_validation import (
    Stage2Checkpoint,
    admission_decision,
    atomic_save_json,
    classify_pair,
    deterministic_leakage_check,
    stable_dev_holdout_split,
    summarize_pairs,
)
from output_paths import stage1_learning_packets_file, stage1_level_packets_file, stage_output_dir
from template.prompt_template import (
    SELF_CPE_PROMPT,
    STAGE2_NEIGHBOR_GATE_PROMPT,
    STAGE2_PRINCIPLE_QUALITY_PROMPT,
    STAGE2_PRINCIPLE_QUALITY_REPAIR_PROMPT,
)
from utils.io_utils import load_json
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_id(dataset: str, question_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}:{question_id}".encode("utf-8")).hexdigest()[:16]
    return f"candidate_{digest}"


def _source_context(source_dsd: Mapping[str, Any]) -> str:
    return (
        f"Core Skill: {source_dsd.get(CORE_FIELD, '')}. "
        f"Constraints: {source_dsd.get(CONDITION_FIELD, '')}. "
        f"Objective: {source_dsd.get(TARGET_FIELD, '')}."
    )


def _aggregate_outcomes(units: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    for unit in units:
        pairs.extend(unit.get("dev_pairs", []) if isinstance(unit.get("dev_pairs"), list) else [])
        pairs.extend(unit.get("holdout_pairs", []) if isinstance(unit.get("holdout_pairs"), list) else [])
    return summarize_pairs(pairs)


def _estimate_model_calls(unit: Mapping[str, Any]) -> int:
    """Count auditable protocol calls from persisted trace structure."""
    calls = 0
    if unit.get("content"):
        calls += 1  # Self-CPE
    if ((unit.get("generation_result") or {}).get("generation_trace") or {}).get("repair_used"):
        calls += 1
    if unit.get("diff_source") == "stage2_generated":
        calls += 1
    if unit.get("quality_check"):
        calls += 1
    calls += 2 * len(unit.get("quality_repair_history", []) or [])
    if unit.get("gate_result"):
        calls += 1
    repair_history = unit.get("repair_history", []) or []
    calls += 3 * len(repair_history)  # diagnosis + patch + post-patch audit
    successful_reruns = sum(1 for item in repair_history if (item.get("post_repair_quality") or {}).get("quality_pass"))
    dev_pairs = unit.get("dev_pairs", []) or []
    holdout_pairs = unit.get("holdout_pairs", []) or []
    # Each pair executes baseline/guided; each Solver-v1 run has reasoner+finalizer calls.
    calls += 4 * (len(dev_pairs) * (1 + successful_reruns) + len(holdout_pairs))
    return calls


async def _bounded_as_completed(
    items: Sequence[Dict[str, Any]],
    limit: int,
    factory: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> AsyncIterator[Tuple[Dict[str, Any], asyncio.Task]]:
    """Yield completed work while keeping at most ``limit`` items in flight."""
    iterator = iter(items)
    active: Dict[asyncio.Task, Dict[str, Any]] = {}

    def fill_one() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        active[asyncio.create_task(factory(item))] = item
        return True

    for _ in range(max(1, min(int(limit), len(items)))):
        fill_one()
    while active:
        done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
        completed_batch = [(active.pop(task), task) for task in done]
        for _ in completed_batch:
            fill_one()
        for item, task in completed_batch:
            yield item, task


async def _paired_validate(
    pipeline: Any,
    neighbors: Sequence[Dict[str, Any]],
    principle: Dict[str, str],
    *,
    seed: int,
    semaphore: asyncio.Semaphore,
    candidate_id: str = "-",
    split: str = "validation",
) -> List[Dict[str, Any]]:
    solver_text = render_solver_memory(principle)

    async def run_one(neighbor: Dict[str, Any]) -> Dict[str, Any]:
        question = str(neighbor.get("question_text", ""))
        expected = str(neighbor.get("expected_answer", ""))
        neighbor_id = str(neighbor.get("question_id", ""))
        urls = list(pipeline.execute.urls)
        start = int(hashlib.sha256(f"{candidate_id}:{neighbor_id}:{seed}".encode()).hexdigest(), 16) % max(1, len(urls))
        attempt_urls = [urls[(start + offset) % len(urls)] for offset in range(min(2, len(urls)))]
        transport_trace: List[Dict[str, Any]] = []
        baseline: Dict[str, Any] = {}
        injected: Dict[str, Any] = {}
        async with semaphore:
            for endpoint_attempt, endpoint in enumerate(attempt_urls, 1):
                with llm_request_context(candidate_id, f"{split}_baseline/neighbor={neighbor_id}/endpoint_attempt={endpoint_attempt}"):
                    baseline = await pipeline.execute.solve_v1(question, seed=seed, url_override=endpoint)
                if baseline.get("infrastructure_success"):
                    with llm_request_context(candidate_id, f"{split}_injected/neighbor={neighbor_id}/endpoint_attempt={endpoint_attempt}"):
                        injected = await pipeline.execute.solve_v1(
                            question, principle_text=solver_text, seed=seed, url_override=endpoint,
                        )
                transport_trace.append({
                    "endpoint_attempt": endpoint_attempt, "endpoint": endpoint,
                    "baseline_success": bool(baseline.get("infrastructure_success")),
                    "injected_success": bool(injected.get("infrastructure_success")),
                })
                if baseline.get("infrastructure_success") and injected.get("infrastructure_success"):
                    break
                # Discard the entire pair and rerun both branches elsewhere.
                baseline, injected = {}, {}
        if not baseline or not injected:
            return {
                "question_id": neighbor_id, "question_text": question, "expected_answer": expected,
                "baseline_correct": None, "injected_correct": None,
                "outcome": "infrastructure_failed", "seed": seed,
                "solver_view_text": solver_text, "transport_trace": transport_trace,
            }
        baseline_output = str(baseline.get("composed_output", ""))
        injected_output = str(injected.get("composed_output", ""))
        baseline_correct = pipeline.execute.check_answer(baseline_output, expected)
        injected_correct = pipeline.execute.check_answer(injected_output, expected)
        return {
            "question_id": str(neighbor.get("question_id", "")),
            "question_text": question,
            "expected_answer": expected,
            "baseline_correct": baseline_correct,
            "injected_correct": injected_correct,
            "outcome": classify_pair(baseline_correct, injected_correct),
            "solver_protocol_version": baseline.get("protocol_version", "solver_v1"),
            "seed": seed,
            "solver_view_text": solver_text,
            "baseline": baseline,
            "injected": injected,
            "transport_trace": transport_trace,
        }

    return list(await asyncio.gather(*(run_one(item) for item in neighbors)))


async def _quality_check(
    pipeline: Any,
    record: Dict[str, Any],
    diff_list: List[Dict[str, Any]],
    principle: Dict[str, str],
) -> Dict[str, Any]:
    deterministic = deterministic_leakage_check(
        principle,
        source_question=str(record.get("question_text", "")),
        expected_answer=str(record.get("expected_answer", "")),
    )
    llm = await pipeline.plan.audit_stage2_principle(
        question=str(record.get("question_text", "")),
        chosen_trajectory=str(record.get("chosen_trajectory", "")),
        rejected_trajectory=str(record.get("hard_negative_trajectory", "")),
        diff_list=diff_list,
        principle=principle,
    )
    # Hard rejection is based on literal, auditable leakage.  The semantic LLM
    # flag remains diagnostic: earlier audits confused a reusable operation
    # learned from the correct trajectory with copying the source answer.
    leakage = bool(deterministic.get("detected"))
    protocol_and_content_pass = bool(
        llm.get("protocol_compliant")
        and llm.get("source_fidelity_pass")
        and llm.get("trigger_decidable")
        and llm.get("required_evidence_decidable")
        and llm.get("exclusion_decidable")
    )
    quality_pass = bool(protocol_and_content_pass and not leakage)
    return {
        "quality_pass": quality_pass,
        "source_fidelity_pass": bool(llm.get("source_fidelity_pass")),
        "answer_leakage_detected": leakage,
        "llm_leakage_suspected": bool(llm.get("answer_leakage_detected")),
        "leakage_decision_policy": "deterministic_literal_leakage_is_authoritative",
        "deterministic_leakage": deterministic,
        "llm_audit": llm,
    }


async def _process_candidate(
    pipeline: Any,
    registry: DSDRegistry,
    record: Dict[str, Any],
    *,
    dataset: str,
    round_id: int,
    seed: int,
    generation_semaphore: asyncio.Semaphore,
    solver_semaphore: asyncio.Semaphore,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    prepared_candidate: Optional[Dict[str, Any]] = None,
    prepare_only: bool = False,
    neighbor_validation: bool = True,
    gate_only: bool = False,
) -> Dict[str, Any]:
    candidate_id = _candidate_id(dataset, str(record.get("question_id", "")))
    request_id = f"{candidate_id}/q={record.get('question_id', '')}"

    def mark(stage: str) -> None:
        logger.debug("Stage 2 candidate transition: candidate=%s stage=%s", request_id, stage)
        if progress_callback:
            progress_callback(str(record.get("question_id", "")), stage)
    base: Dict[str, Any] = {
        "candidate_id": candidate_id,
        "question_id": str(record.get("question_id", "")),
        "source_question_text": str(record.get("question_text", "")),
        "source_dsd": record.get("source_dsd", {}),
        "source_round": round_id,
        "source_dataset": dataset,
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
    }
    if prepared_candidate is not None:
        frozen = dict(prepared_candidate)
        if str(frozen.get("question_id", "")) != base["question_id"]:
            return {**base, "status": "rejected", "admission": {"reasons": ["candidate_bank_id_mismatch"]}}
        if frozen.get("status") != "candidate_ready":
            return frozen
        base.update(frozen)
        base["status"] = "pending"
        principle = dict(base.get("content") or {})
        diff_list = list(base.get("diff") or [])
        quality = dict(base.get("quality_check") or {})
    else:
        diff_list = (record.get("contrast") or {}).get("differences", [])
        if not isinstance(diff_list, list) or not diff_list:
            mark("diff_analysis")
            with llm_request_context(request_id, "diff_analysis"):
                async with generation_semaphore:
                    diff = await pipeline.plan.diff_analysis(
                        str(record.get("question_text", "")), str(record.get("chosen_trajectory", "")),
                        str(record.get("hard_negative_trajectory", "")), record.get("source_dsd", {}),
                        str(record.get("divergence_step_text", "")), int(record.get("divergence_step_idx", 0) or 0),
                    )
            diff_list = diff.get("differences", []) if diff.get("status") == "success" else []
            base["diff_source"] = "stage2_generated"
        else:
            base["diff_source"] = "stage1_reused"
        base["diff"] = diff_list
        if not diff_list:
            return {**base, "status": "rejected", "admission": {"reasons": ["no_valid_diff"]}}

        mark("principle_generation")
        with llm_request_context(request_id, "principle_generation"):
            async with generation_semaphore:
                generated = await pipeline.plan.self_cpe_extract(
                    question=base["source_question_text"], diff_list=diff_list, source_dsd=base["source_dsd"]
                )
        base["generation_result"] = generated
        if generated.get("status") != "success":
            return {**base, "status": "rejected", "generation_result": generated,
                    "admission": {"reasons": ["principle_generation_failed"]}}
        principle = generated["principle"]
        _, schema_errors = validate_principle_content(principle, require_extended=True)
        base["schema_check"] = {"valid": not schema_errors, "errors": schema_errors}
        if schema_errors:
            return {**base, "status": "rejected", "content": principle, "admission": {"reasons": ["schema_invalid"]}}

        mark("principle_quality_audit")
        with llm_request_context(request_id, "principle_quality_audit"):
            async with generation_semaphore:
                quality = await _quality_check(pipeline, record, diff_list, principle)
        quality_repairs: List[Dict[str, Any]] = []
        if not quality["quality_pass"]:
            mark("principle_quality_repair")
            with llm_request_context(request_id, "principle_quality_repair"):
                async with generation_semaphore:
                    repaired = await pipeline.plan.repair_stage2_principle_quality(
                        question=base["source_question_text"], source_dsd=base["source_dsd"],
                        diff_list=diff_list, principle=principle, audit=quality,
                    )
            quality_repairs.append({"pre_repair_audit": quality, "repair_result": repaired})
            if repaired.get("status") == "success":
                principle = repaired["principle"]
                mark("principle_quality_reaudit")
                with llm_request_context(request_id, "principle_quality_reaudit"):
                    async with generation_semaphore:
                        quality = await _quality_check(pipeline, record, diff_list, principle)
        base["quality_check"] = quality
        base["quality_repair_history"] = quality_repairs
        if not quality["quality_pass"]:
            return {**base, "status": "rejected", "content": principle,
                    "admission": {"reasons": ["principle_quality_failed"]}}

    common_views = build_memory_views(
        principle, source_dsd=base["source_dsd"],
        validation={"neighbor_validation": None},
        audit={"source_question_id": base["question_id"], "diff": diff_list, "quality_check": quality,
               "principle_generation": base.get("generation_result", {})},
    )
    if prepare_only:
        return {**base, **common_views, "content": principle, "status": "candidate_ready",
                "qualified_for_memory": False, "admission": {"reasons": ["frozen_candidate"]}}
    if not neighbor_validation:
        return {
            **base, **common_views, "content": principle,
            "principle": principle["action_guidance"], "principle_text": principle["action_guidance"],
            "status": "active", "qualified_for_memory": True,
            "neighbor_validation": False, "operation": "add",
            "admission": {"status": "active", "qualified_for_memory": True, "reasons": []},
        }

    candidate_neighbors = registry.get_neighbors(base["question_id"], top_k=12)
    routing_view = build_memory_views(principle, source_dsd=base["source_dsd"])["routing_view"]
    base["neighbor_candidates"] = candidate_neighbors
    minimum_candidate_neighbors = 1 if gate_only else 4
    if len(candidate_neighbors) < minimum_candidate_neighbors:
        views = build_memory_views(
            principle,
            source_dsd=base["source_dsd"],
            validation={"candidate_neighbors": candidate_neighbors},
            audit={"source_question_id": base["question_id"], "diff": diff_list, "quality_check": quality,
                   "principle_generation": base.get("generation_result", {})},
        )
        return {
            **base,
            **views,
            "content": principle,
            "status": "provisional",
            "admission": {"reasons": ["insufficient_validation_evidence"]},
        }
    mark("application_gate")
    with llm_request_context(request_id, "application_gate"):
        async with generation_semaphore:
            gate = await pipeline.plan.judge_stage2_neighbors(
                source_question=base["source_question_text"],
                source_dsd=base["source_dsd"],
                routing_view=routing_view,
                neighbors=candidate_neighbors,
            )
    base["gate_result"] = gate
    if not gate.get("protocol_compliant"):
        return {
            **base,
            "status": "rejected",
            "content": principle,
            "admission": {"reasons": ["neighbor_gate_protocol_failed"]},
        }
    selected = set(gate.get("selected_ids", []))
    accepted = [item for item in candidate_neighbors if str(item.get("question_id", "")) in selected][:6]
    base["accepted_neighbors"] = accepted
    if gate_only:
        views = build_memory_views(
            principle,
            source_dsd=base["source_dsd"],
            validation={"gate_result": gate, "accepted_neighbors": accepted, "validation_mode": "gate-only"},
            audit={"source_question_id": base["question_id"], "diff": diff_list, "quality_check": quality,
                   "principle_generation": base.get("generation_result", {})},
        )
        accepted_for_use = len(accepted) >= 1
        return {
            **base, **views, "content": principle,
            "principle": principle["action_guidance"], "principle_text": principle["action_guidance"],
            "status": "active" if accepted_for_use else "provisional",
            "qualified_for_memory": accepted_for_use,
            "neighbor_validation": "gate-only", "operation": "add",
            "admission": {
                "status": "active" if accepted_for_use else "provisional",
                "qualified_for_memory": accepted_for_use,
                "reasons": [] if accepted_for_use else ["no_applicable_neighbor"],
            },
        }
    if len(accepted) < 4:
        views = build_memory_views(
            principle,
            source_dsd=base["source_dsd"],
            validation={"gate_result": gate, "accepted_neighbors": accepted},
            audit={"source_question_id": base["question_id"], "diff": diff_list, "quality_check": quality,
                   "principle_generation": base.get("generation_result", {})},
        )
        return {
            **base,
            **views,
            "content": principle,
            "status": "provisional",
            "admission": {"reasons": ["insufficient_validation_evidence"]},
        }

    dev, holdout = stable_dev_holdout_split(accepted, candidate_id=candidate_id, seed=seed)
    repair_history: List[Dict[str, Any]] = []
    mark("dev_paired_validation")
    dev_pairs = await _paired_validate(
        pipeline, dev, principle, seed=seed, semaphore=solver_semaphore,
        candidate_id=request_id, split="dev",
    )
    dev_summary = summarize_pairs(dev_pairs)

    for depth in range(1, 3):
        failures = [pair for pair in dev_pairs if pair["outcome"] in {"loss", "both_wrong"}]
        if not failures or (dev_summary["gain_count"] >= 1 and dev_summary["loss_count"] == 0):
            break
        diagnosis_cases = [
            {
                "question_text": pair["question_text"],
                "expected_answer": pair["expected_answer"],
                "response": pair["injected"].get("composed_output", "")[:1200],
                "outcome": pair["outcome"],
            }
            for pair in failures
        ]
        mark(f"dev_repair_round_{depth}")
        with llm_request_context(request_id, f"dev_repair_round_{depth}"):
            async with generation_semaphore:
                diagnosis = await pipeline.plan.repair_diagnose(
                    principle=principle,
                    failed_validations=diagnosis_cases,
                    source_context=_source_context(base["source_dsd"]),
                )
                patch = await pipeline.plan.repair_patch(
                    principle=principle,
                    diagnosis_result=diagnosis,
                    failed_example=diagnosis_cases[0],
                    source_dsd=base["source_dsd"],
                )
        history_item = {"depth": depth, "dev_summary_before": dev_summary, "diagnosis": diagnosis, "patch": patch}
        if patch.get("status") != "success" or patch.get("principle", {}).get("repair_status") == "unfixable":
            repair_history.append(history_item)
            break
        repaired_principle = dict(patch["principle"])
        repaired_principle.pop("repair_status", None)
        with llm_request_context(request_id, f"dev_repair_quality_round_{depth}"):
            async with generation_semaphore:
                post_quality = await _quality_check(pipeline, record, diff_list, repaired_principle)
        history_item["post_repair_quality"] = post_quality
        repair_history.append(history_item)
        if not post_quality["quality_pass"]:
            break
        principle = repaired_principle
        quality = post_quality
        dev_pairs = await _paired_validate(
            pipeline, dev, principle, seed=seed, semaphore=solver_semaphore,
            candidate_id=request_id, split=f"dev_rerun_{depth}",
        )
        dev_summary = summarize_pairs(dev_pairs)

    # Holdout is touched exactly once, after all repair decisions are frozen.
    mark("holdout_paired_validation")
    holdout_pairs = await _paired_validate(
        pipeline, holdout, principle, seed=seed, semaphore=solver_semaphore,
        candidate_id=request_id, split="holdout",
    )
    holdout_summary = summarize_pairs(holdout_pairs)
    admission = admission_decision(
        schema_valid=True,
        source_fidelity_pass=quality["source_fidelity_pass"],
        answer_leakage_detected=quality["answer_leakage_detected"],
        accepted_neighbors=len(accepted),
        holdout_summary=holdout_summary,
    )
    validation_view = {
        "neighbor_selection": {"candidate_count": len(candidate_neighbors), "accepted_count": len(accepted)},
        "application_gate": gate,
        "dev_neighbor_ids": [item.get("question_id") for item in dev],
        "holdout_neighbor_ids": [item.get("question_id") for item in holdout],
        "dev_pairs": dev_pairs,
        "dev_summary": dev_summary,
        "holdout_pairs": holdout_pairs,
        "holdout_summary": holdout_summary,
    }
    audit_view = {
        "source_question_id": base["question_id"],
        "source_question": base["source_question_text"],
        "source_dsd": base["source_dsd"],
        "diff": diff_list,
        "quality_check": quality,
        "principle_generation": base.get("generation_result", {}),
        "repair_history": repair_history,
        "solver_protocol_version": "solver_v1",
        "validation_seed": seed,
    }
    views = build_memory_views(
        principle,
        source_dsd=base["source_dsd"],
        validation=validation_view,
        audit=audit_view,
    )
    return {
        **base,
        **views,
        "content": principle,
        "principle": principle["action_guidance"],
        "principle_text": principle["action_guidance"],
        "quality_check": quality,
        "repair_history": repair_history,
        "dev_pairs": dev_pairs,
        "dev_summary": dev_summary,
        "holdout_pairs": holdout_pairs,
        "holdout_summary": holdout_summary,
        "status": admission["status"],
        "qualified_for_memory": admission["qualified_for_memory"],
        "admission": admission,
        "neighbor_cases": holdout_pairs,
        "validated_neighbors": holdout_pairs,
        "neighbor_pass_count": holdout_summary["gain_count"] + holdout_summary["both_correct_count"],
        "neighbor_total_count": holdout_summary["pair_count"],
        "neighbor_pass_ratio": holdout_summary["injected_accuracy"],
        "neighbor_gate_pass": True,
        "rank_score": holdout_summary["net_gain"],
        "operation": "repair" if repair_history else "add",
    }


async def run_stage2_v3(
    pipeline: Any,
    *,
    dataset: str,
    round_id: int,
    validation_concurrency: int = 8,
    backbone: str = "",
    max_records: int = 0,
    run_tag: str = "",
    dsd_cache_dir: str = "",
    rebuild_dsd: bool = False,
    prepare_dsd_only: bool = False,
    prepare_candidates_only: bool = False,
    candidate_bank: str = "",
    neighbor_validation: bool = True,
    gate_only: bool = False,
    selection_file: str = "",
    artifact_dir: str = "",
    candidate_concurrency: int = 0,
) -> Dict[str, Any]:
    started = time.monotonic()
    output_root = pipeline.config.paths.output_dir
    level_digits = "".join(ch for ch in str(dataset) if ch.isdigit())
    level = int(level_digits) if level_digits else round_id
    packet_new = stage1_level_packets_file(output_root, level, backbone) if backbone else None
    packet_old = stage1_learning_packets_file(output_root, dataset)
    packet_path = packet_new if packet_new and packet_new.exists() else packet_old
    source_packets = load_json(str(packet_path)) if packet_path.exists() else []
    if not isinstance(source_packets, list) or not source_packets:
        return {"dataset": dataset, "round_id": round_id, "status": "empty_stage1"}

    seed = int(os.environ.get("MEMUA_STAGE2_VALIDATION_SEED", "42"))
    selection_ids = set()
    selection_expected_count = 0
    selection_hash = ""
    if selection_file:
        selection_path = Path(selection_file)
        if not selection_path.exists():
            raise ValueError(f"Stage-2 selection file does not exist: {selection_file}")
        selection_payload = load_json(str(selection_path))
        if isinstance(selection_payload, dict):
            selection_expected_count = int(selection_payload.get("expected_count", 0) or 0)
            selection_payload = selection_payload.get("question_ids", [])
        if not isinstance(selection_payload, list):
            raise ValueError(f"Stage-2 selection file must contain a question_id list: {selection_file}")
        if len(selection_payload) != len({str(value) for value in selection_payload}):
            raise ValueError(f"Stage-2 selection file contains duplicate question_id values: {selection_file}")
        selection_ids = {str(value) for value in selection_payload}
        selection_expected_count = selection_expected_count or len(selection_ids)
        if selection_expected_count != len(selection_ids):
            raise ValueError(
                f"Stage-2 selection expected {selection_expected_count} unique IDs, got {len(selection_ids)}: {selection_file}"
            )
        selection_hash = _sha256_bytes(selection_path.read_bytes())
    packets = canonicalize_stage1_packets(source_packets, sorted(selection_ids) if selection_ids else None)
    if selection_ids:
        canonical_ids = {str(packet.get("question_id", "")) for packet in packets}
        missing_ids = sorted(selection_ids - canonical_ids)
        eligible_ids = {
            str(packet.get("question_id", "")) for packet in packets if stage1_packet_is_candidate(packet)
        }
        ineligible_ids = sorted(selection_ids - eligible_ids)
        if missing_ids or ineligible_ids or len(packets) != selection_expected_count:
            raise ValueError(
                "Stage-2 Pilot selection did not resolve to the requested unique eligible packets: "
                f"expected={selection_expected_count} canonical={len(packets)} "
                f"missing={missing_ids[:5]} ineligible={ineligible_ids[:5]}"
            )
    input_hash = _sha256_bytes(
        json.dumps(packets, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    )
    config_payload = {
        "schema_version": SCHEMA_VERSION,
        "neighbor_candidate_k": 12,
        "neighbor_accept_k": 6,
        "minimum_neighbors": 4,
        "validation_seed": seed,
        "solver_protocol": "solver_v1",
        "solver_max_tokens": 8192,
        "model": pipeline.execute.model,
        "max_records": max(0, int(max_records or 0)),
        "run_tag": str(run_tag or ""),
        "prepare_candidates_only": bool(prepare_candidates_only),
        "neighbor_validation": bool(neighbor_validation),
        "gate_only": bool(gate_only),
        "candidate_bank": str(candidate_bank or ""),
        "selection_file": str(selection_file or ""),
        "selection_hash": selection_hash,
    }
    prompt_hash = _sha256_bytes(
        "\n".join(
            [
                SELF_CPE_PROMPT,
                STAGE2_PRINCIPLE_QUALITY_PROMPT,
                STAGE2_PRINCIPLE_QUALITY_REPAIR_PROMPT,
                STAGE2_NEIGHBOR_GATE_PROMPT,
            ]
        ).encode("utf-8")
    )
    config_payload["prompt_hash"] = prompt_hash
    config_hash = _sha256_bytes(json.dumps(config_payload, sort_keys=True).encode("utf-8"))
    run_id = f"{dataset}_r{round_id}_{input_hash[:8]}_{config_hash[:8]}"
    artifact_root = Path(artifact_dir) if artifact_dir else stage_output_dir(output_root, "stage2", dataset, "")
    safe_run_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_tag or "")).strip("._")
    if safe_run_tag:
        artifact_root = artifact_root / "runs" / safe_run_tag
    staging_root = Path(pipeline.memory.memory_dir) / "staging" / run_id
    staging_units_file = staging_root / "stage2_units.json"
    checkpoint = Stage2Checkpoint(Path(pipeline.memory.memory_dir), run_id)
    checkpoint_state = checkpoint.load()
    completed_summary = artifact_root / "summary.json"
    if checkpoint_state.get("current_step") == "complete" and completed_summary.exists():
        value = load_json(str(completed_summary))
        if isinstance(value, dict):
            return value
    existing_units = []
    if staging_units_file.exists():
        existing_units = load_json(str(staging_units_file))
        if not isinstance(existing_units, list):
            existing_units = []
    completed = {str(item.get("question_id", "")) for item in existing_units}

    pipeline.memory.load()
    legacy_import = {"imported": 0, "skipped": 0, "already_present": 0}
    if Path(pipeline.memory.memory_dir).name == "v3" and not pipeline.memory.memory.principles:
        legacy_import = pipeline.memory.import_legacy_as_provisional(Path(pipeline.memory.memory_dir).parent)
    registry = DSDRegistry(
        dataset=dataset, round_id=round_id,
        output_root=artifact_root if (artifact_dir or dsd_cache_dir) else output_root, backbone=backbone,
        cache_dir=Path(dsd_cache_dir) if dsd_cache_dir else None, rebuild_cache=rebuild_dsd,
    )
    dsd_concurrency = max(1, int(os.environ.get("MEMUA_STAGE2_DSD_CONCURRENCY", "32")))
    registry_summary = await registry.build(packets, pipeline.plan, concurrency=dsd_concurrency)
    registry.save()
    if prepare_dsd_only:
        summary = {
            "status": "dsd_ready", "dataset": dataset, "round_id": round_id,
            "registry_summary": registry_summary, "input_hash": input_hash,
            "runtime_seconds": time.monotonic() - started,
        }
        atomic_save_json(artifact_root / "summary.json", summary)
        return summary
    frozen_by_question: Dict[str, Dict[str, Any]] = {}
    candidate_bank_hash = ""
    if candidate_bank and Path(candidate_bank).exists():
        bank_payload = load_json(candidate_bank)
        bank_units = bank_payload.get("candidates", []) if isinstance(bank_payload, dict) else bank_payload
        if not isinstance(bank_units, list):
            raise ValueError(f"Candidate bank must contain a candidate list: {candidate_bank}")
        candidate_bank_hash = str(bank_payload.get("candidate_bank_hash", "")) if isinstance(bank_payload, dict) else _sha256_bytes(
            json.dumps(bank_units, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        frozen_by_question = {str(item.get("question_id", "")): item for item in bank_units if isinstance(item, dict)}
    records = [
        record for record in registry.records
        if record.get("stage2_candidate", False) and str(record.get("question_id", "")) not in completed
        and (not selection_ids or str(record.get("question_id", "")) in selection_ids)
    ]
    records.sort(key=lambda item: str(item.get("question_id", "")))
    if selection_ids and len(records) + len(completed) != selection_expected_count:
        raise ValueError(
            "Stage-2 Pilot preflight failed before model calls: "
            f"expected={selection_expected_count} pending={len(records)} completed={len(completed)}"
        )
    if frozen_by_question and not prepare_candidates_only:
        missing = [str(record.get("question_id", "")) for record in records if str(record.get("question_id", "")) not in frozen_by_question]
        if missing:
            raise ValueError(f"Frozen candidate bank is missing {len(missing)} Stage-2 candidates; first={missing[:5]}")
    if max_records and int(max_records) > 0:
        remaining_limit = max(0, int(max_records) - len(existing_units))
        records = records[:remaining_limit]
    generation_limit = max(1, int(os.environ.get("MEMUA_STAGE2_GENERATION_CONCURRENCY", str(validation_concurrency))))
    candidate_limit = int(candidate_concurrency or 0)
    if candidate_limit <= 0:
        candidate_limit = int(os.environ.get("MEMUA_STAGE2_CANDIDATE_CONCURRENCY", "0") or 0)
    if candidate_limit <= 0:
        candidate_limit = generation_limit
    generation_semaphore = asyncio.Semaphore(generation_limit)
    solver_semaphore = asyncio.Semaphore(max(1, validation_concurrency))
    checkpoint.save({
        "run_id": run_id,
        "dataset": dataset,
        "round_id": round_id,
        "current_step": "candidate_processing",
        "completed_candidate_ids": sorted(completed),
        "pending_candidate_ids": [str(item.get("question_id", "")) for item in records],
        "failed_candidate_ids": checkpoint_state.get("failed_candidate_ids", []),
        "store_snapshot_id": f"round_{round_id - 1}" if round_id > 1 else None,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "prompt_hash": prompt_hash,
        "model": pipeline.execute.model,
        "seed": seed,
        "temperature": 0.0,
        "solver_protocol_version": "solver_v1",
        "staging_path": str(staging_root),
    })

    progress = tqdm(
        total=len(records),
        desc=f"Stage 2 candidate processing [{dataset}]",
        unit="candidate",
        dynamic_ncols=True,
    )

    stage_entered: Dict[str, int] = {}
    current_stage: Dict[str, str] = {}

    def report_candidate_stage(question_id: str, stage: str) -> None:
        current_stage[question_id] = stage
        stage_entered[stage] = stage_entered.get(stage, 0) + 1
        progress.set_postfix_str(
            " ".join([
                f"candidate={question_id}", f"stage={stage}",
                f"gen={stage_entered.get('principle_generation', 0)}",
                f"audit={stage_entered.get('principle_quality_audit', 0)}",
                f"gate={stage_entered.get('application_gate', 0)}",
                f"dev={stage_entered.get('dev_paired_validation', 0)}",
                f"holdout={stage_entered.get('holdout_paired_validation', 0)}",
            ]),
            refresh=True,
        )

    async def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
        return await _process_candidate(
            pipeline,
            registry,
            record,
            dataset=dataset,
            round_id=round_id,
            seed=seed,
            generation_semaphore=generation_semaphore,
            solver_semaphore=solver_semaphore,
            progress_callback=report_candidate_stage,
            prepared_candidate=frozen_by_question.get(str(record.get("question_id", ""))) if frozen_by_question else None,
            prepare_only=prepare_candidates_only,
            neighbor_validation=neighbor_validation,
            gate_only=gate_only,
        )

    units = list(existing_units)
    failed_ids = list(checkpoint_state.get("failed_candidate_ids", []))
    logger.info(
        "Stage 2 bounded candidate workers: candidates=%d candidate_concurrency=%d generation_concurrency=%d solver_concurrency=%d",
        len(records), candidate_limit, generation_limit, max(1, validation_concurrency),
    )
    async for record, task in _bounded_as_completed(records, candidate_limit, process_record):
        try:
            unit = await task
        except Exception as exc:
            question_id = str(record.get("question_id", ""))
            failed_ids.append(f"{question_id}:runtime:{type(exc).__name__}:{exc}")
            current_stage.pop(question_id, None)
            progress.set_postfix_str(f"candidate={question_id} stage=failed error={type(exc).__name__}", refresh=True)
            progress.update(1)
            continue
        current_stage.pop(str(unit.get("question_id", "")), None)
        units.append(unit)
        units.sort(key=lambda item: str(item.get("question_id", "")))
        atomic_save_json(staging_units_file, units)
        completed = {str(item.get("question_id", "")) for item in units}
        final_reason = ",".join((unit.get("admission") or {}).get("reasons", [])) or unit.get("status", "done")
        progress.set_postfix_str(
            f"candidate={unit.get('question_id', '?')} stage=complete status={unit.get('status', '?')} reason={final_reason[:32]}",
            refresh=True,
        )
        progress.update(1)
        checkpoint.save({
            "run_id": run_id,
            "dataset": dataset,
            "round_id": round_id,
            "current_step": "candidate_processing",
            "completed_candidate_ids": sorted(completed),
            "pending_candidate_ids": sorted(
                str(item.get("question_id", "")) for item in records if str(item.get("question_id", "")) not in completed
            ),
            "failed_candidate_ids": failed_ids,
            "store_snapshot_id": f"round_{round_id - 1}" if round_id > 1 else None,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "prompt_hash": prompt_hash,
            "model": pipeline.execute.model,
            "seed": seed,
            "temperature": 0.0,
            "solver_protocol_version": "solver_v1",
            "staging_path": str(staging_root),
        })
    progress.close()

    if prepare_candidates_only:
        if not candidate_bank:
            raise ValueError("--prepare-candidates-only requires --candidate-bank")
        bank_payload = {
            "schema": "frozen-candidate-bank-v1", "dataset": dataset, "round_id": round_id,
            "input_hash": input_hash, "prompt_hash": prompt_hash,
            "candidate_count": len(units), "candidates": units,
        }
        candidate_bank_hash = _sha256_bytes(
            json.dumps(units, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        bank_payload["candidate_bank_hash"] = candidate_bank_hash
        atomic_save_json(Path(candidate_bank), bank_payload)

    # Persist analysis artifacts before mutating the v3 long-term store.
    artifact_map = {
        "01_candidates/candidates.json": units,
        "02_schema_checks/schema_checks.json": [
            {"candidate_id": u.get("candidate_id"), "schema_check": u.get("schema_check"), "quality_check": u.get("quality_check")}
            for u in units
        ],
        "03_neighbor_candidates/neighbor_candidates.json": [
            {"candidate_id": u.get("candidate_id"), "neighbors": u.get("neighbor_candidates", [])} for u in units
        ],
        "04_gate_results/gate_results.json": [
            {"candidate_id": u.get("candidate_id"), "gate_result": u.get("gate_result", {})} for u in units
        ],
        "05_dev_paired_validation/dev_pairs.json": [
            {"candidate_id": u.get("candidate_id"), "pairs": u.get("dev_pairs", []), "summary": u.get("dev_summary", {})} for u in units
        ],
        "06_repair_history/repair_history.json": [
            {"candidate_id": u.get("candidate_id"), "history": u.get("repair_history", [])} for u in units
        ],
        "07_holdout_paired_validation/holdout_pairs.json": [
            {"candidate_id": u.get("candidate_id"), "pairs": u.get("holdout_pairs", []), "summary": u.get("holdout_summary", {})} for u in units
        ],
        "08_memory_decisions/memory_decisions.json": [
            {"candidate_id": u.get("candidate_id"), "status": u.get("status"), "admission": u.get("admission", {})} for u in units
        ],
    }
    for relative, payload in artifact_map.items():
        atomic_save_json(artifact_root / relative, payload)

    decisions = []
    if not prepare_candidates_only:
        for unit in units:
            # The live overnight Store is active-only. Rejected/provisional
            # candidates remain fully visible in the audit artifacts above.
            if unit.get("status") != "active" or not isinstance(unit.get("content"), dict):
                continue
            decision = pipeline.memory.commit_stage2_candidate(unit, round_id, dataset)
            decisions.append({
                "candidate_id": unit.get("candidate_id", ""),
                "question_id": unit.get("question_id", ""),
                **decision,
            })
        pipeline.memory.build_indices()
        pipeline.memory.save_indices()
        pipeline.memory.save()
        pipeline.memory.snapshot(round_id)
    atomic_save_json(artifact_root / "09_store_commit/store_commit.json", decisions)

    status_counts: Dict[str, int] = {}
    for unit in units:
        status = str(unit.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    operation_counts: Dict[str, int] = {}
    for decision in decisions:
        operation = str(decision.get("operation", "unknown"))
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
    aggregate = _aggregate_outcomes(units)
    total_model_calls = sum(_estimate_model_calls(unit) for unit in units)
    summary = {
        "status": "candidates_ready" if prepare_candidates_only else "complete",
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "round_id": round_id,
        "run_id": run_id,
        "source_input_packets": len(source_packets),
        "input_packets": len(packets),
        "candidate_count": len(units),
        "max_records": max(0, int(max_records or 0)),
        "run_tag": safe_run_tag,
        "neighbor_validation": bool(neighbor_validation),
        "gate_only": bool(gate_only),
        "selection_file": str(selection_file or ""),
        "selection_hash": selection_hash,
        "candidate_bank": str(candidate_bank or ""),
        "candidate_bank_hash": candidate_bank_hash,
        "candidate_concurrency": candidate_limit,
        "generation_concurrency": generation_limit,
        "stage_entered_counts": stage_entered,
        "status_counts": status_counts,
        "operation_counts": operation_counts,
        "legacy_import": legacy_import,
        "schema_valid_count": sum(1 for u in units if (u.get("schema_check") or {}).get("valid") is True),
        "quality_pass_count": sum(1 for u in units if (u.get("quality_check") or {}).get("quality_pass") is True),
        "gate_protocol_compliant_count": sum(1 for u in units if (u.get("gate_result") or {}).get("protocol_compliant") is True),
        "insufficient_neighbor_count": sum(
            1 for u in units if "insufficient_validation_evidence" in (u.get("admission") or {}).get("reasons", [])
        ),
        "repair_round_count": sum(len(u.get("repair_history", [])) for u in units),
        "estimated_model_api_calls": total_model_calls,
        "average_model_api_calls_per_candidate": total_model_calls / len(units) if units else 0.0,
        "parse_status_counts": {
            "principle_parse_failed": sum(
                1 for u in units if (u.get("generation_result") or {}).get("status") == "schema_failed"
            ),
            "quality_parse_failed": sum(
                1 for u in units if ((u.get("quality_check") or {}).get("llm_audit") or {}).get("status") == "parse_failed"
            ),
            "gate_parse_failed": sum(
                1 for u in units if (u.get("gate_result") or {}).get("status") == "parse_failed"
            ),
        },
        "principle_generation_repair_count": sum(
            1 for u in units
            if (((u.get("generation_result") or {}).get("generation_trace") or {}).get("repair_used") is True)
        ),
        "principle_generation_parse_stage_counts": {
            stage: sum(1 for u in units if str((u.get("generation_result") or {}).get("parse_stage", "unknown")) == stage)
            for stage in sorted({str((u.get("generation_result") or {}).get("parse_stage", "unknown")) for u in units})
        },
        **aggregate,
        "registry_summary": registry_summary,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "prompt_hash": prompt_hash,
        "runtime_seconds": time.monotonic() - started,
        "store_root": str(pipeline.memory.memory_dir),
        "checkpoint": str(checkpoint.latest),
    }
    atomic_save_json(artifact_root / "summary.json", summary)
    checkpoint.save({
        "run_id": run_id,
        "dataset": dataset,
        "round_id": round_id,
        "current_step": "complete",
        "completed_candidate_ids": sorted(str(item.get("question_id", "")) for item in units),
        "pending_candidate_ids": [],
        "failed_candidate_ids": failed_ids,
        "store_snapshot_id": f"round_{round_id}",
        "input_hash": input_hash,
        "config_hash": config_hash,
        "prompt_hash": prompt_hash,
        "model": pipeline.execute.model,
        "seed": seed,
        "temperature": 0.0,
        "solver_protocol_version": "solver_v1",
        "staging_path": str(staging_root),
    })
    return summary
