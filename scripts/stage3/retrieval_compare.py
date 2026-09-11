#!/usr/bin/env python3
"""Compare Stage-3 retrieval protocols on one frozen Memory snapshot.

The evaluator reuses baseline drafts and only calls the solver when a memory is
actually selected.  Every question/protocol record is an atomic checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.matcher import check_answer_for_item
from core.config import Config
from core.request_context import llm_request_context
from modules.execute_module import (
    ExecuteModule,
    SOLVER_V1_FINALIZER_MAX_TOKENS,
    SOLVER_V1_FINALIZER_TEMPERATURE,
    SOLVER_V1_TOP_P,
    filter_healthy_urls,
    _completion_tokens,
    _finalizer_format_valid,
    _finish_reason,
    _repetition_diagnostics,
)
from modules.memory_module import MemoryModule
from modules.plan_module import PlanModule
from modules.principle_schema import principle_content_from_item, render_solver_condition_card
from modules.reranker_module import BGECrossEncoderReranker
from modules.stage3_retrieval import (
    BGEHybridFacetIndex,
    DenseTextIndex,
    PROTOCOLS,
    SEMANTIC_HYBRID_PROTOCOLS,
    correction_text,
    deduplicate_source_groups,
    reciprocal_rank_fusion,
    retrieve_current,
    retrieve_hierarchical,
    retrieve_three_facet_union,
    retrieve_strict,
    serializable_entries,
    soft_draft_rerank,
)
from scripts.stage3.evaluate import AFTER_DRAFT_MAX_DRAFT_CHARS, compact_after_draft_text
from template.prompt_template import FORCED_TIMING_AFTER_DRAFT_PROMPT, SOLVER_V1_FINALIZER_PROMPT
from utils.io_utils import atomic_save_json, load_json


logger = logging.getLogger("stage3.retrieval_compare")
SEEDS = (42, 43, 44)
CALIBRATION_QUOTAS = {"Level 1": 30, "Level 2": 30, "Level 3": 30, "Level 4": 10}


DRAFT_SIGNATURE_PROMPT = """You are extracting a short error-oriented signature from a mathematical draft.
Do not solve the problem and do not use an answer key. Describe only what is observable in the draft.
Return raw JSON with exactly these string fields:
{
  "attempted_method": "the main method used by the draft",
  "observed_failure": "a concrete inconsistency, unsupported step, or empty string if none is observable",
  "missing_or_invalid_check": "a check the draft omitted or performed incorrectly, or empty string"
}

Current problem:
{question}

Draft:
{draft}
"""


DRAFT_MEMORY_JUDGE_PROMPT = """You are a conservative routing judge. Decide whether each candidate memory addresses an error that is actually observable in THIS draft.

Do not solve the problem. Do not assume the draft is wrong merely because a candidate describes an error. Topic similarity is insufficient. Return one judgment for every candidate ID. Output raw JSON only.

Accept a candidate only when:
- its mathematical structure matches the current problem;
- all required evidence is observable;
- no exclusion condition holds;
- at least one stated failure signal, or the same underlying failure, is visible in the draft;
- its repair action directly addresses that visible failure without replacing unrelated correct work.

Current problem:
{question}

Current draft:
{draft}

Candidate memories:
{candidates_json}

Schema:
{{
  "judgments": [
    {{
      "memory_id": "...",
      "problem_structure_match": true,
      "required_evidence_met": true,
      "exclusion_triggered": false,
      "failure_signal_observed": true,
      "repair_action_relevant": true,
      "confidence": 0.0,
      "draft_evidence": "short exact description of the observed draft step",
      "reason": "short reason"
    }}
  ]
}}
"""


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalise_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _safe_json_object(plan: PlanModule, text: str) -> Dict[str, Any]:
    parsed = plan._extract_json(text)
    return parsed if isinstance(parsed, dict) else {}


def _candidate_payload(entry: Mapping[str, Any]) -> Dict[str, Any] | None:
    item = entry.get("item", {}) if isinstance(entry.get("item"), dict) else {}
    content, schema_source = principle_content_from_item(item)
    if not content or item.get("status") != "active":
        return None
    return {
        "memory_id": str(entry.get("memory_id") or item.get("memory_id", "")),
        "content": content,
        "schema_source": schema_source,
        "retrieval": dict(entry),
    }


def _draft_candidate_view(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    content = candidate.get("content", {})
    item = (candidate.get("retrieval") or {}).get("item", {})
    return {
        "memory_id": candidate.get("memory_id", ""),
        "source_structure": {
            "core_skill": item.get("core_skill", ""),
            "context_constraints": item.get("context_constraints", ""),
            "task_objective": item.get("task_objective", ""),
        },
        "trigger_condition": content.get("trigger_condition", ""),
        "required_evidence": content.get("required_evidence", []),
        "exclusion_condition": content.get("exclusion_condition", ""),
        "scope": content.get("scope", ""),
        "failure_mode": content.get("failure_mode", ""),
        "failure_signals": content.get("failure_signals", []),
        "action_guidance": content.get("action_guidance", ""),
        "verification_step": content.get("verification_step", ""),
    }


def parse_draft_judgments(raw: str, expected_ids: Sequence[str], plan: PlanModule) -> Dict[str, Any]:
    if hasattr(plan, "_extract_gate_json"):
        envelope, source_stage = plan._extract_gate_json(raw)
        parsed: Any = envelope
    else:
        parsed = plan._extract_json(raw)
        source_stage = "legacy_parser"
    if isinstance(parsed, dict) and isinstance(parsed.get("judgments"), list):
        values = parsed["judgments"]
        parse_stage = source_stage if source_stage != "legacy_parser" else "object"
    elif isinstance(parsed, list):
        values = parsed
        parse_stage = "direct_list"
    else:
        values = []
        parse_stage = "invalid"
    expected = [str(value) for value in expected_ids]
    returned = [str(value.get("memory_id", "")) for value in values if isinstance(value, dict)]
    errors = []
    if set(returned) != set(expected) or len(returned) != len(expected):
        errors.append("candidate_id_set_mismatch")
    accepted = []
    judgments = []
    for value in values:
        if not isinstance(value, dict):
            errors.append("judgment_not_object")
            continue
        memory_id = str(value.get("memory_id", ""))
        boolean_fields = (
            "problem_structure_match", "required_evidence_met", "exclusion_triggered",
            "failure_signal_observed", "repair_action_relevant",
        )
        if any(not isinstance(value.get(field), bool) for field in boolean_fields):
            errors.append(f"{memory_id}:invalid_boolean")
        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
            errors.append(f"{memory_id}:invalid_confidence")
        is_accepted = (
            value.get("problem_structure_match") is True
            and value.get("required_evidence_met") is True
            and value.get("exclusion_triggered") is False
            and value.get("failure_signal_observed") is True
            and value.get("repair_action_relevant") is True
            and confidence >= 0.80
        )
        row = dict(value)
        row["confidence"] = confidence
        row["accepted"] = is_accepted
        judgments.append(row)
        if is_accepted:
            accepted.append(memory_id)
    compliant = parse_stage != "invalid" and not errors
    if not compliant:
        accepted = []
    return {
        "protocol_compliant": compliant,
        "parse_stage": parse_stage,
        "protocol_errors": errors,
        "judgments": judgments,
        "selected_ids": accepted,
        "raw_response": raw,
    }


def _render_draft_signature_prompt(question: str, draft: str) -> str:
    """Render the two named slots without treating the JSON example as format syntax."""
    before_question, remainder = DRAFT_SIGNATURE_PROMPT.split("{question}", 1)
    between_slots, after_draft = remainder.split("{draft}", 1)
    return before_question + question + between_slots + draft + after_draft


def _load_baseline_records(path: Path) -> Dict[int, Dict[str, Any]]:
    file_path = path / "eval_results.json" if path.is_dir() else path
    rows = load_json(str(file_path))
    if not isinstance(rows, list) or len(rows) != 400:
        raise ValueError(f"Baseline must contain exactly 400 questions: {file_path}")
    records = {int(row["idx"]): row for row in rows}
    for idx in range(400):
        row = records.get(idx, {})
        rollouts = row.get("rollouts", [])
        if len(rollouts) != 3 or [int(value.get("seed", -1)) for value in rollouts] != list(SEEDS):
            raise ValueError(f"Baseline idx={idx} does not contain seeds 42/43/44")
        if any(value.get("status") != "complete" for value in rollouts):
            raise ValueError(f"Baseline idx={idx} contains infrastructure failure")
    return records


def _load_source_questions(root: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for path in sorted(root.rglob("*learning_packets.json")):
        try:
            rows = load_json(str(path))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("question_id", "")).strip()
            question = str(row.get("question_text") or row.get("question") or "").strip()
            if question_id and question:
                values.setdefault(question_id, question)
    return values


def _build_manifests(
    baseline: Mapping[int, Mapping[str, Any]],
    current_guided_dir: Path,
    output_root: Path,
    source_questions: Mapping[str, str],
    memories: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    calibration_path = output_root / "calibration_100.json"
    heldout_path = output_root / "heldout_300.json"
    if calibration_path.exists() and heldout_path.exists():
        return {
            "calibration": load_json(str(calibration_path)),
            "heldout": load_json(str(heldout_path)),
        }

    guided_rows = load_json(str(current_guided_dir / "eval_results.json"))
    prior_used = {
        int(row["idx"])
        for row in guided_rows
        if isinstance(row, dict) and (row.get("retrieval") or {}).get("used")
    }
    selected = set(prior_used)
    by_level: Dict[str, List[int]] = defaultdict(list)
    for idx, row in baseline.items():
        by_level[str(row.get("difficulty_level", ""))].append(idx)
    for level, quota in CALIBRATION_QUOTAS.items():
        already = sum(1 for idx in selected if baseline[idx].get("difficulty_level") == level)
        candidates = [idx for idx in by_level[level] if idx not in selected]
        candidates.sort(key=lambda idx: hashlib.sha256(f"{idx}|retrieval_calibration_v1".encode()).hexdigest())
        selected.update(candidates[: max(0, quota - already)])
    if len(selected) > 100:
        # Preserve all prior-used cases, then use stable order for the remainder.
        extras = sorted(
            selected - prior_used,
            key=lambda idx: hashlib.sha256(f"{idx}|retrieval_calibration_trim_v1".encode()).hexdigest(),
        )
        selected = prior_used | set(extras[: 100 - len(prior_used)])
    if len(selected) < 100:
        remainder = [idx for idx in baseline if idx not in selected]
        remainder.sort(key=lambda idx: hashlib.sha256(f"{idx}|retrieval_calibration_fill_v1".encode()).hexdigest())
        selected.update(remainder[: 100 - len(selected)])

    source_text_to_ids: Dict[str, List[str]] = defaultdict(list)
    for item in memories:
        qid = str(item.get("question_id", ""))
        if qid in source_questions:
            source_text_to_ids[_normalise_question(source_questions[qid])].append(qid)

    def record(idx: int) -> Dict[str, Any]:
        row = baseline[idx]
        overlaps = source_text_to_ids.get(_normalise_question(row.get("question", "")), [])
        return {
            "idx": idx,
            "difficulty_level": row.get("difficulty_level", ""),
            "prior_current_retrieval_used": idx in prior_used,
            "exact_memory_source_overlap": bool(overlaps),
            "overlap_source_question_ids": overlaps,
        }

    calibration = [record(idx) for idx in sorted(selected)]
    heldout = [record(idx) for idx in sorted(set(baseline) - selected)]
    atomic_save_json(calibration_path, calibration)
    atomic_save_json(heldout_path, heldout)
    return {"calibration": calibration, "heldout": heldout}


def _raw_question_entries(index: DenseTextIndex, memory: Any, question: str, item_by_id: Mapping[str, Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    result = []
    for row in index.search(memory, question, top_k=top_k):
        item = item_by_id[row["memory_id"]]
        result.append(
            {
                "memory_id": row["memory_id"],
                "item": item,
                "score": row["score"],
                "retrieval_score": row["score"],
                "raw_question_similarity": row["score"],
                "retrieval_method": "r0_single_vector",
            }
        )
    return result


async def _question_gate(plan: PlanModule, question: str, entries: Sequence[Dict[str, Any]], gate_top_k: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = [value for value in (_candidate_payload(entry) for entry in entries[:gate_top_k]) if value]
    if not candidates:
        return [], {"protocol_compliant": True, "selected_ids": [], "judgments": [], "reason": "no_candidates"}
    gate = await plan.judge_memory_applicability(question, candidates)
    selected = set(gate.get("selected_ids", [])) if gate.get("protocol_compliant") else set()
    return [candidate for candidate in candidates if candidate["memory_id"] in selected], gate


async def _adapt_one(plan: PlanModule, question: str, candidate: Dict[str, Any], gate_reason: Mapping[str, Any]) -> Dict[str, Any]:
    adapter = await plan.adapt_memory_for_question(question, candidate, dict(gate_reason))
    if adapter.get("status") != "use" or not adapter.get("condition_card"):
        return {"status": "reject", "adapter": adapter, "solver_text": ""}
    return {
        "status": "use",
        "adapter": adapter,
        "solver_text": render_solver_condition_card(adapter["condition_card"], 1),
    }


async def _draft_signature(plan: PlanModule, question: str, draft: str, seed: int) -> Dict[str, Any]:
    prompt = _render_draft_signature_prompt(
        question,
        compact_after_draft_text(draft, AFTER_DRAFT_MAX_DRAFT_CHARS),
    )
    raw = await plan._call_llm(prompt, temperature=0.0, max_tokens=768, seed=seed)
    parsed = _safe_json_object(plan, raw)
    fields = ("attempted_method", "observed_failure", "missing_or_invalid_check")
    valid = bool(parsed) and all(isinstance(parsed.get(field), str) for field in fields)
    return {"valid": valid, "signature": {field: str(parsed.get(field, "")).strip() for field in fields}, "raw_response": raw, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()}


async def _draft_judge(plan: PlanModule, question: str, draft: str, candidates: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    payload = [_draft_candidate_view(candidate) for candidate in candidates]
    prompt = DRAFT_MEMORY_JUDGE_PROMPT.format(
        question=question,
        draft=compact_after_draft_text(draft, AFTER_DRAFT_MAX_DRAFT_CHARS),
        candidates_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    raw = await plan._call_llm(prompt, temperature=0.0, max_tokens=2048, seed=seed)
    result = parse_draft_judgments(raw, [candidate["memory_id"] for candidate in candidates], plan)
    result["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()
    return result


async def _run_after_draft(
    execute: ExecuteModule,
    question_item: Dict[str, Any],
    baseline_rollout: Dict[str, Any],
    solver_text: str,
    seed: int,
    endpoints: Sequence[str],
    idx: int,
) -> Dict[str, Any]:
    if not solver_text.strip():
        return {
            "seed": seed,
            "status": "complete",
            "correct": bool(baseline_rollout.get("correct")),
            "principle_injected": False,
            "experience_decision": "no_memory_reused_baseline",
            "baseline_rollout": baseline_rollout,
        }
    question = str(question_item.get("question", ""))
    draft = str((((baseline_rollout.get("response") or {}).get("reasoner") or {}).get("output") or "")).strip()
    prompt = FORCED_TIMING_AFTER_DRAFT_PROMPT.format(
        question=question,
        previous_draft=compact_after_draft_text(draft),
        principle_card=solver_text,
    )
    first = (idx + SEEDS.index(seed)) % len(endpoints)
    ordered = [endpoints[first]]
    if len(endpoints) > 1:
        ordered.append(endpoints[(first + 1) % len(endpoints)])
    trace = []
    for endpoint_attempt, endpoint in enumerate(ordered, 1):
        with llm_request_context(f"retrieval_compare_{idx}_seed{seed}", f"forced_c/endpoint_attempt={endpoint_attempt}"):
            reasoner_meta, reasoner_url = await execute._call_with_router_metadata(
                prompt=prompt,
                temperature=0.2,
                top_p=SOLVER_V1_TOP_P,
                max_tokens=4096,
                repetition_penalty=1.0,
                seed=seed,
                url_override=endpoint,
                timeout_override=300,
            )
        reasoner = str(reasoner_meta.get("text") or "").strip()
        if not reasoner:
            trace.append({"endpoint": endpoint, "stage": "reasoner", "success": False})
            continue
        finalizer_prompt = SOLVER_V1_FINALIZER_PROMPT.format(question=question, reasoner_output=reasoner)
        with llm_request_context(f"retrieval_compare_{idx}_seed{seed}", f"finalizer/endpoint_attempt={endpoint_attempt}"):
            finalizer_meta, finalizer_url = await execute._call_with_router_metadata(
                prompt=finalizer_prompt,
                temperature=SOLVER_V1_FINALIZER_TEMPERATURE,
                top_p=SOLVER_V1_TOP_P,
                max_tokens=SOLVER_V1_FINALIZER_MAX_TOKENS,
                repetition_penalty=1.0,
                seed=seed,
                url_override=reasoner_url or endpoint,
                timeout_override=60,
            )
        finalizer = str(finalizer_meta.get("text") or "").strip()
        if not finalizer:
            trace.append({"endpoint": endpoint, "stage": "finalizer", "success": False})
            continue
        composed = f"{reasoner}\n\n{finalizer}"
        return {
            "seed": seed,
            "status": "complete",
            "correct": check_answer_for_item(composed, question_item),
            "principle_injected": True,
            "experience_decision": "corrected_or_verified",
            "solver_prompt": prompt,
            "response": {
                "reasoner": {
                    "output": reasoner,
                    "finish_reason": _finish_reason(reasoner_meta),
                    "completion_tokens": _completion_tokens(reasoner_meta),
                    "repetition": _repetition_diagnostics(reasoner),
                },
                "finalizer": {
                    "output": finalizer,
                    "finish_reason": _finish_reason(finalizer_meta),
                    "format_valid": _finalizer_format_valid(finalizer),
                },
                "final_answer": finalizer,
                "solver_url": reasoner_url,
                "finalizer_url": finalizer_url,
            },
            "baseline_rollout": baseline_rollout,
            "transport_trace": trace,
        }
    return {"seed": seed, "status": "infrastructure_failed", "correct": False, "principle_injected": True, "baseline_rollout": baseline_rollout, "transport_trace": trace}


def _gate_reason(gate: Mapping[str, Any], memory_id: str) -> Dict[str, Any]:
    for row in gate.get("judgments", []):
        if str(row.get("memory_id", "")) == memory_id:
            return dict(row)
    return {"accepted": False}


def _select_draft_candidate(candidates: Sequence[Dict[str, Any]], judgment: Mapping[str, Any]) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    by_id = {
        str(row.get("memory_id", "")): row
        for row in judgment.get("judgments", [])
        if isinstance(row, dict) and row.get("accepted")
    }
    ranked = [candidate for candidate in candidates if candidate["memory_id"] in by_id]
    ranked.sort(
        key=lambda candidate: (
            float(by_id[candidate["memory_id"]].get("confidence", 0.0)),
            float((candidate.get("retrieval") or {}).get("score", 0.0)),
        ),
        reverse=True,
    )
    if not ranked:
        return None, {}
    chosen = ranked[0]
    return chosen, dict(by_id[chosen["memory_id"]])


async def _process_question(
    *,
    protocol: str,
    idx: int,
    baseline_record: Dict[str, Any],
    memory: Any,
    raw_index: DenseTextIndex,
    correction_index: DenseTextIndex,
    item_by_id: Mapping[str, Dict[str, Any]],
    semantics: Dict[str, Any],
    plan: PlanModule,
    execute: ExecuteModule,
    endpoints: Sequence[str],
    output_path: Path,
    facet_pool_k: int = 30,
    candidate_top_k: int = 12,
    rerank_top_k: int = 12,
    gate_top_k: int = 6,
    max_injected: int = 1,
    cross_encoder: BGECrossEncoderReranker | None = None,
    hybrid_index: BGEHybridFacetIndex | None = None,
    rrf_rank_constant: int = 60,
    draft_structure_weight: float = 0.70,
    draft_error_weight: float = 0.30,
) -> Dict[str, Any]:
    if output_path.exists():
        old = load_json(str(output_path))
        if old.get("status") in {"complete", "infrastructure_failed"}:
            old_draft_records = (old.get("retrieval") or {}).get("draft_records", [])
            has_legacy_gate_parse_failure = protocol == "r4_draft_verified" and any(
                "parse_stage" not in (record.get("judgment") or {})
                and "candidate_id_set_mismatch" in ((record.get("judgment") or {}).get("protocol_errors") or [])
                for record in old_draft_records
                if isinstance(record, dict)
            )
            if not has_legacy_gate_parse_failure:
                return old
    question = str(baseline_record.get("question", ""))
    question_item = {
        "question": question,
        "answer": baseline_record.get("expected_answer", ""),
        "difficulty_level": baseline_record.get("difficulty_level", ""),
    }

    dedup_removed: List[Dict[str, Any]] = []
    pre_rerank_entries: List[Dict[str, Any]] = []
    if protocol == "r0_single_vector":
        entries = _raw_question_entries(raw_index, memory, question, item_by_id, candidate_top_k)
    elif protocol == "r1_current_three_facet":
        entries = retrieve_current(memory, semantics, top_k=candidate_top_k, threshold=0.55)
    elif protocol == "r2_strict_three_facet":
        entries = retrieve_strict(memory, semantics, top_k=candidate_top_k, threshold=0.55)
    elif protocol in {"r3_hierarchical_three_facet", "r4_draft_verified", "r5_dual_retrieval", "e0_r3_current"}:
        entries = retrieve_hierarchical(
            memory, semantics,
            top_k=30 if protocol == "r5_dual_retrieval" else candidate_top_k,
            pool_per_facet=facet_pool_k,
        )
    elif protocol == "e1_r3_source_dedup":
        wide = retrieve_hierarchical(
            memory, semantics, top_k=len(item_by_id), pool_per_facet=facet_pool_k,
        )
        entries, dedup_removed = deduplicate_source_groups(wide, top_k=candidate_top_k)
    elif protocol in {"e2_three_facet_high_recall", "e3_cross_encoder"}:
        wide = retrieve_three_facet_union(
            memory, semantics, top_k=len(item_by_id), pool_per_facet=facet_pool_k,
        )
        entries, dedup_removed = deduplicate_source_groups(wide, top_k=candidate_top_k)
    elif protocol in {"h1_dense_sparse", "h2_dense_sparse_multivector", "h3_draft_soft_rerank"}:
        if hybrid_index is None:
            raise RuntimeError(f"{protocol} requires a BGEHybridFacetIndex")
        modalities = ("dense", "sparse") if protocol == "h1_dense_sparse" else ("dense", "sparse", "multivector")
        wide = hybrid_index.retrieve(
            semantics,
            modalities=modalities,
            pool_per_facet=facet_pool_k,
            top_k=len(item_by_id),
            rank_constant=rrf_rank_constant,
        )
        entries, dedup_removed = deduplicate_source_groups(wide, top_k=candidate_top_k)
    else:
        raise ValueError(f"Unsupported retrieval protocol: {protocol}")

    if protocol in {"e3_cross_encoder", "h3_draft_soft_rerank"}:
        if cross_encoder is None:
            raise RuntimeError(f"{protocol} requires a cross encoder")
        pre_rerank_entries = list(entries)
        candidates = [value for value in (_candidate_payload(entry) for entry in entries) if value]
        ranked = cross_encoder.rank(question, semantics, candidates, top_k=rerank_top_k)
        entries = []
        for candidate in ranked:
            entry = dict(candidate["retrieval"])
            entry.update(
                rerank_score=float(candidate["rerank_score"]),
                pre_rerank_score=float(candidate["pre_rerank_score"]),
                score=float(candidate["rerank_score"]),
                retrieval_score=float(candidate["rerank_score"]),
                retrieval_method="bge_cross_encoder",
            )
            entries.append(entry)

    base_trace: Dict[str, Any] = {
        "protocol": protocol,
        "query_semantics": semantics,
        "retrieved": serializable_entries(entries),
        "candidate_ids": [str(row.get("memory_id", "")) for row in entries],
        "pre_rerank": serializable_entries(pre_rerank_entries),
        "dedup_removed": dedup_removed,
        "parameters": {
            "facet_pool_k": facet_pool_k,
            "candidate_top_k": candidate_top_k,
            "rerank_top_k": rerank_top_k,
            "gate_top_k": gate_top_k,
            "max_injected": max_injected,
        },
    }
    rollouts = []
    selected_ids = []
    gate_records = []
    adapter_records = []
    draft_records = []

    question_gate_protocols = {
        "r0_single_vector", "r1_current_three_facet", "r2_strict_three_facet",
        "r3_hierarchical_three_facet", "e0_r3_current", "e1_r3_source_dedup",
        "e2_three_facet_high_recall", "e3_cross_encoder", "h1_dense_sparse",
        "h2_dense_sparse_multivector",
    }
    if protocol in question_gate_protocols:
        accepted, gate = await _question_gate(plan, question, entries, gate_top_k)
        gate_records.append(gate)
        chosen = accepted[:max_injected]
        adapted = {"status": "reject", "solver_text": "", "adapter": {}}
        if chosen:
            adapted = await _adapt_one(plan, question, chosen[0], _gate_reason(gate, chosen[0]["memory_id"]))
            adapter_records.append(adapted.get("adapter", {}))
        solver_text = adapted.get("solver_text", "")
        if solver_text:
            selected_ids.append(chosen[0]["memory_id"])
        for baseline_rollout in baseline_record["rollouts"]:
            seed = int(baseline_rollout["seed"])
            result = await _run_after_draft(execute, question_item, baseline_rollout, solver_text, seed, endpoints, idx)
            result["selected_memory_id"] = chosen[0]["memory_id"] if solver_text and chosen else ""
            rollouts.append(result)
    else:
        for baseline_rollout in baseline_record["rollouts"]:
            seed = int(baseline_rollout["seed"])
            draft = str((((baseline_rollout.get("response") or {}).get("reasoner") or {}).get("output") or "")).strip()
            seed_entries = entries
            signature_record: Dict[str, Any] = {}
            if protocol in {"r5_dual_retrieval", "h3_draft_soft_rerank"}:
                signature_record = await _draft_signature(plan, question, draft, seed)
                signature = signature_record.get("signature", {})
                signature_text = "Attempted method: {attempted_method}\nObserved failure: {observed_failure}\nMissing or invalid check: {missing_or_invalid_check}".format(**signature)
                has_error_signal = bool(
                    str(signature.get("observed_failure", "")).strip()
                    or str(signature.get("missing_or_invalid_check", "")).strip()
                )
                correction = (
                    correction_index.search(memory, signature_text, top_k=max(facet_pool_k, len(entries)))
                    if signature_record.get("valid") and has_error_signal else []
                )
                if protocol == "r5_dual_retrieval":
                    seed_entries = reciprocal_rank_fusion(entries, correction, top_k=rerank_top_k)
                else:
                    correction_scores = {str(row["memory_id"]): float(row["score"]) for row in correction}
                    seed_entries = soft_draft_rerank(
                        entries,
                        correction_scores,
                        structure_weight=draft_structure_weight,
                        error_weight=draft_error_weight,
                        top_k=rerank_top_k,
                    )
            candidates = [value for value in (_candidate_payload(entry) for entry in seed_entries[:gate_top_k]) if value]
            judgment = await _draft_judge(plan, question, draft, candidates, seed) if candidates else {
                "protocol_compliant": True, "selected_ids": [], "judgments": [], "raw_response": "", "reason": "no_candidates"
            }
            draft_records.append({"seed": seed, "signature": signature_record, "judgment": judgment, "reranked": serializable_entries(seed_entries)})
            chosen, selected_judgment = _select_draft_candidate(candidates, judgment)
            adapted = {"status": "reject", "solver_text": "", "adapter": {}}
            if chosen:
                selected_judgment = {**selected_judgment, "accepted": True}
                adapted = await _adapt_one(plan, question, chosen, selected_judgment)
                adapter_records.append({"seed": seed, **adapted.get("adapter", {})})
            solver_text = adapted.get("solver_text", "")
            if solver_text and chosen:
                selected_ids.append(chosen["memory_id"])
            result = await _run_after_draft(execute, question_item, baseline_rollout, solver_text, seed, endpoints, idx)
            result["selected_memory_id"] = chosen["memory_id"] if solver_text and chosen else ""
            rollouts.append(result)

    status = "complete" if all(row.get("status") == "complete" for row in rollouts) else "infrastructure_failed"
    record = {
        "idx": idx,
        "question": question,
        "expected_answer": baseline_record.get("expected_answer", ""),
        "difficulty_level": baseline_record.get("difficulty_level", ""),
        "protocol": protocol,
        "status": status,
        "retrieval": {
            **base_trace,
            "selected_ids": selected_ids,
            "used": bool(selected_ids),
            "gate_records": gate_records,
            "adapter_records": adapter_records,
            "draft_records": draft_records,
        },
        "rollouts": rollouts,
        "pass1_correct": bool(rollouts[0].get("correct")),
        "pass3_correct": any(bool(row.get("correct")) for row in rollouts),
    }
    atomic_save_json(output_path, record)
    return record


def _bootstrap_ci(deltas: Sequence[float], seed: int = 42, repeats: int = 10000) -> List[float]:
    values = np.asarray(list(deltas), dtype=np.float64)
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for pos in range(repeats):
        means[pos] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _summarise(records: Sequence[Dict[str, Any]], baseline: Mapping[int, Mapping[str, Any]], excluded_overlap: set[int]) -> Dict[str, Any]:
    def metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        gain = loss = both_correct = both_wrong = 0
        injected_rollouts = 0
        question_deltas = []
        pass3_deltas = []
        injected_questions = 0
        for row in rows:
            base = baseline[int(row["idx"])]
            if row.get("retrieval", {}).get("used"):
                injected_questions += 1
            guided_correct = []
            base_correct = []
            for guided_rollout, baseline_rollout in zip(row["rollouts"], base["rollouts"]):
                g = bool(guided_rollout.get("correct"))
                b = bool(baseline_rollout.get("correct"))
                guided_correct.append(g)
                base_correct.append(b)
                injected_rollouts += int(bool(guided_rollout.get("principle_injected")))
                if g and not b:
                    gain += 1
                elif b and not g:
                    loss += 1
                elif g and b:
                    both_correct += 1
                else:
                    both_wrong += 1
            question_deltas.append((sum(guided_correct) - sum(base_correct)) / len(SEEDS))
            pass3_deltas.append(float(any(guided_correct)) - float(any(base_correct)))
        discordant = gain + loss
        return {
            "questions": len(rows),
            "injected_questions": injected_questions,
            "injection_coverage": injected_questions / len(rows) if rows else 0.0,
            "injected_rollouts": injected_rollouts,
            "gain": gain,
            "loss": loss,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "helpful_precision": gain / discordant if discordant else None,
            "harm_rate_among_discordant": loss / discordant if discordant else None,
            "net_gain": gain - loss,
            "question_macro_delta": float(np.mean(question_deltas)) if question_deltas else 0.0,
            "question_macro_delta_ci95": _bootstrap_ci(question_deltas),
            "pass1_correct": sum(bool(row.get("pass1_correct")) for row in rows),
            "pass3_correct": sum(bool(row.get("pass3_correct")) for row in rows),
            "pass3_delta": int(sum(pass3_deltas)),
            "pass3_delta_ci95": _bootstrap_ci(pass3_deltas),
            "infrastructure_failed": sum(row.get("status") != "complete" for row in rows),
        }

    all_rows = list(records)
    transfer_rows = [row for row in records if int(row["idx"]) not in excluded_overlap]
    return {"all": metrics(all_rows), "transfer_only": metrics(transfer_rows)}


def _write_report(root: Path, summary: Dict[str, Any], protocols: Sequence[str] = PROTOCOLS) -> None:
    lines = [
        "# Stage 3 检索协议对比",
        "",
        "主指标排除与Memory来源题完全相同的测试题；全部题目的结果同时保留为描述性结果。",
        "",
        "| 协议 | Transfer题数 | 注入题数 | gain | loss | net | 题目宏平均变化 | 95% CI | pass@1 | pass@3 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for protocol in protocols:
        value = (
            summary.get("protocols", {}).get(protocol, {}).get("heldout", {}).get("transfer_only", {})
            or summary.get("protocols", {}).get(protocol, {}).get("combined", {}).get("transfer_only", {})
        )
        ci = value.get("question_macro_delta_ci95", [0, 0])
        lines.append(
            f"| {protocol} | {value.get('questions', 0)} | {value.get('injected_questions', 0)} | "
            f"{value.get('gain', 0)} | {value.get('loss', 0)} | {value.get('net_gain', 0)} | "
            f"{value.get('question_macro_delta', 0):.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{value.get('pass1_correct', 0)} | {value.get('pass3_correct', 0)} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _effect_rows(records: Sequence[Dict[str, Any]], baseline: Mapping[int, Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    question_rows = []
    candidate_totals: Dict[str, Counter] = defaultdict(Counter)
    candidate_questions: Dict[str, set[int]] = defaultdict(set)
    for row in records:
        idx = int(row["idx"])
        base = baseline[idx]
        counts = Counter()
        for guided, control in zip(row.get("rollouts", []), base.get("rollouts", [])):
            g = bool(guided.get("correct"))
            b = bool(control.get("correct"))
            label = "gain" if g and not b else "loss" if b and not g else "both_correct" if g else "both_wrong"
            counts[label] += 1
            if guided.get("principle_injected"):
                memory_id = str(guided.get("selected_memory_id", ""))
                if memory_id:
                    candidate_totals[memory_id][label] += 1
                    candidate_questions[memory_id].add(idx)
        delta = counts["gain"] - counts["loss"]
        label = "helpful" if delta > 0 else "harmful" if delta < 0 else "neutral"
        question_rows.append(
            {
                "idx": idx,
                "difficulty_level": row.get("difficulty_level", ""),
                "retrieval_used": bool((row.get("retrieval") or {}).get("used")),
                "selected_ids": (row.get("retrieval") or {}).get("selected_ids", []),
                "gain": counts["gain"],
                "loss": counts["loss"],
                "both_correct": counts["both_correct"],
                "both_wrong": counts["both_wrong"],
                "net_gain": delta,
                "effect_label": label,
            }
        )
    candidate_rows = [
        {
            "memory_id": memory_id,
            "question_indices": sorted(candidate_questions[memory_id]),
            "gain": counts["gain"],
            "loss": counts["loss"],
            "both_correct": counts["both_correct"],
            "both_wrong": counts["both_wrong"],
            "net_gain": counts["gain"] - counts["loss"],
        }
        for memory_id, counts in sorted(candidate_totals.items())
    ]
    return question_rows, candidate_rows


def _recommendation(protocol_summaries: Mapping[str, Any]) -> Dict[str, Any]:
    ranked = []
    for protocol, summary in protocol_summaries.items():
        value = (summary.get("heldout") or {}).get("transfer_only", {})
        if not value:
            continue
        coverage_ok = value.get("injected_questions", 0) >= 24
        safe = value.get("loss", 0) <= value.get("gain", 0)
        ci = value.get("question_macro_delta_ci95", [-1.0, 1.0])
        freeze_eligible = (
            coverage_ok
            and safe
            and value.get("question_macro_delta", 0.0) > 0.0
            and ci[0] >= 0.0
            and value.get("pass3_delta", 0) >= 0
            and value.get("infrastructure_failed", 0) == 0
        )
        harm = value.get("harm_rate_among_discordant")
        ranked.append(
            {
                "protocol": protocol,
                "freeze_eligible": freeze_eligible,
                "coverage_ok": coverage_ok,
                "safe_gain_loss": safe,
                "injected_questions": value.get("injected_questions", 0),
                "gain": value.get("gain", 0),
                "loss": value.get("loss", 0),
                "net_gain": value.get("net_gain", 0),
                "question_macro_delta": value.get("question_macro_delta", 0.0),
                "harm_rate_among_discordant": 1.0 if harm is None else harm,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["freeze_eligible"],
            -row["harm_rate_among_discordant"],
            row["net_gain"],
            row["question_macro_delta"],
            row["injected_questions"],
        ),
        reverse=True,
    )
    return {
        "ranking": ranked,
        "recommended_protocol": ranked[0]["protocol"] if ranked and ranked[0]["freeze_eligible"] else None,
        "decision": "freeze_best" if ranked and ranked[0]["freeze_eligible"] else "none_meets_freeze_criteria",
    }


def _write_representative_cases(
    root: Path,
    records_by_protocol: Mapping[str, Sequence[Dict[str, Any]]],
    baseline: Mapping[int, Mapping[str, Any]],
) -> None:
    lines = ["# 检索协议代表案例", "", "每个协议最多列出3个净收益和3个净损失题。", ""]
    for protocol in records_by_protocol:
        cases = []
        for row in records_by_protocol.get(protocol, []):
            idx = int(row["idx"])
            gains = losses = 0
            for guided, control in zip(row.get("rollouts", []), baseline[idx].get("rollouts", [])):
                g, b = bool(guided.get("correct")), bool(control.get("correct"))
                gains += int(g and not b)
                losses += int(b and not g)
            if gains != losses:
                cases.append((gains - losses, row))
        positive = sorted((value for value in cases if value[0] > 0), key=lambda value: value[0], reverse=True)[:3]
        negative = sorted((value for value in cases if value[0] < 0), key=lambda value: value[0])[:3]
        lines.extend([f"## {protocol}", ""])
        for delta, row in positive + negative:
            direction = "收益" if delta > 0 else "损失"
            trace = row.get("retrieval", {})
            lines.extend(
                [
                    f"### {direction}：题号 {row['idx']}，net={delta}",
                    "",
                    f"- 题目：{str(row.get('question', '')).replace(chr(10), ' ')[:500]}",
                    f"- 选中Memory：{', '.join(dict.fromkeys(trace.get('selected_ids', []))) or '无'}",
                    f"- 初始候选：{', '.join(trace.get('candidate_ids', [])[:6]) or '无'}",
                ]
            )
            draft_records = trace.get("draft_records", [])
            if draft_records:
                evidence = []
                for draft_row in draft_records:
                    for judgment in (draft_row.get("judgment") or {}).get("judgments", []):
                        if judgment.get("accepted"):
                            evidence.append(str(judgment.get("draft_evidence", "")))
                lines.append(f"- 草稿证据：{'；'.join(value for value in evidence if value) or '无明确证据'}")
            lines.append("")
    (root / "representative_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay_saved_draft_gates(source_root: Path, output_root: Path, gate_top_k: int = 6) -> Dict[str, Any]:
    """Reparse saved R4/R5 responses without issuing model or solver calls."""
    replay_rows = []
    for protocol in ("r4_draft_verified", "r5_dual_retrieval"):
        question_dir = source_root / "protocols" / protocol / "questions"
        for path in sorted(question_dir.glob("*.json"), key=lambda value: int(value.stem)):
            record = load_json(str(path))
            for draft_row in (record.get("retrieval") or {}).get("draft_records", []):
                judgment = draft_row.get("judgment") or {}
                raw = str(judgment.get("raw_response", ""))
                expected = [
                    str(row.get("memory_id", ""))
                    for row in (draft_row.get("reranked") or [])[:gate_top_k]
                    if row.get("memory_id")
                ]
                # Use a real parser instance method without constructing PlanModule.
                class _Parser:
                    _json_candidates = PlanModule._json_candidates
                    _extract_json_value = PlanModule._extract_json_value
                    _extract_gate_json = PlanModule._extract_gate_json

                parsed = parse_draft_judgments(raw, expected, _Parser())
                replay_rows.append(
                    {
                        "protocol": protocol,
                        "idx": int(record["idx"]),
                        "seed": draft_row.get("seed"),
                        "expected_ids": expected,
                        "old_parse_stage": judgment.get("parse_stage", "missing"),
                        "replayed": parsed,
                    }
                )
    summary = {
        "records": len(replay_rows),
        "protocol_compliant": sum(bool(row["replayed"].get("protocol_compliant")) for row in replay_rows),
        "selected_calls": sum(bool(row["replayed"].get("selected_ids")) for row in replay_rows),
        "selected_judgments": sum(len(row["replayed"].get("selected_ids", [])) for row in replay_rows),
        "model_calls": 0,
    }
    target = output_root / "parser_replay"
    target.mkdir(parents=True, exist_ok=True)
    atomic_save_json(target / "r4_r5_gate_replay.json", replay_rows)
    atomic_save_json(target / "summary.json", summary)
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 retrieval protocol comparison")
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--baseline-cache-dir", required=True)
    parser.add_argument("--current-guided-dir", required=True)
    parser.add_argument("--stage1-packets-root", default="output/stage1/02_learning_packets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--phase", choices=["parser-replay", "smoke", "calibration", "heldout", "all", "report"], default="all")
    parser.add_argument("--suite", choices=["legacy", "semantic-hybrid"], default="legacy")
    parser.add_argument("--protocol", choices=["all", *PROTOCOLS, *SEMANTIC_HYBRID_PROTOCOLS], default="all")
    parser.add_argument("--source-comparison-root", default="output/stage3/retrieval_protocol_compare_v1")
    parser.add_argument("--facet-pool-k", type=int, default=30)
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--rerank-top-k", type=int, default=12)
    parser.add_argument("--gate-top-k", type=int, default=6)
    parser.add_argument("--max-injected", type=int, default=1)
    parser.add_argument("--reranker-model-path", default="models/reranker/bge-reranker-v2-m3")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--hybrid-model-path", default="/home/share/models/modelscope/BAAI/bge-m3")
    parser.add_argument("--hybrid-device", default="cpu")
    parser.add_argument("--rrf-rank-constant", type=int, default=60)
    parser.add_argument("--draft-structure-weight", type=float, default=0.70)
    parser.add_argument("--draft-error-weight", type=float, default=0.30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_injected != 1:
        raise ValueError("This experiment fixes --max-injected=1 to isolate retrieval precision")
    if args.candidate_top_k < args.rerank_top_k or args.rerank_top_k < args.gate_top_k:
        raise ValueError("Require candidate_top_k >= rerank_top_k >= gate_top_k")

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(root / "raw.log", encoding="utf-8")],
    )
    replay_summary = replay_saved_draft_gates(
        Path(args.source_comparison_root), root, gate_top_k=args.gate_top_k,
    )
    logger.info("R4/R5离线解析回放完成：%s", replay_summary)
    if args.phase == "parser-replay":
        return
    baseline = _load_baseline_records(Path(args.baseline_cache_dir))
    config = Config.from_core_file()
    memory_module = MemoryModule(config, memory_dir=args.memory_dir)
    memory_module.load()
    if not memory_module.is_indexed:
        memory_module.build_indices()
    memory = memory_module.memory
    active = [item for item in memory.principles if item.get("status") == "active"]
    item_by_id = {str(item["memory_id"]): item for item in active}
    source_questions = _load_source_questions(Path(args.stage1_packets_root))
    missing = sorted({str(item.get("question_id", "")) for item in active} - set(source_questions))
    if missing:
        raise ValueError(f"Missing source question text for {len(missing)} Memory source IDs: {missing[:10]}")

    raw_index = DenseTextIndex.build(
        memory,
        [str(item["memory_id"]) for item in active],
        [source_questions[str(item["question_id"])] for item in active],
    )
    correction_index = DenseTextIndex.build(
        memory,
        [str(item["memory_id"]) for item in active],
        [correction_text(item) for item in active],
    )
    manifests = _build_manifests(
        baseline, Path(args.current_guided_dir), root, source_questions, active,
    )
    overlap = {
        int(row["idx"])
        for phase_rows in manifests.values()
        for row in phase_rows
        if row.get("exact_memory_source_overlap")
    }

    if args.phase == "report":
        phase_names = ("calibration", "heldout")
    elif args.phase == "all":
        phase_names = ("smoke", "calibration", "heldout")
    else:
        phase_names = (args.phase,)
    suite_protocols = PROTOCOLS if args.suite == "legacy" else SEMANTIC_HYBRID_PROTOCOLS
    protocols = suite_protocols if args.protocol == "all" else (args.protocol,)
    if any(protocol not in suite_protocols for protocol in protocols):
        raise ValueError(f"Protocol {protocols} does not belong to suite {args.suite}")

    needs_cross_encoder = any(protocol in {"e3_cross_encoder", "h3_draft_soft_rerank"} for protocol in protocols)
    needs_hybrid = any(protocol.startswith("h") for protocol in protocols)
    cross_encoder = (
        BGECrossEncoderReranker(args.reranker_model_path, device=args.reranker_device)
        if needs_cross_encoder else None
    )
    hybrid_index = (
        BGEHybridFacetIndex(
            args.hybrid_model_path,
            active,
            device=args.hybrid_device,
            cache_path=root / "shared" / "bge_m3_hybrid_passages.pkl",
        )
        if needs_hybrid else None
    )

    manifest = {
        "schema_version": "stage3_retrieval_compare.2",
        "suite": args.suite,
        "protocols": list(protocols),
        "memory_dir": str(Path(args.memory_dir).resolve()),
        "memory_hash": _hash_json(active),
        "active_memory_count": len(active),
        "baseline_cache_dir": str(Path(args.baseline_cache_dir).resolve()),
        "baseline_hash": _hash_json(list(baseline.values())),
        "seeds": list(SEEDS),
        "temperature": 0.2,
        "source_overlap_count": len(overlap),
        "source_overlap_indices": sorted(overlap),
        "calibration_count": len(manifests["calibration"]),
        "heldout_count": len(manifests["heldout"]),
        "retrieval_parameters": {
            "facet_pool_k": args.facet_pool_k,
            "candidate_top_k": args.candidate_top_k,
            "rerank_top_k": args.rerank_top_k,
            "gate_top_k": args.gate_top_k,
            "max_injected": args.max_injected,
            "rrf_rank_constant": args.rrf_rank_constant,
            "draft_structure_weight": args.draft_structure_weight,
            "draft_error_weight": args.draft_error_weight,
            "reranker_model_path": str(Path(args.reranker_model_path).resolve()),
            "hybrid_model_path": str(Path(args.hybrid_model_path).resolve()),
        },
        "prompt_hashes": {
            "draft_signature": hashlib.sha256(DRAFT_SIGNATURE_PROMPT.encode()).hexdigest(),
            "draft_memory_judge": hashlib.sha256(DRAFT_MEMORY_JUDGE_PROMPT.encode()).hexdigest(),
            "forced_c": hashlib.sha256(FORCED_TIMING_AFTER_DRAFT_PROMPT.encode()).hexdigest(),
        },
    }
    old_manifest_path = root / "manifest.json"
    if old_manifest_path.exists() and args.resume:
        old_manifest = load_json(str(old_manifest_path))
        for key in ("memory_hash", "baseline_hash", "seeds", "temperature", "prompt_hashes", "retrieval_parameters"):
            if old_manifest.get(key) != manifest.get(key):
                raise ValueError(f"Resume manifest mismatch: {key}")
    atomic_save_json(old_manifest_path, manifest)

    if args.phase != "report":
        urls = [f"http://127.0.0.1:{port}/v1/chat/completions" for port in args.ports]
        healthy = filter_healthy_urls(urls)
        if len(healthy) != len(urls):
            raise RuntimeError(f"Expected {len(urls)} healthy endpoints, found {len(healthy)}")
        plan = PlanModule(config, urls=healthy, model_override=args.model, timeout=120, seed=42)
        execute = ExecuteModule(config, urls=healthy, model_override=args.model, timeout=300, seed=42)
        semantics_dir = root / "shared" / "semantics"
        semantics_dir.mkdir(parents=True, exist_ok=True)
        semantic_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

        async def semantics_for(idx: int) -> Dict[str, Any]:
            path = semantics_dir / f"{idx}.json"
            if path.exists():
                return load_json(str(path))["semantics"]
            async with semantic_locks[idx]:
                if path.exists():
                    return load_json(str(path))["semantics"]
                with llm_request_context(f"retrieval_compare_{idx}", "routing_semantics"):
                    value = await plan.extract_semantics(str(baseline[idx].get("question", "")))
                atomic_save_json(path, {"idx": idx, "semantics": value})
                return value

        for phase_name in phase_names:
            rows = manifests["calibration"][:10] if phase_name == "smoke" else manifests[phase_name]
            for protocol in protocols:
                question_dir = root / "protocols" / protocol / "questions"
                question_dir.mkdir(parents=True, exist_ok=True)
                semaphore = asyncio.Semaphore(max(1, args.concurrency))
                completed_count = 0
                count_lock = asyncio.Lock()
                progress = tqdm(total=len(rows), desc=f"{phase_name}:{protocol}", unit="题")

                async def task(row: Dict[str, Any]) -> None:
                    nonlocal completed_count
                    idx = int(row["idx"])
                    path = question_dir / f"{idx}.json"
                    async with semaphore:
                        semantics = await semantics_for(idx)
                        await _process_question(
                            protocol=protocol,
                            idx=idx,
                            baseline_record=baseline[idx],
                            memory=memory,
                            raw_index=raw_index,
                            correction_index=correction_index,
                            item_by_id=item_by_id,
                            semantics=semantics,
                            plan=plan,
                            execute=execute,
                            endpoints=healthy,
                            output_path=path,
                            facet_pool_k=args.facet_pool_k,
                            candidate_top_k=args.candidate_top_k,
                            rerank_top_k=args.rerank_top_k,
                            gate_top_k=args.gate_top_k,
                            max_injected=args.max_injected,
                            cross_encoder=cross_encoder,
                            hybrid_index=hybrid_index,
                            rrf_rank_constant=args.rrf_rank_constant,
                            draft_structure_weight=args.draft_structure_weight,
                            draft_error_weight=args.draft_error_weight,
                        )
                    async with count_lock:
                        completed_count += 1
                        progress.update(1)
                        logger.info("[%s/%s] 完成 %d/%d，题号=%d", phase_name, protocol, completed_count, len(rows), idx)

                logger.info("开始协议 %s，阶段=%s，题数=%d", protocol, phase_name, len(rows))
                await asyncio.gather(*(task(row) for row in rows))
                progress.close()

    calibration_ids = {int(row["idx"]) for row in manifests["calibration"]}
    heldout_ids = {int(row["idx"]) for row in manifests["heldout"]}
    summary: Dict[str, Any] = {"schema_version": "stage3_retrieval_compare_summary.1", "protocols": {}}
    records_by_protocol: Dict[str, List[Dict[str, Any]]] = {}
    for protocol in suite_protocols:
        question_dir = root / "protocols" / protocol / "questions"
        records = []
        if question_dir.exists():
            for path in sorted(question_dir.glob("*.json"), key=lambda p: int(p.stem)):
                value = load_json(str(path))
                if isinstance(value, dict):
                    records.append(value)
        records_by_protocol[protocol] = records
        if records:
            calibration_records = [row for row in records if int(row["idx"]) in calibration_ids]
            heldout_records = [row for row in records if int(row["idx"]) in heldout_ids]
            summary["protocols"][protocol] = {
                "calibration": _summarise(calibration_records, baseline, overlap),
                "heldout": _summarise(heldout_records, baseline, overlap),
                "combined": _summarise(records, baseline, overlap),
            }
        else:
            summary["protocols"][protocol] = {}
        if records:
            atomic_save_json(root / "protocols" / protocol / "eval_results.json", records)
            atomic_save_json(root / "protocols" / protocol / "summary.json", summary["protocols"][protocol])
            question_effects, candidate_effects = _effect_rows(records, baseline)
            effects_dir = root / "effects" / protocol
            effects_dir.mkdir(parents=True, exist_ok=True)
            atomic_save_json(effects_dir / "question_effects.json", question_effects)
            atomic_save_json(effects_dir / "candidate_effects.json", candidate_effects)
    summary["source_overlap_indices"] = sorted(overlap)
    summary["recommendation"] = _recommendation(summary["protocols"])
    atomic_save_json(root / "summary.json", summary)
    _write_report(root, summary, suite_protocols)
    _write_representative_cases(root, records_by_protocol, baseline)
    logger.info("检索协议实验结果：%s", root / "report.md")


if __name__ == "__main__":
    asyncio.run(main())
