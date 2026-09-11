"""Strict post-draft Stage-3 transfer experiment.

This module is intentionally isolated from the production Stage-3 evaluator.
It reuses frozen baseline drafts, performs three-facet joint retrieval, and
compares a memory-specific local correction with an equal-cost generic check.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from benchmarks.matcher import check_answer_for_item
from core.config import Config
from core.request_context import llm_request_context
from modules.execute_module import (
    ExecuteModule,
    SOLVER_V1_FINALIZER_MAX_TOKENS,
    SOLVER_V1_FINALIZER_TEMPERATURE,
    SOLVER_V1_PROMPT_HASHES,
    SOLVER_V1_TOP_P,
    filter_healthy_urls,
)
from modules.memory_module import MemoryModule
from modules.principle_schema import principle_content_from_item
from modules.reranker_module import BGECrossEncoderReranker
from modules.stage3_retrieval import retrieve_three_facet_joint
from template.prompt_template import (
    POSTDRAFT_GENERIC_LOCATOR_PROMPT,
    POSTDRAFT_GENERIC_REVISION_PROMPT,
    POSTDRAFT_MEMORY_REVISION_PROMPT,
    POSTDRAFT_MEMORY_VERIFIER_PROMPT,
    SOLVER_V1_FINALIZER_PROMPT,
)
from utils.io_utils import atomic_save_json, load_json

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "postdraft_strict_transfer.1"
EXACT_SOURCE_OVERLAP = {11, 91, 321, 373}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _prompt_hash(prompt: str) -> str:
    return _sha256_bytes(prompt.strip().encode("utf-8"))


def extract_json_object(text: str) -> Dict[str, Any] | None:
    """Parse an object, a fenced object, or the first complete JSON object."""
    value = str(text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for pos, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _exact_quote_in_draft(quote: Any, draft: str) -> bool:
    quote_text = _normalized_text(quote)
    return bool(quote_text) and quote_text in _normalized_text(draft)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def validate_memory_judgment(
    parsed: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    draft: str,
) -> Dict[str, Any]:
    """Fail-closed validator for the strict draft applicability response."""
    if not isinstance(parsed, Mapping):
        return {"accepted": False, "reason": "unparseable_json", "selected_memory_id": None}
    selected = parsed.get("selected_memory_id")
    if selected is None or str(selected).strip().lower() in {"", "null", "none"}:
        return {"accepted": False, "reason": "model_rejected", "selected_memory_id": None}
    selected = str(selected).strip()
    by_id = {str(row.get("memory_id", "")): row for row in candidates}
    if selected not in by_id:
        return {"accepted": False, "reason": "id_not_in_top3", "selected_memory_id": selected}
    required = {
        "problem_structure_match": True,
        "all_required_evidence_present": True,
        "exclusion_applies": False,
        "action_applicable": True,
    }
    for field, expected in required.items():
        if parsed.get(field) is not expected:
            return {"accepted": False, "reason": f"invalid_{field}", "selected_memory_id": selected}
    candidate = by_id[selected]
    content = candidate.get("content", {})
    signals = _strings(content.get("failure_signals"))
    matched = str(parsed.get("matched_failure_signal") or "").strip()
    if matched not in signals:
        return {"accepted": False, "reason": "failure_signal_not_exact", "selected_memory_id": selected}
    quote = str(parsed.get("draft_error_quote") or "").strip()
    if not _exact_quote_in_draft(quote, draft):
        return {"accepted": False, "reason": "draft_quote_not_exact", "selected_memory_id": selected}
    return {
        "accepted": True,
        "reason": "strict_conditions_satisfied",
        "selected_memory_id": selected,
        "matched_failure_signal": matched,
        "draft_error_quote": quote,
        "candidate": candidate,
    }


def validate_generic_locator(parsed: Mapping[str, Any] | None, draft: str) -> Dict[str, Any]:
    if not isinstance(parsed, Mapping):
        return {"issue_found": False, "valid": False, "reason": "unparseable_json"}
    if parsed.get("issue_found") is not True:
        return {"issue_found": False, "valid": True, "reason": "no_issue"}
    quote = str(parsed.get("draft_error_quote") or "").strip()
    if not _exact_quote_in_draft(quote, draft):
        return {"issue_found": False, "valid": False, "reason": "draft_quote_not_exact"}
    issue = str(parsed.get("issue_description") or "").strip()
    if not issue:
        return {"issue_found": False, "valid": False, "reason": "missing_issue_description"}
    return {"issue_found": True, "valid": True, "draft_error_quote": quote, "issue_description": issue}


def _candidate_for_reranker(entry: Mapping[str, Any]) -> Dict[str, Any] | None:
    item = entry.get("item", {})
    content, schema_source = principle_content_from_item(item)
    if not content:
        return None
    return {
        "memory_id": str(entry.get("memory_id") or item.get("memory_id", "")),
        "content": content,
        "schema_source": schema_source,
        "retrieval": dict(entry),
    }


def _routing_payload(candidates: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for candidate in candidates:
        content = candidate.get("content", {})
        rows.append({
            "memory_id": candidate.get("memory_id"),
            "trigger_condition": content.get("trigger_condition", ""),
            "required_evidence": content.get("required_evidence", []),
            "exclusion_condition": content.get("exclusion_condition", ""),
            "scope": content.get("scope", ""),
            "failure_signals": content.get("failure_signals", []),
        })
    return rows


def _baseline_item(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "question": str(record.get("question") or ""),
        "answer": str(record.get("expected_answer") or ""),
        "difficulty_level": record.get("difficulty_level", ""),
    }


def _baseline_draft(rollout: Mapping[str, Any]) -> str:
    return str((((rollout.get("response") or {}).get("reasoner") or {}).get("output")) or "").strip()


def validate_baseline_cache(
    baseline_dir: Path,
    *,
    seeds: Sequence[int],
    temperature: float,
) -> tuple[list[Dict[str, Any]], Dict[tuple[int, int], Dict[str, Any]], Dict[str, Any]]:
    results_path = baseline_dir / "eval_results.json"
    summary_path = baseline_dir / "eval_summary.json"
    prompt_path = baseline_dir / "prompt_snapshot.json"
    for path in (results_path, summary_path, prompt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline cache artifact: {path}")
    records = load_json(results_path)
    summary = load_json(summary_path)
    prompt = load_json(prompt_path)
    if not isinstance(records, list) or len(records) != 400:
        raise ValueError(f"Baseline cache must contain 400 questions, got {len(records) if isinstance(records, list) else 'non-list'}")
    if int(summary.get("infrastructure_failure_count", -1)) != 0:
        raise ValueError("Baseline cache contains infrastructure failures")
    if list(map(int, summary.get("seeds", []))) != list(map(int, seeds)):
        raise ValueError(f"Baseline seed mismatch: {summary.get('seeds')} != {list(seeds)}")
    if not math.isclose(float(summary.get("temperature", -1)), float(temperature), abs_tol=1e-9):
        raise ValueError("Baseline temperature mismatch")
    prompt_hashes = summary.get("prompt_hashes", {})
    if prompt_hashes.get("reasoner_base") != SOLVER_V1_PROMPT_HASHES["reasoner_base"]:
        raise ValueError("Baseline Solver prompt hash does not match current frozen Solver-v1.1")
    cache: Dict[tuple[int, int], Dict[str, Any]] = {}
    for expected_idx, record in enumerate(records):
        if int(record.get("idx", -1)) != expected_idx or record.get("status") != "complete":
            raise ValueError(f"Incomplete or unordered baseline record at position {expected_idx}")
        rollouts = record.get("rollouts", [])
        if [int(row.get("seed", -1)) for row in rollouts] != list(map(int, seeds)):
            raise ValueError(f"Baseline rollout seed mismatch for idx={expected_idx}")
        for row in rollouts:
            if row.get("status") != "complete" or not _baseline_draft(row):
                raise ValueError(f"Incomplete baseline rollout idx={expected_idx} seed={row.get('seed')}")
            cache[(expected_idx, int(row["seed"]))] = row
    metadata = {
        "results_sha256": _sha256_file(results_path),
        "prompt_snapshot_sha256": _sha256_file(prompt_path),
        "question_order_hash": _stable_hash([{"idx": r["idx"], "question": r["question"]} for r in records]),
        "prompt_hashes": prompt_hashes,
        "model": prompt.get("model", summary.get("model", "llama")),
        "temperature": float(temperature),
        "seeds": list(seeds),
        "count": len(records),
    }
    return records, cache, metadata


def _split_indices(split_root: Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, list[int]]:
    calibration = [int(row["idx"]) for row in load_json(split_root / "calibration_100.json")]
    heldout = [int(row["idx"]) for row in load_json(split_root / "heldout_300.json")]
    if len(calibration) != 100 or len(heldout) != 300 or set(calibration) & set(heldout):
        raise ValueError("Invalid 100/300 Stage-3 split")
    preferred = [77, 62, 378, 11, 91]
    extras = sorted(
        (idx for idx in calibration if idx not in preferred),
        key=lambda idx: hashlib.sha256(f"postdraft_smoke::{idx}".encode()).hexdigest(),
    )[:5]
    smoke = preferred + extras
    if any(idx < 0 or idx >= len(records) for idx in smoke):
        raise ValueError("Smoke split contains an unknown baseline index")
    return {"smoke": smoke, "calibration": calibration, "heldout": heldout}


def _bootstrap_ci(values: Sequence[float], seed: int = 42, repeats: int = 10000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        means[i] = rng.choice(array, size=len(array), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_effects(rows: Sequence[Mapping[str, Any]], *, seeds_per_question: int) -> Dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "complete"]
    accepted = [row for row in completed if row.get("accepted")]
    def pair(left: str, right: str) -> Dict[str, int]:
        return {
            "gain": sum(bool(row[left]) and not bool(row[right]) for row in accepted),
            "loss": sum(not bool(row[left]) and bool(row[right]) for row in accepted),
            "both_correct": sum(bool(row[left]) and bool(row[right]) for row in accepted),
            "both_wrong": sum(not bool(row[left]) and not bool(row[right]) for row in accepted),
        }
    by_question: Dict[int, list[Mapping[str, Any]]] = {}
    for row in accepted:
        by_question.setdefault(int(row["idx"]), []).append(row)
    question_deltas = []
    harmful = 0
    for group in by_question.values():
        delta = sum(int(bool(row["memory_correct"])) - int(bool(row["generic_correct"])) for row in group) / max(1, len(group))
        question_deltas.append(delta)
        if len(group) >= seeds_per_question and sum(int(bool(row["memory_correct"])) - int(bool(row["generic_correct"])) for row in group) <= -2:
            harmful += 1
    memory_generic = pair("memory_correct", "generic_correct")
    memory_baseline = pair("memory_correct", "baseline_correct")
    generic_baseline = pair("generic_correct", "baseline_correct")
    seed_values = sorted({int(row["seed"]) for row in completed})
    def pass_metrics(field: str) -> Dict[str, Any]:
        grouped: Dict[int, Dict[int, bool]] = {}
        for row in completed:
            grouped.setdefault(int(row["idx"]), {})[int(row["seed"])] = bool(row[field])
        first = seed_values[:1]
        first3 = seed_values[:3]
        return {
            "pass1_correct": sum(any(values.get(seed, False) for seed in first) for values in grouped.values()),
            "pass3_correct": sum(any(values.get(seed, False) for seed in first3) for values in grouped.values()),
            "pass10_correct": sum(any(values.get(seed, False) for seed in seed_values) for values in grouped.values()),
            "question_total": len(grouped),
            "seed_accuracy": {
                str(seed): sum(values.get(seed, False) for values in grouped.values()) / max(1, len(grouped))
                for seed in seed_values
            },
        }
    return {
        "total_pairs": len(completed),
        "accepted_pairs": len(accepted),
        "rejected_pairs": len(completed) - len(accepted),
        "infrastructure_failed": len(rows) - len(completed),
        "quote_valid_count": sum(bool(row.get("quote_valid")) for row in accepted),
        "quote_valid_rate": (sum(bool(row.get("quote_valid")) for row in accepted) / len(accepted)) if accepted else 0.0,
        "memory_vs_generic": memory_generic,
        "memory_vs_baseline": memory_baseline,
        "generic_vs_baseline": generic_baseline,
        "question_count_with_accept": len(by_question),
        "question_macro_delta_vs_generic": float(np.mean(question_deltas)) if question_deltas else 0.0,
        "question_macro_delta_ci95": _bootstrap_ci(question_deltas),
        "obviously_harmful_question_count": harmful,
        "obviously_harmful_question_rate": harmful / len(by_question) if by_question else 0.0,
        "baseline_metrics": pass_metrics("baseline_correct"),
        "generic_metrics": pass_metrics("generic_correct"),
        "memory_metrics": pass_metrics("memory_correct"),
    }


class PostDraftRunner:
    def __init__(self, config: Config, args: Any) -> None:
        self.config = config
        self.args = args
        self.root = Path(args.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seeds = tuple(int(value) for value in args.seeds)
        self.urls = filter_healthy_urls([f"http://127.0.0.1:{port}/v1/chat/completions" for port in args.ports])
        if len(self.urls) != len(args.ports):
            raise RuntimeError(f"Expected {len(args.ports)} healthy endpoints, found {len(self.urls)}")
        self.execute = ExecuteModule(config, urls=self.urls, model_override=args.model, timeout=300, seed=42)
        self.records, self.baseline, self.baseline_meta = validate_baseline_cache(
            Path(args.baseline_cache_dir), seeds=self.seeds, temperature=args.temperature,
        )
        self.splits = _split_indices(Path(args.split_root), self.records)
        for name, indices in self.splits.items():
            atomic_save_json(self.root / "splits" / f"{name}.json", [
                {"idx": idx, "difficulty_level": self.records[idx].get("difficulty_level"), "excluded_exact_source_overlap": idx in EXACT_SOURCE_OVERLAP}
                for idx in indices
            ])
        self.memories: Dict[str, MemoryModule] = {}
        self.memory_meta: Dict[str, Dict[str, Any]] = {}
        for arm, directory in (("172", args.memory_172_dir), ("264", args.memory_264_dir)):
            if args.memory_arm not in {"both", arm}:
                continue
            module = MemoryModule(config, memory_dir=directory)
            module.load()
            if not module.is_indexed:
                module.build_indices()
            count = sum(item.get("status") == "active" for item in module.memory.principles)
            expected = int(arm)
            if count != expected:
                raise ValueError(f"M{arm} must contain {expected} active memories, got {count}")
            principle_path = Path(directory) / "principles" / "principles.json"
            self.memories[arm] = module
            self.memory_meta[arm] = {"directory": str(Path(directory).resolve()), "count": count, "sha256": _sha256_file(principle_path)}
        self.reranker = BGECrossEncoderReranker(
            args.reranker_model_path, device=args.reranker_device, batch_size=16,
        )
        self.semaphore = asyncio.Semaphore(max(1, args.concurrency))
        self.prompt_hashes = {
            "memory_verifier": _prompt_hash(POSTDRAFT_MEMORY_VERIFIER_PROMPT),
            "memory_revision": _prompt_hash(POSTDRAFT_MEMORY_REVISION_PROMPT),
            "generic_locator": _prompt_hash(POSTDRAFT_GENERIC_LOCATOR_PROMPT),
            "generic_revision": _prompt_hash(POSTDRAFT_GENERIC_REVISION_PROMPT),
            "finalizer": SOLVER_V1_PROMPT_HASHES["finalizer"],
        }
        self.run_hash = _stable_hash({
            "schema": SCHEMA_VERSION, "baseline": self.baseline_meta,
            "memory": self.memory_meta, "prompts": self.prompt_hashes,
            "candidate_top_k": args.candidate_top_k, "rerank_top_k": args.rerank_top_k,
            "max_injected": args.max_injected, "temperature": args.temperature,
            "seeds": self.seeds, "model": args.model,
        })

    def _semantics(self, idx: int) -> Dict[str, Any]:
        path = Path(self.args.split_root) / "shared" / "semantics" / f"{idx}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen query semantics for idx={idx}: {path}")
        payload = load_json(path)
        semantics = payload.get("semantics", payload)
        required = ("core_skill", "context_constraints", "task_objective")
        if any(not str(semantics.get(field, "")).strip() for field in required):
            raise ValueError(f"Incomplete query semantics for idx={idx}")
        return semantics

    def _retrieve(self, arm: str, idx: int) -> Dict[str, Any]:
        path = self.root / "retrieval_cache" / f"memory_{arm}" / f"{idx}.json"
        if path.is_file():
            payload = load_json(path)
            if payload.get("cache_hash") == self.run_hash:
                return payload
        semantics = self._semantics(idx)
        raw, removed = retrieve_three_facet_joint(
            self.memories[arm].memory, semantics, top_k=self.args.candidate_top_k,
        )
        candidates = [row for row in (_candidate_for_reranker(entry) for entry in raw) if row]
        ranked = self.reranker.rank(
            self.records[idx]["question"], semantics, candidates, top_k=self.args.rerank_top_k,
        )
        payload = {
            "cache_hash": self.run_hash, "idx": idx, "memory_arm": arm,
            "query_semantics": semantics, "joint_candidates": candidates,
            "source_duplicates_removed": removed, "top_candidates": ranked,
        }
        atomic_save_json(path, payload)
        return payload

    def _attempt_urls(self, idx: int, seed: int, label: str) -> list[str]:
        start = int(hashlib.sha256(f"{idx}:{seed}:{label}".encode()).hexdigest(), 16) % len(self.urls)
        return [self.urls[start]] + ([self.urls[(start + 1) % len(self.urls)]] if len(self.urls) > 1 else [])

    async def _call(
        self, *, prompt: str, idx: int, seed: int, stage: str,
        max_tokens: int, temperature: float = 0.0,
    ) -> Dict[str, Any]:
        trace = []
        for attempt, endpoint in enumerate(self._attempt_urls(idx, seed, stage), 1):
            with llm_request_context(f"postdraft_q{idx}_s{seed}", f"{stage}/endpoint_attempt={attempt}"):
                meta, used = await self.execute._call_with_router_metadata(
                    prompt=prompt, temperature=temperature, top_p=SOLVER_V1_TOP_P,
                    max_tokens=max_tokens, repetition_penalty=1.0, seed=seed,
                    url_override=endpoint, timeout_override=self.args.timeout,
                )
            text = str(meta.get("text") or "").strip()
            trace.append({"attempt": attempt, "endpoint": endpoint, "used": used, "success": bool(text)})
            if text:
                return {"success": True, "text": text, "metadata": meta, "endpoint": used, "transport_trace": trace, "prompt": prompt}
        return {"success": False, "text": "", "transport_trace": trace, "prompt": prompt}

    async def _finalize(self, *, question: str, reasoning: str, idx: int, seed: int, stage: str) -> Dict[str, Any]:
        prompt = SOLVER_V1_FINALIZER_PROMPT.format(question=question, reasoner_output=reasoning)
        return await self._call(
            prompt=prompt, idx=idx, seed=seed, stage=stage,
            max_tokens=SOLVER_V1_FINALIZER_MAX_TOKENS,
            temperature=SOLVER_V1_FINALIZER_TEMPERATURE,
        )

    async def _generic(self, idx: int, seed: int, draft: str) -> Dict[str, Any]:
        path = self.root / "generic_control" / f"{idx}_{seed}.json"
        cache_hash = _stable_hash({"run": self.run_hash, "kind": "generic", "idx": idx, "seed": seed})
        if path.is_file():
            old = load_json(path)
            if old.get("cache_hash") == cache_hash and old.get("status") in {"complete", "infrastructure_failed"}:
                return old
        question = self.records[idx]["question"]
        locator_prompt = POSTDRAFT_GENERIC_LOCATOR_PROMPT.format(question=question, draft=draft)
        locator_call = await self._call(prompt=locator_prompt, idx=idx, seed=seed, stage="generic_locator", max_tokens=768)
        if not locator_call["success"]:
            result = {"cache_hash": cache_hash, "status": "infrastructure_failed", "locator": locator_call}
            atomic_save_json(path, result)
            return result
        locator = validate_generic_locator(extract_json_object(locator_call["text"]), draft)
        revision_prompt = POSTDRAFT_GENERIC_REVISION_PROMPT.format(
            question=question, draft=draft,
            draft_error_quote=locator.get("draft_error_quote", "No concrete error was located."),
            issue_description=locator.get("issue_description", "No concrete issue was located; retain the draft."),
        )
        revision = await self._call(
            prompt=revision_prompt, idx=idx, seed=seed, stage="generic_revision",
            max_tokens=4096, temperature=self.args.temperature,
        )
        if not revision["success"]:
            result = {"cache_hash": cache_hash, "status": "infrastructure_failed", "locator": locator_call, "locator_validation": locator, "revision": revision}
            atomic_save_json(path, result)
            return result
        finalizer = await self._finalize(question=question, reasoning=revision["text"], idx=idx, seed=seed, stage="generic_finalizer")
        if not finalizer["success"]:
            result = {"cache_hash": cache_hash, "status": "infrastructure_failed", "locator": locator_call, "locator_validation": locator, "revision": revision, "finalizer": finalizer}
            atomic_save_json(path, result)
            return result
        item = _baseline_item(self.records[idx])
        result = {
            "cache_hash": cache_hash, "status": "complete", "locator": locator_call,
            "locator_validation": locator, "revision": revision, "finalizer": finalizer,
            "correct": check_answer_for_item(f"{revision['text']}\n{finalizer['text']}", item),
            "final_answer": finalizer["text"],
        }
        atomic_save_json(path, result)
        return result

    async def _pair(self, phase: str, arm: str, idx: int, seed: int, retrieval: Mapping[str, Any]) -> Dict[str, Any]:
        path = self.root / "checkpoints" / phase / f"memory_{arm}" / f"{idx}_{seed}.json"
        pair_hash = _stable_hash({"run": self.run_hash, "phase": phase, "arm": arm, "idx": idx, "seed": seed, "candidates": [row.get("memory_id") for row in retrieval["top_candidates"]]})
        if self.args.resume and path.is_file():
            old = load_json(path)
            if old.get("checkpoint_hash") == pair_hash and old.get("status") in {"complete", "infrastructure_failed"}:
                return old
        baseline = self.baseline[(idx, seed)]
        draft = _baseline_draft(baseline)
        question = self.records[idx]["question"]
        top = retrieval["top_candidates"][: self.args.rerank_top_k]
        verifier_prompt = POSTDRAFT_MEMORY_VERIFIER_PROMPT.format(
            question=question, draft=draft,
            candidates=json.dumps(_routing_payload(top), ensure_ascii=False, indent=2),
        )
        verifier = await self._call(prompt=verifier_prompt, idx=idx, seed=seed, stage=f"memory_{arm}_verifier", max_tokens=1024)
        baseline_correct = bool(baseline.get("correct"))
        base = {
            "checkpoint_hash": pair_hash, "idx": idx, "seed": seed, "phase": phase,
            "memory_arm": arm, "baseline_correct": baseline_correct,
            "baseline_rollout": baseline, "retrieval": {"top_candidates": top}, "verifier": verifier,
            "excluded_exact_source_overlap": idx in EXACT_SOURCE_OVERLAP,
        }
        if not verifier["success"]:
            result = {**base, "status": "infrastructure_failed", "accepted": False}
            atomic_save_json(path, result)
            return result
        validation = validate_memory_judgment(extract_json_object(verifier["text"]), top, draft)
        if not validation["accepted"]:
            result = {
                **base, "status": "complete", "accepted": False, "judgment_validation": validation,
                "quote_valid": False, "memory_correct": baseline_correct,
                "generic_correct": baseline_correct, "baseline_reused_exactly": True,
            }
            atomic_save_json(path, result)
            return result
        selected = validation["candidate"]
        content = selected["content"]
        memory_prompt = POSTDRAFT_MEMORY_REVISION_PROMPT.format(
            question=question, draft=draft,
            draft_error_quote=validation["draft_error_quote"],
            matched_failure_signal=validation["matched_failure_signal"],
            action_guidance=content.get("action_guidance", ""),
            verification_step=content.get("verification_step", ""),
        )
        memory_first = int(hashlib.sha256(f"{idx}:{seed}".encode()).hexdigest(), 16) % 2 == 0

        async def memory_branch() -> Dict[str, Any]:
            revision = await self._call(
                prompt=memory_prompt, idx=idx, seed=seed, stage=f"memory_{arm}_revision",
                max_tokens=4096, temperature=self.args.temperature,
            )
            if not revision["success"]:
                return {"status": "infrastructure_failed", "revision": revision}
            finalizer = await self._finalize(question=question, reasoning=revision["text"], idx=idx, seed=seed, stage=f"memory_{arm}_finalizer")
            if not finalizer["success"]:
                return {"status": "infrastructure_failed", "revision": revision, "finalizer": finalizer}
            return {
                "status": "complete", "revision": revision, "finalizer": finalizer,
                "correct": check_answer_for_item(f"{revision['text']}\n{finalizer['text']}", _baseline_item(self.records[idx])),
                "final_answer": finalizer["text"],
            }

        if memory_first:
            memory_result = await memory_branch()
            generic = await self._generic(idx, seed, draft)
        else:
            generic = await self._generic(idx, seed, draft)
            memory_result = await memory_branch()
        if memory_result.get("status") != "complete" or generic.get("status") != "complete":
            result = {**base, "status": "infrastructure_failed", "accepted": True, "judgment_validation": {key: value for key, value in validation.items() if key != "candidate"}, "generic": generic, "memory": memory_result}
            atomic_save_json(path, result)
            return result
        result = {
            **base, "status": "complete", "accepted": True,
            "selected_memory_id": validation["selected_memory_id"],
            "judgment_validation": {key: value for key, value in validation.items() if key != "candidate"},
            "quote_valid": True, "injected_fields": {
                "draft_error_quote": validation["draft_error_quote"],
                "matched_failure_signal": validation["matched_failure_signal"],
                "action_guidance": content.get("action_guidance", ""),
                "verification_step": content.get("verification_step", ""),
            },
            "generic": generic, "memory": memory_result,
            "generic_correct": bool(generic["correct"]), "memory_correct": bool(memory_result["correct"]),
            "baseline_reused_exactly": True, "execution_order": "memory_then_generic" if memory_first else "generic_then_memory",
        }
        atomic_save_json(path, result)
        atomic_save_json(self.root / "draft_judgments" / phase / f"memory_{arm}" / f"{idx}_{seed}.json", {"verifier": verifier, "validation": result["judgment_validation"]})
        atomic_save_json(self.root / "memory_revision" / phase / f"memory_{arm}" / f"{idx}_{seed}.json", memory_result)
        return result

    async def run_phase(self, phase: str, arm: str) -> Dict[str, Any]:
        indices = self.splits[phase]
        seeds = self.seeds[:3] if phase == "smoke" else self.seeds
        logger.info("[严格草稿验证] 阶段=%s Memory=M%s，题目=%d，seed=%d", phase, arm, len(indices), len(seeds))
        retrievals = {}
        for idx in tqdm(indices, desc=f"{phase}/M{arm} 三维联合召回+精排", unit="题"):
            retrievals[idx] = self._retrieve(arm, idx)
        rows: list[Dict[str, Any]] = []
        lock = asyncio.Lock()
        progress = tqdm(total=len(indices) * len(seeds), desc=f"{phase}/M{arm} 草稿验证", unit="pair")

        async def task(idx: int, seed: int) -> None:
            async with self.semaphore:
                row = await self._pair(phase, arm, idx, seed, retrievals[idx])
            async with lock:
                rows.append(row)
                progress.update(1)

        await asyncio.gather(*(task(idx, seed) for idx in indices for seed in seeds))
        progress.close()
        rows.sort(key=lambda row: (int(row["idx"]), int(row["seed"])))
        metric_rows = rows
        if phase == "heldout":
            metric_rows = [row for row in rows if not row.get("excluded_exact_source_overlap")]
        summary = summarize_effects(metric_rows, seeds_per_question=len(seeds))
        summary.update({"phase": phase, "memory_arm": arm, "question_count": len(indices), "seed_count": len(seeds)})
        summary["excluded_exact_source_overlap_pairs"] = len(rows) - len(metric_rows)
        if arm == "264" and "172" in self.memories:
            ids172 = {str(item.get("memory_id", "")) for item in self.memories["172"].memory.principles if item.get("status") == "active"}
            extra_rows = [row for row in metric_rows if row.get("accepted") and str(row.get("selected_memory_id", "")) not in ids172]
            extra_effect = summarize_effects(extra_rows, seeds_per_question=len(seeds))
            summary["added_92_effect"] = extra_effect
        if phase == "smoke":
            summary["gate_passed"] = bool(
                summary["infrastructure_failed"] == 0
                and summary["accepted_pairs"] > 0
                and summary["rejected_pairs"] > 0
                and summary["quote_valid_rate"] == 1.0
            )
        elif phase == "calibration":
            pair = summary["memory_vs_generic"]
            summary["gate_passed"] = bool(
                pair["gain"] > pair["loss"]
                and summary["question_macro_delta_vs_generic"] > 0
                and summary["obviously_harmful_question_rate"] <= 0.10
                and summary["quote_valid_rate"] == 1.0
                and summary["infrastructure_failed"] == 0
            )
        else:
            summary["gate_passed"] = summary["infrastructure_failed"] == 0
        atomic_save_json(self.root / "effects" / f"{phase}_memory_{arm}_pairs.json", rows)
        atomic_save_json(self.root / "effects" / f"{phase}_memory_{arm}_summary.json", summary)
        logger.info("[严格草稿验证] 完成=%s M%s accept=%d gain/loss=%d/%d gate=%s", phase, arm, summary["accepted_pairs"], summary["memory_vs_generic"]["gain"], summary["memory_vs_generic"]["loss"], summary["gate_passed"])
        return summary

    def _write_report(self, summaries: Mapping[str, Any]) -> None:
        lines = ["# Stage 3 草稿后严格验证报告", "", f"- 协议：`{SCHEMA_VERSION}`", f"- Baseline：400题 × {len(self.seeds)} seeds，严格复用", "- 主要比较：Memory Treatment vs Generic Control", ""]
        for key, value in summaries.items():
            if not isinstance(value, Mapping) or "memory_vs_generic" not in value:
                continue
            pair = value["memory_vs_generic"]
            lines.extend([
                f"## {key}", "",
                f"- 接受/完成：{value['accepted_pairs']} / {value['total_pairs']}",
                f"- Memory vs Generic gain/loss：{pair['gain']} / {pair['loss']}",
                f"- 题目级平均变化：{value['question_macro_delta_vs_generic']:.4f}",
                f"- 明显有害题目比例：{value['obviously_harmful_question_rate']:.2%}",
                f"- 草稿引用合法率：{value['quote_valid_rate']:.2%}",
                f"- 是否通过门槛：{value['gate_passed']}", "",
            ])
        (self.root / "report.md").write_text("\n".join(lines), encoding="utf-8")
        cases = [
            "# Representative post-draft cases", "",
            "Cases are selected from completed accepted pairs. `gain` and `loss` are defined against the equal-cost generic control.", "",
        ]
        selected_rows: list[tuple[str, Mapping[str, Any], str]] = []
        for path in sorted((self.root / "effects").glob("*_pairs.json")):
            for row in load_json(path):
                if not row.get("accepted") or row.get("status") != "complete":
                    continue
                label = "gain" if row.get("memory_correct") and not row.get("generic_correct") else "loss" if not row.get("memory_correct") and row.get("generic_correct") else "neutral"
                if label in {"gain", "loss"} and sum(value[2] == label for value in selected_rows) < 5:
                    selected_rows.append((path.stem, row, label))
        for source, row, label in selected_rows:
            injected = row.get("injected_fields", {})
            cases.extend([
                f"## {label.upper()} — {source}, idx={row.get('idx')}, seed={row.get('seed')}", "",
                f"- Memory: `{row.get('selected_memory_id')}`",
                f"- Draft quote: {injected.get('draft_error_quote', '')}",
                f"- Matched signal: {injected.get('matched_failure_signal', '')}",
                f"- Action: {injected.get('action_guidance', '')}",
                f"- Verification: {injected.get('verification_step', '')}",
                f"- Baseline / Generic / Memory correct: {row.get('baseline_correct')} / {row.get('generic_correct')} / {row.get('memory_correct')}", "",
            ])
        (self.root / "representative_cases.md").write_text("\n".join(cases), encoding="utf-8")

    async def run(self) -> Dict[str, Any]:
        atomic_save_json(self.root / "manifest.json", {
            "schema_version": SCHEMA_VERSION, "run_hash": self.run_hash,
            "baseline": self.baseline_meta, "memories": self.memory_meta,
            "splits": {key: len(value) for key, value in self.splits.items()},
            "prompt_hashes": self.prompt_hashes, "candidate_top_k": self.args.candidate_top_k,
            "rerank_top_k": self.args.rerank_top_k, "max_injected": self.args.max_injected,
            "adapter_enabled": False, "solver_use_ignore_enabled": False,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        requested_phases = [self.args.postdraft_phase] if self.args.postdraft_phase != "all" else ["smoke", "calibration", "heldout"]
        summaries: Dict[str, Any] = {}
        active_arms = list(self.memories)
        for phase in requested_phases:
            phase_summaries = {}
            for arm in active_arms:
                if phase == "heldout":
                    calibration = summaries.get(f"calibration_m{arm}")
                    if calibration is None:
                        calibration_path = self.root / "effects" / f"calibration_memory_{arm}_summary.json"
                        calibration = load_json(calibration_path) if calibration_path.is_file() else None
                    if not calibration or not calibration.get("gate_passed"):
                        logger.warning("M%s未通过100题校准，不进入heldout", arm)
                        continue
                    if arm == "264" and "172" in self.memories:
                        m172 = summaries.get("calibration_m172")
                        if m172 is None:
                            p172 = self.root / "effects" / "calibration_memory_172_summary.json"
                            m172 = load_json(p172) if p172.is_file() else None
                        marginal = calibration.get("added_92_effect", {}).get("memory_vs_generic", {})
                        marginal_nonnegative = int(marginal.get("gain", 0)) >= int(marginal.get("loss", 0))
                        if (
                            not m172
                            or calibration["question_macro_delta_vs_generic"] < m172["question_macro_delta_vs_generic"]
                            or not marginal_nonnegative
                        ):
                            logger.warning("M264校准结果差于M172或新增92条净效应为负，不进入heldout")
                            continue
                summary = await self.run_phase(phase, arm)
                summaries[f"{phase}_m{arm}"] = summary
                phase_summaries[arm] = summary
            if phase in {"smoke", "calibration"} and self.args.postdraft_phase == "all":
                if "172" in active_arms and not phase_summaries.get("172", {}).get("gate_passed"):
                    logger.warning("M172未通过%s门槛，流程按协议停止", phase)
                    break
                if phase == "smoke":
                    active_arms = [arm for arm in active_arms if phase_summaries.get(arm, {}).get("gate_passed")]
        atomic_save_json(self.root / "summary.json", summaries)
        self._write_report(summaries)
        return summaries


async def run_postdraft_transfer(config: Config, args: Any) -> Dict[str, Any]:
    if int(args.max_injected) != 1:
        raise ValueError("postdraft-transfer requires --max-injected 1")
    if int(args.rerank_top_k) != 3:
        logger.warning("Strict verifier was designed for Top-3; using requested Top-%d", args.rerank_top_k)
    runner = PostDraftRunner(config, args)
    return await runner.run()
