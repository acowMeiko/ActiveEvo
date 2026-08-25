"""Empirically validate generated Principles on their source questions.

This module is deliberately read-only with respect to the Memory Store.  It
compares a frozen Solver under baseline, direct-Principle, and production-like
Gate/Adapter interventions, and persists pair-level evidence for inspection.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from benchmarks.matcher import check_answer
from core.config import Config
from core.request_context import llm_request_context
from modules.execute_module import (
    ExecuteModule,
    SOLVER_V1_PROMPT_HASHES,
    SOLVER_V1_PROTOCOL,
    filter_healthy_urls,
)
from modules.plan_module import PlanModule
from modules.principle_schema import render_solver_condition_card, render_solver_memory
from utils.io_utils import atomic_save_json, load_json

logger = logging.getLogger(__name__)

ARMS = ("baseline", "direct", "adapted")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_endpoint_index(question_id: str, temperature: float, seed: int, size: int) -> int:
    payload = f"{question_id}:{temperature:.6f}:{seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % size


def _record_id(arm: str, question_id: str, temperature: float, seed: int, candidate_id: str = "") -> str:
    owner = candidate_id if arm != "baseline" else question_id
    return f"{arm}:{owner}:t{temperature:.6f}:s{seed}"


def _experience_decision(text: str) -> str:
    match = re.search(r"Experience Decision:\s*(use|ignore)\b", text or "", flags=re.IGNORECASE)
    return match.group(1).lower() if match else "missing"


def paired_outcomes(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Classify matched seeds into gain/loss/both-correct/both-wrong."""
    base_by_seed = {int(item["seed"]): bool(item.get("correct")) for item in baseline}
    treatment_by_seed = {int(item["seed"]): bool(item.get("correct")) for item in treatment}
    if set(base_by_seed) != set(treatment_by_seed):
        raise ValueError("Paired validation requires identical seed sets")
    counts = Counter()
    for seed in sorted(base_by_seed):
        before, after = base_by_seed[seed], treatment_by_seed[seed]
        outcome = (
            "gain" if not before and after else
            "loss" if before and not after else
            "both_correct" if before else
            "both_wrong"
        )
        counts[outcome] += 1
    return {name: counts[name] for name in ("gain", "loss", "both_correct", "both_wrong")}


def effect_label(outcomes: Mapping[str, int]) -> str:
    gains, losses = int(outcomes.get("gain", 0)), int(outcomes.get("loss", 0))
    if gains > losses:
        return "source_helpful"
    if losses > gains:
        return "source_harmful"
    return "source_inconclusive"


def robust_effect_label(labels: Iterable[str]) -> str:
    values = list(labels)
    if values and all(value == "source_helpful" for value in values):
        return "robust_source_helpful"
    if any(value == "source_harmful" for value in values):
        return "temperature_sensitive_or_harmful"
    return "inconclusive"


def inherited_adapted_record(
    baseline: Mapping[str, Any],
    *,
    candidate_id: str,
    temperature: float,
    seed: int,
) -> Dict[str, Any]:
    """Represent fail-closed Gate/Adapter behavior without another Solver call."""
    return {
        **dict(baseline),
        "record_id": _record_id(
            "adapted", str(baseline["question_id"]), temperature, seed, candidate_id
        ),
        "candidate_id": candidate_id,
        "arm": "adapted",
        "injected": False,
        "experience_decision": "gate_or_adapter_reject",
        "derived_from_baseline": baseline["record_id"],
    }


def load_validation_inputs(artifact_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load every frozen candidate and its source bundle without modifying either."""
    bank_dir = artifact_root / "candidate_banks"
    bank_paths = sorted(bank_dir.glob("level*.json"))
    if not bank_paths:
        raise FileNotFoundError(f"No candidate banks found under {bank_dir}")
    candidates: List[Dict[str, Any]] = []
    hashes: Dict[str, str] = {}
    seen_ids = set()
    for bank_path in bank_paths:
        payload = load_json(bank_path)
        level = str(payload.get("level") or bank_path.stem)
        bundle_path = artifact_root / level / "00_learning_bundles" / "bundles.json"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing source bundles for {level}: {bundle_path}")
        bundles = load_json(bundle_path)
        bundle_by_id = {
            str(item.get("question_id", "")): item
            for item in bundles if isinstance(item, dict)
        }
        hashes[str(bank_path.relative_to(artifact_root))] = _sha256_file(bank_path)
        hashes[str(bundle_path.relative_to(artifact_root))] = _sha256_file(bundle_path)
        for raw in payload.get("candidates", []):
            if not isinstance(raw, dict):
                continue
            candidate_id = str(raw.get("candidate_id", ""))
            question_id = str(raw.get("question_id", ""))
            if not candidate_id or candidate_id in seen_ids:
                raise ValueError(f"Missing or duplicate candidate_id: {candidate_id!r}")
            if question_id not in bundle_by_id:
                raise ValueError(f"Candidate {candidate_id} has no source bundle for question {question_id}")
            content = raw.get("content")
            if not isinstance(content, dict):
                raise ValueError(f"Candidate {candidate_id} has no canonical content object")
            candidates.append({
                **raw,
                "candidate_id": candidate_id,
                "question_id": question_id,
                "level": level,
                "source_bundle": bundle_by_id[question_id],
            })
            seen_ids.add(candidate_id)
    candidates.sort(key=lambda item: item["candidate_id"])
    return candidates, hashes


def _candidate_for_gate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    bundle = candidate["source_bundle"]
    return {
        "memory_id": candidate["candidate_id"],
        "content": candidate["content"],
        "source_item": {
            "question": bundle["question_text"],
            "source_question_text": bundle["question_text"],
            "source_dsd": candidate.get("source_dsd", bundle.get("source_dsd", {})),
        },
    }


def _prompt_privacy_violations(
    prompt: str,
    candidate: Mapping[str, Any],
    *,
    adapted: bool,
) -> List[str]:
    """Detect metadata/trajectory leakage; mathematical overlap is reported separately."""
    lowered = (prompt or "").lower()
    violations = []
    forbidden_literals = {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "retrieval_score": "retrieval_score",
        "quality_audit": "quality_audit",
        "expected_answer_label": "expected_answer",
        "source_rollout_label": "source rollout",
    }
    for label, value in forbidden_literals.items():
        if value and value.lower() in lowered:
            violations.append(label)
    bundle = candidate.get("source_bundle", {})
    for rollout in bundle.get("rollouts", []):
        response = " ".join(str(rollout.get("response", "")).split())
        if len(response) >= 80 and response[:80].lower() in lowered:
            violations.append("source_trajectory_text")
            break
    if adapted:
        raw_action = " ".join(str(candidate.get("content", {}).get("action_guidance", "")).split())
        if len(raw_action) >= 80 and raw_action[:80].lower() in lowered:
            violations.append("raw_principle_copied_verbatim")
    return sorted(set(violations))


def _source_answer_overlap(text: str, candidate: Mapping[str, Any]) -> bool:
    """Flag source-answer literals that are absent from the question itself."""
    bundle = candidate["source_bundle"]
    answer = str(bundle.get("expected_answer", "")).strip().strip("$")
    question = str(bundle.get("question_text", ""))
    if not answer or answer in question:
        return False
    return answer in (text or "")


async def validate_source_principles(
    config: Config,
    *,
    artifact_root: Path,
    output_dir: Path,
    ports: Sequence[int],
    model: str,
    seeds: Sequence[int],
    temperatures: Sequence[float],
    concurrency: int = 2,
    timeout: int = 600,
    expected_candidates: int = 0,
    expected_questions: int = 0,
) -> Dict[str, Any]:
    """Run the three-arm source-question diagnostic without touching Memory."""
    started = time.monotonic()
    candidates, input_hashes = load_validation_inputs(Path(artifact_root))
    question_ids = sorted({item["question_id"] for item in candidates})
    sample_manifest_path = Path(artifact_root) / "sample_manifest.json"
    sample_manifest = load_json(str(sample_manifest_path)) if sample_manifest_path.exists() else {}
    expected_question_ids = sorted({
        str(value) for value in (
            sample_manifest.get("question_ids", []) if isinstance(sample_manifest, dict) else []
        ) if str(value).strip()
    }) or question_ids
    if expected_candidates and len(candidates) != expected_candidates:
        raise ValueError(f"Expected {expected_candidates} candidates, found {len(candidates)}")
    if expected_questions and len(expected_question_ids) != expected_questions:
        raise ValueError(
            f"Expected {expected_questions} manifest questions, found {len(expected_question_ids)}"
        )
    seeds = tuple(int(seed) for seed in seeds)
    temperatures = tuple(float(value) for value in temperatures)
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Validation seeds must be non-empty and unique")
    if not temperatures:
        raise ValueError("At least one validation temperature is required")

    urls = [f"http://127.0.0.1:{int(port)}/v1/chat/completions" for port in ports]
    healthy = filter_healthy_urls(urls)
    if len(healthy) != len(urls):
        raise RuntimeError(f"Expected {len(urls)} healthy Llama endpoints, found {len(healthy)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "artifact_root": str(Path(artifact_root).resolve()),
        "input_hashes": input_hashes,
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "question_ids": question_ids,
        "expected_question_ids": expected_question_ids,
        "model": model,
        "ports": list(map(int, ports)),
        "seeds": list(seeds),
        "temperatures": list(temperatures),
        "arms": list(ARMS),
        "solver_protocol": SOLVER_V1_PROTOCOL,
        "prompt_hashes": SOLVER_V1_PROMPT_HASHES,
        "memory_write": False,
        "answer_dependent_retry": False,
    }
    run_hash = hashlib.sha256(
        json.dumps(run_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = load_json(manifest_path)
        if previous.get("run_hash") != run_hash:
            raise ValueError("Validation output exists with a different run configuration")
    manifest = {**run_payload, "run_hash": run_hash}
    atomic_save_json(manifest_path, manifest)

    execute = ExecuteModule(config, urls=healthy, model_override=model, timeout=timeout, seed=seeds[0])
    plan = PlanModule(config, urls=healthy, model_override=model, timeout=min(timeout, 120), seed=seeds[0])
    request_semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    transport_failures: List[Dict[str, Any]] = []

    cards_path = output_dir / "adapted_cards.json"
    cards_payload = load_json(cards_path) if cards_path.exists() else []
    cards: Dict[str, Dict[str, Any]] = {
        str(item.get("candidate_id", "")): item
        for item in cards_payload if isinstance(item, dict) and item.get("candidate_id")
    }
    cards_lock = asyncio.Lock()
    expected_rollout_count = (
        len(question_ids) * len(temperatures) * len(seeds)
        + len(candidates) * len(temperatures) * len(seeds) * 2
    )

    async def prepare_card(candidate: Dict[str, Any]) -> Dict[str, Any]:
        candidate_id = candidate["candidate_id"]
        if candidate_id in cards:
            return cards[candidate_id]
        gate_candidate = _candidate_for_gate(candidate)
        async with request_semaphore:
            with llm_request_context(candidate_id, "source_validation/application_gate"):
                gate = await plan.judge_memory_applicability(
                    candidate["source_bundle"]["question_text"], [gate_candidate]
                )
        judgments = gate.get("judgments", [])
        judgment = judgments[0] if judgments else {"accepted": False, "reason": "missing_gate_judgment"}
        adapter = {
            "status": "reject", "condition_card": {}, "schema_errors": [],
            "reason": "gate_rejected", "raw_response": "",
        }
        if gate.get("protocol_compliant") and judgment.get("accepted"):
            async with request_semaphore:
                with llm_request_context(candidate_id, "source_validation/adapter"):
                    adapter = await plan.adapt_memory_for_question(
                        candidate["source_bundle"]["question_text"], gate_candidate, judgment
                    )
        card = adapter.get("condition_card", {}) if adapter.get("status") == "use" else {}
        solver_text = render_solver_condition_card(card) if card else ""
        result = {
            "candidate_id": candidate_id,
            "question_id": candidate["question_id"],
            "audit_decision": (candidate.get("quality_audit") or {}).get("decision", ""),
            "gate": gate,
            "adapter": adapter,
            "injected": bool(solver_text),
            "solver_text": solver_text,
            "privacy_violations": _prompt_privacy_violations(
                solver_text, candidate, adapted=True
            ) if solver_text else [],
            "source_answer_overlap": _source_answer_overlap(solver_text, candidate),
        }
        async with cards_lock:
            cards[candidate_id] = result
            atomic_save_json(cards_path, [cards[key] for key in sorted(cards)])
            logger.info(
                "Source validation cards: %d/%d candidate=%s gate=%s adapter=%s",
                len(cards), len(candidates), candidate_id,
                "accept" if judgment.get("accepted") else "reject",
                adapter.get("status", "reject"),
            )
        return result

    await asyncio.gather(*(prepare_card(candidate) for candidate in candidates))

    rollout_path = output_dir / "rollouts.json"
    previous_rollouts = load_json(rollout_path) if rollout_path.exists() else []
    rollout_by_id: Dict[str, Dict[str, Any]] = {
        str(item.get("record_id", "")): item
        for item in previous_rollouts
        if isinstance(item, dict) and item.get("record_id") and item.get("status") in {"complete", "infrastructure_failed"}
    }
    rollout_lock = asyncio.Lock()

    async def persist(record: Dict[str, Any]) -> Dict[str, Any]:
        async with rollout_lock:
            was_new = record["record_id"] not in rollout_by_id
            rollout_by_id[record["record_id"]] = record
            atomic_save_json(rollout_path, [rollout_by_id[key] for key in sorted(rollout_by_id)])
            with (output_dir / "transport_failures.jsonl").open("w", encoding="utf-8") as handle:
                for failure in transport_failures:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if was_new and (len(rollout_by_id) == 1 or len(rollout_by_id) % 10 == 0):
                logger.info(
                    "Source validation rollouts: %d/%d last=%s",
                    len(rollout_by_id), expected_rollout_count, record["record_id"],
                )
        return record

    async def solve(
        candidate: Dict[str, Any], arm: str, temperature: float, seed: int,
        principle_text: str,
    ) -> Dict[str, Any]:
        question_id = candidate["question_id"]
        candidate_id = candidate["candidate_id"] if arm != "baseline" else ""
        record_id = _record_id(arm, question_id, temperature, seed, candidate_id)
        if record_id in rollout_by_id:
            return rollout_by_id[record_id]
        primary = _stable_endpoint_index(question_id, temperature, seed, len(healthy))
        attempt_urls = [healthy[primary]]
        if len(healthy) > 1:
            attempt_urls.append(healthy[(primary + 1) % len(healthy)])
        trace = []
        response: Dict[str, Any] = {}
        async with request_semaphore:
            for endpoint_attempt, endpoint in enumerate(attempt_urls, 1):
                with llm_request_context(
                    candidate_id or question_id,
                    f"source_validation/{arm}/t={temperature}/seed={seed}/attempt={endpoint_attempt}",
                ):
                    response = await execute.solve_v1(
                        candidate["source_bundle"]["question_text"],
                        principle_text=principle_text,
                        seed=seed,
                        url_override=endpoint,
                        reasoner_temperature=temperature,
                        reasoner_timeout=min(timeout, 300),
                        finalizer_timeout=min(timeout, 60),
                    )
                trace.append({
                    "endpoint_attempt": endpoint_attempt,
                    "endpoint": endpoint,
                    "success": bool(response.get("infrastructure_success")),
                    "failed_stage": response.get("infrastructure_stage", ""),
                })
                if response.get("infrastructure_success"):
                    break
        infrastructure_success = bool(response.get("infrastructure_success"))
        if not infrastructure_success:
            transport_failures.append({
                "record_id": record_id, "candidate_id": candidate_id,
                "question_id": question_id, "arm": arm,
                "temperature": temperature, "seed": seed, "transport_trace": trace,
            })
        composed = str(response.get("composed_output", ""))
        solver_prompt = str((response.get("reasoner") or {}).get("prompt", ""))
        record = {
            "record_id": record_id,
            "candidate_id": candidate_id,
            "question_id": question_id,
            "level": candidate["level"],
            "arm": arm,
            "temperature": temperature,
            "seed": seed,
            "status": "complete" if infrastructure_success else "infrastructure_failed",
            "correct": check_answer(composed, candidate["source_bundle"]["expected_answer"])
            if infrastructure_success else False,
            "injected": bool(principle_text),
            "experience_decision": _experience_decision(
                str((response.get("reasoner") or {}).get("output", ""))
            ) if principle_text else "baseline",
            "response": response,
            "solver_prompt": solver_prompt,
            "privacy_violations": _prompt_privacy_violations(
                solver_prompt, candidate, adapted=arm == "adapted"
            ),
            "source_answer_overlap": _source_answer_overlap(principle_text, candidate),
            "transport_trace": trace,
        }
        return await persist(record)

    baseline_tasks: Dict[Tuple[str, float, int], asyncio.Task] = {}
    representative = {}
    for candidate in candidates:
        representative.setdefault(candidate["question_id"], candidate)

    def baseline_task(question_id: str, temperature: float, seed: int) -> asyncio.Task:
        key = (question_id, temperature, seed)
        if key not in baseline_tasks:
            baseline_tasks[key] = asyncio.create_task(
                solve(representative[question_id], "baseline", temperature, seed, "")
            )
        return baseline_tasks[key]

    for question_id in question_ids:
        for temperature in temperatures:
            for seed in seeds:
                baseline_task(question_id, temperature, seed)
    await asyncio.gather(*baseline_tasks.values())

    async def validate_candidate(candidate: Dict[str, Any]) -> None:
        direct_text = render_solver_memory(candidate["content"])
        card = cards[candidate["candidate_id"]]
        adapted_text = card.get("solver_text", "") if card.get("injected") else ""
        tasks = []
        for temperature in temperatures:
            for seed in seeds:
                tasks.append(solve(candidate, "direct", temperature, seed, direct_text))
                if adapted_text:
                    tasks.append(solve(candidate, "adapted", temperature, seed, adapted_text))
                else:
                    baseline = await baseline_task(candidate["question_id"], temperature, seed)
                    record_id = _record_id("adapted", candidate["question_id"], temperature, seed, candidate["candidate_id"])
                    if record_id not in rollout_by_id:
                        inherited = inherited_adapted_record(
                            baseline,
                            candidate_id=candidate["candidate_id"],
                            temperature=temperature,
                            seed=seed,
                        )
                        await persist(inherited)
        if tasks:
            await asyncio.gather(*tasks)
        logger.info("Source validation candidate complete: %s", candidate["candidate_id"])

    await asyncio.gather(*(validate_candidate(candidate) for candidate in candidates))

    effects = []
    for candidate in candidates:
        candidate_rows = []
        for temperature in temperatures:
            baseline = [
                rollout_by_id[_record_id("baseline", candidate["question_id"], temperature, seed)]
                for seed in seeds
            ]
            row: Dict[str, Any] = {
                "temperature": temperature,
                "baseline_correct_at_10": sum(bool(item["correct"]) for item in baseline),
                "baseline_pass1": bool(baseline[0]["correct"]),
                "baseline_pass10": any(bool(item["correct"]) for item in baseline),
            }
            for arm in ("direct", "adapted"):
                treatment = [
                    rollout_by_id[_record_id(arm, candidate["question_id"], temperature, seed, candidate["candidate_id"])]
                    for seed in seeds
                ]
                outcomes = paired_outcomes(baseline, treatment)
                correct = sum(bool(item["correct"]) for item in treatment)
                row[arm] = {
                    "correct_at_10": correct,
                    "pass1": bool(treatment[0]["correct"]),
                    "pass10": any(bool(item["correct"]) for item in treatment),
                    "delta_correct": correct - row["baseline_correct_at_10"],
                    "paired_outcomes": outcomes,
                    "effect_label": effect_label(outcomes),
                    "injected_rollout_count": sum(bool(item.get("injected")) for item in treatment),
                    "solver_use_count": sum(item.get("experience_decision") == "use" for item in treatment),
                    "solver_ignore_count": sum(item.get("experience_decision") == "ignore" for item in treatment),
                }
            candidate_rows.append(row)
        audit = candidate.get("quality_audit") or {}
        effects.append({
            "candidate_id": candidate["candidate_id"],
            "question_id": candidate["question_id"],
            "level": candidate["level"],
            "quality_status": candidate.get("status", ""),
            "audit_decision": audit.get("decision", ""),
            "audit_reasons": audit.get("reasons", []),
            "stage1_correct_count": (candidate["source_bundle"].get("capability_signal") or {}).get("correct_count"),
            "stage1_success_rate": (candidate["source_bundle"].get("capability_signal") or {}).get("success_rate"),
            "source_answer_overlap_direct": _source_answer_overlap(
                render_solver_memory(candidate["content"]), candidate
            ),
            "source_answer_overlap_adapted": cards[candidate["candidate_id"]].get("source_answer_overlap", False),
            "gate_protocol_compliant": bool((cards[candidate["candidate_id"]].get("gate") or {}).get("protocol_compliant")),
            "gate_selected": bool((cards[candidate["candidate_id"]].get("gate") or {}).get("selected_ids")),
            "adapter_status": (cards[candidate["candidate_id"]].get("adapter") or {}).get("status", "reject"),
            "temperatures": candidate_rows,
            "direct_robust_label": robust_effect_label(row["direct"]["effect_label"] for row in candidate_rows),
            "adapted_robust_label": robust_effect_label(row["adapted"]["effect_label"] for row in candidate_rows),
        })
    atomic_save_json(output_dir / "candidate_effects.json", effects)

    audit_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for effect in effects:
        audit_groups[effect["audit_decision"] or "unknown"].append(effect)
    audit_calibration = {}
    for decision, group in sorted(audit_groups.items()):
        audit_calibration[decision] = {
            "candidate_count": len(group),
            "direct_robust_helpful": sum(item["direct_robust_label"] == "robust_source_helpful" for item in group),
            "adapted_robust_helpful": sum(item["adapted_robust_label"] == "robust_source_helpful" for item in group),
            "direct_harmful_or_sensitive": sum(item["direct_robust_label"] == "temperature_sensitive_or_harmful" for item in group),
            "adapted_harmful_or_sensitive": sum(item["adapted_robust_label"] == "temperature_sensitive_or_harmful" for item in group),
        }
    atomic_save_json(output_dir / "audit_calibration.json", audit_calibration)

    all_rollouts = list(rollout_by_id.values())
    helpful_question_ids = {
        item["question_id"] for item in effects
        if item["direct_robust_label"] == "robust_source_helpful"
        or item["adapted_robust_label"] == "robust_source_helpful"
    }
    harmful_effects = [
        item for item in effects
        if item["direct_robust_label"] == "temperature_sensitive_or_harmful"
        or item["adapted_robust_label"] == "temperature_sensitive_or_harmful"
    ]
    summary = {
        "status": "complete",
        "run_hash": run_hash,
        "candidate_count": len(candidates),
        "source_question_count": len(question_ids),
        "manifest_question_count": len(expected_question_ids),
        "no_reliable_principle_question_count": len(set(expected_question_ids) - set(question_ids)),
        "quality_pass_count": sum(item.get("status") == "quality_pass" for item in candidates),
        "quality_rejected_count": sum(
            item.get("status") in {"hard_rejected", "audit_flagged", "rejected"}
            for item in candidates
        ),
        "temperatures": list(temperatures),
        "seeds": list(seeds),
        "rollout_count": len(all_rollouts),
        "infrastructure_failure_count": sum(item.get("status") != "complete" for item in all_rollouts),
        "gate_protocol_compliant_count": sum(item["gate_protocol_compliant"] for item in effects),
        "gate_selected_count": sum(item["gate_selected"] for item in effects),
        "adapter_use_count": sum(item["adapter_status"] == "use" for item in effects),
        "direct_robust_helpful_count": sum(item["direct_robust_label"] == "robust_source_helpful" for item in effects),
        "adapted_robust_helpful_count": sum(item["adapted_robust_label"] == "robust_source_helpful" for item in effects),
        "robust_helpful_question_count": len(helpful_question_ids),
        "robust_helpful_question_rate": (
            len(helpful_question_ids) / len(expected_question_ids) if expected_question_ids else 0.0
        ),
        "robust_helpful_question_ids": sorted(helpful_question_ids),
        "harmful_or_sensitive_candidate_count": len(harmful_effects),
        "harmful_or_sensitive_candidate_ids": sorted(
            item["candidate_id"] for item in harmful_effects
        ),
        "direct_source_answer_overlap_count": sum(item["source_answer_overlap_direct"] for item in effects),
        "adapted_source_answer_overlap_count": sum(item["source_answer_overlap_adapted"] for item in effects),
        "memory_store_modified": False,
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_save_json(output_dir / "summary.json", summary)

    report_lines = [
        "# Stage 2 Source-Question Principle Validation",
        "",
        "> This is a source-question diagnostic. It does not establish cross-question generalization and does not modify Memory.",
        "",
        f"- Candidates: {summary['candidate_count']} ({summary['quality_pass_count']} audit-pass, {summary['quality_rejected_count']} audit-rejected)",
        f"- Source questions with candidates: {summary['source_question_count']} / {summary['manifest_question_count']}",
        f"- Temperatures: {', '.join(map(str, temperatures))}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- Infrastructure failures: {summary['infrastructure_failure_count']}",
        f"- Direct robust helpful: {summary['direct_robust_helpful_count']} / {len(candidates)}",
        f"- Adapted robust helpful: {summary['adapted_robust_helpful_count']} / {len(candidates)}",
        f"- Questions with at least one robust helpful Principle: {summary['robust_helpful_question_count']} / {summary['manifest_question_count']}",
        f"- Harmful or temperature-sensitive candidates: {summary['harmful_or_sensitive_candidate_count']} / {len(candidates)}",
        f"- Gate protocol compliant: {summary['gate_protocol_compliant_count']} / {len(candidates)}",
        f"- Adapter produced usable cards: {summary['adapter_use_count']} / {len(candidates)}",
        "",
        "| Candidate | Audit | Direct | Adapted | Gate | Adapter |",
        "|---|---|---|---|---:|---|",
    ]
    for item in effects:
        report_lines.append(
            f"| {item['candidate_id']} | {item['audit_decision']} | {item['direct_robust_label']} | "
            f"{item['adapted_robust_label']} | {'yes' if item['gate_selected'] else 'no'} | {item['adapter_status']} |"
        )
    report_lines.extend(["", "## Audit calibration", "", "```json", json.dumps(audit_calibration, ensure_ascii=False, indent=2), "```", ""])
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Source validation complete: %s", output_dir)
    return summary
