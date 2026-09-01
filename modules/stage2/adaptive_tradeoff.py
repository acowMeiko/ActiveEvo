"""Isolated draft-aware Principle intervention experiment.

This module does not build or mutate a Memory Store.  It reuses frozen source
questions, source Principles, and unguided Llama drafts, then compares one
adaptive intervention protocol against a no-Memory process control.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.matcher import answers_equivalent, check_answer, extract_final_answer
from core.request_context import llm_request_context
from modules.execute_module import (
    SOLVER_V1_FINALIZER_MAX_TOKENS,
    SOLVER_V1_FINALIZER_TEMPERATURE,
    SOLVER_V1_REASONER_MAX_TOKENS,
    SOLVER_V1_TOP_P,
    _completion_tokens,
    _finalizer_format_valid,
    _finish_reason,
    _repetition_diagnostics,
)
from modules.stage2.model_client import parse_json_object
from modules.stage2.principle_test import (
    _PairedRunner,
    _effect_label,
    _gather_with_progress,
    _healthy_actor_urls,
    _json_hash,
    _paired_counts,
    _stable_endpoint_index,
    _summarize_effect_rows,
)
from modules.stage2.protocol_calibration import (
    _baseline_by_seed,
    _compact_current_draft,
    _forced_prompt_privacy_violations,
    _forced_question_effects,
    _load_forced_timing_cache,
    _load_intervention_sources,
)
from template.prompt_template import SOLVER_V1_FINALIZER_PROMPT
from utils.io_utils import atomic_save_json, load_json


logger = logging.getLogger(__name__)

ADAPTIVE_VERSION = "principle_adaptive_tradeoff.1"
SCORING_VERSION = "reasoner_support.2"
ROUTE_DECISIONS = {"keep", "local_repair", "restart"}


PRINCIPLE_ROUTE_PROMPT = r'''
You are a conservative diagnostic router for a mathematical solver.

Your job is only to compare the existing draft with the supplied diagnostic
Principle. Do not solve the problem, calculate a new answer, or rewrite the
draft.

Choose exactly one decision:
- keep: the draft does not exhibit the described failure, an exclusion applies,
  or the evidence is insufficient.
- local_repair: the main method remains viable, but a specific local step is
  wrong, missing, or insufficiently checked.
- restart: the draft's target, representation, formula, or overall method is
  invalid and cannot be repaired locally.

Safety rules:
1. Default to keep.
2. local_repair and restart require at least one exact draft quotation.
3. restart additionally requires a method-level failure and an explanation of
   why a local edit cannot make the method valid.
4. If an exclusion applies, return keep.
5. Do not infer that the Principle is reliable merely because it was supplied.
6. A warning sign or an omitted optional detail is not itself an error. State
   the concrete mathematical contradiction before intervening.

Return raw JSON only:
{{
  "decision": "keep|local_repair|restart",
  "trigger_satisfied": true,
  "exclusion_hit": false,
  "matched_evidence": [
    {{
      "principle_requirement": "...",
      "problem_fact": "...",
      "draft_span": "exact short quotation from the draft",
      "why_incorrect": "the concrete contradiction, not merely a warning sign"
    }}
  ],
  "error_scope": "none|local|method_level",
  "reason": "brief diagnostic reason",
  "repair_instruction": "brief instruction; empty when decision is keep"
}}

Current problem:
{question}

Existing unguided draft:
{draft}

Principle diagnostic fields:
{routing_view}
'''


NO_MEMORY_ROUTE_PROMPT = r'''
You are a conservative diagnostic router for a mathematical solver.

Inspect the existing draft using only the current problem and ordinary
mathematical verification. No external Principle or Memory is available.
Do not solve the problem, calculate a new answer, or rewrite the draft.

Choose exactly one decision:
- keep: no concrete error is established.
- local_repair: the main method remains viable, but a specific local step is
  wrong, missing, or insufficiently checked.
- restart: the target, representation, formula, or overall method is invalid
  and cannot be repaired locally.

Safety rules:
1. Default to keep.
2. local_repair and restart require at least one exact draft quotation.
3. restart additionally requires a method-level failure and an explanation of
   why a local edit cannot make the method valid.
4. A warning sign or an omitted optional detail is not itself an error. State
   the concrete mathematical contradiction before intervening.

Return raw JSON only with exactly this schema:
{{
  "decision": "keep|local_repair|restart",
  "trigger_satisfied": true,
  "exclusion_hit": false,
  "matched_evidence": [
    {{
      "principle_requirement": "generic mathematical requirement",
      "problem_fact": "fact from the current problem",
      "draft_span": "exact short quotation from the draft",
      "why_incorrect": "the concrete mathematical contradiction"
    }}
  ],
  "error_scope": "none|local|method_level",
  "reason": "brief diagnostic reason",
  "repair_instruction": "brief instruction; empty when decision is keep"
}}

Current problem:
{question}

Existing unguided draft:
{draft}
'''


ADAPTIVE_EXECUTION_PROMPT = r'''
You are revising an existing mathematical solution.

Intervention mode: {decision}

Mode rules:
- local_repair: preserve every valid part of the draft and correct only the
  diagnosed local problem.
- restart: discard the invalid method and derive a complete solution again.

The diagnostic result identifies where intervention is allowed:
{diagnosis}

Conditional guidance:
{guidance}

Use the guidance as a fallible suggestion. Verify it against the current
problem. Do not obey it when it conflicts with the problem facts. Do not mention
Memory, retrieval, audit, candidates, or source examples.

Return exactly these sections:

Intervention Mode:
{decision}

Method:
[one coherent method]

Reasoning:
[complete compact derivation]

Check:
[direct verification]

Derived Result:
[the answer supported by the derivation; do not use a box]

Current problem:
{question}

Existing unguided draft:
{draft}
'''


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def principle_routing_view(content: Mapping[str, Any]) -> dict[str, Any]:
    """Fields used to decide whether and how strongly to intervene."""
    return {
        "trigger_condition": content.get("trigger_condition", ""),
        "required_evidence": content.get("required_evidence", []),
        "exclusion_condition": content.get("exclusion_condition", ""),
        "failure_mode": content.get("failure_mode", ""),
        "failure_signals": content.get("failure_signals", []),
    }


def principle_action_view(content: Mapping[str, Any]) -> dict[str, str]:
    """Only executable guidance; routing and provenance never reach the solver."""
    return {
        "action_guidance": _clean_text(content.get("action_guidance")),
        "verification_step": _clean_text(content.get("verification_step")),
    }


def validate_route(value: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate and conservatively downgrade an LLM route decision."""
    if not isinstance(value, Mapping):
        return {
            "decision": "keep", "trigger_satisfied": False,
            "exclusion_hit": False, "matched_evidence": [],
            "error_scope": "none", "reason": "route_not_an_object",
            "repair_instruction": "",
        }, ["route_must_be_object"]

    errors: list[str] = []
    decision = _clean_text(value.get("decision")).lower()
    if decision not in ROUTE_DECISIONS:
        errors.append("invalid_decision")
        decision = "keep"
    trigger = value.get("trigger_satisfied")
    exclusion = value.get("exclusion_hit")
    if not isinstance(trigger, bool):
        errors.append("trigger_satisfied_must_be_boolean")
        trigger = False
    if not isinstance(exclusion, bool):
        errors.append("exclusion_hit_must_be_boolean")
        exclusion = False
    scope = _clean_text(value.get("error_scope")).lower()
    if scope not in {"none", "local", "method_level"}:
        errors.append("invalid_error_scope")
        scope = "none"

    raw_evidence = value.get("matched_evidence")
    evidence: list[dict[str, str]] = []
    if not isinstance(raw_evidence, list):
        errors.append("matched_evidence_must_be_list")
        raw_evidence = []
    for index, item in enumerate(raw_evidence):
        if not isinstance(item, Mapping):
            errors.append(f"matched_evidence_{index}_must_be_object")
            continue
        normalized = {
            "principle_requirement": _clean_text(item.get("principle_requirement")),
            "problem_fact": _clean_text(item.get("problem_fact")),
            "draft_span": _clean_text(item.get("draft_span")),
            "why_incorrect": _clean_text(item.get("why_incorrect")),
        }
        if all(normalized.values()):
            evidence.append(normalized)
        else:
            errors.append(f"matched_evidence_{index}_incomplete")

    reason = _clean_text(value.get("reason"))
    repair = _clean_text(value.get("repair_instruction"))

    # Fail closed.  A malformed or unsupported intervention becomes keep, not
    # a content rejection and never an aggressive restart.
    unsafe = bool(errors) or exclusion
    if decision != "keep":
        unsafe = unsafe or not trigger or not evidence or not repair
    if decision == "local_repair" and scope != "local":
        unsafe = True
    if decision == "restart" and scope != "method_level":
        unsafe = True
    if unsafe:
        decision = "keep"
        scope = "none"
        repair = ""

    return {
        "decision": decision,
        "trigger_satisfied": bool(trigger),
        "exclusion_hit": bool(exclusion),
        "matched_evidence": evidence,
        "error_scope": scope,
        "reason": reason or ("kept_by_safety_policy" if unsafe else ""),
        "repair_instruction": repair,
    }, errors


def _derived_result(reasoner_output: str) -> str:
    match = re.search(
        r"Derived Result:\s*(.+?)(?:\n\s*\n|\Z)",
        reasoner_output or "", flags=re.IGNORECASE | re.DOTALL,
    )
    return _clean_text(match.group(1)) if match else ""


def _answer_supported(reasoner_output: str, finalizer_output: str) -> bool:
    derived = _derived_result(reasoner_output)
    final_answer = extract_final_answer(finalizer_output, allow_short_direct=False)
    if not final_answer:
        return False
    compact_final = str(final_answer).replace(" ", "")
    if derived:
        derived_candidates = [derived]
        # Models often write "the result is <expression>".  Compare the final
        # expression as mathematics instead of requiring identical surface text.
        for marker in (" is ", " = "):
            if marker in derived:
                derived_candidates.append(derived.rsplit(marker, 1)[-1].rstrip(". "))
        if any(
            answers_equivalent(value, str(final_answer))
            or compact_final in value.replace(" ", "")
            for value in derived_candidates
        ):
            return True
    return compact_final in (reasoner_output or "").replace(" ", "")


def parse_route_output(text: str) -> tuple[Any, str]:
    """Parse raw JSON, repairing only invalid JSON backslash escaping."""
    parsed = parse_json_object(text)
    if isinstance(parsed, Mapping) and parsed.get("decision"):
        return parsed, "raw_json"
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text or "")
    parsed = parse_json_object(repaired)
    if isinstance(parsed, Mapping) and parsed.get("decision"):
        return parsed, "escaped_invalid_backslashes"
    return parsed, "failed"


class _AdaptiveRunner:
    def __init__(
        self, config: Any, *, root: Path, urls: Sequence[str], model: str,
        concurrency: int, timeout: int,
    ) -> None:
        self.root = Path(root)
        self.urls = list(urls)
        self.model = model
        self.timeout = int(timeout)
        self.runner = _PairedRunner(
            config, output_dir=self.root / "internal_solver_state",
            urls=urls, model=model, concurrency=concurrency, timeout=timeout,
        )
        # Hold one slot for the complete route -> optional repair -> finalizer
        # transaction.  A per-HTTP-call semaphore lets hundreds of route calls
        # overtake pending finalizers and produces a severe tail-latency convoy.
        self.transaction_semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    def _endpoints(self, key: str, seed: int, temperature: float) -> list[str]:
        primary = _stable_endpoint_index(key, temperature, seed, len(self.urls))
        values = [self.urls[primary]]
        if len(self.urls) > 1:
            values.append(self.urls[(primary + 1) % len(self.urls)])
        return values

    async def _call(
        self, *, prompt: str, key: str, stage: str, seed: int,
        temperature: float, max_tokens: int, timeout: int,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        response: dict[str, Any] = {}
        endpoint_used = ""
        trace: list[dict[str, Any]] = []
        for attempt, endpoint in enumerate(self._endpoints(key, seed, temperature), 1):
            with llm_request_context(key, f"adaptive_tradeoff/{stage}/seed={seed}/attempt={attempt}"):
                response, endpoint_used = await self.runner.execute._call_with_router_metadata(
                    prompt=prompt,
                    temperature=temperature,
                    top_p=SOLVER_V1_TOP_P,
                    max_tokens=max_tokens,
                    repetition_penalty=1.0,
                    seed=seed,
                    url_override=endpoint,
                    timeout_override=min(self.timeout, timeout),
                )
            success = bool(response.get("text"))
            trace.append({"endpoint": endpoint, "attempt": attempt, "success": success})
            if success:
                break
        return response, endpoint_used, trace

    async def route(
        self, *, arm: str, question_id: str, question: str, draft: str,
        seed: int, candidate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_id = str((candidate or {}).get("candidate_id", ""))
        name = f"{candidate_id}_s{seed}.json" if candidate_id else f"{question_id}_s{seed}.json"
        path = self.root / "routes" / arm / name
        if arm == "adaptive_memory":
            routing_view = principle_routing_view((candidate or {}).get("content", {}))
            prompt = PRINCIPLE_ROUTE_PROMPT.format(
                question=question,
                draft=_compact_current_draft(draft),
                routing_view=json.dumps(routing_view, ensure_ascii=False, indent=2),
            )
        else:
            prompt = NO_MEMORY_ROUTE_PROMPT.format(
                question=question, draft=_compact_current_draft(draft),
            )
        input_hash = _json_hash({
            "version": ADAPTIVE_VERSION, "arm": arm, "prompt": prompt,
            "model": self.model, "seed": seed, "temperature": 0.0,
        })
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash and saved.get("status") == "complete":
                return saved

        key = candidate_id or question_id
        response, endpoint, trace = await self._call(
            prompt=prompt, key=f"{arm}:{key}", stage=f"{arm}/route",
            seed=seed, temperature=0.0, max_tokens=1024, timeout=120,
        )
        text = str(response.get("text") or "").strip()
        parsed, parse_stage = parse_route_output(text)
        route, errors = validate_route(parsed)
        status = "complete" if text else "infrastructure_failed"
        if status != "complete":
            route, errors = validate_route(None)
        record = {
            "input_hash": input_hash,
            "status": status,
            "arm": arm,
            "question_id": question_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "route": route,
            "parse_success": bool(text) and not errors,
            "parse_stage": parse_stage,
            "schema_errors": errors,
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output": text,
            "finish_reason": _finish_reason(response),
            "completion_tokens": _completion_tokens(response),
            "endpoint": endpoint,
            "transport_trace": trace,
        }
        atomic_save_json(path, record)
        return record

    async def execute(
        self, *, arm: str, question_id: str, question: str, expected_answer: str,
        draft: str, seed: int, temperature: float, route_record: Mapping[str, Any],
        baseline_record: Mapping[str, Any], candidate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_id = str((candidate or {}).get("candidate_id", ""))
        name = f"{candidate_id}_s{seed}.json" if candidate_id else f"{question_id}_s{seed}.json"
        path = self.root / "rollouts" / arm / name
        route = dict(route_record.get("route") or {})
        decision = str(route.get("decision", "keep"))

        if arm == "adaptive_memory":
            action = principle_action_view((candidate or {}).get("content", {}))
            guidance = "\n".join([
                f"Recommended procedure: {action['action_guidance']}",
                f"Verification procedure: {action['verification_step']}",
            ])
        else:
            guidance = "Use only this diagnosis-derived instruction:\n" + str(
                route.get("repair_instruction", "")
            )

        prompt = ""
        if decision != "keep":
            prompt = ADAPTIVE_EXECUTION_PROMPT.format(
                decision=decision,
                diagnosis=json.dumps(route, ensure_ascii=False, indent=2),
                guidance=guidance,
                question=question,
                draft=_compact_current_draft(draft),
            )
        input_hash = _json_hash({
            "version": ADAPTIVE_VERSION, "arm": arm, "question_id": question_id,
            "candidate_id": candidate_id, "seed": seed, "temperature": temperature,
            "route_hash": route_record.get("input_hash", ""), "guidance": guidance,
            "execution_prompt": prompt,
        })
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash and saved.get("status") == "complete":
                if saved.get("route_decision") != "keep" and saved.get("scoring_version") != SCORING_VERSION:
                    reasoner_output = str((saved.get("reasoner") or {}).get("output", ""))
                    finalizer_output = str((saved.get("finalizer") or {}).get("output", ""))
                    composed = f"{reasoner_output}\n\n{finalizer_output}".strip()
                    raw_correct = check_answer(composed, expected_answer)
                    supported = _answer_supported(reasoner_output, finalizer_output)
                    saved["raw_correct"] = bool(raw_correct)
                    saved["reasoning_supported"] = bool(supported)
                    saved["support_status"] = "supported" if supported else "unsupported_final_answer"
                    saved["correct"] = bool(raw_correct and supported)
                    saved["scoring_version"] = SCORING_VERSION
                    atomic_save_json(path, saved)
                return saved

        initial_correct = bool(baseline_record.get("correct"))
        if decision == "keep":
            record = {
                "input_hash": input_hash,
                "status": "complete",
                "arm": arm,
                "question_id": question_id,
                "candidate_id": candidate_id,
                "seed": seed,
                "temperature": temperature,
                "route_decision": "keep",
                "route_record": str(
                    self.root / "routes" / arm
                    / (f"{candidate_id}_s{seed}.json" if candidate_id else f"{question_id}_s{seed}.json")
                ),
                "correct": initial_correct,
                "raw_correct": initial_correct,
                "reasoning_supported": True,
                "support_status": "reused_unguided_draft",
                "scoring_version": SCORING_VERSION,
                "final_answer": extract_final_answer(
                    str(((baseline_record.get("response") or {}).get("finalizer") or {}).get("output", "")),
                    allow_short_direct=False,
                ),
                "reasoner": (baseline_record.get("response") or {}).get("reasoner", {}),
                "finalizer": (baseline_record.get("response") or {}).get("finalizer", {}),
                "prompt": "",
                "prompt_hash": "",
                "principle_action_injected": False,
                "transport_trace": [],
                "privacy_violations": [],
            }
            atomic_save_json(path, record)
            return record

        key = candidate_id or question_id
        reasoner_meta, endpoint, trace = await self._call(
            prompt=prompt, key=f"{arm}:{key}", stage=f"{arm}/execute",
            seed=seed, temperature=temperature,
            max_tokens=SOLVER_V1_REASONER_MAX_TOKENS, timeout=300,
        )
        reasoner_output = str(reasoner_meta.get("text") or "").strip()
        finalizer_meta: dict[str, Any] = {}
        finalizer_output = ""
        if reasoner_output:
            finalizer_prompt = SOLVER_V1_FINALIZER_PROMPT.format(
                question=question, reasoner_output=reasoner_output,
            )
            finalizer_meta, final_endpoint, final_trace = await self._call(
                prompt=finalizer_prompt, key=f"{arm}:{key}", stage=f"{arm}/finalizer",
                seed=seed, temperature=SOLVER_V1_FINALIZER_TEMPERATURE,
                max_tokens=SOLVER_V1_FINALIZER_MAX_TOKENS, timeout=60,
            )
            endpoint = final_endpoint or endpoint
            trace.extend(final_trace)
            finalizer_output = str(finalizer_meta.get("text") or "").strip()

        success = bool(reasoner_output and finalizer_output)
        composed = f"{reasoner_output}\n\n{finalizer_output}".strip()
        raw_correct = check_answer(composed, expected_answer) if success else False
        supported = _answer_supported(reasoner_output, finalizer_output) if success else False
        record = {
            "input_hash": input_hash,
            "status": "complete" if success else "infrastructure_failed",
            "arm": arm,
            "question_id": question_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "temperature": temperature,
            "route_decision": decision,
            "correct": bool(raw_correct and supported),
            "raw_correct": bool(raw_correct),
            "reasoning_supported": bool(supported),
            "support_status": "supported" if supported else "unsupported_final_answer",
            "scoring_version": SCORING_VERSION,
            "final_answer": extract_final_answer(finalizer_output, allow_short_direct=False),
            "reasoner": {
                "output": reasoner_output,
                "derived_result": _derived_result(reasoner_output),
                "finish_reason": _finish_reason(reasoner_meta),
                "completion_tokens": _completion_tokens(reasoner_meta),
                "hit_token_limit": _finish_reason(reasoner_meta) == "length",
                "repetition": _repetition_diagnostics(reasoner_output),
                "usage": reasoner_meta.get("usage"),
            },
            "finalizer": {
                "output": finalizer_output,
                "finish_reason": _finish_reason(finalizer_meta),
                "completion_tokens": _completion_tokens(finalizer_meta),
                "format_valid": _finalizer_format_valid(finalizer_output),
                "usage": finalizer_meta.get("usage"),
            },
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "principle_action_injected": arm == "adaptive_memory",
            "endpoint": endpoint,
            "transport_trace": trace,
            "privacy_violations": (
                _forced_prompt_privacy_violations(prompt, candidate=candidate or {})
                if candidate else []
            ),
        }
        atomic_save_json(path, record)
        return record


def _baseline_draft(record: Mapping[str, Any]) -> str:
    return str(((record.get("response") or {}).get("reasoner") or {}).get("output", "")).strip()


def _summary_for_arm(
    records: Sequence[Mapping[str, Any]], *, original_baseline: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    complete = [row for row in records if row.get("status") == "complete"]
    routes = Counter(str(row.get("route_decision", "unknown")) for row in complete)
    rescue = 0
    damage = 0
    unsupported = 0
    for row in complete:
        base = original_baseline[(str(row["question_id"]), int(row["seed"]))]
        before = bool(base.get("correct"))
        after = bool(row.get("correct"))
        rescue += int(not before and after)
        damage += int(before and not after)
        unsupported += int(bool(row.get("raw_correct")) and not bool(row.get("reasoning_supported")))
    return {
        "record_count": len(records),
        "complete_count": len(complete),
        "infrastructure_failure_count": len(records) - len(complete),
        "route_counts": dict(routes),
        "rescue": rescue,
        "damage": damage,
        "net_rescue": rescue - damage,
        "unsupported_raw_correct_count": unsupported,
        "principle_action_injected_count": sum(bool(row.get("principle_action_injected")) for row in complete),
    }


async def run_adaptive_tradeoff_compare(
    config: Any,
    *,
    test_root: Path,
    source_roots: Sequence[Path],
    comparison_cache_root: Path,
    ports: Sequence[int],
    model: str,
    validation_concurrency: int,
    timeout: int,
    smoke_only: bool = False,
) -> dict[str, Any]:
    """Run one adaptive B/C trade-off protocol without touching prior runs."""
    root = Path(test_root)
    data = _load_intervention_sources(source_roots)
    bundles = data["bundles"]
    all_candidates = data["candidates"]
    full_seeds = tuple(int(seed) for seed in config.stage2.principle_test_solver_seeds)
    temperature = float(config.stage2.principle_test_primary_temperature)
    baseline = _baseline_by_seed(data["p0_rollouts"])
    if not all_candidates:
        raise RuntimeError("No audited Principle candidates are available")

    candidate_question_ids = {str(item["question_id"]) for item in all_candidates.values()}
    for qid in candidate_question_ids:
        for seed in full_seeds:
            if (qid, seed) not in baseline:
                raise RuntimeError(f"Missing unguided baseline question={qid} seed={seed}")

    # Validate the frozen comparison cache without modifying it.  This also
    # proves that the same question/seed controls used by A/B/C remain intact.
    timing_cache = _load_forced_timing_cache(
        comparison_cache_root, bundles=bundles, seeds=full_seeds,
    )
    if timing_cache["manifest_candidate_ids"] != set(all_candidates):
        raise RuntimeError("Candidate IDs differ between sources and comparison cache")

    # A full run first proves the exact live path on two candidates and three
    # seeds.  Hash-compatible artifacts are then reused by the full run.
    if not smoke_only:
        logger.info("[自适应介入] 先运行2条Principle×3 seeds真实Smoke Test")
        smoke = await run_adaptive_tradeoff_compare(
            config,
            test_root=root,
            source_roots=source_roots,
            comparison_cache_root=comparison_cache_root,
            ports=ports,
            model=model,
            validation_concurrency=validation_concurrency,
            timeout=timeout,
            smoke_only=True,
        )
        errors = []
        for arm in ("adaptive_no_memory", "adaptive_memory"):
            stats = smoke.get(arm, {})
            if int(stats.get("infrastructure_failure_count", 0)):
                errors.append(f"{arm}:infrastructure_failure")
            if int(stats.get("unsupported_raw_correct_count", 0)):
                errors.append(f"{arm}:unsupported_raw_correct")
        expected_routes = {
            "adaptive_no_memory": smoke.get("adaptive_no_memory", {}).get("record_count", 0),
            "adaptive_memory": smoke.get("adaptive_memory", {}).get("record_count", 0),
        }
        for arm, expected in expected_routes.items():
            if int(smoke.get("route_parse_success", {}).get(arm, 0)) != int(expected):
                errors.append(f"{arm}:route_parse_incomplete")
        validation = {
            "passed": not errors,
            "errors": errors,
            "expected_candidates": 2,
            "expected_seeds": [42, 43, 44],
            "summary": smoke,
        }
        atomic_save_json(root / "smoke_validation.json", validation)
        if errors:
            raise RuntimeError(
                "Adaptive trade-off Smoke Test failed; full run stopped: "
                + ", ".join(errors)
            )
        logger.info("[自适应介入] Smoke Test通过，开始完整74条×10 seeds实验")

    selected = list(sorted(all_candidates.values(), key=lambda item: str(item["candidate_id"])))
    seeds = full_seeds
    if smoke_only:
        selected = selected[:2]
        seeds = full_seeds[:3]
    selected_qids = sorted({str(item["question_id"]) for item in selected})

    views = {}
    for candidate in selected:
        cid = str(candidate["candidate_id"])
        views[cid] = {
            "candidate_id": cid,
            "question_id": str(candidate["question_id"]),
            "routing_view": principle_routing_view(candidate["content"]),
            "action_view": principle_action_view(candidate["content"]),
        }
        atomic_save_json(root / "principle_views" / f"{cid}.json", views[cid])

    manifest = {
        "schema_version": ADAPTIVE_VERSION,
        "run_scope": "smoke" if smoke_only else "full",
        "source_manifests": data["manifests"],
        "comparison_cache_root": str(comparison_cache_root),
        "question_count": len(candidate_question_ids),
        "candidate_count": len(all_candidates),
        "run_candidate_count": len(selected),
        "run_question_count": len(selected_qids),
        "candidate_ids": sorted(all_candidates),
        "run_candidate_ids": [str(item["candidate_id"]) for item in selected],
        "seeds": list(seeds),
        "temperature": temperature,
        "model": model,
        "ports": list(ports),
        "formal_control": "adaptive_no_memory",
        "secondary_control": "original_unguided_baseline",
    }
    atomic_save_json(root / "manifest.json", manifest)

    urls = _healthy_actor_urls(ports)
    runner = _AdaptiveRunner(
        config, root=root, urls=urls, model=model,
        concurrency=validation_concurrency, timeout=timeout,
    )

    async def control_task(qid: str, seed: int) -> dict[str, Any]:
        async with runner.transaction_semaphore:
            base = baseline[(qid, seed)]
            draft = _baseline_draft(base)
            route = await runner.route(
                arm="adaptive_no_memory", question_id=qid,
                question=str(bundles[qid]["question_text"]), draft=draft, seed=seed,
            )
            return await runner.execute(
                arm="adaptive_no_memory", question_id=qid,
                question=str(bundles[qid]["question_text"]),
                expected_answer=str(bundles[qid]["expected_answer"]),
                draft=draft, seed=seed, temperature=temperature,
                route_record=route, baseline_record=base,
            )

    async def memory_task(candidate: Mapping[str, Any], seed: int) -> dict[str, Any]:
        async with runner.transaction_semaphore:
            qid = str(candidate["question_id"])
            base = baseline[(qid, seed)]
            draft = _baseline_draft(base)
            route = await runner.route(
                arm="adaptive_memory", question_id=qid,
                question=str(bundles[qid]["question_text"]), draft=draft,
                seed=seed, candidate=candidate,
            )
            return await runner.execute(
                arm="adaptive_memory", question_id=qid,
                question=str(bundles[qid]["question_text"]),
                expected_answer=str(bundles[qid]["expected_answer"]),
                draft=draft, seed=seed, temperature=temperature,
                route_record=route, baseline_record=base, candidate=candidate,
            )

    control_rows = await _gather_with_progress(
        [control_task(qid, seed) for qid in selected_qids for seed in seeds],
        description="自适应无Memory对照", unit="题目-seed",
    )
    memory_rows = await _gather_with_progress(
        [memory_task(candidate, seed) for candidate in selected for seed in seeds],
        description="草稿感知Principle介入", unit="候选-seed",
    )
    atomic_save_json(root / "rollouts" / "adaptive_no_memory.json", control_rows)
    atomic_save_json(root / "rollouts" / "adaptive_memory.json", memory_rows)

    control_map = {
        (str(row["question_id"]), int(row["seed"])): row for row in control_rows
    }
    memory_map = {
        (str(row["candidate_id"]), int(row["seed"])): row for row in memory_rows
    }
    effect_rows: list[dict[str, Any]] = []
    original_effect_rows: list[dict[str, Any]] = []
    for candidate in selected:
        cid = str(candidate["candidate_id"])
        qid = str(candidate["question_id"])
        treatment = [memory_map[(cid, seed)] for seed in seeds]
        process_control = [control_map[(qid, seed)] for seed in seeds]
        original_control = [baseline[(qid, seed)] for seed in seeds]
        for control_name, controls, destination in (
            ("adaptive_no_memory", process_control, effect_rows),
            ("original_unguided_baseline", original_control, original_effect_rows),
        ):
            valid_pairs = [
                (control, guided) for control, guided in zip(controls, treatment)
                if control.get("status") == "complete" and guided.get("status") == "complete"
            ]
            valid_control = [item[0] for item in valid_pairs]
            valid_treatment = [item[1] for item in valid_pairs]
            paired = _paired_counts(valid_control, valid_treatment) if valid_pairs else {
                "gain": 0, "loss": 0, "both_correct": 0, "both_wrong": 0,
            }
            control_correct = sum(bool(item.get("correct")) for item in valid_control)
            guided_correct = sum(bool(item.get("correct")) for item in valid_treatment)
            delta = guided_correct - control_correct
            destination.append({
                "arm": "adaptive_memory",
                "control": control_name,
                "candidate_id": cid,
                "question_id": qid,
                "level": candidate.get("level", ""),
                "evaluated": bool(valid_pairs),
                "paired_seed_count": len(valid_pairs),
                "infrastructure_failure_count": len(seeds) - len(valid_pairs),
                "control_correct": control_correct,
                "guided_correct": guided_correct,
                "delta_correct": delta,
                "accuracy_delta": delta / len(valid_pairs) if valid_pairs else 0.0,
                "paired": paired,
                "effect_label": _effect_label(delta),
                "route_counts": dict(Counter(str(item.get("route_decision")) for item in valid_treatment)),
                "unsupported_raw_correct_count": sum(
                    bool(item.get("raw_correct")) and not bool(item.get("reasoning_supported"))
                    for item in valid_treatment
                ),
                "privacy_violations": sorted({
                    violation for item in valid_treatment
                    for violation in item.get("privacy_violations", [])
                }),
            })

    question_rows = _forced_question_effects(effect_rows)
    atomic_save_json(root / "effects" / "candidate_effects_vs_process_control.json", effect_rows)
    atomic_save_json(root / "effects" / "candidate_effects_vs_original_baseline.json", original_effect_rows)
    atomic_save_json(root / "effects" / "question_effects_vs_process_control.json", question_rows)

    summary = {
        "schema_version": ADAPTIVE_VERSION,
        "run_scope": "smoke" if smoke_only else "full",
        "question_count": len(selected_qids),
        "candidate_count": len(selected),
        "seeds": list(seeds),
        "temperature": temperature,
        "adaptive_no_memory": _summary_for_arm(
            control_rows, original_baseline=baseline,
        ),
        "adaptive_memory": _summary_for_arm(
            memory_rows, original_baseline=baseline,
        ),
        "effects_vs_process_control": _summarize_effect_rows(effect_rows),
        "effects_vs_original_baseline": _summarize_effect_rows(original_effect_rows),
        "route_parse_success": {
            "adaptive_no_memory": sum(
                bool(load_json(
                    root / "routes" / "adaptive_no_memory" / f"{qid}_s{seed}.json"
                ).get("parse_success"))
                for qid in selected_qids for seed in seeds
            ),
            "adaptive_memory": sum(
                bool(load_json(
                    root / "routes" / "adaptive_memory"
                    / f"{candidate['candidate_id']}_s{seed}.json"
                ).get("parse_success"))
                for candidate in selected for seed in seeds
            ),
        },
        "output_root": str(root),
    }
    atomic_save_json(root / "summary.json", summary)

    lines = [
        "# 草稿感知 Principle 自适应介入实验",
        "",
        f"- 运行范围：{'Smoke Test' if smoke_only else '完整实验'}",
        f"- 题目：{len(selected_qids)}",
        f"- Principle：{len(selected)}",
        f"- seeds：{list(seeds)}",
        "",
        "## 与同流程无Memory对照",
        "",
        f"- Macro delta: {summary['effects_vs_process_control']['question_mean_accuracy_delta']:.4f}",
        f"- Gain/Loss: {summary['effects_vs_process_control']['gain']}/{summary['effects_vs_process_control']['loss']}",
        f"- Bootstrap 95% CI: {summary['effects_vs_process_control']['paired_bootstrap_question_95ci']}",
        "",
        "## 安全性",
        "",
        f"- Memory rescue/damage: {summary['adaptive_memory']['rescue']}/{summary['adaptive_memory']['damage']}",
        f"- No-Memory rescue/damage: {summary['adaptive_no_memory']['rescue']}/{summary['adaptive_no_memory']['damage']}",
        f"- Memory routes: {summary['adaptive_memory']['route_counts']}",
        f"- Unsupported raw-correct: {summary['adaptive_memory']['unsupported_raw_correct_count']}",
        "",
        "主结论必须使用同流程无Memory对照；原始Baseline只作为整体能力参考。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
