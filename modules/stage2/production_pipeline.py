"""Recoverable Stage-2 production pipeline for source-validated Principles.

This module deliberately keeps three different claims separate:

* generation/format validity: a DeepSeek candidate has the canonical shape;
* source helpfulness: Forced-C improves the same Llama on the source question;
* transfer readiness: a helpful candidate also passes an independent Kimi audit.

Only the last group is indexed for the subsequent Stage-3 transfer experiment.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from benchmarks.matcher import check_answer, extract_final_answer
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
from modules.stage1_sampler import STAGE1_SOLVER_PROTOCOL, prompt_hashes
from modules.stage2.model_client import Stage2ModelClient
from modules.stage2.principle_generation import repair_principle
from modules.stage2.principle_test import (
    _PairedRunner,
    _gather_with_progress,
    _healthy_actor_urls,
    _json_hash,
    _load_bundles_by_level,
    _paired_counts,
    _stable_endpoint_index,
    generate_raw_candidates,
)
from modules.stage2.protocol_calibration import (
    HARD_AUDIT_DIMENSIONS,
    _audit_prompt,
    _decision,
    _format_check,
    _parse_audit,
    render_forced_principle_card,
)
from template.prompt_template import (
    FORCED_TIMING_AFTER_DRAFT_PROMPT,
    SOLVER_V1_FINALIZER_PROMPT,
)
from utils.io_utils import atomic_save_json, load_json


logger = logging.getLogger(__name__)

PRODUCTION_VERSION = "stage2_production_1200.1"
PRODUCTION_SALT = "stage2_production_1200_v1"
CORRECT_COUNTS = tuple(range(2, 9))
PRODUCTION_QUOTAS = {2: 386, 3: 209, 4: 179, 5: 128, 6: 104, 7: 95, 8: 99}
BATCH300_QUOTAS = {2: 96, 3: 52, 4: 45, 5: 32, 6: 26, 7: 24, 8: 25}
DEFAULT_SELECTION = Path(
    "output/stage1/selections/learning_zone_correct2_8_seed42/learning_zone_1258.json"
)
DEFAULT_BASELINE_FILES = (
    Path("output/stage1/01_sampling/llama_solver_v11_lz218_unseen_unified_sampling_results.json"),
    Path("output/stage1/01_sampling/llama_solver_v11_lz1000_unseen_unified_sampling_results.json"),
)
DEFAULT_PROTOCOL_ROOTS = (
    Path("output/stage2/principle_protocol_l23_v1"),
    Path("output/stage2/principle_protocol_l23_v1_batch2"),
)
SEEDS = tuple(range(42, 52))
TEMPERATURE = 0.7

GENERIC_AFTER_DRAFT_PROMPT = r"""
You are reviewing a complete mathematical draft without access to any stored
experience. Check only for arithmetic mistakes, missing problem conditions,
invalid shortcuts, and inconsistency between the derivation and its result.

If the draft is valid, retain it and state that no correction was needed. If a
concrete error is found, identify the exact faulty step and make the smallest
necessary correction. Do not ignore the draft and start a random new solution.
Do not state or box the final answer; a separate finalizer will do that.

Output exactly these sections:

Draft Assessment:
[the exact checked step or issue]

Revised Method:
[method]

Revised Reasoning:
[complete corrected or retained derivation]

Verification:
[direct verification]

Current Problem:
{question}

Previous Draft:
{previous_draft}
"""

FAILURE_ANALYSIS_PROMPT = """You diagnose why generated mathematical advice did not
produce a clear improvement for a weaker solver. Return raw JSON only. Do not repair the
Principle and do not solve the source question again.

Choose one primary category:
- mathematical_or_factual_error
- over_specific
- over_general
- failure_not_present_in_draft
- damaged_valid_reasoning
- solver_did_not_execute_advice
- random_or_insufficient_evidence

Question: {question}
Candidate Principles and paired effects: {candidates}

Return:
{{"primary_category": "...", "secondary_categories": [], "reason": "...",
  "evidence_candidate_ids": []}}
"""


def _stable_key(question_id: str) -> str:
    return hashlib.sha256(f"{question_id}:{PRODUCTION_SALT}".encode("utf-8")).hexdigest()


def _load_excluded_ids(roots: Sequence[Path]) -> set[str]:
    excluded: set[str] = set()
    for root in roots:
        path = Path(root) / "manifests" / "source40.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing protocol-calibration manifest: {path}")
        excluded.update(map(str, load_json(path).get("source_question_ids", [])))
    if len(excluded) != 40:
        raise RuntimeError(f"Expected exactly 40 protocol-calibration questions; found {len(excluded)}")
    return excluded


def _bundle_map(config: Any, backbone: str) -> dict[str, dict[str, Any]]:
    by_level, _ = _load_bundles_by_level(config, backbone)
    output: dict[str, dict[str, Any]] = {}
    for level, bundles in by_level.items():
        for bundle in bundles:
            qid = str(bundle["question_id"])
            record = dict(bundle)
            record["level"] = str(record.get("level") or level)
            output[qid] = record
    return output


def prepare_production_target(
    config: Any,
    *,
    production_root: Path,
    target_size: int,
    backbone: str = "llama",
    selection_path: Path = DEFAULT_SELECTION,
    exclude_roots: Sequence[Path] = DEFAULT_PROTOCOL_ROOTS,
) -> dict[str, Any]:
    """Freeze the 1200-question pool and its nested 300-question prefix."""
    if target_size not in {300, 1200}:
        raise ValueError("Production target_size must be 300 or 1200")
    root = Path(production_root)
    selection = load_json(selection_path)
    if not isinstance(selection, list) or len(selection) != 1258:
        raise RuntimeError(f"Expected the frozen 1258-question Learning Zone: {selection_path}")
    excluded = _load_excluded_ids(exclude_roots)
    bundle_by_id = _bundle_map(config, backbone)
    strata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in selection:
        qid = str(item.get("question_id"))
        if qid in excluded:
            continue
        evidence = item.get("selection_evidence") or {}
        correct_count = int(evidence.get("correct_count", -1))
        if correct_count not in CORRECT_COUNTS:
            raise RuntimeError(f"Selection contains non-Learning-Zone question={qid}")
        if qid not in bundle_by_id:
            raise RuntimeError(f"Missing original Stage-1 bundle for question={qid}")
        bundle = dict(bundle_by_id[qid])
        observed = int((bundle.get("capability_signal") or {}).get("correct_count", -1))
        if observed != correct_count:
            raise RuntimeError(
                f"Stage-1 evidence mismatch question={qid}: selection={correct_count}, bundle={observed}"
            )
        bundle["production_correct_count"] = correct_count
        strata[correct_count].append(bundle)
    if sum(map(len, strata.values())) != 1218:
        raise RuntimeError("Removing the frozen 40-question calibration set must leave 1218 questions")
    for values in strata.values():
        values.sort(key=lambda row: _stable_key(str(row["question_id"])))

    production = [row for count in CORRECT_COUNTS for row in strata[count][:PRODUCTION_QUOTAS[count]]]
    reserve = [row for count in CORRECT_COUNTS for row in strata[count][PRODUCTION_QUOTAS[count]:]]
    batch300 = [
        row for count in CORRECT_COUNTS
        for row in strata[count][:BATCH300_QUOTAS[count]]
    ]
    if len(production) != 1200 or len(reserve) != 18 or len(batch300) != 300:
        raise AssertionError("Frozen production quota arithmetic is inconsistent")
    if not {str(row["question_id"]) for row in batch300}.issubset(
        {str(row["question_id"]) for row in production}
    ):
        raise AssertionError("Batch 300 is not nested in production 1200")

    def manifest(name: str, rows: Sequence[Mapping[str, Any]], quotas: Mapping[int, int]) -> dict:
        payload = {
            "schema_version": PRODUCTION_VERSION,
            "name": name,
            "selection_salt": PRODUCTION_SALT,
            "backbone": backbone,
            "question_count": len(rows),
            "correct_count_quotas": {str(k): int(v) for k, v in quotas.items()},
            "excluded_protocol_question_ids": sorted(excluded),
            "source_question_ids": [str(row["question_id"]) for row in rows],
            "source_bundle_hashes": {str(row["question_id"]): _json_hash(row) for row in rows},
        }
        payload["manifest_hash"] = _json_hash(payload)
        return payload

    manifests = {
        "production1200": manifest("production1200", production, PRODUCTION_QUOTAS),
        "batch300": manifest("batch300", batch300, BATCH300_QUOTAS),
        "reserve18": manifest(
            "reserve18", reserve,
            {count: len(strata[count]) - PRODUCTION_QUOTAS[count] for count in CORRECT_COUNTS},
        ),
    }
    for name, payload in manifests.items():
        path = root / "manifests" / f"{name}.json"
        if path.exists() and load_json(path).get("manifest_hash") != payload["manifest_hash"]:
            raise RuntimeError(f"Frozen production manifest differs: {path}")
        atomic_save_json(path, payload)

    target_rows = batch300 if target_size == 300 else production
    target_root = root / f"batch_{target_size:04d}"
    compatibility_manifest = dict(manifests["batch300" if target_size == 300 else "production1200"])
    compatibility_manifest["target_count"] = target_size
    atomic_save_json(target_root / "manifests" / "source40.json", compatibility_manifest)
    atomic_save_json(target_root / "source_bundles" / "source40_bundles.json", target_rows)
    atomic_save_json(target_root / "source_bundles" / "reserve20_bundles.json", [])
    summary = {
        "production_root": str(root),
        "target_root": str(target_root),
        "target_size": target_size,
        "production_count": 1200,
        "reserve_count": 18,
        "excluded_count": 40,
        "target_manifest_hash": compatibility_manifest["manifest_hash"],
        "correct_count_quotas": compatibility_manifest["correct_count_quotas"],
    }
    atomic_save_json(target_root / "prepare_summary.json", summary)
    return summary


async def generate_production_candidates(
    config: Any, *, target_root: Path, concurrency: int,
) -> dict[str, Any]:
    summary = await generate_raw_candidates(
        config, test_root=Path(target_root), concurrency=concurrency,
    )
    bundles = {
        str(row["question_id"]): row
        for row in load_json(Path(target_root) / "source_bundles" / "source40_bundles.json")
    }
    bank = load_json(Path(target_root) / "raw_candidates" / "candidate_bank.json")
    valid = 0
    invalid = 0
    for candidate in bank.get("candidates", []):
        check = _format_check(candidate, bundles[str(candidate["question_id"])])
        atomic_save_json(
            Path(target_root) / "format_checks" / f"{candidate['candidate_id']}.json", check,
        )
        if check["hard_invalid"]:
            invalid += 1
        else:
            valid += 1
    result = {**summary, "format_valid": valid, "format_invalid": invalid}
    atomic_save_json(Path(target_root) / "generation_summary.json", result)
    return result


def production_status(production_root: Path, target_size: int) -> dict[str, Any]:
    target_root = Path(production_root) / f"batch_{target_size:04d}"
    result: dict[str, Any] = {"target_root": str(target_root), "exists": target_root.exists()}
    for name, path in {
        "prepare": target_root / "prepare_summary.json",
        "generation": target_root / "generation_summary.json",
        "validation": target_root / "local_validation" / "summary.json",
        "audit_store": target_root / "audit_store_summary.json",
    }.items():
        result[name] = load_json(path) if path.exists() else {"status": "not_started"}
    return result


def _load_stage1_baseline(
    target_root: Path,
    baseline_files: Sequence[Path] = DEFAULT_BASELINE_FILES,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load the independent Solver-v1.1 baseline and enforce its frozen protocol."""
    target_ids = {
        str(row["question_id"])
        for row in load_json(Path(target_root) / "source_bundles" / "source40_bundles.json")
    }
    current_hashes = prompt_hashes()
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for path in baseline_files:
        payload = load_json(path)
        config = payload.get("config") or {}
        if float(config.get("temperature", -1)) != TEMPERATURE:
            raise RuntimeError(f"Baseline temperature mismatch in {path}")
        if tuple(map(int, config.get("rollout_seeds", []))) != SEEDS:
            raise RuntimeError(f"Baseline seeds mismatch in {path}")
        for question in payload.get("results", []):
            qid = str(question.get("question_id"))
            if qid not in target_ids:
                continue
            if int(question.get("successful_trials", -1)) != len(SEEDS):
                raise RuntimeError(f"Incomplete baseline question={qid} in {path}")
            for trial in question.get("trial_records", []):
                seed = int(trial.get("seed", -1))
                if seed not in SEEDS:
                    continue
                if trial.get("prompt_version") != STAGE1_SOLVER_PROTOCOL:
                    raise RuntimeError(f"Baseline Solver protocol mismatch question={qid} seed={seed}")
                if trial.get("prompt_hashes") != current_hashes:
                    raise RuntimeError(f"Baseline Prompt hash mismatch question={qid} seed={seed}")
                if str(trial.get("model", "")) != "llama":
                    raise RuntimeError(f"Baseline model mismatch question={qid} seed={seed}")
                draft = str(trial.get("reasoner_output") or "").strip()
                finalizer = str(trial.get("finalizer_output") or "").strip()
                if not draft or not finalizer:
                    raise RuntimeError(f"Baseline output missing question={qid} seed={seed}")
                key = (qid, seed)
                if key in output:
                    raise RuntimeError(f"Duplicate baseline question={qid} seed={seed}")
                output[key] = {
                    "question_id": qid,
                    "seed": seed,
                    "correct": bool(trial.get("is_correct")),
                    "draft": draft,
                    "finalizer_output": finalizer,
                    "final_answer": str(trial.get("final_answer", "")),
                    "source_file": str(path),
                    "prompt_hashes": current_hashes,
                    "temperature": TEMPERATURE,
                }
    required = {(qid, seed) for qid in target_ids for seed in SEEDS}
    missing = sorted(required - set(output))
    if missing:
        raise RuntimeError(f"Independent Stage-1 baseline is incomplete: missing={missing[:10]}")
    atomic_save_json(
        Path(target_root) / "local_validation" / "original_baseline" / "manifest.json",
        {
            "question_count": len(target_ids),
            "record_count": len(output),
            "temperature": TEMPERATURE,
            "seeds": list(SEEDS),
            "prompt_hashes": current_hashes,
            "source_files": [str(path) for path in baseline_files],
        },
    )
    return output


def _compact_draft(text: str, max_chars: int = 9000) -> str:
    cleaned = "\n".join(line.rstrip() for line in str(text).splitlines()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    half = max_chars // 2
    return "\n".join([
        cleaned[:half].rstrip(),
        "[... middle of draft omitted to fit context ...]",
        cleaned[-half:].lstrip(),
    ])


async def _run_batched(
    jobs: Sequence[Any],
    worker: Callable[[Any], Awaitable[dict[str, Any]]],
    *,
    description: str,
    batch_size: int = 1000,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for offset in range(0, len(jobs), batch_size):
        batch = jobs[offset: offset + batch_size]
        label = f"{description} {offset + 1}-{offset + len(batch)}/{len(jobs)}"
        results.extend(await _gather_with_progress(
            [worker(job) for job in batch], description=label, unit="次",
        ))
    return results


async def _revision_transaction(
    runner: _PairedRunner,
    *,
    path: Path,
    question_id: str,
    candidate_id: str,
    question: str,
    expected_answer: str,
    seed: int,
    prompt: str,
    stage: str,
    principle_card_hash: str = "",
) -> dict[str, Any]:
    input_hash = _json_hash({
        "version": PRODUCTION_VERSION,
        "stage": stage,
        "prompt": prompt,
        "finalizer_prompt": SOLVER_V1_FINALIZER_PROMPT,
        "model": runner.model,
        "temperature": TEMPERATURE,
        "seed": seed,
    })
    if path.exists():
        try:
            saved = load_json(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            # An externally interrupted process can leave a zero-byte or
            # truncated checkpoint.  Preserve it for audit, then recompute
            # only this logical rollout; never discard the forensic artifact.
            corrupt_dir = path.parent / "corrupt_checkpoints"
            corrupt_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archived = corrupt_dir / f"{path.name}.{stamp}.corrupt"
            path.replace(archived)
            logger.warning(
                "损坏checkpoint已归档并将重新计算: path=%s archived=%s error=%s",
                path, archived, exc,
            )
        else:
            if saved.get("input_hash") == input_hash and saved.get("status") in {
                "complete", "infrastructure_failed",
            }:
                return saved
    endpoints = []
    primary = _stable_endpoint_index(question_id, TEMPERATURE, seed, len(runner.urls))
    endpoints.append(runner.urls[primary])
    if len(runner.urls) > 1:
        endpoints.append(runner.urls[(primary + 1) % len(runner.urls)])
    reasoner_meta: dict[str, Any] = {}
    finalizer_meta: dict[str, Any] = {}
    reasoner_output = ""
    finalizer_output = ""
    endpoint_used = ""
    trace = []
    async with runner.semaphore:
        for attempt, endpoint in enumerate(endpoints, 1):
            with llm_request_context(
                candidate_id or question_id,
                f"production/{stage}/seed={seed}/attempt={attempt}",
            ):
                reasoner_meta, endpoint_used = await runner.execute._call_with_router_metadata(
                    prompt=prompt,
                    temperature=TEMPERATURE,
                    top_p=SOLVER_V1_TOP_P,
                    max_tokens=SOLVER_V1_REASONER_MAX_TOKENS,
                    repetition_penalty=1.0,
                    seed=seed,
                    url_override=endpoint,
                    timeout_override=min(runner.timeout, 300),
                )
            reasoner_output = str(reasoner_meta.get("text") or "").strip()
            if not reasoner_output:
                trace.append({"endpoint": endpoint, "stage": "reasoner", "success": False})
                continue
            finalizer_prompt = SOLVER_V1_FINALIZER_PROMPT.format(
                question=question, reasoner_output=reasoner_output,
            )
            with llm_request_context(
                candidate_id or question_id,
                f"production/{stage}/finalizer/seed={seed}/attempt={attempt}",
            ):
                finalizer_meta, _ = await runner.execute._call_with_router_metadata(
                    prompt=finalizer_prompt,
                    temperature=SOLVER_V1_FINALIZER_TEMPERATURE,
                    top_p=SOLVER_V1_TOP_P,
                    max_tokens=SOLVER_V1_FINALIZER_MAX_TOKENS,
                    repetition_penalty=1.0,
                    seed=seed,
                    url_override=endpoint_used or endpoint,
                    timeout_override=min(runner.timeout, 60),
                )
            finalizer_output = str(finalizer_meta.get("text") or "").strip()
            trace.append({
                "endpoint": endpoint,
                "stage": "transaction",
                "reasoner_success": True,
                "finalizer_success": bool(finalizer_output),
            })
            if finalizer_output:
                break
            reasoner_output = ""
    success = bool(reasoner_output and finalizer_output)
    composed = f"{reasoner_output}\n\n{finalizer_output}".strip()
    record = {
        "input_hash": input_hash,
        "status": "complete" if success else "infrastructure_failed",
        "stage": stage,
        "candidate_id": candidate_id,
        "question_id": question_id,
        "seed": seed,
        "temperature": TEMPERATURE,
        "principle_injected": bool(principle_card_hash),
        "principle_card_hash": principle_card_hash,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "correct": check_answer(composed, expected_answer) if success else False,
        "final_answer": extract_final_answer(finalizer_output, allow_short_direct=False),
        "reasoner": {
            "output": reasoner_output,
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
        "transport_trace": trace,
    }
    atomic_save_json(path, record)
    return record


def _candidate_effect(
    candidate: Mapping[str, Any],
    baseline: Mapping[tuple[str, int], Mapping[str, Any]],
    generic: Sequence[Mapping[str, Any]],
    guided: Sequence[Mapping[str, Any]],
    *,
    version: str = "raw",
) -> dict[str, Any]:
    qid = str(candidate["question_id"])
    base_rows = [baseline[(qid, seed)] for seed in SEEDS]
    if len(generic) != len(SEEDS) or len(guided) != len(SEEDS) or any(
        row.get("status") != "complete" for row in list(generic) + list(guided)
    ):
        return {
            "candidate_id": candidate["candidate_id"], "question_id": qid,
            "version": version, "evaluated": False,
            "effect_label": "infrastructure_failed",
        }
    baseline_correct = sum(bool(row["correct"]) for row in base_rows)
    generic_correct = sum(bool(row["correct"]) for row in generic)
    guided_correct = sum(bool(row["correct"]) for row in guided)
    vs_original = _paired_counts(base_rows, guided)
    vs_generic = _paired_counts(generic, guided)
    delta_original = guided_correct - baseline_correct
    delta_generic = guided_correct - generic_correct
    helpful = (
        delta_generic >= 2 and delta_original >= 0
        and vs_generic["gain"] > vs_generic["loss"]
    )
    harmful = delta_original <= -2 or delta_generic <= -2
    label = "source_helpful" if helpful else "source_harmful" if harmful else "inconclusive"
    return {
        "candidate_id": candidate["candidate_id"],
        "question_id": qid,
        "level": candidate.get("level", ""),
        "version": version,
        "evaluated": True,
        "baseline_correct_at_10": baseline_correct,
        "generic_correct_at_10": generic_correct,
        "guided_correct_at_10": guided_correct,
        "delta_vs_original": delta_original,
        "delta_vs_generic": delta_generic,
        "paired_vs_original": vs_original,
        "paired_vs_generic": vs_generic,
        "effect_label": label,
    }


async def validate_production_candidates(
    config: Any,
    *,
    target_root: Path,
    ports: Sequence[int],
    model: str,
    concurrency: int,
    timeout: int,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    version: str = "raw",
) -> dict[str, Any]:
    root = Path(target_root)
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    baseline = _load_stage1_baseline(root)
    bank = load_json(root / "raw_candidates" / "candidate_bank.json")
    selected = list(candidates) if candidates is not None else [
        candidate for candidate in bank.get("candidates", [])
        if not load_json(root / "format_checks" / f"{candidate['candidate_id']}.json")["hard_invalid"]
    ]
    urls = _healthy_actor_urls(ports)
    runner = _PairedRunner(
        config, output_dir=root / "local_validation" / "runner",
        urls=urls, model=model, concurrency=concurrency, timeout=timeout,
    )

    generic_jobs = [(qid, seed) for qid in sorted(bundles) for seed in SEEDS]

    async def generic_worker(job: tuple[str, int]) -> dict[str, Any]:
        qid, seed = job
        base = baseline[(qid, seed)]
        prompt = GENERIC_AFTER_DRAFT_PROMPT.format(
            question=bundles[qid]["question_text"], previous_draft=_compact_draft(base["draft"]),
        )
        return await _revision_transaction(
            runner,
            path=root / "local_validation" / "generic_self_check" / f"{qid}_s{seed}.json",
            question_id=qid, candidate_id="", question=bundles[qid]["question_text"],
            expected_answer=str(bundles[qid]["expected_answer"]), seed=seed,
            prompt=prompt, stage="generic_self_check",
        )

    generic_records = await _run_batched(
        generic_jobs, generic_worker, description="Stage2通用草稿复核",
    )
    generic_by_key = {(str(row["question_id"]), int(row["seed"])): row for row in generic_records}

    guided_jobs = [(candidate, seed) for candidate in selected for seed in SEEDS]

    async def guided_worker(job: tuple[Mapping[str, Any], int]) -> dict[str, Any]:
        candidate, seed = job
        qid = str(candidate["question_id"])
        candidate_id = str(candidate["candidate_id"])
        content = candidate["content"]
        card = render_forced_principle_card(content)
        card_hash = hashlib.sha256(card.encode("utf-8")).hexdigest()
        prompt = FORCED_TIMING_AFTER_DRAFT_PROMPT.format(
            question=bundles[qid]["question_text"],
            previous_draft=_compact_draft(baseline[(qid, seed)]["draft"]),
            principle_card=card,
        )
        suffix = "" if version == "raw" else f"_{version}"
        return await _revision_transaction(
            runner,
            path=root / "local_validation" / "forced_c" / version / f"{candidate_id}{suffix}_s{seed}.json",
            question_id=qid, candidate_id=candidate_id,
            question=bundles[qid]["question_text"],
            expected_answer=str(bundles[qid]["expected_answer"]), seed=seed,
            prompt=prompt, stage=f"forced_c/{version}", principle_card_hash=card_hash,
        )

    guided_records = await _run_batched(
        guided_jobs, guided_worker, description=f"Stage2 Forced-C验证({version})",
    )
    guided_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in guided_records:
        guided_by_candidate[str(row["candidate_id"])].append(row)
    effects = []
    for candidate in selected:
        qid = str(candidate["question_id"])
        generic = [generic_by_key[(qid, seed)] for seed in SEEDS]
        guided = sorted(guided_by_candidate[str(candidate["candidate_id"])], key=lambda row: int(row["seed"]))
        effects.append(_candidate_effect(
            candidate, baseline, generic, guided, version=version,
        ))
    effect_path = root / "local_validation" / f"candidate_effects_{version}.json"
    atomic_save_json(effect_path, effects)
    summary = {
        "version": version,
        "candidate_count": len(selected),
        "effect_counts": dict(Counter(row["effect_label"] for row in effects)),
        "baseline_record_count": len(baseline),
        "generic_record_count": len(generic_records),
        "guided_record_count": len(guided_records),
        "temperature": TEMPERATURE,
        "seeds": list(SEEDS),
    }
    if version == "raw":
        atomic_save_json(root / "local_validation" / "summary.json", summary)
    else:
        atomic_save_json(root / "local_validation" / f"summary_{version}.json", summary)
    return summary


async def _kimi_audit_once(
    client: Stage2ModelClient,
    *,
    bundle: Mapping[str, Any],
    candidate: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    format_check = _format_check(candidate, bundle)
    if format_check["hard_invalid"]:
        return {
            "status": "format_invalid", "decision": "reject",
            "format_check": format_check, "response": None, "audit": {},
        }
    evidence_ids = set(map(str, candidate.get("supporting_wrong_rollout_ids", []))) | set(
        map(str, candidate.get("contrast_correct_rollout_ids", []))
    )
    prompt = _audit_prompt(bundle, candidate, format_check["semantic_hints"])
    with llm_request_context(str(candidate["candidate_id"]), stage):
        response = await client.auditor(
            prompt, auditor_id="kimi", thinking_enabled=False,
            max_tokens=2048, json_mode=True,
        )
    audit, errors = _parse_audit(response.get("parsed"), evidence_ids)
    if errors:
        return {
            "status": "protocol_failed", "decision": "protocol_failed",
            "format_check": format_check, "response": response,
            "audit": audit, "protocol_errors": errors,
        }
    return {
        "status": "complete", "decision": _decision(audit),
        "format_check": format_check, "response": response, "audit": audit,
        "protocol_errors": [],
    }


async def _deepseek_failure_analyses(
    config: Any,
    *,
    root: Path,
    bundles: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for effect in effects:
        if effect.get("effect_label") == "source_helpful":
            continue
        candidate = candidates.get(str(effect.get("candidate_id")))
        if candidate:
            grouped[str(effect["question_id"])].append({
                "candidate_id": candidate["candidate_id"],
                "content": candidate["content"],
                "effect": dict(effect),
            })
    client = Stage2ModelClient(config, generator_id="deepseek")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(item: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
        qid, rows = item
        path = root / "failure_analysis" / "deepseek" / f"{qid}.json"
        prompt = FAILURE_ANALYSIS_PROMPT.format(
            question=bundles[qid]["question_text"],
            candidates=json.dumps(rows, ensure_ascii=False),
        )
        input_hash = _json_hash({"version": PRODUCTION_VERSION, "prompt": prompt})
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash:
                return saved
        async with semaphore:
            with llm_request_context(qid, "production/failure_analysis/deepseek"):
                response = await client.auditor(
                    prompt, auditor_id="deepseek", thinking_enabled=False,
                    max_tokens=1536, json_mode=True,
                )
        record = {
            "input_hash": input_hash, "question_id": qid,
            "candidate_ids": [row["candidate_id"] for row in rows],
            "analysis": response.get("parsed") or {}, "response": response,
            "status": "complete" if response.get("parsed") else "protocol_failed",
        }
        atomic_save_json(path, record)
        return record

    return await _run_batched(
        list(sorted(grouped.items())), one,
        description="Stage2 DeepSeek失败归因", batch_size=300,
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


async def audit_and_store_production(
    config: Any,
    *,
    target_root: Path,
    ports: Sequence[int],
    model: str,
    validation_concurrency: int,
    audit_concurrency: int,
    timeout: int,
    kimi_max_calls: int = 300,
    diagnostic_sample_size: int = 30,
) -> dict[str, Any]:
    root = Path(target_root)
    bundles = {
        str(row["question_id"]): row
        for row in load_json(root / "source_bundles" / "source40_bundles.json")
    }
    bank = load_json(root / "raw_candidates" / "candidate_bank.json")
    candidates = {str(row["candidate_id"]): row for row in bank.get("candidates", [])}
    raw_effects = load_json(root / "local_validation" / "candidate_effects_raw.json")
    effects_by_id = {str(row["candidate_id"]): row for row in raw_effects}
    helpful = sorted(
        (candidates[str(row["candidate_id"])] for row in raw_effects
         if row.get("effect_label") == "source_helpful"),
        key=lambda row: hashlib.sha256(str(row["candidate_id"]).encode()).hexdigest(),
    )
    diagnostic_budget = min(max(0, int(diagnostic_sample_size)), int(kimi_max_calls))
    positive_budget = max(0, int(kimi_max_calls) - diagnostic_budget)
    # Keep a small reserve for reaudit. Existing evidence suggests repairs are
    # uncommon; 30 slots prevent an initial wave from consuming the whole gate.
    reaudit_reserve = min(30, max(0, positive_budget // 5))
    initial_limit = min(len(helpful), max(0, positive_budget - reaudit_reserve))
    initial_candidates = helpful[:initial_limit]
    awaiting = helpful[initial_limit:]
    client = Stage2ModelClient(config, generator_id="deepseek")
    semaphore = asyncio.Semaphore(max(1, audit_concurrency))

    async def initial_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
        cid = str(candidate["candidate_id"])
        path = root / "kimi_audits" / "initial" / f"{cid}.json"
        input_hash = _json_hash({
            "version": PRODUCTION_VERSION,
            "candidate": candidate,
            "bundle_hash": _json_hash(bundles[str(candidate["question_id"])]),
        })
        if path.exists():
            saved = load_json(path)
            if saved.get("input_hash") == input_hash:
                return saved
        async with semaphore:
            result = await _kimi_audit_once(
                client, bundle=bundles[str(candidate["question_id"])],
                candidate=candidate, stage="production/kimi_audit",
            )
        record = {"input_hash": input_hash, "candidate_id": cid, **result}
        atomic_save_json(path, record)
        return record

    initial_records = await _run_batched(
        initial_candidates, initial_audit,
        description="Stage2 Kimi正收益盲审", batch_size=300,
    )
    audit_by_id = {str(row["candidate_id"]): row for row in initial_records}

    repair_candidates: list[dict[str, Any]] = []
    repair_records: dict[str, dict[str, Any]] = {}
    for candidate in initial_candidates:
        cid = str(candidate["candidate_id"])
        audit_record = audit_by_id[cid]
        if audit_record.get("decision") != "repair":
            continue
        path = root / "repairs" / f"{cid}.json"
        input_hash = _json_hash({
            "version": PRODUCTION_VERSION,
            "candidate": candidate,
            "audit": audit_record.get("audit"),
        })
        if path.exists() and load_json(path).get("input_hash") == input_hash:
            repair = load_json(path)
        else:
            with llm_request_context(cid, "production/deepseek_repair"):
                response = await repair_principle(
                    client, bundles[str(candidate["question_id"])],
                    dict(candidate), audit_record.get("audit") or {},
                )
            repair = {"input_hash": input_hash, "candidate_id": cid, **response}
            atomic_save_json(path, repair)
        repair_records[cid] = repair
        if repair.get("success"):
            repaired = {
                **candidate,
                "content": repair["content"],
                "version": "repaired",
                "parent_content_hash": _json_hash(candidate["content"]),
            }
            check = _format_check(repaired, bundles[str(candidate["question_id"])])
            atomic_save_json(root / "format_checks" / "repaired" / f"{cid}.json", check)
            if not check["hard_invalid"]:
                repair_candidates.append(repaired)

    repaired_effects: dict[str, dict[str, Any]] = {}
    if repair_candidates:
        await validate_production_candidates(
            config, target_root=root, ports=ports, model=model,
            concurrency=validation_concurrency, timeout=timeout,
            candidates=repair_candidates, version="repaired",
        )
        repaired_effects = {
            str(row["candidate_id"]): row
            for row in load_json(root / "local_validation" / "candidate_effects_repaired.json")
        }

    initial_call_count = len(initial_records)
    remaining_positive_budget = max(0, positive_budget - initial_call_count)
    reaudit_candidates = [
        candidate for candidate in repair_candidates
        if repaired_effects.get(str(candidate["candidate_id"]), {}).get("effect_label")
        == "source_helpful"
    ][:remaining_positive_budget]

    async def reaudit(candidate: Mapping[str, Any]) -> dict[str, Any]:
        cid = str(candidate["candidate_id"])
        path = root / "kimi_audits" / "repaired" / f"{cid}.json"
        input_hash = _json_hash({"version": PRODUCTION_VERSION, "candidate": candidate})
        if path.exists() and load_json(path).get("input_hash") == input_hash:
            return load_json(path)
        async with semaphore:
            result = await _kimi_audit_once(
                client, bundle=bundles[str(candidate["question_id"])],
                candidate=candidate, stage="production/kimi_reaudit",
            )
        record = {"input_hash": input_hash, "candidate_id": cid, **result}
        atomic_save_json(path, record)
        return record

    reaudit_records = await _run_batched(
        reaudit_candidates, reaudit,
        description="Stage2 Kimi修复版复审", batch_size=100,
    )
    reaudit_by_id = {str(row["candidate_id"]): row for row in reaudit_records}

    # Failure analysis is diagnostic and never changes admission decisions.
    failure_analyses = await _deepseek_failure_analyses(
        config, root=root, bundles=bundles, candidates=candidates,
        effects=raw_effects, concurrency=audit_concurrency,
    )
    harmful = [row for row in raw_effects if row.get("effect_label") == "source_harmful"]
    inconclusive = [row for row in raw_effects if row.get("effect_label") == "inconclusive"]
    stable = lambda row: hashlib.sha256(str(row["candidate_id"]).encode()).hexdigest()
    harmful_quota = diagnostic_budget // 2
    inconclusive_quota = diagnostic_budget - harmful_quota
    diagnostic_effects = (
        sorted(harmful, key=stable)[:harmful_quota]
        + sorted(inconclusive, key=stable)[:inconclusive_quota]
    )

    async def diagnostic_kimi(effect: Mapping[str, Any]) -> dict[str, Any]:
        cid = str(effect["candidate_id"])
        candidate = candidates[cid]
        path = root / "failure_analysis" / "kimi_sample" / f"{cid}.json"
        prompt = FAILURE_ANALYSIS_PROMPT.format(
            question=bundles[str(candidate["question_id"])]["question_text"],
            candidates=json.dumps([{"candidate_id": cid, "content": candidate["content"], "effect": effect}], ensure_ascii=False),
        )
        input_hash = _json_hash({"version": PRODUCTION_VERSION, "prompt": prompt})
        if path.exists() and load_json(path).get("input_hash") == input_hash:
            return load_json(path)
        async with semaphore:
            with llm_request_context(cid, "production/failure_analysis/kimi"):
                response = await client.auditor(
                    prompt, auditor_id="kimi", thinking_enabled=False,
                    max_tokens=1024, json_mode=True,
                )
        record = {
            "input_hash": input_hash, "candidate_id": cid,
            "effect_label": effect.get("effect_label"),
            "analysis": response.get("parsed") or {}, "response": response,
            "status": "complete" if response.get("parsed") else "protocol_failed",
        }
        atomic_save_json(path, record)
        return record

    diagnostic_records = await _run_batched(
        diagnostic_effects, diagnostic_kimi,
        description="Stage2 Kimi非正收益抽查", batch_size=30,
    )

    pending: list[dict[str, Any]] = []
    not_validated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    awaiting_ids = {str(row["candidate_id"]) for row in awaiting}
    repaired_by_id = {str(row["candidate_id"]): row for row in repair_candidates}
    for cid, candidate in sorted(candidates.items()):
        raw_effect = effects_by_id.get(cid, {"effect_label": "protocol_failed"})
        content = candidate["content"]
        parent_version_id = ""
        status = str(raw_effect.get("effect_label", "protocol_failed"))
        audit_record = audit_by_id.get(cid)
        if cid in awaiting_ids:
            status = "awaiting_audit"
        elif raw_effect.get("effect_label") == "source_helpful":
            if not audit_record:
                status = "awaiting_audit"
            elif audit_record.get("decision") == "pass":
                status = "pending_transfer_validation"
            elif audit_record.get("decision") == "reject":
                status = "content_rejected"
            elif audit_record.get("decision") == "protocol_failed":
                status = "protocol_failed"
            elif audit_record.get("decision") == "repair":
                repaired = repaired_by_id.get(cid)
                repaired_effect = repaired_effects.get(cid, {})
                final_audit = reaudit_by_id.get(cid)
                if repaired:
                    content = repaired["content"]
                    parent_version_id = f"{cid}_raw_{_json_hash(candidate['content'])[:8]}"
                if repaired_effect.get("effect_label") != "source_helpful":
                    status = str(repaired_effect.get("effect_label", "protocol_failed"))
                elif not final_audit:
                    status = "awaiting_audit"
                elif final_audit.get("decision") == "pass":
                    status = "pending_transfer_validation"
                elif final_audit.get("decision") == "protocol_failed":
                    status = "protocol_failed"
                else:
                    status = "content_rejected"
        content_hash = _json_hash(content)
        version_kind = "repaired" if cid in repaired_by_id and content == repaired_by_id[cid]["content"] else "raw"
        record = {
            "memory_id": cid,
            "version_id": f"{cid}_{version_kind}_{content_hash[:8]}",
            "parent_version_id": parent_version_id,
            "operation": "repair" if version_kind == "repaired" else "add",
            "status": status,
            "question_id": str(candidate["question_id"]),
            "level": candidate.get("level", ""),
            "content": content,
            "content_hash": content_hash,
            "source_effect": repaired_effects.get(cid, raw_effect),
            "raw_source_effect": raw_effect,
            "audit_result": reaudit_by_id.get(cid) or audit_record or {},
            "intervention_protocol": "forced_c_v1",
            "schema_version": "source_validated_memory.1",
        }
        if status == "pending_transfer_validation":
            pending.append(record)
        else:
            not_validated[status].append(record)

    memory_root = root / "memory"
    _write_jsonl(memory_root / "pending_transfer_validation" / "store.jsonl", pending)
    atomic_save_json(
        memory_root / "pending_transfer_validation" / "retrieval_index.json",
        {
            "schema_version": "source_validated_memory_index.1",
            "memory_ids": [row["memory_id"] for row in pending],
            "memory_count": len(pending),
            "intervention_protocol": "forced_c_v1",
        },
    )
    for status, rows in sorted(not_validated.items()):
        _write_jsonl(memory_root / "not_validated" / f"{status}.jsonl", rows)
    all_records = pending + [row for rows in not_validated.values() for row in rows]
    _write_jsonl(memory_root / "all_records.jsonl", all_records)

    kimi_calls = len(initial_records) + len(reaudit_records) + len(diagnostic_records)
    summary = {
        "schema_version": PRODUCTION_VERSION,
        "question_count": len(bundles),
        "raw_candidate_count": len(candidates),
        "raw_effect_counts": dict(Counter(row.get("effect_label") for row in raw_effects)),
        "initial_helpful_count": len(helpful),
        "initial_kimi_audit_count": len(initial_records),
        "repair_count": len(repair_records),
        "repaired_validation_count": len(repair_candidates),
        "kimi_reaudit_count": len(reaudit_records),
        "diagnostic_kimi_count": len(diagnostic_records),
        "diagnostic_kimi_budget": diagnostic_budget,
        "kimi_call_count": kimi_calls,
        "kimi_max_calls": int(kimi_max_calls),
        "pending_transfer_validation_count": len(pending),
        "not_validated_counts": {key: len(rows) for key, rows in sorted(not_validated.items())},
        "deepseek_failure_analysis_question_count": len(failure_analyses),
    }
    if kimi_calls > int(kimi_max_calls):
        raise AssertionError("Kimi call budget exceeded")
    atomic_save_json(root / "audit_store_summary.json", summary)
    _write_production_report(root, summary)
    return summary


def _write_production_report(root: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Stage 2 首批生产报告", "",
        f"- 来源题：{summary['question_count']}",
        f"- Raw Principle：{summary['raw_candidate_count']}",
        f"- 来源题正收益候选：{summary['initial_helpful_count']}",
        f"- Kimi总调用：{summary['kimi_call_count']} / {summary['kimi_max_calls']}",
        f"- 待迁移验证Memory：{summary['pending_transfer_validation_count']}",
        "", "## Raw效果分布", "",
    ]
    for key, value in sorted((summary.get("raw_effect_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 未通过分布", ""])
    for key, value in sorted((summary.get("not_validated_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    path = root / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_production_complete(
    config: Any,
    *,
    production_root: Path,
    target_size: int,
    backbone: str,
    ports: Sequence[int],
    model: str,
    generation_concurrency: int,
    validation_concurrency: int,
    audit_concurrency: int,
    timeout: int,
    kimi_max_calls: int,
) -> dict[str, Any]:
    prepared = prepare_production_target(
        config, production_root=production_root,
        target_size=target_size, backbone=backbone,
    )
    target_root = Path(prepared["target_root"])

    # A real two-question end-to-end smoke precedes the expensive batch.  It
    # has an isolated artifact root so it cannot freeze or contaminate the
    # immutable 300/1200-question Candidate Bank.  All steps are recoverable.
    smoke_root = target_root / "smoke_0002"
    smoke_bundles_path = smoke_root / "source_bundles" / "source40_bundles.json"
    if not smoke_bundles_path.exists():
        target_bundles = load_json(target_root / "source_bundles" / "source40_bundles.json")
        smoke_bundles = list(target_bundles[:2])
        smoke_ids = [str(row["question_id"]) for row in smoke_bundles]
        smoke_manifest = {
            "schema_version": PRODUCTION_VERSION,
            "name": "production_smoke_0002",
            "question_count": len(smoke_bundles),
            "source_question_ids": smoke_ids,
            "source_bundle_hashes": {
                str(row["question_id"]): _json_hash(row) for row in smoke_bundles
            },
        }
        smoke_manifest["manifest_hash"] = _json_hash(smoke_manifest)
        atomic_save_json(smoke_root / "manifests" / "source40.json", smoke_manifest)
        atomic_save_json(smoke_bundles_path, smoke_bundles)
        atomic_save_json(smoke_root / "source_bundles" / "reserve20_bundles.json", [])
    logger.info("[Stage2生产预检] 开始2题端到端Smoke，结果目录=%s", smoke_root)
    smoke_generation = await generate_production_candidates(
        config, target_root=smoke_root, concurrency=min(2, generation_concurrency),
    )
    smoke_validation = await validate_production_candidates(
        config, target_root=smoke_root, ports=ports, model=model,
        concurrency=min(2, validation_concurrency), timeout=timeout,
    )
    smoke_audit = await audit_and_store_production(
        config, target_root=smoke_root, ports=ports, model=model,
        validation_concurrency=min(2, validation_concurrency),
        audit_concurrency=min(2, audit_concurrency), timeout=timeout,
        kimi_max_calls=min(30, kimi_max_calls), diagnostic_sample_size=0,
    )
    smoke = {
        "generation": smoke_generation,
        "validation": smoke_validation,
        "audit_store": smoke_audit,
    }
    atomic_save_json(smoke_root / "complete_summary.json", smoke)
    logger.info("[Stage2生产预检] 2题Smoke完成；开始正式%s题批次", target_size)
    generation = await generate_production_candidates(
        config, target_root=target_root, concurrency=generation_concurrency,
    )
    validation = await validate_production_candidates(
        config, target_root=target_root, ports=ports, model=model,
        concurrency=validation_concurrency, timeout=timeout,
    )
    audit_store = await audit_and_store_production(
        config, target_root=target_root, ports=ports, model=model,
        validation_concurrency=validation_concurrency,
        audit_concurrency=audit_concurrency, timeout=timeout,
        kimi_max_calls=kimi_max_calls,
    )
    result = {
        "prepare": prepared,
        "smoke": smoke,
        "generation": generation,
        "validation": validation,
        "audit_store": audit_store,
    }
    atomic_save_json(target_root / "complete_summary.json", result)
    return result
