"""Utilities for the frozen DeepSeek/Qwen Stage-2 generator comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence

from modules.stage2.model_client import Stage2ModelClient
from utils.io_utils import atomic_save_json, load_json


LEVELS = ("level1", "level2", "level3", "level4")


def _stable_key(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{question_id}{seed}".encode("utf-8")).hexdigest()


def build_balanced_manifest(
    packets_by_level: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    per_level: int,
    seed: int = 42,
    excluded_question_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """Select the same stable, balanced question set for both generators."""
    excluded = {str(value) for value in excluded_question_ids}
    by_level: Dict[str, list[str]] = {}
    for level in LEVELS:
        ids = {
            str(item.get("idx", item.get("question_id", "")))
            for item in packets_by_level.get(level, [])
            if str(item.get("idx", item.get("question_id", ""))).strip()
        } - excluded
        ordered = sorted(ids, key=lambda value: (_stable_key(value, seed), value))
        if len(ordered) < per_level:
            raise ValueError(f"{level} has only {len(ordered)} eligible questions; need {per_level}")
        by_level[level] = ordered[:per_level]
    flat = [question_id for level in LEVELS for question_id in by_level[level]]
    return {
        "policy": "sha256_balanced",
        "seed": seed,
        "per_level": per_level,
        "actual_count": len(flat),
        "excluded_question_ids": sorted(excluded),
        "question_ids_by_level": by_level,
        "question_ids": flat,
        "manifest_hash": hashlib.sha256(
            json.dumps(by_level, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


async def run_qwen_protocol_probe(
    config: Any, *, output_dir: Path, repeats: int = 3
) -> Dict[str, Any]:
    """Verify that Qwen returns separate reasoning and final JSON content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = Stage2ModelClient(config, generator_id="qwen")
    prompt = """Think through the protocol carefully, then return a final JSON object in
the answer channel. The final answer must be exactly an object with keys protocol_ok
(boolean) and note (string). Do not put JSON in the thinking block."""
    records = []
    for index in range(repeats):
        result = await client.generator(
            prompt,
            thinking_enabled=True,
            max_tokens=config.stage2.qwen_generation_analysis_max_tokens,
        )
        parsed = result.get("parsed", {})
        record = {
            "index": index,
            "endpoint": result.get("endpoint", ""),
            "reasoning_content": result.get("reasoning", ""),
            "final_content": result.get("content", result.get("raw", "")),
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage"),
            "parse_source": result.get("parse_source"),
            "attempts": result.get("attempts", []),
            "passed": bool(
                result.get("reasoning")
                and result.get("content", result.get("raw", ""))
                and result.get("finish_reason") == "stop"
                and isinstance(parsed, dict)
                and parsed.get("protocol_ok") is True
            ),
        }
        records.append(record)
        atomic_save_json(output_dir / "probe_records.json", records)
    summary = {
        "repeats": repeats,
        "passed": sum(bool(item["passed"]) for item in records),
        "all_passed": bool(records) and all(bool(item["passed"]) for item in records),
        "thinking_disabled_count": 0,
    }
    atomic_save_json(output_dir / "probe_summary.json", summary)
    return summary


def candidate_bank_summary(root: Path) -> Dict[str, Any]:
    candidates = []
    for path in sorted((root / "candidate_banks").glob("level*.json")):
        payload = load_json(str(path))
        candidates.extend(payload.get("candidates", []) if isinstance(payload, dict) else [])
    question_ids = {str(item.get("question_id", "")) for item in candidates}
    return {
        "candidate_count": len(candidates),
        "question_count": len(question_ids - {""}),
        "schema_valid_count": sum(not item.get("routing_schema_errors") for item in candidates),
        "quality_pass_count": sum(item.get("status") == "quality_pass" for item in candidates),
        "hard_rejected_count": sum(item.get("status") == "hard_rejected" for item in candidates),
        "audit_flagged_count": sum(item.get("status") == "audit_flagged" for item in candidates),
        "thinking_complete_count": sum(
            bool((item.get("generation") or {}).get("reasoning_content"))
            and (item.get("generation") or {}).get("finish_reason") == "stop"
            for item in candidates
        ),
    }


def _audit_metric(candidates: Sequence[Mapping[str, Any]], dimension: str) -> tuple[int, int]:
    passed = total = 0
    for candidate in candidates:
        audit = candidate.get("quality_audit") or {}
        value = (audit.get("dimensions") or {}).get(dimension)
        if isinstance(value, dict) and isinstance(value.get("pass"), bool):
            total += 1
            passed += int(value["pass"])
    return passed, total


def load_candidates(root: Path) -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    for path in sorted((root / "candidate_banks").glob("level*.json")):
        payload = load_json(str(path))
        if isinstance(payload, dict):
            candidates.extend(
                dict(item) for item in payload.get("candidates", []) if isinstance(item, dict)
            )
    return candidates


def generator_quality_summary(
    artifact_root: Path, source_validation_dir: Path | None = None
) -> Dict[str, Any]:
    """Aggregate candidate quality by source question, not raw candidate count."""
    candidates = load_candidates(artifact_root)
    by_question: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_question[str(candidate.get("question_id", ""))].append(candidate)
    generated_question_ids = set(by_question) - {""}
    manifest_path = artifact_root / "sample_manifest.json"
    manifest = load_json(str(manifest_path)) if manifest_path.exists() else {}
    question_ids = {
        str(value) for value in (
            manifest.get("question_ids", []) if isinstance(manifest, dict) else []
        ) if str(value).strip()
    } or generated_question_ids
    schema_legal_questions = {
        question_id for question_id, rows in by_question.items()
        if any(not row.get("routing_schema_errors") for row in rows)
    }
    leakage_count = sum(
        bool(((item.get("quality_audit") or {}).get("deterministic_leakage") or {}).get("detected"))
        for item in candidates
    )
    trigger_pass, trigger_total = _audit_metric(candidates, "trigger_decidability")
    evidence_pass, evidence_total = _audit_metric(candidates, "evidence_grounding")
    fidelity_pass, fidelity_total = _audit_metric(candidates, "source_fidelity")
    summary: Dict[str, Any] = {
        "artifact_root": str(artifact_root.resolve()),
        "candidate_count": len(candidates),
        "question_count": len(question_ids),
        "question_with_candidate_count": len(generated_question_ids),
        "no_reliable_principle_question_count": len(question_ids - generated_question_ids),
        "legal_principle_question_count": len(schema_legal_questions),
        "legal_principle_question_rate": (
            len(schema_legal_questions) / len(question_ids) if question_ids else 0.0
        ),
        "answer_leakage_count": leakage_count,
        "answer_leakage_rate": leakage_count / len(candidates) if candidates else 0.0,
        "trigger_decidable_count": trigger_pass,
        "trigger_audited_count": trigger_total,
        "trigger_decidable_rate": trigger_pass / trigger_total if trigger_total else 0.0,
        "evidence_grounded_count": evidence_pass,
        "evidence_audited_count": evidence_total,
        "evidence_grounded_rate": evidence_pass / evidence_total if evidence_total else 0.0,
        "source_fidelity_pass_count": fidelity_pass,
        "source_fidelity_audited_count": fidelity_total,
        # This is an audit proxy, not a verified mathematical-error rate.
        "audit_math_fact_error_proxy_rate": (
            (fidelity_total - fidelity_pass) / fidelity_total if fidelity_total else 0.0
        ),
    }
    if source_validation_dir and (source_validation_dir / "candidate_effects.json").exists():
        effects = load_json(str(source_validation_dir / "candidate_effects.json"))
        effects = [item for item in effects if isinstance(item, dict)]
        helpful_questions = {
            str(item.get("question_id", "")) for item in effects
            if item.get("direct_robust_label") == "robust_source_helpful"
            or item.get("adapted_robust_label") == "robust_source_helpful"
        } - {""}
        harmful = [
            item for item in effects
            if item.get("direct_robust_label") == "temperature_sensitive_or_harmful"
            or item.get("adapted_robust_label") == "temperature_sensitive_or_harmful"
        ]
        gain = loss = 0
        by_level: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "candidate_count": 0, "robust_helpful_count": 0, "harmful_or_sensitive_count": 0,
        })
        for item in effects:
            level_row = by_level[str(item.get("level", "unknown"))]
            level_row["candidate_count"] += 1
            is_helpful = (
                item.get("direct_robust_label") == "robust_source_helpful"
                or item.get("adapted_robust_label") == "robust_source_helpful"
            )
            is_harmful = item in harmful
            level_row["robust_helpful_count"] += int(is_helpful)
            level_row["harmful_or_sensitive_count"] += int(is_harmful)
            for temperature in item.get("temperatures", []):
                outcomes = ((temperature.get("adapted") or {}).get("paired_outcomes") or {})
                gain += int(outcomes.get("gain", 0))
                loss += int(outcomes.get("loss", 0))
        summary.update({
            "source_validated": True,
            "robust_helpful_question_count": len(helpful_questions),
            "robust_helpful_question_rate": (
                len(helpful_questions) / len(question_ids) if question_ids else 0.0
            ),
            "harmful_or_sensitive_candidate_count": len(harmful),
            "harmful_or_sensitive_candidate_rate": (
                len(harmful) / len(effects) if effects else 0.0
            ),
            "adapted_gain": gain,
            "adapted_loss": loss,
            "adapted_net_gain": gain - loss,
            "by_level": dict(sorted(by_level.items())),
        })
    else:
        summary["source_validated"] = False
    return summary


def compare_generators(
    *,
    deepseek_root: Path,
    qwen_root: Path,
    output_dir: Path,
    deepseek_validation: Path | None = None,
    qwen_validation: Path | None = None,
) -> Dict[str, Any]:
    """Apply the frozen winner cascade to paired generator artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    deepseek = generator_quality_summary(deepseek_root, deepseek_validation)
    qwen = generator_quality_summary(qwen_root, qwen_validation)
    metrics = {"deepseek": deepseek, "qwen": qwen}
    winner = ""
    reason = ""
    if any(item["answer_leakage_count"] for item in metrics.values()):
        eligible = [name for name, item in metrics.items() if item["answer_leakage_count"] == 0]
        if len(eligible) == 1:
            winner, reason = eligible[0], "only_generator_with_zero_answer_leakage"
    if not winner and all(item.get("source_validated") for item in metrics.values()):
        ranking = sorted(
            metrics,
            key=lambda name: (
                metrics[name].get("robust_helpful_question_rate", 0.0),
                -metrics[name].get("harmful_or_sensitive_candidate_rate", 1.0),
                metrics[name].get("adapted_net_gain", 0),
                name == "qwen",
            ),
            reverse=True,
        )
        winner = ranking[0]
        reason = "frozen_quality_cascade"
    result = {
        "metrics": metrics,
        "winner": winner,
        "winner_reason": reason,
        "selection_ready": bool(winner),
        "warning": "math fact error is represented only by an audit proxy unless independently annotated",
    }
    atomic_save_json(output_dir / "generator_comparison.json", result)
    lines = [
        "# Stage 2 Generator Comparison", "",
        f"- Winner: {winner or 'not selected'}", f"- Reason: {reason or 'source validation incomplete'}", "",
        "| Metric | DeepSeek | Qwen3.8 |", "|---|---:|---:|",
    ]
    for key in (
        "question_count", "candidate_count", "legal_principle_question_rate",
        "answer_leakage_rate", "trigger_decidable_rate", "evidence_grounded_rate",
        "robust_helpful_question_rate", "harmful_or_sensitive_candidate_rate",
        "adapted_net_gain",
    ):
        lines.append(f"| {key} | {deepseek.get(key, 'N/A')} | {qwen.get(key, 'N/A')} |")
    (output_dir / "generator_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
