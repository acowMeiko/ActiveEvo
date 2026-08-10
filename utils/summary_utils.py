from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_div(numerator: float, denominator: float) -> float:
    return (numerator / denominator) if denominator else 0.0


def compact_runtime_metrics(runtime_metrics: Optional[Dict[str, Any]], sample_count: int) -> Dict[str, Any]:
    raw = dict(runtime_metrics or {})
    wall_time_sec = float(raw.get("wall_time_sec") or 0.0)
    total_tokens = int(raw.get("total_tokens") or 0)
    scope_count = max(int(sample_count or 0), 0)
    return {
        "wall_time_sec": wall_time_sec,
        "llm_calls": int(raw.get("llm_calls") or 0),
        "total_tokens": total_tokens,
        "token_source": raw.get("token_source", "unknown"),
        "avg_wall_time_per_sample": _safe_div(wall_time_sec, scope_count),
        "avg_tokens_per_sample": _safe_div(total_tokens, scope_count),
    }


def build_stage3_summary(
    results: List[Dict[str, Any]],
    *,
    scope_total: int,
    correct: int,
    memory_experience_count: int,
    dataset: Optional[str] = None,
    mode: Optional[str] = None,
    eval_level: Optional[int] = None,
    test_scope_key: Optional[str] = None,
    runtime_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    valid = [r for r in results if r.get("status") == "success"]
    retrieved = [r for r in valid if r.get("retrieval_used")]
    retrieved_count = len(retrieved)
    retrieved_correct = sum(1 for r in retrieved if r.get("is_correct"))

    meta: Dict[str, Any] = {
        "summary_version": "stage3.v2",
        "dataset": dataset,
        "mode": mode or next((r.get("mode") for r in valid if r.get("mode")), None),
        "eval_level": eval_level,
        "test_scope_key": test_scope_key or "default",
    }
    meta = {key: value for key, value in meta.items() if value is not None}

    return {
        "meta": meta,
        "performance": {
            "scope_total": int(scope_total),
            "correct": int(correct),
            "accuracy": _safe_div(correct, scope_total),
        },
        "retrieval": {
            "memory_experience_count": int(memory_experience_count),
            "retrieved_samples": retrieved_count,
            "retrieval_used_ratio": _safe_div(retrieved_count, scope_total),
            "retrieved_correct": retrieved_correct,
            "retrieved_accuracy": _safe_div(retrieved_correct, retrieved_count),
        },
        "efficiency": compact_runtime_metrics(runtime_metrics, scope_total),
    }
