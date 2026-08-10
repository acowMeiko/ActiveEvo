from __future__ import annotations

import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional


_CURRENT_COLLECTOR: ContextVar[Optional["RuntimeMetricsCollector"]] = ContextVar(
    "memua_runtime_metrics_collector",
    default=None,
)


def estimate_tokens(text: Any) -> int:
    """Small dependency-free token estimate used when the API omits usage."""
    value = str(text or "")
    if not value:
        return 0
    cjk = sum(1 for ch in value if "\u4e00" <= ch <= "\u9fff")
    non_cjk = "".join(ch if not ("\u4e00" <= ch <= "\u9fff") else " " for ch in value)
    pieces = re.findall(r"\w+|[^\s\w]", non_cjk, flags=re.UNICODE)
    return int(cjk + len(pieces))


class RuntimeMetricsCollector:
    def __init__(self, stage: str, metadata: Optional[Dict[str, Any]] = None):
        self.stage = stage
        self.metadata = dict(metadata or {})
        self.started_at = time.time()
        self.started_perf = time.perf_counter()
        self.finished_at: Optional[float] = None
        self.wall_time_sec: Optional[float] = None
        self.llm_calls = 0
        self.successful_llm_calls = 0
        self.failed_llm_calls = 0
        self.api_duration_sec = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated_prompt_tokens = 0
        self.estimated_completion_tokens = 0
        self.estimated_total_tokens = 0
        self.usage_reported_calls = 0

    def record_llm_call(
        self,
        *,
        prompt: str,
        response: str = "",
        usage: Optional[Dict[str, Any]] = None,
        duration_sec: float = 0.0,
        success: bool = True,
    ) -> None:
        self.llm_calls += 1
        self.api_duration_sec += float(duration_sec or 0.0)
        if success:
            self.successful_llm_calls += 1
        else:
            self.failed_llm_calls += 1

        est_prompt = estimate_tokens(prompt)
        est_completion = estimate_tokens(response)
        self.estimated_prompt_tokens += est_prompt
        self.estimated_completion_tokens += est_completion
        self.estimated_total_tokens += est_prompt + est_completion

        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            self.usage_reported_calls += 1
            p = int(prompt_tokens or 0)
            c = int(completion_tokens or 0)
            t = int(total_tokens if total_tokens is not None else p + c)
            self.prompt_tokens += p
            self.completion_tokens += c
            self.total_tokens += t

    def finish(self) -> Dict[str, Any]:
        self.finished_at = time.time()
        self.wall_time_sec = time.perf_counter() - self.started_perf
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        total_tokens = self.total_tokens if self.usage_reported_calls else self.estimated_total_tokens
        prompt_tokens = self.prompt_tokens if self.usage_reported_calls else self.estimated_prompt_tokens
        completion_tokens = self.completion_tokens if self.usage_reported_calls else self.estimated_completion_tokens
        return {
            "stage": self.stage,
            "metadata": self.metadata,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_time_sec": self.wall_time_sec,
            "llm_calls": self.llm_calls,
            "successful_llm_calls": self.successful_llm_calls,
            "failed_llm_calls": self.failed_llm_calls,
            "api_duration_sec": self.api_duration_sec,
            "usage_reported_calls": self.usage_reported_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_source": "api_usage" if self.usage_reported_calls else "estimated",
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "estimated_completion_tokens": self.estimated_completion_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
            "api_prompt_tokens": self.prompt_tokens,
            "api_completion_tokens": self.completion_tokens,
            "api_total_tokens": self.total_tokens,
        }


@contextmanager
def collect_runtime_metrics(stage: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[RuntimeMetricsCollector]:
    collector = RuntimeMetricsCollector(stage=stage, metadata=metadata)
    token = _CURRENT_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        collector.finish()
        _CURRENT_COLLECTOR.reset(token)


def current_collector() -> Optional[RuntimeMetricsCollector]:
    return _CURRENT_COLLECTOR.get()


def record_llm_call(**kwargs: Any) -> None:
    collector = current_collector()
    if collector is not None:
        collector.record_llm_call(**kwargs)
