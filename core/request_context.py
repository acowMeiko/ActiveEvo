"""Task-local context attached to LLM calls for auditable logs."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_candidate_id: ContextVar[str] = ContextVar("llm_candidate_id", default="-")
_call_stage: ContextVar[str] = ContextVar("llm_call_stage", default="unspecified")


@contextmanager
def llm_request_context(candidate_id: str, call_stage: str) -> Iterator[None]:
    """Set candidate/stage labels without leaking them across asyncio tasks."""
    candidate_token = _candidate_id.set(str(candidate_id or "-"))
    stage_token = _call_stage.set(str(call_stage or "unspecified"))
    try:
        yield
    finally:
        _call_stage.reset(stage_token)
        _candidate_id.reset(candidate_token)


def request_context_label() -> str:
    return f"candidate={_candidate_id.get()} stage={_call_stage.get()}"


def current_request_stage() -> str:
    return _call_stage.get()
