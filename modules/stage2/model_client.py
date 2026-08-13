"""Role-specific Stage-2 model clients."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from core.api_call import async_llm_call


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*|```", "", text or "").strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class Stage2ModelClient:
    def __init__(self, config: Any):
        self.config = config.stage2

    async def generator(self, prompt: str) -> Dict[str, Any]:
        raw = await async_llm_call(
            prompt=prompt,
            model=self.config.generator_model,
            url=self.config.generator_url,
            api_key=os.environ.get(self.config.generator_key_env, ""),
            temperature=0.0,
            max_tokens=self.config.generation_max_tokens,
            timeout=600,
        )
        return {"raw": raw or "", "parsed": parse_json_object(raw or "")}

    async def auditor(self, prompt: str) -> Dict[str, Any]:
        raw = await async_llm_call(
            prompt=prompt,
            model=self.config.auditor_model,
            url=self.config.auditor_url,
            api_key=os.environ.get(self.config.auditor_key_env, "EMPTY"),
            temperature=0.0,
            max_tokens=self.config.audit_max_tokens,
            timeout=600,
        )
        return {"raw": raw or "", "parsed": parse_json_object(raw or "")}
