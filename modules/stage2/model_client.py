"""Role-specific Stage-2 model clients."""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List

from core.api_call import async_llm_call_with_metadata


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


def _choice_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured-output fields without treating hidden reasoning as JSON."""
    provider = result.get("raw", {}) if isinstance(result, dict) else {}
    choices = provider.get("choices", []) if isinstance(provider, dict) else []
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    reasoning = message.get("reasoning_content", message.get("reasoning", ""))
    if not isinstance(reasoning, str):
        reasoning = ""
    return {
        "content": content.strip(),
        "reasoning": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "usage": provider.get("usage") if isinstance(provider, dict) else None,
        "provider_response": provider,
        "transport_attempts": result.get("transport_attempts", []),
    }


class Stage2ModelClient:
    def __init__(self, config: Any, *, generator_id: str = "deepseek"):
        self.config = config.stage2
        if generator_id not in {"deepseek", "qwen", "kimi"}:
            raise ValueError(f"Unsupported Stage-2 generator: {generator_id}")
        self.generator_id = generator_id
        self._generator_cursor = 0
        self._generator_lock = asyncio.Lock()
        configured_urls = getattr(self.config, "auditor_urls", ())
        self._auditor_urls: List[str] = [str(url).strip() for url in configured_urls if str(url).strip()]
        if not self._auditor_urls:
            self._auditor_urls = [self.config.auditor_url]
        self._auditor_cursor = 0
        self._auditor_lock = asyncio.Lock()

    @property
    def requires_full_thinking(self) -> bool:
        return self.generator_id in {"qwen", "kimi"}

    async def _ordered_generator_urls(self) -> List[str]:
        if self.generator_id == "deepseek":
            return [self.config.generator_url]
        if self.generator_id == "kimi":
            return await self._ordered_auditor_urls()
        urls = [str(url).strip() for url in self.config.qwen_generator_urls if str(url).strip()]
        if not urls:
            raise ValueError("Qwen generator has no configured endpoint")
        async with self._generator_lock:
            start = self._generator_cursor % len(urls)
            self._generator_cursor += 1
        return urls[start:] + urls[:start]

    async def _ordered_auditor_urls(self) -> List[str]:
        """Return a round-robin order, with each endpoint appearing exactly once."""
        async with self._auditor_lock:
            start = self._auditor_cursor % len(self._auditor_urls)
            self._auditor_cursor += 1
        return self._auditor_urls[start:] + self._auditor_urls[:start]

    async def generator(
        self,
        prompt: str,
        *,
        thinking_enabled: bool | None = None,
        max_tokens: int | None = None,
        accept_reasoning_only: bool = False,
    ) -> Dict[str, Any]:
        if self.generator_id == "qwen":
            # Qwen3.8 is a thinking model.  Never disable thinking merely to
            # make JSON easier to parse, and never treat reasoning_content as
            # the final structured answer.
            use_thinking = True
            requested_tokens = max_tokens or self.config.qwen_generation_max_tokens
            model = self.config.qwen_generator_model
            key = os.environ.get(self.config.qwen_generator_key_env, "EMPTY")
        elif self.generator_id == "kimi":
            # Kimi diagnoses with thinking, but the second-stage synthesis is
            # a constrained JSON serialization task and may explicitly turn it
            # off.  Final structured data is always parsed from content, never
            # from reasoning_content.
            use_thinking = True if thinking_enabled is None else bool(thinking_enabled)
            requested_tokens = max_tokens or self.config.generation_max_tokens
            model = self.config.auditor_model
            key = os.environ.get(self.config.auditor_key_env, "")
        else:
            use_thinking = (
                self.config.generator_synthesis_thinking_enabled
                if thinking_enabled is None
                else bool(thinking_enabled)
            )
            requested_tokens = max_tokens or self.config.generation_max_tokens
            model = self.config.generator_model
            key = os.environ.get(self.config.generator_key_env, "")

        attempts: List[Dict[str, Any]] = []
        last_details: Dict[str, Any] = {
            "content": "", "reasoning": "", "finish_reason": None,
            "usage": None, "provider_response": {},
        }
        urls = await self._ordered_generator_urls()
        endpoint = urls[0]
        for endpoint_index, url in enumerate(urls):
            # vLLM reserves one terminal token when checking prompt + output
            # against max_model_len.  A request that arithmetically equals the
            # context limit can therefore fail as one token too long.  Keep a
            # one-token safety margin without changing the configured protocol
            # or invalidating existing generation checkpoints.
            effective_tokens = max(1, requested_tokens - 1) if self.generator_id == "qwen" else requested_tokens
            token_budgets = [effective_tokens]
            if self.generator_id == "qwen" and requested_tokens < self.config.qwen_generation_retry_max_tokens:
                token_budgets.append(max(1, self.config.qwen_generation_retry_max_tokens - 1))
            if (
                self.generator_id == "kimi"
                and use_thinking
                and requested_tokens < getattr(
                    self.config,
                    "role_compare_kimi_generation_retry_max_tokens",
                    requested_tokens,
                )
            ):
                token_budgets.append(
                    getattr(
                        self.config,
                        "role_compare_kimi_generation_retry_max_tokens",
                        requested_tokens,
                    )
                )
            for token_budget in token_budgets:
                kwargs: Dict[str, Any]
                if self.generator_id == "qwen":
                    kwargs = {
                        "temperature": self.config.qwen_generator_temperature,
                        "top_p": self.config.qwen_generator_top_p,
                        "top_k": self.config.qwen_generator_top_k,
                        "min_p": self.config.qwen_generator_min_p,
                        "presence_penalty": self.config.qwen_generator_presence_penalty,
                        "repetition_penalty": self.config.qwen_generator_repetition_penalty,
                        "seed": self.config.generator_seed,
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                            "preserve_thinking": False,
                        },
                    }
                    if self.config.qwen_generator_reasoning_effort:
                        kwargs["reasoning_effort"] = self.config.qwen_generator_reasoning_effort
                elif self.generator_id == "kimi":
                    kwargs = {
                        "temperature": (
                            self.config.kimi_thinking_temperature
                            if use_thinking else self.config.kimi_nonthinking_temperature
                        ),
                        "top_p": (
                            self.config.kimi_thinking_top_p
                            if use_thinking else self.config.kimi_nonthinking_top_p
                        ),
                        "seed": self.config.generator_seed,
                        "thinking": {"type": "enabled" if use_thinking else "disabled"},
                    }
                    # The compact synthesis step is a JSON serialization task.
                    # Asking the provider for a JSON object here avoids wasting
                    # its answer budget on Markdown fences or prose.
                    if not use_thinking:
                        kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs = {
                        "temperature": 0.0,
                        "top_p": 0.95,
                        "seed": self.config.generator_seed,
                        "thinking": {"type": "enabled" if use_thinking else "disabled"},
                    }
                result = await async_llm_call_with_metadata(
                    prompt=prompt,
                    model=model,
                    url=url,
                    api_key=key,
                    max_tokens=token_budget,
                    retries=self.config.generator_retries if self.generator_id == "deepseek" else 1,
                    timeout=self.config.generator_timeout,
                    request_logprobs=False,
                    include_raw=True,
                    **kwargs,
                )
                details = _choice_metadata(result)
                last_details = details
                content = details["content"]
                truncated = details.get("finish_reason") == "length"
                attempts.append({
                    "endpoint": url,
                    "max_tokens": token_budget,
                    "finish_reason": details.get("finish_reason"),
                    "reasoning_nonempty": bool(details["reasoning"]),
                    "content_nonempty": bool(content),
                    "reasoning_content": details["reasoning"],
                    "final_content": content,
                    "usage": details.get("usage"),
                    "transport_attempts": details.get("transport_attempts", []),
                    "failure_reason": (
                        "thinking_truncated" if truncated else
                        "missing_final_content" if not content else ""
                    ),
                })
                endpoint = url
                if accept_reasoning_only and details["reasoning"]:
                    # Diagnosis uses the reasoning trace itself as evidence;
                    # it does not require a duplicate answer-channel render.
                    # Accepting the first non-empty trace avoids repeating the
                    # same long diagnosis across every local endpoint.
                    return {
                        "raw": content,
                        "parsed": parse_json_object(content),
                        "parse_source": "message.content",
                        "endpoint": url,
                        "generator_id": self.generator_id,
                        "thinking_enabled": use_thinking,
                        "reasoning_only_accepted": not bool(content),
                        "attempts": attempts,
                        **details,
                    }
                if content and not truncated:
                    return {
                        "raw": content,
                        "parsed": parse_json_object(content),
                        "parse_source": "message.content",
                        "endpoint": url,
                        "generator_id": self.generator_id,
                        "thinking_enabled": use_thinking,
                        "attempts": attempts,
                        **details,
                    }
                # A length/empty-content retry remains a full thinking call.
                # Transport failover uses the next endpoint only after the
                # larger-budget attempt has also failed.
            if endpoint_index + 1 < len(urls):
                continue
        final = attempts[-1] if attempts else {}
        return {
            "raw": "",
            "parsed": {},
            "parse_source": "message.content",
            "endpoint": endpoint,
            "generator_id": self.generator_id,
            "thinking_enabled": use_thinking,
            "attempts": attempts,
            "content": last_details.get("content", ""),
            "reasoning": last_details.get("reasoning", ""),
            "finish_reason": last_details.get("finish_reason", final.get("finish_reason")),
            "usage": last_details.get("usage"),
            "provider_response": last_details.get("provider_response", {}),
            "failure_reason": final.get("failure_reason", "generator_failed"),
        }

    async def auditor(
        self,
        prompt: str,
        *,
        auditor_id: str = "kimi",
        thinking_enabled: bool = True,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        """Call an independent auditor without leaking provider-specific options.

        ``kimi`` uses Moonshot's fixed K2.6 sampling contract.  ``qwen`` is
        retained only for compatibility with historical experiments.
        Structured data is parsed exclusively from final ``message.content``;
        hidden reasoning is always retained as audit evidence and never parsed.
        """
        if auditor_id not in {"deepseek", "qwen", "kimi"}:
            raise ValueError(f"Unsupported Stage-2 auditor: {auditor_id}")
        attempted_urls: List[str] = []
        last_details: Dict[str, Any] = {
            "content": "", "reasoning": "", "finish_reason": None,
            "usage": None, "provider_response": {},
        }
        urls = (
            [self.config.generator_url]
            if auditor_id == "deepseek" else await self._ordered_auditor_urls()
        )
        call_attempts: List[Dict[str, Any]] = []
        for url in urls:
            attempted_urls.append(url)
            if auditor_id == "deepseek":
                model = self.config.generator_model
                api_key = os.environ.get(self.config.generator_key_env, "")
                extra = {
                    "thinking": {
                        "type": "enabled" if thinking_enabled else "disabled"
                    }
                }
                if json_mode:
                    extra["response_format"] = {"type": "json_object"}
                temperature = 0.0
                top_p = 0.95
            elif auditor_id == "kimi":
                model = self.config.auditor_model
                api_key = os.environ.get(self.config.auditor_key_env, "")
                extra = {
                    "thinking": {
                        "type": "enabled" if thinking_enabled else "disabled"
                    },
                }
                if json_mode:
                    extra["response_format"] = {"type": "json_object"}
                temperature = (
                    getattr(self.config, "kimi_thinking_temperature", 1.0)
                    if thinking_enabled else
                    getattr(self.config, "kimi_nonthinking_temperature", 0.6)
                )
                top_p = (
                    getattr(self.config, "kimi_thinking_top_p", 0.95)
                    if thinking_enabled else
                    getattr(self.config, "kimi_nonthinking_top_p", 0.95)
                )
            else:
                model = self.config.auditor_model
                api_key = os.environ.get(self.config.auditor_key_env, "EMPTY")
                extra = {
                    "reasoning_effort": getattr(
                        self.config, "auditor_reasoning_effort", "medium"
                    ),
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "preserve_thinking": False,
                    },
                }
                temperature = 0.0
                top_p = 0.95
            token_budgets = [max_tokens or self.config.audit_max_tokens]
            if auditor_id == "qwen":
                retry_budget = getattr(
                    self.config, "audit_retry_max_tokens", self.config.audit_max_tokens
                )
                if retry_budget > self.config.audit_max_tokens:
                    token_budgets.append(retry_budget)
            endpoint_had_provider_response = False
            for token_budget in token_budgets:
                result = await async_llm_call_with_metadata(
                    prompt=prompt,
                    model=model,
                    url=url,
                    api_key=api_key,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=token_budget,
                    retries=(
                        self.config.generator_retries
                        if auditor_id == "deepseek" else self.config.auditor_retries_per_endpoint
                    ),
                    timeout=self.config.auditor_timeout,
                    request_logprobs=False,
                    include_raw=True,
                    **extra,
                )
                details = _choice_metadata(result)
                last_details = details
                raw = details["content"]
                provider = details.get("provider_response", {})
                endpoint_had_provider_response = bool(
                    isinstance(provider, dict) and provider.get("choices")
                )
                call_attempts.append({
                    "endpoint": url,
                    "max_tokens": token_budget,
                    "finish_reason": details.get("finish_reason"),
                    "reasoning_nonempty": bool(details.get("reasoning")),
                    "content_nonempty": bool(raw),
                    "transport_attempts": details.get("transport_attempts", []),
                })
                if raw:
                    return {
                        "raw": raw,
                        "parsed": parse_json_object(raw),
                        "endpoint": url,
                        "attempted_endpoints": attempted_urls,
                        "auditor_id": auditor_id,
                        "attempts": call_attempts,
                        **details,
                    }
                # A valid Qwen response with only reasoning is a length/final-
                # content failure, not an endpoint outage. Expand on the same
                # endpoint once and never duplicate it on another GPU.
                if auditor_id in {"qwen", "kimi"} and endpoint_had_provider_response:
                    continue
                break
            if auditor_id in {"qwen", "kimi"} and endpoint_had_provider_response:
                break
        return {
            "raw": "",
            "parsed": {},
            "endpoint": "",
            "attempted_endpoints": attempted_urls,
            "auditor_id": auditor_id,
            "attempts": call_attempts,
            **last_details,
        }

    async def format_audit(
        self, prompt: str, *, formatter_id: str = "kimi"
    ) -> Dict[str, Any]:
        """Serialize a completed critique without performing a new audit.

        The formatter receives only critic text and the JSON contract.  It is
        therefore safe to use DeepSeek as a serializer for a Kimi critique:
        it cannot inspect or revise its own generated Principle.
        """
        return await self.auditor(
            prompt,
            auditor_id=formatter_id,
            thinking_enabled=False,
            max_tokens=getattr(self.config, "audit_formatter_max_tokens", 3072),
            json_mode=True,
        )

    async def kimi_generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 8192,
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        """Use Kimi for blind variant construction or current-question binding."""
        return await self.auditor(
            prompt,
            auditor_id="kimi",
            thinking_enabled=True,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
