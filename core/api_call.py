# core/api_call.py
import json
import logging
import asyncio
import math
import time
from typing import Optional, Dict, Any

from core.runtime_metrics import record_llm_call
from core.request_context import request_context_label

# 尝试导入异步库
try:
    import aiohttp
    _ASYNC_OK = True
except ImportError:
    _ASYNC_OK = False

logger = logging.getLogger(__name__)


def _normalize_chat_completions_url(url: str) -> str:
    """兼容传入 base URL (如 .../v1) 或完整 chat/completions URL。"""
    normalized = (url or "").strip().rstrip("/")
    if not normalized:
        return normalized

    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    if normalized.endswith("/v1/"):
        return f"{normalized}chat/completions"

    return normalized


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
        return "\n".join(parts).strip()
    return ""

# ================== 异步接口 (核心) ==================

def _first_choice(data: Dict[str, Any]) -> Dict[str, Any]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _coerce_logprob(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _top_logprobs_from_entry(entry: Dict[str, Any]) -> list:
    top = entry.get("top_logprobs")
    if isinstance(top, list):
        return [x for x in top if isinstance(x, dict)]
    if isinstance(top, dict):
        return [{"token": token, "logprob": logprob} for token, logprob in top.items()]
    return []


def _iter_logprob_entries(data: Dict[str, Any]) -> list:
    choice = _first_choice(data)
    logprobs = choice.get("logprobs") if isinstance(choice, dict) else None
    if not isinstance(logprobs, dict):
        return []

    content = logprobs.get("content")
    if isinstance(content, list):
        return [x for x in content if isinstance(x, dict)]

    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    if isinstance(tokens, list) and isinstance(token_logprobs, list):
        entries = []
        for i, token in enumerate(tokens):
            entry = {"token": token, "logprob": token_logprobs[i] if i < len(token_logprobs) else None}
            if isinstance(top_logprobs, list) and i < len(top_logprobs):
                entry["top_logprobs"] = top_logprobs[i]
            entries.append(entry)
        return entries
    return []


def _normalized_entropy_from_top_logprobs(top_entries: list) -> Optional[float]:
    logps = []
    for item in top_entries:
        logprob = _coerce_logprob(item.get("logprob"))
        if logprob is not None:
            logps.append(logprob)
    if len(logps) <= 1:
        return None

    max_logp = max(logps)
    weights = [math.exp(lp - max_logp) for lp in logps]
    total = sum(weights)
    if total <= 0:
        return None

    probs = [w / total for w in weights]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0:
        return None
    return max(0.0, min(1.0, entropy / max_entropy))


def _logprob_entropy_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    entries = _iter_logprob_entries(data)
    token_entropies = []
    token_logprobs = []
    available_top_k = []

    for entry in entries:
        logprob = _coerce_logprob(entry.get("logprob"))
        if logprob is not None:
            token_logprobs.append(logprob)

        top_entries = _top_logprobs_from_entry(entry)
        entropy = _normalized_entropy_from_top_logprobs(top_entries)
        if entropy is not None:
            token_entropies.append(entropy)
            available_top_k.append(len(top_entries))

    normalized_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 1.0
    avg_neg_logprob = -sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
    return {
        "available": bool(token_entropies),
        "normalized_entropy": max(0.0, min(1.0, normalized_entropy)),
        "trajectory_entropy": max(0.0, min(1.0, normalized_entropy)),
        "token_count": len(entries),
        "entropy_token_count": len(token_entropies),
        "avg_neg_logprob": avg_neg_logprob,
        "avg_top_logprobs": sum(available_top_k) / len(available_top_k) if available_top_k else 0.0,
        "method": "top_logprobs_normalized_entropy" if token_entropies else "unavailable_default_max_entropy",
    }


def _metadata_response(
    prompt: str,
    data: Dict[str, Any],
    duration_sec: float,
    include_raw: bool = False,
    record_metrics: bool = True,
) -> Dict[str, Any]:
    text = _parse_response(data)
    result = {
        "text": text,
        "usage": data.get("usage") if isinstance(data, dict) else None,
        "entropy": _logprob_entropy_summary(data if isinstance(data, dict) else {}),
        "duration_sec": duration_sec,
    }
    if include_raw:
        result["raw"] = data
    if record_metrics:
        record_llm_call(
            prompt=prompt,
            response=text,
            usage=result["usage"],
            duration_sec=duration_sec,
            success=bool(text),
        )
    return result


async def async_llm_call(
    prompt: str,
    model: str,              # 必填
    url: str,                # 必填
    api_key: Optional[str] = None, # 选填
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    retries: int = 3,
    timeout: int = 120,
    session: Optional["aiohttp.ClientSession"] = None,
    **kwargs
) -> str:
    """
    通用异步 LLM 调用接口
    兼容本地 vLLM (无需 Key) 和 外部商用 API (需 Key)
    """
    if not _ASYNC_OK:
        logger.error("aiohttp is required for async calls.")
        return ""

    request_url = _normalize_chat_completions_url(url)

    # 1. 构建 Header
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    if api_key and api_key.strip() and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    # 2. 构建 Payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        **kwargs
    }

    # 3. Session 管理
    manage_session = False
    if session is None:
        session = aiohttp.ClientSession()
        manage_session = True

    # 设置超时
    timeout_obj = aiohttp.ClientTimeout(total=timeout, connect=20)

    try:
        for attempt in range(1, retries + 1):
            attempt_start = time.perf_counter()
            try:
                logger.debug(
                    "LLM request start (%s, endpoint_attempt=%d/%d) url=%s model=%s",
                    request_context_label(), attempt, retries, request_url, model,
                )
                async with session.post(request_url, headers=headers, json=payload, timeout=timeout_obj) as resp:

                    if resp.status == 200:
                        data = await resp.json()
                        parsed_text = _parse_response(data)
                        record_llm_call(
                            prompt=prompt,
                            response=parsed_text,
                            usage=data.get("usage") if isinstance(data, dict) else None,
                            duration_sec=time.perf_counter() - attempt_start,
                            success=bool(parsed_text),
                        )
                        if not parsed_text:
                            logger.warning(
                                f"API returned 200 but empty parsed content. Model: {model} | URL: {request_url} | "
                                f"ResponsePreview: {json.dumps(data, ensure_ascii=False)[:500]}"
                            )
                        return parsed_text

                    error_text = await resp.text()
                    record_llm_call(
                        prompt=prompt,
                        response="",
                        duration_sec=time.perf_counter() - attempt_start,
                        success=False,
                    )
                    logger.warning(
                        "API error HTTP %d (%s) model=%s url=%s body=%s",
                        resp.status, request_context_label(), model, request_url, error_text[:200],
                    )

                    if resp.status == 404:
                        logger.error(
                            "HTTP 404: 请检查 strong_model_url 是否指向 /v1/chat/completions，"
                            f"以及模型名 `{model}` 是否在服务端可用。"
                        )
                        return ""

                    if resp.status in [429, 500, 502, 503, 504]:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    if resp.status in [401, 400, 403]:
                        logger.error(f"Critical API Error: {error_text}")
                        return ""

            except asyncio.TimeoutError:
                record_llm_call(
                    prompt=prompt,
                    response="",
                    duration_sec=time.perf_counter() - attempt_start,
                    success=False,
                )
                logger.warning(
                    "Timeout (%s, endpoint_attempt=%d/%d) calling %s",
                    request_context_label(), attempt, retries, request_url,
                )
                await asyncio.sleep(2)
            except Exception as e:
                record_llm_call(
                    prompt=prompt,
                    response="",
                    duration_sec=time.perf_counter() - attempt_start,
                    success=False,
                )
                logger.error(
                    "Network exception (%s, endpoint_attempt=%d/%d) calling %s: %s",
                    request_context_label(), attempt, retries, request_url, e,
                )
                await asyncio.sleep(2)

        logger.error(
            "Failed to get response from %s (%s) after %d endpoint attempt(s).",
            model, request_context_label(), retries,
        )
        return ""

    finally:
        if manage_session and session:
            await session.close()

# ================== 同步接口 (保留用于简单测试) ==================

async def async_llm_call_with_metadata(
    prompt: str,
    model: str,
    url: str,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    retries: int = 3,
    timeout: int = 120,
    session: Optional["aiohttp.ClientSession"] = None,
    request_logprobs: bool = True,
    top_logprobs: int = 5,
    include_raw: bool = False,
    **kwargs
) -> Dict[str, Any]:
    if not _ASYNC_OK:
        logger.error("aiohttp is required for async calls.")
        return _metadata_response(prompt, {}, 0.0, include_raw=include_raw)

    request_url = _normalize_chat_completions_url(url)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key and api_key.strip() and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        **kwargs,
    }
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = max(1, int(top_logprobs))

    manage_session = False
    if session is None:
        session = aiohttp.ClientSession()
        manage_session = True

    timeout_obj = aiohttp.ClientTimeout(total=timeout, connect=20)
    logprob_retry_used = False

    try:
        for attempt in range(1, retries + 1):
            attempt_start = time.perf_counter()
            try:
                logger.debug(
                    "LLM request start (%s, endpoint_attempt=%d/%d) url=%s model=%s",
                    request_context_label(), attempt, retries, request_url, model,
                )
                async with session.post(request_url, headers=headers, json=payload, timeout=timeout_obj) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = _metadata_response(
                            prompt,
                            data,
                            time.perf_counter() - attempt_start,
                            include_raw=include_raw,
                        )
                        if not result["text"]:
                            logger.warning(
                                f"API returned 200 but empty parsed content. Model: {model} | URL: {request_url} | "
                                f"ResponsePreview: {json.dumps(data, ensure_ascii=False)[:500]}"
                            )
                        return result

                    error_text = await resp.text()
                    record_llm_call(
                        prompt=prompt,
                        response="",
                        duration_sec=time.perf_counter() - attempt_start,
                        success=False,
                    )

                    if (
                        resp.status == 400
                        and request_logprobs
                        and not logprob_retry_used
                        and ("logprob" in error_text.lower() or "top_logprobs" in error_text.lower())
                    ):
                        payload.pop("logprobs", None)
                        payload.pop("top_logprobs", None)
                        logprob_retry_used = True
                        logger.warning("Server rejected logprobs; retrying without entropy metadata.")
                        continue

                    logger.warning(
                        "API error HTTP %d (%s) model=%s url=%s body=%s",
                        resp.status, request_context_label(), model, request_url, error_text[:200],
                    )

                    if resp.status == 404:
                        logger.error("HTTP 404: check whether the model endpoint and model name are correct.")
                        return _metadata_response(
                            prompt, {}, time.perf_counter() - attempt_start,
                            include_raw=include_raw, record_metrics=False
                        )
                    if resp.status in [429, 500, 502, 503, 504]:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    if resp.status in [401, 400, 403]:
                        logger.error(f"Critical API Error: {error_text}")
                        return _metadata_response(
                            prompt, {}, time.perf_counter() - attempt_start,
                            include_raw=include_raw, record_metrics=False
                        )

            except asyncio.TimeoutError:
                record_llm_call(
                    prompt=prompt,
                    response="",
                    duration_sec=time.perf_counter() - attempt_start,
                    success=False,
                )
                logger.warning(
                    "Timeout (%s, endpoint_attempt=%d/%d) calling %s",
                    request_context_label(), attempt, retries, request_url,
                )
                await asyncio.sleep(2)
            except Exception as e:
                record_llm_call(
                    prompt=prompt,
                    response="",
                    duration_sec=time.perf_counter() - attempt_start,
                    success=False,
                )
                logger.error(
                    "Network exception (%s, endpoint_attempt=%d/%d) calling %s: %s",
                    request_context_label(), attempt, retries, request_url, e,
                )
                await asyncio.sleep(2)

        logger.error(
            "Failed to get response from %s (%s) after %d endpoint attempt(s).",
            model, request_context_label(), retries,
        )
        return _metadata_response(prompt, {}, 0.0, include_raw=include_raw, record_metrics=False)

    finally:
        if manage_session and session:
            await session.close()


def llm_call(prompt: str, model: str, url: str, api_key: Optional[str] = None, **kwargs) -> str:
    import requests
    request_url = _normalize_chat_completions_url(url)
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip() and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **kwargs
    }
    try:
        resp = requests.post(request_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        parsed_text = _parse_response(data)
        if not parsed_text:
            logger.warning(
                f"Sync API returned 200 but empty parsed content. Model: {model} | URL: {request_url} | "
                f"ResponsePreview: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        return parsed_text
    except Exception as e:
        logger.error(f"Sync call failed: {e}")
        return ""

def _parse_response(data: Dict[str, Any]) -> str:
    if not data:
        return ""

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}

        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        if isinstance(message, dict):
            text = _extract_text_from_content(message.get("content"))
            if text:
                return text

            reasoning_text = _extract_text_from_content(message.get("reasoning_content"))
            if reasoning_text:
                return reasoning_text

        delta = first_choice.get("delta") if isinstance(first_choice, dict) else None
        if isinstance(delta, dict):
            delta_text = _extract_text_from_content(delta.get("content"))
            if delta_text:
                return delta_text

        top_level_text = first_choice.get("text") if isinstance(first_choice, dict) else None
        if isinstance(top_level_text, str) and top_level_text.strip():
            return top_level_text.strip()

    output = data.get("output")
    if isinstance(output, list):
        output_parts = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content")
                text = _extract_text_from_content(content)
                if text:
                    output_parts.append(text)
        merged_output = "\n".join(output_parts).strip()
        if merged_output:
            return merged_output

    direct_text = data.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    return ""
