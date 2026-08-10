"""
modules/execute_module.py — ExecuteModule 类

Agent 的「执行」模块，负责：
  - 不稳定样本采样 (多卡 Round-Robin 负载均衡)
  - 困难负样本挖掘
  - 直接答题 / 经验引导答题
  - 答案校验
"""
import asyncio
import hashlib
import logging
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from core.api_call import async_llm_call, async_llm_call_with_metadata
from core.config import Config
from template.prompt_template import (
    DIRECT_ANSWER_PROMPT,
    GUIDED_ANSWER_PROMPT,
    SOLVER_V1_FINALIZER_PROMPT,
    SOLVER_V1_GUIDED_REASONER_PROMPT,
    SOLVER_V1_REASONER_PROMPT,
    WEAK_BASELINE_TEST_PROMPT,
)
from benchmarks.matcher import check_answer, extract_final_answer

logger = logging.getLogger(__name__)

SOLVER_V1_PROTOCOL = "solver_v1.1"
SOLVER_V1_REASONER_MAX_TOKENS = 8192
SOLVER_V1_FINALIZER_MAX_TOKENS = 128
SOLVER_V1_REASONER_TEMPERATURE = 0.0
SOLVER_V1_FINALIZER_TEMPERATURE = 0.0
SOLVER_V1_TOP_P = 0.95
SOLVER_V1_REASONER_REPETITION_PENALTY = 1.0


def _prompt_hash(template: str) -> str:
    return hashlib.sha256(template.strip().encode("utf-8")).hexdigest()


SOLVER_V1_PROMPT_HASHES = {
    "reasoner_base": _prompt_hash(SOLVER_V1_REASONER_PROMPT),
    "reasoner_guided": _prompt_hash(SOLVER_V1_GUIDED_REASONER_PROMPT),
    "finalizer": _prompt_hash(SOLVER_V1_FINALIZER_PROMPT),
}


def _finish_reason(metadata: Dict[str, Any]) -> str:
    raw = metadata.get("raw") or {}
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return str(choices[0].get("finish_reason") or "")
    return ""


def _completion_tokens(metadata: Dict[str, Any]) -> Optional[int]:
    usage = metadata.get("usage") or {}
    value = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return int(value) if isinstance(value, (int, float)) else None


def _repetition_diagnostics(text: str) -> Dict[str, Any]:
    """Cheap, model-independent signals for mechanical output loops."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    words = normalized.split()
    windows = [" ".join(words[i:i + 8]) for i in range(max(0, len(words) - 7))]
    repeated_windows = len(windows) - len(set(windows))
    repeat_ratio = repeated_windows / len(windows) if windows else 0.0
    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in (text or "").splitlines() if line.strip()]
    repeated_lines = len(lines) - len(set(lines))
    consecutive_pattern_words = 0
    consecutive_repeat_count = 0
    consecutive_pattern_preview = ""
    # A contiguous block repeated at least three times is much stronger
    # evidence of a mechanical loop than a global repeated-token ratio.
    for pattern_size in range(4, min(64, len(words) // 3) + 1):
        for start in range(0, len(words) - 3 * pattern_size + 1):
            pattern = words[start:start + pattern_size]
            count = 1
            cursor = start + pattern_size
            while cursor + pattern_size <= len(words) and words[cursor:cursor + pattern_size] == pattern:
                count += 1
                cursor += pattern_size
            if count >= 3 and pattern_size * count > consecutive_pattern_words * consecutive_repeat_count:
                consecutive_pattern_words = pattern_size
                consecutive_repeat_count = count
                consecutive_pattern_preview = " ".join(pattern)[:240]
    return {
        "word_count": len(words),
        "repeated_8gram_count": repeated_windows,
        "repeated_8gram_ratio": round(repeat_ratio, 6),
        "repeated_line_count": repeated_lines,
        "likely_loop": bool(repeat_ratio >= 0.08 or repeated_lines >= 3),
        "consecutive_pattern_words": consecutive_pattern_words,
        "consecutive_repeat_count": consecutive_repeat_count,
        "consecutive_pattern_preview": consecutive_pattern_preview,
        "hard_loop": bool(consecutive_repeat_count >= 3),
    }


def _reasoner_format_valid(text: str) -> bool:
    value = text or ""
    positions = [value.find("Method:"), value.find("Reasoning:"), value.find("Check:")]
    return all(pos >= 0 for pos in positions) and positions == sorted(positions)


def _finalizer_format_valid(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return len(lines) == 1 and bool(re.fullmatch(r"Final Answer:\s*\\boxed\{.+\}", lines[0]))


def _extract_host_port(url: str) -> tuple[Optional[str], Optional[int]]:
    normalized = str(url or "").strip()
    if not normalized:
        return None, None
    parsed = urlparse(normalized)
    return parsed.hostname, parsed.port


def is_url_reachable(url: str, timeout: float = 1.5) -> bool:
    """Best-effort TCP probe used to remove obviously dead endpoints before routing."""
    host, port = _extract_host_port(url)
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def filter_healthy_urls(urls: Sequence[str], timeout: float = 1.5) -> List[str]:
    healthy = [url for url in urls if is_url_reachable(url, timeout=timeout)]
    if healthy:
        dropped = [url for url in urls if url not in healthy]
        if dropped:
            logger.warning("Skipping unreachable model endpoints at startup: %s", dropped)
        return healthy
    return list(urls)


# ── 答案校验工具 ──────────────────────────────────────────────


def normalize_answer(text: str) -> str:
    """规范化答案文本，用于比对。"""
    cleaned = (text or "").strip().lower()
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\displaystyle", "")
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\,", "").replace(" ", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(r"\(", "").replace(r"\)", "")
    cleaned = cleaned.replace(r"\[", "").replace(r"\]", "")
    cleaned = re.sub(r"^finalanswer:?", "", cleaned)
    cleaned = cleaned.strip(".,;:!?")
    return cleaned.replace("\\", "")


def check_answer_match(generated: str, expected: str) -> bool:
    """校验模型答案是否正确（统一使用 benchmark 严格 matcher）。"""
    return check_answer(generated, expected)


# ── URL 轮询器 ───────────────────────────────────────────────


class URLRouter:
    """Round-Robin URL 路由器，支持多卡负载均衡。"""

    def __init__(self, urls: List[str]):
        self.urls = urls
        self._counter = 0
        self._cooldown_until: Dict[str, float] = {}
        self._failure_counts: Dict[str, int] = {url: 0 for url in urls}
        self.cooldown_seconds = max(30, int(os.environ.get("MEMUA_URL_COOLDOWN_SECONDS", "180")))

    def next(self) -> str:
        if not self.urls:
            raise ValueError("No model endpoints configured.")
        total = len(self.urls)
        now = time.time()
        for _ in range(total):
            url = self.urls[self._counter % total]
            self._counter += 1
            if self._cooldown_until.get(url, 0) <= now:
                return url
        url = self.urls[self._counter % total]
        self._counter += 1
        return url

    def report_success(self, url: str) -> None:
        self._failure_counts[url] = 0
        self._cooldown_until.pop(url, None)

    def report_failure(self, url: str, reason: str = "") -> None:
        failures = self._failure_counts.get(url, 0) + 1
        self._failure_counts[url] = failures
        self._cooldown_until[url] = time.time() + self.cooldown_seconds
        logger.warning(
            "Endpoint temporarily quarantined for %ss after failure #%d: %s%s",
            self.cooldown_seconds,
            failures,
            url,
            f" | {reason}" if reason else "",
        )

    def attempt_budget(self) -> int:
        """Maximum total endpoint attempts for one logical request."""
        configured = max(1, int(os.environ.get("MEMUA_MAX_ENDPOINT_ATTEMPTS", "2")))
        return min(len(self.urls), configured) if self.urls else 0

    @staticmethod
    def build_urls(base_url: str, ports: Optional[Sequence[int]] = None) -> List[str]:
        """从基础 URL + 端口列表构建多个 API endpoint。"""
        if not ports:
            return [base_url]
        urls = []
        for port in ports:
            if re.search(r":\d+(?=/|$)", base_url):
                urls.append(re.sub(r":\d+(?=/|$)", f":{port}", base_url))
            else:
                urls.append(re.sub(r"^(https?://[^/]+)", rf"\1:{port}", base_url))
        return urls


# ── ExecuteModule 类 ──────────────────────────────────────────


class ExecuteModule:
    """Agent 的执行模块。

    使用方式:
        exe = ExecuteModule(config, ports=[8000, 8001, ...])

        # Stage 1: 不稳定样本采样
        sampling_path = await exe.sample_unstable("dpo_level1", concurrency=10)

        # Stage 1: 困难负样本挖掘
        mining_path = exe.mine_hard_negatives("dpo_level1")

        # Stage 3: 答题
        answer = await exe.answer_direct(question)
        answer = await exe.answer_guided(question, principle_text)
        is_correct = exe.check_answer(answer, expected)
    """

    def __init__(
        self,
        config: Config,
        ports: Optional[Sequence[int]] = None,
        urls: Optional[Sequence[str]] = None,
        model_override: Optional[str] = None,
        seed: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.config = config
        hp = config.hyperparams
        self.model = model_override or hp.weak_model_name
        self.api_key = hp.weak_model_key or "EMPTY"
        self.timeout = int(timeout) if timeout is not None else hp.weak_timeout
        self.seed = seed
        base_url = hp.weak_model_url
        initial_urls = list(urls) if urls else URLRouter.build_urls(base_url, ports)
        self.urls = filter_healthy_urls(initial_urls)
        self.router = URLRouter(self.urls)

    async def _call_with_router(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
        url_override: Optional[str] = None,
    ) -> str:
        attempts = 1 if url_override else max(1, self.router.attempt_budget())
        last_response = ""
        for _ in range(attempts):
            url = url_override or self.router.next()
            request_seed = self.seed if seed is None else seed
            response = await async_llm_call(
                prompt=prompt,
                model=self.model,
                url=url,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                retries=1,
                **({"seed": request_seed} if request_seed is not None else {}),
            )
            if response:
                self.router.report_success(url)
                return response
            logger.warning("Solver request returned empty; switching endpoint without quarantine: %s", url)
            last_response = response
        return last_response

    async def _call_with_router_metadata(
        self,
        *,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        url_override: Optional[str] = None,
        timeout_override: Optional[int] = None,
    ) -> tuple[Dict[str, Any], str]:
        """Call one endpoint and retain auditable generation metadata."""
        attempts = 1 if url_override else max(1, self.router.attempt_budget())
        last: Dict[str, Any] = {"text": "", "usage": None, "raw": {}}
        last_url = ""
        for _ in range(attempts):
            url = url_override or self.router.next()
            last_url = url
            request_seed = self.seed if seed is None else seed
            result = await async_llm_call_with_metadata(
                prompt=prompt,
                model=self.model,
                url=url,
                api_key=self.api_key,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                timeout=int(timeout_override or self.timeout),
                retries=1,
                request_logprobs=False,
                include_raw=True,
                **({"seed": request_seed} if request_seed is not None else {}),
            )
            if result.get("text"):
                self.router.report_success(url)
                return result, url
            # Explicit endpoint calls are controlled by the pair/rollout
            # transaction.  Do not quarantine a healthy but slow GPU here.
            if not url_override:
                logger.warning("Solver request returned empty; switching endpoint without quarantine: %s", url)
            last = result
        return last, last_url

    async def solve_v1(
        self,
        question: str,
        principle_text: str = "",
        *,
        seed: Optional[int] = None,
        url_override: Optional[str] = None,
        reasoner_max_tokens: int = SOLVER_V1_REASONER_MAX_TOKENS,
        finalizer_max_tokens: int = SOLVER_V1_FINALIZER_MAX_TOKENS,
        reasoner_repetition_penalty: float = SOLVER_V1_REASONER_REPETITION_PENALTY,
        reasoner_temperature: float = SOLVER_V1_REASONER_TEMPERATURE,
        reasoner_timeout: int = 300,
        finalizer_timeout: int = 60,
    ) -> Dict[str, Any]:
        """Run the frozen two-call Solver-v1 protocol without answer retries."""
        guided = bool((principle_text or "").strip())
        if guided:
            reasoner_prompt = SOLVER_V1_GUIDED_REASONER_PROMPT.format(
                question=question,
                principle_text=principle_text.strip(),
            )
            reasoner_hash_key = "reasoner_guided"
        else:
            reasoner_prompt = SOLVER_V1_REASONER_PROMPT.format(question=question)
            reasoner_hash_key = "reasoner_base"

        reasoner_meta, solver_url = await self._call_with_router_metadata(
            prompt=reasoner_prompt,
            temperature=reasoner_temperature,
            top_p=SOLVER_V1_TOP_P,
            max_tokens=reasoner_max_tokens,
            repetition_penalty=reasoner_repetition_penalty,
            seed=seed,
            url_override=url_override,
            timeout_override=reasoner_timeout,
        )
        reasoner_output = str(reasoner_meta.get("text") or "")
        if not reasoner_output:
            return {
                "protocol_version": SOLVER_V1_PROTOCOL, "guided": guided,
                "seed": self.seed if seed is None else seed, "solver_url": solver_url,
                "finalizer_url": "", "infrastructure_success": False,
                "infrastructure_stage": "reasoner", "reasoner": {"output": ""},
                "finalizer": {"output": ""}, "final_answer": "", "composed_output": "",
            }
        finalizer_prompt = SOLVER_V1_FINALIZER_PROMPT.format(
            question=question,
            reasoner_output=reasoner_output,
        )
        finalizer_meta, finalizer_url = await self._call_with_router_metadata(
            prompt=finalizer_prompt,
            temperature=SOLVER_V1_FINALIZER_TEMPERATURE,
            top_p=SOLVER_V1_TOP_P,
            max_tokens=finalizer_max_tokens,
            repetition_penalty=1.0,
            seed=seed,
            url_override=solver_url or url_override,
            timeout_override=finalizer_timeout,
        )
        finalizer_output = str(finalizer_meta.get("text") or "").strip()

        reasoner_tokens = _completion_tokens(reasoner_meta)
        finalizer_tokens = _completion_tokens(finalizer_meta)
        reasoner_finish = _finish_reason(reasoner_meta)
        finalizer_finish = _finish_reason(finalizer_meta)
        repetition = _repetition_diagnostics(reasoner_output)
        composed = reasoner_output.rstrip()
        if finalizer_output:
            composed = f"{composed}\n\n{finalizer_output}" if composed else finalizer_output

        return {
            "protocol_version": SOLVER_V1_PROTOCOL,
            "guided": guided,
            "seed": self.seed if seed is None else seed,
            "solver_url": solver_url,
            "finalizer_url": finalizer_url,
            "prompt_hashes": {
                "reasoner": SOLVER_V1_PROMPT_HASHES[reasoner_hash_key],
                "finalizer": SOLVER_V1_PROMPT_HASHES["finalizer"],
            },
            "reasoner_repetition_penalty": reasoner_repetition_penalty,
            "reasoner_temperature": reasoner_temperature,
            "reasoner": {
                "output": reasoner_output,
                "max_tokens": reasoner_max_tokens,
                "repetition_penalty": reasoner_repetition_penalty,
                "temperature": reasoner_temperature,
                "completion_tokens": reasoner_tokens,
                "finish_reason": reasoner_finish,
                "hit_token_limit": bool(
                    reasoner_finish == "length"
                    or (reasoner_tokens is not None and reasoner_tokens >= reasoner_max_tokens)
                ),
                "format_valid": _reasoner_format_valid(reasoner_output),
                "repetition": repetition,
                "duration_sec": reasoner_meta.get("duration_sec"),
                "usage": reasoner_meta.get("usage"),
            },
            "finalizer": {
                "output": finalizer_output,
                "max_tokens": finalizer_max_tokens,
                "repetition_penalty": 1.0,
                "temperature": SOLVER_V1_FINALIZER_TEMPERATURE,
                "completion_tokens": finalizer_tokens,
                "finish_reason": finalizer_finish,
                "hit_token_limit": bool(
                    finalizer_finish == "length"
                    or (finalizer_tokens is not None and finalizer_tokens >= finalizer_max_tokens)
                ),
                "format_valid": _finalizer_format_valid(finalizer_output),
                "duration_sec": finalizer_meta.get("duration_sec"),
                "usage": finalizer_meta.get("usage"),
            },
            "final_answer": extract_final_answer(finalizer_output, allow_short_direct=False),
            "composed_output": composed,
            "infrastructure_success": bool(finalizer_output),
            "infrastructure_stage": "" if finalizer_output else "finalizer",
        }

    # ================================================================
    # Stage 1: 不稳定样本采样
    # ================================================================

    async def sample_unstable(
        self,
        dataset: str,
        concurrency: int = 10,
        batch_size: int = 5,
        max_trials: int = 15,
        temperature: float = 0.7,
        collect_all_samples: bool = False,
    ) -> Optional[Path]:
        """运行多次推理采样，识别 learning-zone 样本。

        Returns: 输出文件路径 (sampling_results.json)，失败返回 None
        """
        from sample_unstable_questions import ModelTester
        from output_paths import stage1_sampling_file

        tester = ModelTester(self.config, dataset, urls=self.urls)
        tester.model = self.model
        tester.temperature = temperature
        tester.batch_size = batch_size
        tester.max_trials = max_trials
        tester.collect_all_samples = collect_all_samples

        logger.info(
            "Stage 1.1: 不稳定样本采样 (dataset=%s, concurrency=%d, collect_all_samples=%s)",
            dataset, concurrency, collect_all_samples,
        )
        summary = await tester.run_test(concurrency=concurrency)

        if "error" in summary:
            logger.error("采样失败: %s", summary["error"])
            return None

        output_path = stage1_sampling_file(self.config.paths.output_dir, dataset)
        logger.info(
            "采样完成: total=%d, learning_zone=%d -> %s",
            summary.get("total", 0), summary.get("usable_samples", 0), output_path,
        )
        return output_path

    # ================================================================
    # Stage 1: 困难负样本挖掘
    # ================================================================

    def mine_hard_negatives(
        self,
        dataset: str,
        sampling_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """从采样结果中挖掘 hard negative pairs。

        Returns: 输出文件路径，失败返回 None
        """
        from hard_negative_mining import HardNegativeMiner
        from output_paths import stage1_sampling_file, stage1_mining_file

        if sampling_path is None:
            sampling_path = stage1_sampling_file(self.config.paths.output_dir, dataset)

        mining_path = stage1_mining_file(self.config.paths.output_dir, dataset)

        if not Path(sampling_path).exists():
            logger.error("采样结果不存在: %s", sampling_path)
            return None

        logger.info("Stage 1.2: 困难负样本挖掘 (dataset=%s)", dataset)
        miner = HardNegativeMiner(self.config, dataset)
        stats = miner.run_from_file(str(sampling_path), str(mining_path))

        if "error" in stats:
            logger.error("挖掘失败: %s", stats["error"])
            return None

        logger.info("挖掘完成: %s", stats)
        return mining_path

    # ================================================================
    # Stage 3: 答题
    # ================================================================

    async def answer_direct(
        self,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        seed: Optional[int] = None,
        url_override: Optional[str] = None,
    ) -> str:
        """直接回答问题 (无经验引导)。"""
        prompt = DIRECT_ANSWER_PROMPT.format(question=question)
        return await self._call_with_router(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            url_override=url_override,
        )

    async def answer_guided(
        self,
        question: str,
        principle_text: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        seed: Optional[int] = None,
        url_override: Optional[str] = None,
    ) -> str:
        """使用经验引导回答问题。"""
        prompt = GUIDED_ANSWER_PROMPT.format(
            principle_text=principle_text,
            question=question,
        )
        return await self._call_with_router(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            url_override=url_override,
        )

    async def answer_baseline(
        self,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        seed: Optional[int] = None,
        url_override: Optional[str] = None,
    ) -> str:
        """Baseline 答题 (与实验对照组一致的 prompt)。"""
        prompt = WEAK_BASELINE_TEST_PROMPT.format(question=question)
        return await self._call_with_router(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            url_override=url_override,
        )

    def check_answer(self, generated: str, expected: str) -> bool:
        """校验答案是否正确。"""
        return check_answer_match(generated, expected)


# ── 向后兼容的函数式接口 (供 sample_unstable_questions 等脚本使用) ──


async def async_answer_question_directly(
    question: str,
    model: str,
    url: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    """异步直接回答问题 (函数式接口, 向后兼容)。"""
    prompt = DIRECT_ANSWER_PROMPT.format(question=question)
    return await async_llm_call(
        prompt=prompt,
        model=model,
        url=url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
