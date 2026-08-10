"""
modules/stage1_sampler.py — Stage 1 第一阶段：能力边界感知

K-trial 随机采样 → 成功率(SR)计算 → 空间拓扑划分 → level 划分

早停策略: 连续 5 次正确 → 提前终止, SR=1.0, 归入 Master Zone

Zone 分类:
  - Master Zone  (Z_M): SR = 1.0   — 已征服,无需进化
  - Learning Zone (Z_L): 0 < SR < 1 — 当前进化领土
  - Panic Zone   (Z_P): SR = 0     — 暂时无法解决

Learning Zone 按 correct_count 排序等分为 4 个 level.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm.asyncio import tqdm as async_tqdm

from core.config import Config
from core.api_call import async_llm_call, async_llm_call_with_metadata
from modules.execute_module import URLRouter
from template.prompt_template import DIRECT_ANSWER_PROMPT

logger = logging.getLogger(__name__)

# 早停阈值：连续 N 次正确即提前终止
EARLY_STOP_CONSECUTIVE = 5


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %.3f", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using %d", name, raw, default)
        return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ════════════════════════════════════════════════════════════════
# 答案提取与校验
# ════════════════════════════════════════════════════════════════

def _sanitize_text(text: object) -> str:
    """移除控制字符,保留可打印内容。"""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    return cleaned.strip()


def extract_boxed(text: str) -> Optional[str]:
    """安全提取 \\boxed{...},支持嵌套大括号与双反斜杠。"""
    if not text:
        return None
    for pattern in (r"\boxed{", "\\boxed{"):
        idx = text.find(pattern)
        if idx == -1:
            continue
        start = idx + len(pattern)
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    raw = text[start:i].strip().strip("$")
                    return raw.strip()
        return None
    return None


def normalize_answer(text: str) -> str:
    """规范化答案文本用于比对。"""
    cleaned = _sanitize_text(text).lower()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\displaystyle", "")
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\,", "").replace("\\ ", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"^finalanswer:?", "", cleaned)
    cleaned = re.sub(r"^answer:?", "", cleaned)
    cleaned = cleaned.strip(".,;:!?\"'")
    cleaned = cleaned.replace("\\", "")
    return cleaned


def check_answer_correct(generated: str, expected: str) -> Tuple[bool, Optional[str]]:
    """校验答案是否正确。"""
    if not generated or not expected:
        return False, None
    expected_norm = normalize_answer(expected)
    if not expected_norm:
        return False, None

    boxed = extract_boxed(generated)
    if boxed:
        boxed_norm = normalize_answer(boxed)
        if boxed_norm == expected_norm:
            return True, boxed
        if expected_norm in boxed_norm or boxed_norm in expected_norm:
            return True, boxed

    fa_match = re.search(r"Final\s*Answer:\s*(.+?)(?:\n|$)", generated, re.IGNORECASE)
    if fa_match:
        fa_text = fa_match.group(1).strip()
        fa_norm = normalize_answer(fa_text)
        if fa_norm == expected_norm or expected_norm in fa_norm or fa_norm in expected_norm:
            return True, fa_text

    gen_norm = normalize_answer(generated)
    if gen_norm and expected_norm:
        if expected_norm in gen_norm or gen_norm in expected_norm:
            return True, None

    return False, boxed


# ════════════════════════════════════════════════════════════════
# Stage 1 采样器
# ════════════════════════════════════════════════════════════════

class Stage1Sampler:
    """K-trial 采样, 三区划分, 等分四级."""

    def __init__(
        self,
        urls: List[str],
        model: str,
        api_key: str = "EMPTY",
        k_trials: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 600,
        use_entropy_signal: Optional[bool] = None,
        capability_lambda: Optional[float] = None,
        capability_alpha: Optional[float] = None,
        capability_beta: Optional[float] = None,
        request_logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        disable_early_stop_for_entropy: Optional[bool] = None,
    ):
        self.urls = urls
        self.model = model
        self.api_key = api_key
        self.k_trials = k_trials
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.router = URLRouter(urls)
        self.use_entropy_signal = (
            _env_bool("MEMUA_STAGE1_USE_ENTROPY", True)
            if use_entropy_signal is None else bool(use_entropy_signal)
        )
        self.capability_lambda = _clip01(
            _env_float("MEMUA_CAPABILITY_LAMBDA", 0.7)
            if capability_lambda is None else capability_lambda
        )
        self.capability_alpha = _clip01(
            _env_float("MEMUA_CAPABILITY_ALPHA", 0.25)
            if capability_alpha is None else capability_alpha
        )
        self.capability_beta = _clip01(
            _env_float("MEMUA_CAPABILITY_BETA", 0.75)
            if capability_beta is None else capability_beta
        )
        self.request_logprobs = (
            _env_bool("MEMUA_STAGE1_LOGPROBS", True)
            if request_logprobs is None else bool(request_logprobs)
        )
        self.top_logprobs = max(
            1,
            _env_int("MEMUA_STAGE1_TOP_LOGPROBS", 5)
            if top_logprobs is None else int(top_logprobs),
        )
        self.disable_early_stop_for_entropy = (
            _env_bool("MEMUA_STAGE1_DISABLE_EARLY_STOP_FOR_ENTROPY", True)
            if disable_early_stop_for_entropy is None else bool(disable_early_stop_for_entropy)
        )

    async def _single_inference(self, question: str) -> str:
        prompt = DIRECT_ANSWER_PROMPT.format(question=question)
        for _ in range(max(1, len(self.urls))):
            url = self.router.next()
            try:
                response = await async_llm_call(
                    prompt=prompt, model=self.model, url=url,
                    api_key=self.api_key, temperature=self.temperature,
                    max_tokens=self.max_tokens, timeout=self.timeout,
                )
                if response:
                    self.router.report_success(url)
                    return response
                self.router.report_failure(url, "empty response")
            except Exception as e:
                self.router.report_failure(url, str(e))
        return ""

    async def _single_inference_with_metadata(self, question: str) -> Dict[str, Any]:
        prompt = DIRECT_ANSWER_PROMPT.format(question=question)
        for _ in range(max(1, len(self.urls))):
            url = self.router.next()
            try:
                response = await async_llm_call_with_metadata(
                    prompt=prompt,
                    model=self.model,
                    url=url,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    request_logprobs=self.request_logprobs,
                    top_logprobs=self.top_logprobs,
                    include_raw=False,
                )
                if response.get("text"):
                    self.router.report_success(url)
                    return response
                self.router.report_failure(url, "empty response")
            except Exception as e:
                self.router.report_failure(url, str(e))
        return {
            "text": "",
            "entropy": {
                "available": False,
                "normalized_entropy": 1.0,
                "trajectory_entropy": 1.0,
                "token_count": 0,
                "entropy_token_count": 0,
                "avg_neg_logprob": None,
                "method": "request_failed_default_max_entropy",
            },
        }

    @staticmethod
    def sr_zone(correct_count: int, actual_trials: int, early_stopped: bool = False) -> str:
        if actual_trials <= 0:
            return "no_data"
        if early_stopped or correct_count == actual_trials:
            return "master"
        if correct_count == 0:
            return "panic"
        return "learning"

    @staticmethod
    def capability_zone(score: float, alpha: float = 0.25, beta: float = 0.75) -> str:
        if score < alpha:
            return "panic"
        if score > beta:
            return "master"
        return "learning"

    @staticmethod
    def level_from_score(score: float) -> int:
        if score >= 0.75:
            return 1
        if score >= 0.50:
            return 2
        if score >= 0.25:
            return 3
        if score > 0:
            return 4
        return 0

    @classmethod
    def level_from_result(cls, result: Dict[str, Any]) -> int:
        score = result.get("capability_score", result.get("sr", 0.0))
        try:
            return cls.level_from_score(float(score))
        except (TypeError, ValueError):
            return 0

    def finalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        actual_trials = int(result.get("actual_trials") or 0)
        correct_count = int(result.get("correct_count") or 0)
        planned_trials = int(result.get("planned_trials") or self.k_trials or actual_trials or 0)
        denominator = planned_trials if planned_trials > 0 else actual_trials

        sr_actual = correct_count / actual_trials if actual_trials > 0 else 0.0
        sr_fixed_k = correct_count / denominator if denominator > 0 else 0.0
        result["success_rate_actual"] = _clip01(sr_actual)
        result["success_rate_fixed_k"] = _clip01(sr_fixed_k)
        result["sr"] = _clip01(sr_fixed_k if self.use_entropy_signal else sr_actual)

        records = result.get("trial_records") or []
        entropies = []
        available_count = 0
        for record in records:
            entropy_value = record.get("normalized_entropy", record.get("trajectory_entropy", 1.0))
            try:
                entropies.append(_clip01(float(entropy_value)))
            except (TypeError, ValueError):
                entropies.append(1.0)
            if record.get("entropy_available"):
                available_count += 1

        avg_entropy = sum(entropies) / len(entropies) if entropies else 1.0
        confidence_score = 1.0 - avg_entropy
        capability_score = (
            self.capability_lambda * result["success_rate_fixed_k"]
            + (1.0 - self.capability_lambda) * confidence_score
        )

        result["avg_entropy"] = _clip01(avg_entropy)
        result["normalized_entropy"] = result["avg_entropy"]
        result["entropy_available_trials"] = available_count
        result["confidence_score"] = _clip01(confidence_score)
        result["capability_score"] = _clip01(capability_score)
        result["sr_zone"] = self.sr_zone(
            correct_count,
            actual_trials,
            bool(result.get("early_stopped")),
        )
        result["capability_zone"] = self.capability_zone(
            result["capability_score"],
            self.capability_alpha,
            self.capability_beta,
        )
        result["zone"] = result["capability_zone"] if self.use_entropy_signal else result["sr_zone"]
        result["assigned_level"] = self.level_from_result(result)
        return result

    async def sample_one_task(self, task: Dict[str, Any], task_idx: int) -> Dict[str, Any]:
        """对单个任务执行 K-trial 采样（含早停）。

        Returns 包含 zone / correct_count / sr / 全部轨迹.
        """
        question = task.get("question", "")
        expected = task.get("answer", "")

        result = {
            "idx": task_idx,
            "question": question,
            "expected_answer": expected,
            "orig_level": task.get("difficulty_level", ""),
            "all_answers": [],
            "correct_trajectories": [],
            "wrong_trajectories": [],
            "correct_count": 0,
            "wrong_count": 0,
            "sr": 0.0,
            "zone": "panic",
            "trial_records": [],
            "planned_trials": self.k_trials,
            "early_stopped": False,
            "actual_trials": 0,
        }

        if not question or not expected:
            result["zone"] = "no_data"
            result["sr_zone"] = "no_data"
            result["capability_zone"] = "no_data"
            return result

        consecutive_correct = 0

        for trial in range(self.k_trials):
            entropy = {
                "available": False,
                "normalized_entropy": 1.0,
                "trajectory_entropy": 1.0,
                "token_count": 0,
                "entropy_token_count": 0,
                "avg_neg_logprob": None,
                "method": "disabled_default_max_entropy",
            }
            if self.use_entropy_signal:
                metadata = await self._single_inference_with_metadata(question)
                response = metadata.get("text", "")
                entropy = metadata.get("entropy") or entropy
            else:
                response = await self._single_inference(question)
            if not response:
                response = "[EMPTY_RESPONSE]"

            result["actual_trials"] = trial + 1
            is_correct, extracted = check_answer_correct(response, expected)
            result["all_answers"].append(response)
            result["trial_records"].append({
                "trial": trial + 1,
                "response": response,
                "is_correct": bool(is_correct),
                "extracted_answer": extracted,
                "normalized_entropy": entropy.get("normalized_entropy", 1.0),
                "trajectory_entropy": entropy.get("trajectory_entropy", entropy.get("normalized_entropy", 1.0)),
                "entropy_available": bool(entropy.get("available")),
                "entropy_method": entropy.get("method", "unknown"),
                "token_count": entropy.get("token_count", 0),
                "entropy_token_count": entropy.get("entropy_token_count", 0),
                "avg_neg_logprob": entropy.get("avg_neg_logprob"),
            })

            if is_correct:
                result["correct_trajectories"].append(response)
                result["correct_count"] += 1
                consecutive_correct += 1
            else:
                result["wrong_trajectories"].append(response)
                result["wrong_count"] += 1
                consecutive_correct = 0

            # 早停: 连续 5 次正确 → 认定为 Master
            early_stop_enabled = not (self.use_entropy_signal and self.disable_early_stop_for_entropy)
            if early_stop_enabled and consecutive_correct >= EARLY_STOP_CONSECUTIVE:
                result["early_stopped"] = True
                break

        return self.finalize_result(result)

    @staticmethod
    def _load_checkpoint(checkpoint_path: Optional[str]) -> List[Dict[str, Any]]:
        if not checkpoint_path:
            return []
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            return []
        try:
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            return data.get("results", [])
        except Exception:
            return []

    async def sample_all(
        self, tasks: Sequence[Dict[str, Any]], concurrency: int = 10,
        checkpoint_path: Optional[str] = None,
        collect_all_samples: bool = False,
    ) -> Dict[str, Any]:
        """全量采样并划分三区 + 等分四级.

        Args:
            tasks: 待采样任务列表
            concurrency: 并发数
            checkpoint_path: 增量 checkpoint 路径
            collect_all_samples: True = 收集所有 zone (No Learning Zone 消融).
                False = 只收集 learning-zone (默认).
        """
        semaphore = asyncio.Semaphore(concurrency)

        # ── 断点续跑：加载已完成 ──
        cached_results = self._load_checkpoint(checkpoint_path)
        completed_ids: set = {r.get("idx") for r in cached_results if r.get("idx") is not None}
        pending = [(i, t) for i, t in enumerate(tasks) if i not in completed_ids]

        if completed_ids:
            logger.info("Checkpoint: %d 已完成, %d 待处理", len(completed_ids), len(pending))

        async def sample_one(task: Dict, idx: int) -> Dict:
            async with semaphore:
                return await self.sample_one_task(task, idx)

        logger.info(
            "Stage 1.1: K-trial 采样 (K=%d, temp=%.1f, early_stop=%d, concurrency=%d, total=%d, skip=%d)",
            self.k_trials, self.temperature, EARLY_STOP_CONSECUTIVE, concurrency,
            len(tasks), len(completed_ids),
        )

        # 新跑的结果（内存中），与 cached_results 合并写入 checkpoint
        new_results: List[Dict] = []
        lock = asyncio.Lock()

        async def run_and_checkpoint(idx: int, task: Dict) -> Dict:
            r = await sample_one(task, idx)
            async with lock:
                new_results.append(r)
                if checkpoint_path:
                    all_r = sorted(cached_results + new_results, key=lambda x: x.get("idx", -1))
                    ckpt = Path(checkpoint_path)
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    tmp = ckpt.with_suffix(".tmp")
                    payload = {"checkpoint": True, "results": all_r}
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    tmp.replace(ckpt)
            return r

        coros = [run_and_checkpoint(i, t) for i, t in pending]
        for coro in async_tqdm(asyncio.as_completed(coros), total=len(coros), desc="K-trial Sampling"):
            await coro

        # 合并所有结果
        results = sorted(cached_results + new_results, key=lambda r: r.get("idx", -1))

        # ── 分区统计 ──
        master = [r for r in results if r.get("zone") == "master"]
        learning = [r for r in results if r.get("zone") == "learning"]
        panic = [r for r in results if r.get("zone") == "panic"]
        no_data = [r for r in results if r.get("zone") == "no_data"]
        early_stopped = [r for r in results if r.get("early_stopped")]

        # ── 按论文 SR 阈值分四级 ──
        # Level 1: SR >= 0.75  (近掌握, 偶尔失误可诊断)
        # Level 2: 0.50 <= SR < 0.75
        # Level 3: 0.25 <= SR < 0.50
        # Level 4: SR < 0.25   (近恐慌, 信噪比低)
        level_data: Dict[str, List[Dict]] = {"1": [], "2": [], "3": [], "4": []}

        source = [r for r in results if r.get("zone") != "no_data"] if collect_all_samples else learning
        if collect_all_samples:
            logger.info("No Learning Zone ablation: collecting %d samples (all zones except no_data)", len(source))

        for r in source:
            level = self.level_from_result(r)
            if level <= 0:
                continue
            r["assigned_level"] = level
            level_data[str(level)].append(r)

        for lv in ("1", "2", "3", "4"):
            items = level_data[lv]
            if items:
                logger.info(
                    "  Level %s: %d samples, correct_count [%d, %d]",
                    lv, len(items),
                    min(r["correct_count"] for r in items),
                    max(r["correct_count"] for r in items),
                )

        summary = {
            "total": len(results),
            "master_zone": len(master),
            "learning_zone": len(learning),
            "panic_zone": len(panic),
            "no_data": len(no_data),
            "early_stopped": len(early_stopped),
            "k_trials": self.k_trials,
            "temperature": self.temperature,
            "capability_signal_enabled": self.use_entropy_signal,
            "capability_lambda": self.capability_lambda,
            "capability_alpha": self.capability_alpha,
            "capability_beta": self.capability_beta,
            "entropy_available_trials": sum(int(r.get("entropy_available_trials") or 0) for r in results),
            "sr_zone_counts": {
                "master": sum(1 for r in results if r.get("sr_zone") == "master"),
                "learning": sum(1 for r in results if r.get("sr_zone") == "learning"),
                "panic": sum(1 for r in results if r.get("sr_zone") == "panic"),
                "no_data": sum(1 for r in results if r.get("sr_zone") == "no_data"),
            },
            "capability_zone_counts": {
                "master": sum(1 for r in results if r.get("capability_zone") == "master"),
                "learning": sum(1 for r in results if r.get("capability_zone") == "learning"),
                "panic": sum(1 for r in results if r.get("capability_zone") == "panic"),
                "no_data": sum(1 for r in results if r.get("capability_zone") == "no_data"),
            },
            "levels": {lv: len(level_data[lv]) for lv in ("1", "2", "3", "4")},
        }

        logger.info(
            "Stage 1.1 完成: total=%d | master=%d | learning=%d | panic=%d | early_stop=%d",
            summary["total"], summary["master_zone"], summary["learning_zone"],
            summary["panic_zone"], summary["early_stopped"],
        )
        logger.info(
            "  Level 分配: L1=%d L2=%d L3=%d L4=%d",
            summary["levels"]["1"], summary["levels"]["2"],
            summary["levels"]["3"], summary["levels"]["4"],
        )

        return {
            "config": {
                "k_trials": self.k_trials,
                "temperature": self.temperature,
                "early_stop_consecutive": EARLY_STOP_CONSECUTIVE,
                "use_entropy_signal": self.use_entropy_signal,
                "request_logprobs": self.request_logprobs,
                "top_logprobs": self.top_logprobs,
                "disable_early_stop_for_entropy": self.disable_early_stop_for_entropy,
                "capability_formula": "S = lambda * F + (1 - lambda) * (1 - H)",
                "capability_lambda": self.capability_lambda,
                "capability_alpha": self.capability_alpha,
                "capability_beta": self.capability_beta,
            },
            "results": results,
            "summary": summary,
            "level_data": level_data,
        }

    @staticmethod
    def save(data: Dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("采样结果已保存: %s", output_path)
        return output_path
