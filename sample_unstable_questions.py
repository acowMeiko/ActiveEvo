#!/usr/bin/env python
"""
训练题不稳定样本采样脚本 (8卡极限集群榨干版)
- 支持多节点 Round-Robin 负载均衡 (--ports)
- 内置防截断自愈机制 (Auto-Healing)
- Early Stopping: 有对有错时提前终止
"""
import os
import sys
import json
import argparse
import asyncio
import re
from pathlib import Path
from typing import Optional, Sequence
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, str(Path(__file__).parent))

from core.config import Config
from utils.io_utils import load_json, save_json
from output_paths import stage1_sampling_file
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_urls(base_url: str, ports: Optional[Sequence[int]] = None) -> list[str]:
    if not ports:
        return [base_url]

    urls = []
    for port in ports:
        if re.search(r':\d+(?=/|$)', base_url):
            urls.append(re.sub(r':\d+(?=/|$)', f':{port}', base_url))
        else:
            urls.append(re.sub(r'^(https?://[^/]+)', rf'\1:{port}', base_url))
    return urls


class ModelTester:
    def __init__(self, config: Config, dataset_name: str, urls: Optional[Sequence[str]] = None):
        self.config = config
        self.dataset_name = dataset_name
        dataset_suffix = dataset_name.replace("dpo_level", "")
        self.dataset_path = config.paths.data_dir / "input" / "V3" / f"dpo_level_{dataset_suffix}.json"
        
        hp = config.hyperparams
        self.model = hp.weak_model_name
        self.key = hp.weak_model_key
        self.temperature = 0.7
        
        # 负载均衡核心属性
        self.urls = urls if urls else [hp.weak_model_url]
        self.req_counter = 0  # 保留兼容字段
        
        self.batch_size = 5
        self.max_trials = 15
        self.max_retries = max(1, int(os.environ.get("MEMUA_STAGE1_MAX_RETRIES", "2")))
        self.request_timeout = max(30, int(os.environ.get("MEMUA_STAGE1_REQUEST_TIMEOUT", "300")))
        self.request_max_tokens = max(256, int(os.environ.get("MEMUA_STAGE1_MAX_TOKENS", "8192")))
        self.collect_all_samples = False

    def _normalize_response(self, response: str) -> str:
        if not response:
            return ""

        text = str(response).strip().replace("\r\n", "\n")

        if "UnicodeEncodeError:" in text:
            return ""

        bad_prefixes = ["Toiven", "ToThisHeaderCode", "Toacklemb"]
        for prefix in bad_prefixes:
            if text.startswith(prefix):
                break

        final_matches = list(re.finditer(r"Final Answer:\s*.*", text))
        if final_matches:
            last = final_matches[-1]
            prefix = text[:last.start()].strip()
            final_line = last.group(0).split("\n", 1)[0].strip()
            text = f"{prefix}\n{final_line}".strip() if prefix else final_line

        return text

    def _is_malformed_response(self, response: str) -> bool:
        if not response: return True
        text = response.strip()
        if not text: return True
        if "UnicodeEncodeError:" in text: return True
        if re.search(r"[\u4e00-\u9fff]", text) and "Final Answer:" not in text: return True
        if "Final Answer:" not in text: return True
        return False

    def _looks_truncated_response(self, response: str) -> bool:
        """Heuristic truncation detector with lower false-positive rate.

        We only retry when the tail strongly suggests the output was cut off,
        instead of requiring a punctuation terminator for every response.
        """
        text = (response or "").strip()
        if not text:
            return True

        # 1) Common hard signs of being cut in the middle.
        tail_markers = (
            "Final Answer:",
            "Therefore",
            "Thus",
            "So ",
            "Hence",
            "because",
            "which means",
            "we get",
        )
        lowered = text[-120:].lower()
        if any(lowered.endswith(m.lower()) for m in tail_markers):
            return True

        # 2) Unbalanced structures near the end often indicate truncation.
        if text.count("(") > text.count(")"):
            return True
        if text.count("[") > text.count("]"):
            return True
        if text.count("{") > text.count("}"):
            return True
        if text.count("$") % 2 == 1:
            return True

        # 3) If code fence starts but does not close.
        if text.count("```") % 2 == 1:
            return True

        # 4) If the answer already has a usable final answer, avoid retry noise.
        if "Final Answer:" in text:
            final_part = text.split("Final Answer:", 1)[-1].strip()
            if final_part:
                return False

        # 5) Keep a conservative fallback for obviously abrupt punctuation-free tails,
        # but do not trigger on alnum/CJK endings (which are often valid).
        if text.endswith((":", "，", ",", ";", "；", "-", "(", "[", "{")):
            return True

        return False
        
    async def generate_batch(self, question: str, batch_size: int) -> list:
        from modules.execute_module import URLRouter, async_answer_question_directly, filter_healthy_urls

        router = URLRouter(filter_healthy_urls(self.urls))

        async def generate_single(idx_in_batch: int):
            max_retries = self.max_retries
            for attempt in range(max_retries):
                # 【核心逻辑 1】多节点 Round-Robin 分配 URL，并跳过已熔断节点
                target_url = router.next()
                self.req_counter += 1

                try:
                    response = await async_answer_question_directly(
                        question=question,
                        model=self.model,
                        url=target_url,
                        temperature=self.temperature,
                        max_tokens=self.request_max_tokens,
                        api_key=self.key,
                        timeout=self.request_timeout,
                    )
                    if response:
                        router.report_success(target_url)
                    else:
                        router.report_failure(target_url, "empty response")
                    
                    # 【核心逻辑 2】截断自愈检测（降误报版）
                    if self._looks_truncated_response(response):
                        if attempt < max_retries - 1:
                            logger.debug(f"[{idx_in_batch}] 检测到生成截断 (Port: {target_url.split(':')[-1][:4]}), 触发重试 ({attempt+1}/{max_retries})...")
                            await asyncio.sleep(2)
                            continue
                            
                    return idx_in_batch, response, None
                except Exception as e:
                    router.report_failure(target_url, str(e))
                    if attempt == max_retries - 1:
                        return idx_in_batch, None, f"{target_url} -> {str(e)}"
                    await asyncio.sleep(2)
        
        tasks = [generate_single(i) for i in range(batch_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                valid_results.append((None, None, str(r)))
            else:
                valid_results.append(r)
        
        return valid_results
    
    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """宽松规范化，用于答案比对：去除 LaTeX 装饰、空白，统一小写。"""
        import unicodedata
        cleaned = unicodedata.normalize("NFKC", str(text or "").lower())
        for tok in (r"\left", r"\right", r"\displaystyle", r"\dfrac", r"\tfrac",
                    "$", r"\(", r"\)", r"\[", r"\]"):
            cleaned = cleaned.replace(tok, "")
        cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"^finalanswer:?", "", cleaned)
        cleaned = re.sub(r"^answer:?", "", cleaned)
        cleaned = cleaned.strip(".,;:!?")
        return cleaned

    def check_answer_match(self, generated: str, expected: str) -> bool:
        """使用规范化比对，与 HardNegativeMiner 保持一致，避免字符串包含误判。"""
        if not generated or not expected:
            return False

        expected_norm = self._normalize_for_match(expected)
        if not expected_norm:
            return False

        # 优先从 \boxed{} 提取候选答案
        boxed_matches = re.findall(r'\\boxed\{([^{}]+)\}', generated)
        candidates = [self._normalize_for_match(m) for m in boxed_matches if m.strip()]

        # 兜底：提取 Final Answer 行
        fa_match = re.search(r'Final Answer:\s*(.+)', generated, re.IGNORECASE)
        if fa_match:
            candidates.append(self._normalize_for_match(fa_match.group(1)))

        for candidate in candidates:
            if not candidate:
                continue
            # 精确包含匹配（规范化后）
            if expected_norm in candidate or candidate in expected_norm:
                return True

        return False
    
    async def test_single_question(self, item, idx: int = 0) -> dict:
        if isinstance(item, dict):
            question = item.get("question", item.get("problem", ""))
            expected_answer = item.get("answer", item.get("expected_answer", ""))
        else:
            question = str(item or "")
            expected_answer = str(idx or "")
            idx = 0

        if not question or not expected_answer:
            return {"idx": idx, "status": "skipped", "reason": "missing question or answer"}
        
        all_correct_paths = []
        all_logical_wrong_paths = []
        all_trash_paths = []
        current_trials = 0
        total_generated = 0
        status = "incomplete"
        malformed_count = 0

        while total_generated < self.max_trials:
            current_trials += 1
            remaining_budget = self.max_trials - total_generated
            current_batch_size = min(self.batch_size, remaining_budget)

            batch_results = await self.generate_batch(question, current_batch_size)
            total_generated += len(batch_results)
            
            for _, response, error in batch_results:
                if error:
                    all_trash_paths.append(f"ERROR: {error}")
                    continue

                normalized_response = self._normalize_response(response)
                if not normalized_response:
                    all_trash_paths.append("EMPTY_RESPONSE")
                    continue

                if self._is_malformed_response(normalized_response):
                    malformed_count += 1
                    all_trash_paths.append(f"MALFORMED_RESPONSE: {normalized_response[:500]}")
                    continue

                is_correct = self.check_answer_match(normalized_response, expected_answer)
                if is_correct:
                    all_correct_paths.append(normalized_response)
                else:
                    all_logical_wrong_paths.append(normalized_response)
            
            has_correct = len(all_correct_paths) > 0
            has_logical_wrong = len(all_logical_wrong_paths) > 0
            
            if has_correct and has_logical_wrong:
                status = "early_stop"
                if not self.collect_all_samples:
                    break
            if has_correct and not has_logical_wrong:
                status = "all_correct"
                if not self.collect_all_samples:
                    break
            if not has_correct and has_logical_wrong:
                status = "all_wrong"

        if total_generated >= self.max_trials and status == "all_wrong":
            status = "max_trials_exceeded"

        n_correct = len(all_correct_paths)
        n_wrong = len(all_logical_wrong_paths)
        denom = n_correct + n_wrong
        success_rate = round(n_correct / denom, 4) if denom > 0 else None

        return {
            "idx": idx,
            "question": question,
            "status": status,
            "correct_count": n_correct,
            "wrong_count": n_wrong,
            "success_rate": success_rate,
            "total_trials": current_trials,
            "expected_answer": expected_answer,
            "all_answers": all_correct_paths + all_logical_wrong_paths,
            "trash_count": len(all_trash_paths),
            "trash_answers": all_trash_paths,
            "malformed_count": malformed_count,
            "success": n_correct > 0 and n_wrong > 0,
            "collection_policy": "all_samples_fixed_budget" if self.collect_all_samples else "learning_zone_early_stop",
        }
    
    async def run_test(self, concurrency: int = 3) -> dict:
        if not self.dataset_path.exists():
            logger.error(f"数据集不存在: {self.dataset_path}")
            return {"error": f"Dataset not found: {self.dataset_path}"}
        
        dataset = load_json(str(self.dataset_path))
        logger.info(f"加载数据集: {self.dataset_path}, 共 {len(dataset)} 条")
        logger.info(f"启用的集群端口池: {[u.split(':')[-1][:4] for u in self.urls]}")
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def sem_task(item, idx):
            async with semaphore:
                return await self.test_single_question(item, idx)
        
        tasks = [sem_task(item, i) for i, item in enumerate(dataset)]
        results = []
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Testing"):
            results.append(await coro)
        
        results.sort(key=lambda x: x["idx"])
        
        summary = {
            "dataset": self.dataset_name,
            "total": len(results),
            "early_stop": sum(1 for r in results if r.get("status") == "early_stop"),
            "all_correct": sum(1 for r in results if r.get("status") == "all_correct"),
            "all_wrong": sum(1 for r in results if r.get("status") == "all_wrong"),
            "max_trials_exceeded": sum(1 for r in results if r.get("status") == "max_trials_exceeded"),
            "usable_samples": sum(1 for r in results if r.get("status") == "early_stop"),
            "collection_policy": "all_samples_fixed_budget" if self.collect_all_samples else "learning_zone_early_stop",
        }
        
        output_path = stage1_sampling_file(self.config.paths.output_dir, self.dataset_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json({"summary": summary, "results": results}, output_path)
        return summary


async def main():
    parser = argparse.ArgumentParser(description="训练题不稳定样本采样 - 8卡极限集群榨干版")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"])
    parser.add_argument("--model", "-m", type=str, default="MetaEVO")
    # 【改动】默认并发从 10 飙升到了 120
    parser.add_argument("--concurrency", "-c", type=int, default=120, help="并行跑的题目数 (推荐8卡开 100-120)")
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="每题单轮生成数 (默认 5)")
    parser.add_argument("--max-trials", "-t", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--collect-all-samples", action="store_true",
                        help="Ablation mode: do not stop at the learning-zone boundary; spend the full trial budget on every sample.")
    
    # 接收任意数量的端口
    parser.add_argument("--ports", type=int, nargs='+', default=None,
                        help="多实例本地 API 端口列表，例如 --ports 8100 8101 8102 8103")
    
    args = parser.parse_args()
    config = Config.from_core_file()
    config.setup_logging()
    
    base_url = config.hyperparams.weak_model_url
    urls = build_urls(base_url, args.ports)
        
    logger.info("=" * 70)
    logger.info(f"MetaEvo Stage 1 - 认知边界采样 (8卡集群榨干版)")
    logger.info(f"并发题目数: {args.concurrency} | 单题并发: {args.batch_size} | 瞬间总请求压测: {args.concurrency * args.batch_size}")
    logger.info("=" * 70)
    
    tester = ModelTester(config, args.dataset, urls=urls)
    tester.model = args.model
    tester.temperature = args.temperature
    tester.batch_size = args.batch_size
    tester.max_trials = args.max_trials
    tester.collect_all_samples = args.collect_all_samples
    
    summary = await tester.run_test(concurrency=args.concurrency)
    if "error" in summary:
        logger.error(summary["error"])
        raise SystemExit(1)

    logger.info(f"测试完成! 成功挖掘可用样本: {summary['usable_samples']} / {summary['total']}")


if __name__ == "__main__":
    asyncio.run(main())
