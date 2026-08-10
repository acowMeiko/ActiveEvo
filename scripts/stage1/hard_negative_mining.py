#!/usr/bin/env python
"""
MetaEvo Stage 1 - 子任务 1.2
困难负样本挖掘 (Hard Negative Mining)

输入：output/test_dpo_levelX_v2_results.json (已生成的推理路径)
输出：output/hard_negative_pairs_levelX.json (正负样本对)
"""
import os
import sys
import json
import argparse
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import Config
from utils.io_utils import load_json
from utils.output_paths import stage1_mining_file, stage1_sampling_file
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """困难负样本挖掘器"""

    def __init__(self, config: Config, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.result_path = stage1_sampling_file(config.paths.output_dir, dataset_name)

    def extract_final_answer(self, text: str) -> Optional[str]:
        """从回答中提取最终答案 (支持嵌套大括号的安全提取)。"""
        text = self._sanitize_text(text)
        if not text:
            return None

        if "Final Answer:" in text:
            text = text.split("Final Answer:")[-1].strip()

        boxed_idx = text.find(r'\boxed{')
        if boxed_idx != -1:
            start_idx = boxed_idx + 7
            open_braces = 1
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    open_braces += 1
                elif text[i] == '}':
                    open_braces -= 1

                if open_braces == 0:
                    return text[start_idx:i].strip()

        patterns = [
            r'\$\s*([^\$]+)\$',
            r'答案是?\s*([^\n，。]+)',
            r'answer[:\s]+([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        return text.strip()

    def _sanitize_text(self, text: object) -> str:
        if text is None:
            return ""

        cleaned = str(text)
        cleaned = cleaned.replace("\x08", r"\b")
        cleaned = cleaned.replace("\x0c", r"\f")
        cleaned = cleaned.replace("\u2212", "-")
        cleaned = cleaned.replace("\u2013", "-")
        cleaned = cleaned.replace("\u2014", "-")
        cleaned = unicodedata.normalize("NFKC", cleaned)

        filtered_chars = []
        for ch in cleaned:
            if ch in ("\n", "\t"):
                filtered_chars.append(ch)
                continue
            if unicodedata.category(ch).startswith("C"):
                continue
            filtered_chars.append(ch)

        return "".join(filtered_chars).strip()

    def _normalize_answer(self, text: object) -> str:
        cleaned = self._sanitize_text(text).lower()
        if not cleaned:
            return ""

        cleaned = cleaned.replace("\\left", "")
        cleaned = cleaned.replace("\\right", "")
        cleaned = cleaned.replace("\\displaystyle", "")
        cleaned = cleaned.replace("\\dfrac", "\\frac")
        cleaned = cleaned.replace("\\tfrac", "\\frac")
        cleaned = cleaned.replace("$", "")
        cleaned = cleaned.replace(r"\(", "").replace(r"\)", "")
        cleaned = cleaned.replace(r"\[", "").replace(r"\]", "")
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"^finalanswer:?", "", cleaned)
        cleaned = re.sub(r"^answer:?", "", cleaned)
        cleaned = cleaned.strip(".,;:!?")
        return cleaned

    def _extract_numeric_atoms(self, text: object) -> List[str]:
        normalized = self._normalize_answer(text)
        if not normalized:
            return []

        compact = normalized
        compact = compact.replace("\\frac", " frac ")
        compact = compact.replace("\\sqrt", " sqrt ")
        compact = re.sub(r"[^a-z0-9./+-]+", " ", compact)
        atoms = [token for token in compact.split() if re.search(r"\d", token)]
        return atoms

    def _shares_expected_signal(self, candidate: str, expected: str) -> bool:
        expected_norm = self._normalize_answer(expected)
        candidate_norm = self._normalize_answer(candidate)
        if not expected_norm or not candidate_norm:
            return False

        if len(candidate_norm) < 2 and candidate_norm != expected_norm:
            return False

        if expected_norm in candidate_norm:
            return True

        if expected_norm.replace("\\", "") == candidate_norm.replace("\\", ""):
            return True

        return False

    def _extract_answer_candidates(self, text: str) -> List[str]:
        candidates = []

        final_answer = self.extract_final_answer(text)
        if final_answer:
            candidates.append(final_answer)

        for pattern in [
            r'Final Answer:\s*(.+)',
            r'答案是?\s*([^\n，。]+)',
            r'answer[:\s]+([^\n]+)',
        ]:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                candidate = match.group(1).strip()
                if candidate:
                    candidates.append(candidate)

        tail_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if tail_lines:
            candidates.append(tail_lines[-1])

        deduped = []
        seen = set()
        for candidate in candidates:
            key = self._normalize_answer(candidate)
            if key and key not in seen:
                seen.add(key)
                deduped.append(candidate.strip())
        return deduped

    def _looks_incomplete(self, text: str) -> bool:
        text = self._sanitize_text(text)
        if not text:
            return True
        if "Final Answer:" in text:
            return False
        if text.endswith(("}", ".", "]", "$$", '"', "'", ")", "```")):
            return False
        return True

    # ================================================================
    # 逐步前缀匹配 — 分叉点定位
    # ================================================================

    def _split_steps(self, path: str) -> List[str]:
        """将推理轨迹拆分为逐步推理单元。

        优先级：编号步骤 > 双换行 > 单换行（最低 15 字符/段）
        """
        path = self._sanitize_text(path)
        if not path:
            return []

        # Priority 1: Numbered steps (Step 1: / 1. / 1) etc.)
        numbered_re = re.compile(r'\n(?:Step\s+)?\d+[.)]\s*', re.IGNORECASE)
        if numbered_re.search(path):
            parts = numbered_re.split(path)
            steps = [s.strip() for s in parts if s.strip() and len(s.strip()) >= 10]
            if len(steps) >= 3:
                return steps

        # Priority 2: Double newline
        parts = re.split(r'\n{2,}', path)
        steps = [s.strip() for s in parts if s.strip() and len(s.strip()) >= 10]
        if len(steps) >= 2:
            return steps

        # Priority 3: Single newline with minimum length
        parts = path.split('\n')
        steps = [s.strip() for s in parts if s.strip() and len(s.strip()) >= 15]
        return steps if steps else [path]

    def _steps_match(self, step1: str, step2: str, threshold: float = 0.75) -> bool:
        """判断两个推理步骤是否"等价"（用步骤级 SequenceMatcher）。

        若文本相似度 >= threshold 则视为匹配；
        若相似度略低但数字原子集合完全相同也视为匹配（同一运算，不同表达）。
        """
        s1 = step1.lower().strip()
        s2 = step2.lower().strip()
        if not s1 or not s2:
            return False
        ratio = SequenceMatcher(None, s1, s2).ratio()
        if ratio >= threshold:
            return True
        # Fallback: same numeric atoms but different surface form
        atoms1 = set(self._extract_numeric_atoms(s1))
        atoms2 = set(self._extract_numeric_atoms(s2))
        if atoms1 and atoms2 and atoms1 == atoms2 and ratio >= 0.50:
            return True
        return False

    def _divergence_score(
        self, correct_path: str, wrong_path: str
    ) -> Tuple[float, int, str]:
        """计算两条轨迹的分叉分数。

        Returns:
            (score, divergence_step_idx, divergence_step_text)
            score = 分叉前匹配步数 / 正确路径总步数  (0.0 ~ 1.0)
            divergence_step_idx: 第一个不匹配步的索引 (0-based)
            divergence_step_text: 正确路径在分叉点的步骤原文
        """
        correct_steps = self._split_steps(correct_path)
        wrong_steps = self._split_steps(wrong_path)

        if not correct_steps or not wrong_steps:
            return 0.0, 0, ""

        n_correct = len(correct_steps)
        divergence_idx = 0
        for i, (cs, ws) in enumerate(zip(correct_steps, wrong_steps)):
            if self._steps_match(cs, ws):
                divergence_idx = i + 1
            else:
                break

        score = divergence_idx / n_correct
        div_text = correct_steps[divergence_idx] if divergence_idx < n_correct else ""
        return score, divergence_idx, div_text

    def _is_correct_path(self, answer_text: str, expected_answer: str) -> Tuple[bool, Optional[str]]:
        sanitized = self._sanitize_text(answer_text)
        expected_norm = self._normalize_answer(expected_answer)
        candidates = self._extract_answer_candidates(sanitized)

        for candidate in candidates:
            candidate_norm = self._normalize_answer(candidate)
            if not candidate_norm:
                continue
            if self._shares_expected_signal(candidate_norm, expected_norm):
                return True, candidate

        tail_norm = self._normalize_answer(" ".join(sanitized.splitlines()[-2:]))
        if self._shares_expected_signal(tail_norm, expected_norm):
            return True, tail_norm

        return False, candidates[0] if candidates else None

    def mine_hard_negatives(self, result_item: Dict) -> Dict:
        """核心逻辑：困难负样本挖掘"""
        all_answers = result_item.get("all_answers", [])
        expected_answer = result_item.get("expected_answer", "")

        if not all_answers or not expected_answer:
            return {"anchor_chosen": "", "hard_negatives": [], "error": "missing data"}

        expected_clean = str(expected_answer).strip().lower()

        correct_paths = []
        wrong_paths = []

        for ans in all_answers:
            ans_str = self._sanitize_text(ans)
            if not ans_str or "ERROR" in ans_str or "MALFORMED_RESPONSE" in ans_str or "EMPTY_RESPONSE" in ans_str:
                continue

            if self._looks_incomplete(ans_str):
                continue

            is_correct, final_ans = self._is_correct_path(ans_str, expected_answer)

            if is_correct:
                correct_paths.append(ans_str)
            else:
                if not final_ans:
                    continue
                wrong_paths.append({
                    "path": ans_str,
                    "final_answer": final_ans,
                })

        if not correct_paths or not wrong_paths:
            return {"anchor_chosen": "", "hard_negatives": [], "error": "not a divergent sample"}

        # B-1: 选择最长（推理步骤最完整）的成功轨迹而非最短的
        anchor_chosen = max(correct_paths, key=lambda p: len(p.split('\n')))

        wrong_answers = {}
        for wrong_item in wrong_paths:
            path = wrong_item["path"]
            final_ans = wrong_item.get("final_answer") or "UNKNOWN"
            final_key = self._normalize_answer(final_ans) or "UNKNOWN"

            if final_key == expected_clean:
                continue

            if final_key not in wrong_answers:
                wrong_answers[final_key] = {
                    "answer": final_ans,
                    "paths": [],
                }

            # 逐步前缀匹配：分叉越晚，分数越高
            div_score, div_idx, div_text = self._divergence_score(anchor_chosen, path)
            wrong_answers[final_key]["paths"].append({
                "path": path,
                "similarity": div_score,           # 向后兼容：保留 similarity 键
                "divergence_step_idx": div_idx,
                "divergence_step_text": div_text,
            })

        unique_wrong_paths = []
        for wrong_info in wrong_answers.values():
            paths_info = wrong_info["paths"]
            best_match = max(paths_info, key=lambda x: x["similarity"])
            unique_wrong_paths.append({
                "answer": wrong_info["answer"],
                "path": best_match["path"],
                "similarity": best_match["similarity"],
                "divergence_step_idx": best_match["divergence_step_idx"],
                "divergence_step_text": best_match["divergence_step_text"],
            })

        unique_wrong_paths.sort(key=lambda x: x["similarity"], reverse=True)

        # B-2: 分叉分数过滤：< 0.3 说明分叉太早，价值低
        hard_negatives = [hn for hn in unique_wrong_paths if hn["similarity"] > 0.3][:2]

        return {
            "anchor_chosen": anchor_chosen,
            "hard_negatives": hard_negatives,
        }

    def process_item(self, result_item: Dict) -> List[Dict]:
        """处理单个样本，返回该样本下产生的所有正负对"""
        idx = result_item.get("idx", 0)
        question = result_item.get("question", "")
        expected_answer = result_item.get("expected_answer", "")

        if not question or not expected_answer:
            return []

        mining_result = self.mine_hard_negatives(result_item)

        if not mining_result.get("anchor_chosen") or not mining_result.get("hard_negatives"):
            return []

        pairs = []
        for hn in mining_result["hard_negatives"]:
            pairs.append({
                "idx": idx,
                "question": question,
                "expected_answer": expected_answer,
                "chosen_path": mining_result["anchor_chosen"],
                "rejected_path": hn.get("path", ""),
                "rejected_answer": hn.get("answer", ""),
                "similarity": hn.get("similarity", 0),
                "divergence_step_idx": hn.get("divergence_step_idx", 0),
                "divergence_step_text": hn.get("divergence_step_text", ""),
            })

        return pairs

    def run(self) -> Dict:
        """运行挖掘流程 (多进程加速版，输出格式化 JSON)"""
        import concurrent.futures

        if not self.result_path.exists():
            logger.error(f"结果文件不存在: {self.result_path}")
            return {"error": "file_not_found"}

        result_data = load_json(str(self.result_path))
        results = result_data.get("results", [])

        valid_results = [r for r in results if r.get("success") is True]

        max_workers = max(1, (os.cpu_count() or 4) - 1)

        logger.info(f"加载结果: {self.result_path}")
        logger.info(f"总样本数: {len(results)}")
        logger.info(f"有效运行样本数: {len(valid_results)}")
        logger.info(f"启动多进程加速，分配 CPU 核心数: {max_workers}")

        all_pairs = []
        divergent_count = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_iterator = list(tqdm(
                executor.map(self.process_item, valid_results),
                total=len(valid_results),
                desc="Mining Hard Negatives (Multi-core)",
            ))

        for pairs in results_iterator:
            if pairs:
                divergent_count += 1
                all_pairs.extend(pairs)

        all_pairs.sort(key=lambda x: x.get("idx", 0))

        stats = {
            "total_valid_samples": len(valid_results),
            "divergent_questions": divergent_count,
            "total_pairs_generated": len(all_pairs),
        }

        if all_pairs:
            output_path = stage1_mining_file(self.config.paths.output_dir, self.dataset_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_pairs, f, ensure_ascii=False, indent=4)
            logger.info(f"正负样本对已保存: {output_path}")

        return stats

    def run_from_file(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """从指定结果文件运行挖掘，便于主流程复用。"""
        input_file = Path(input_path)
        if not input_file.exists():
            logger.error(f"结果文件不存在: {input_file}")
            return {"error": "file_not_found"}

        result_data = load_json(str(input_file))
        results = result_data.get("results", []) if isinstance(result_data, dict) else result_data
        if not isinstance(results, list):
            return {"error": "invalid_data_format"}

        valid_results = [r for r in results if isinstance(r, dict) and r.get("success") is True]
        all_pairs = []
        divergent_count = 0

        for item in tqdm(valid_results, desc="Mining Hard Negatives"):
            pairs = self.process_item(item)
            if pairs:
                divergent_count += 1
                all_pairs.extend(pairs)

        all_pairs.sort(key=lambda x: x.get("idx", 0))

        final_output = Path(output_path) if output_path else stage1_mining_file(self.config.paths.output_dir, self.dataset_name)
        final_output.parent.mkdir(parents=True, exist_ok=True)
        with open(final_output, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=4)
        logger.info(f"正负样本对已保存: {final_output}")

        return {
            "total_valid_samples": len(valid_results),
            "divergent_questions": divergent_count,
            "total_pairs_generated": len(all_pairs),
            "output_path": str(final_output),
        }


def main():
    parser = argparse.ArgumentParser(description="困难负样本挖掘 (提取正负对)")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"],
        help="数据集名称",
    )

    args = parser.parse_args()
    config = Config.from_core_file()
    config.setup_logging()

    logger.info("=" * 70)
    logger.info("MetaEvo Stage 1 - 困难负样本挖掘")
    logger.info("=" * 70)
    logger.info(f"数据集: {args.dataset}")
    logger.info("=" * 70)

    miner = HardNegativeMiner(config, args.dataset)
    stats = miner.run()

    if "error" not in stats:
        logger.info("=" * 70)
        logger.info("处理完成!")
        logger.info(f"有效运行样本: {stats.get('total_valid_samples', 0)}")
        logger.info(f"有对有错题目数: {stats.get('divergent_questions', 0)}")
        logger.info(f"共生成正负样本对: {stats.get('total_pairs_generated', 0)}")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
