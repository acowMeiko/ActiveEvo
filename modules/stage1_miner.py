"""
modules/stage1_miner.py — Stage 1 第二阶段：困难负样本挖掘 + Diff 分析 + 打包

流程 (per learning-zone question):
  1. 对每条错误轨迹,在所有正确轨迹中找最佳 anchor (分叉最晚的)
  2. 改进的步骤分割 + 数学原子分叉检测
  3. 过滤 similarity > 0.3, 取 top-2
  4. 调用 LLM diff 分析 (简化版 prompt,无语义字段)
  5. 打包为 learning_packet
"""

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm as async_tqdm

from core.api_call import async_llm_call
from modules.execute_module import URLRouter
from template.prompt_template import STAGE1_DIFF_PROMPT

logger = logging.getLogger(__name__)



# ════════════════════════════════════════════════════════════════
# 步骤分割
# ════════════════════════════════════════════════════════════════

def _split_steps(path: str) -> List[str]:
    """将推理轨迹拆分为步骤序列。

    4 级策略,保证任何格式都有合理的分段粒度。
    """
    if not path or not path.strip():
        return []

    # Level 1: 编号步骤 (Step 1: / 1. / 1) etc.)
    numbered = re.compile(r'\n(?:Step\s+)?\d+[.)]\s*', re.IGNORECASE)
    if numbered.search(path):
        parts = numbered.split(path)
        steps = [s.strip() for s in parts if len(s.strip()) >= 10]
        if len(steps) >= 3:
            return steps

    # Level 2: 双换行
    parts = re.split(r'\n{2,}', path)
    steps = [s.strip() for s in parts if len(s.strip()) >= 10]
    if len(steps) >= 2:
        return steps

    # Level 3: 句子边界 (大写字母开头)
    lines = path.split('\n')
    merged: List[str] = []
    buf = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buf:
                merged.append(buf.strip())
                buf = ""
            continue
        if buf and re.match(r'^[A-Z]', stripped):
            merged.append(buf.strip())
            buf = stripped
        else:
            buf = (buf + " " + stripped).strip() if buf else stripped
    if buf:
        merged.append(buf.strip())
    steps = [s for s in merged if len(s) >= 15]
    if len(steps) >= 2:
        return steps

    # 前三层都分不出来 → 轨迹无步骤结构,返回整段
    return [path]


# ════════════════════════════════════════════════════════════════
# 数学原子提取
# ════════════════════════════════════════════════════════════════

def _extract_math_atoms(text: str) -> set:
    """从文本中提取数学操作相关的原子符号。"""
    text = text.lower()
    atoms = set()

    # 数字
    for m in re.finditer(r'\d+\.?\d*', text):
        atoms.add(("num", m.group()))

    # LaTeX 命令
    for m in re.finditer(r'\\([a-z]+)', text):
        atoms.add(("cmd", m.group(1)))

    # 运算符
    for m in re.finditer(r'[+\-×÷=<>≤≥≠≈±√∫∑∏]', text):
        atoms.add(("op", m.group()))

    # 常见数学函数名
    for m in re.finditer(r'\b(sin|cos|tan|log|ln|exp|sqrt|abs|max|min|gcd|lcm|mod|lim|det|tr)\b', text):
        atoms.add(("fn", m.group()))

    return atoms


# ════════════════════════════════════════════════════════════════
# 分叉检测
# ════════════════════════════════════════════════════════════════

def _steps_diverge(step_a: str, step_b: str) -> bool:
    """判断两步是否已经分叉。"""
    sa = step_a.lower().strip()
    sb = step_b.lower().strip()
    if not sa or not sb:
        return True

    # 文本相似度 ≥ 0.80 → 未分叉
    if SequenceMatcher(None, sa, sb).ratio() >= 0.80:
        return False

    # 数学原子重叠 ≥ 50% → 未分叉
    atoms_a = _extract_math_atoms(sa)
    atoms_b = _extract_math_atoms(sb)
    if atoms_a and atoms_b:
        overlap = len(atoms_a & atoms_b) / max(len(atoms_a | atoms_b), 1)
        if overlap >= 0.50:
            return False

    return True


def divergence_score(correct_path: str, wrong_path: str) -> Tuple[float, int, str]:
    """计算分叉分数 = 分叉前匹配步数 / 正确路径总步数。"""
    correct_steps = _split_steps(correct_path)
    wrong_steps = _split_steps(wrong_path)

    if not correct_steps or not wrong_steps:
        return 0.0, 0, ""

    n_correct = len(correct_steps)
    div_idx = 0
    for i, (cs, ws) in enumerate(zip(correct_steps, wrong_steps)):
        if not _steps_diverge(cs, ws):
            div_idx = i + 1
        else:
            break

    score = div_idx / n_correct
    div_text = correct_steps[div_idx] if div_idx < n_correct else ""
    return score, div_idx, div_text


# ════════════════════════════════════════════════════════════════
# 正负样本挖掘
# ════════════════════════════════════════════════════════════════

def mine_pairs(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """为单个 learning-zone 样本挖掘正负轨迹对。

    对每条错误轨迹在所有正确轨迹中找最佳 anchor。

    Returns:
        [{anchor, wrong_path, similarity, divergence_step_idx, divergence_step_text}, ...]
        按 similarity 降序,取 top-2
    """
    correct_paths = sample.get("correct_trajectories", [])
    wrong_paths = sample.get("wrong_trajectories", [])

    if not correct_paths or not wrong_paths:
        return []

    # 过滤: 只保留看起来完整的轨迹
    def _looks_ok(path: str) -> bool:
        if not path or not path.strip():
            return False
        s = path.strip()
        if s in ("[EMPTY_RESPONSE]", "ERROR", "MALFORMED_RESPONSE"):
            return False
        if len(s) >= 50:
            return True
        return False

    correct_paths = [p for p in correct_paths if _looks_ok(p)]
    wrong_paths = [p for p in wrong_paths if _looks_ok(p)]

    if not correct_paths or not wrong_paths:
        return []

    # 为每条错误轨迹找最佳 anchor
    best_pairs: List[Dict[str, Any]] = []
    for wrong in wrong_paths:
        best_score = 0.0
        best_anchor = ""
        best_div_idx = 0
        best_div_text = ""
        for correct in correct_paths:
            score, div_idx, div_text = divergence_score(correct, wrong)
            if score > best_score:
                best_score = score
                best_anchor = correct
                best_div_idx = div_idx
                best_div_text = div_text
        best_pairs.append({
            "anchor": best_anchor,
            "wrong_path": wrong,
            "similarity": best_score,
            "divergence_step_idx": best_div_idx,
            "divergence_step_text": best_div_text,
        })

    best_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return best_pairs[:2]


# ════════════════════════════════════════════════════════════════
# Diff 分析 + 打包
# ════════════════════════════════════════════════════════════════

class Stage1Miner:
    """困难负样本挖掘 + Diff 分析 + Learning Packet 打包。"""

    def __init__(
        self,
        urls: List[str],
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 600,
    ):
        self.urls = urls
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.router = URLRouter(urls)

    async def _diff_analysis(
        self,
        question: str,
        chosen_path: str,
        rejected_path: str,
        divergence_step_idx: int,
        divergence_step_text: str,
    ) -> Dict[str, Any]:
        """调用 LLM 做 diff 分析。"""
        if divergence_step_idx > 0 and divergence_step_text:
            divergence_hint = (
                f"The two trajectories share the same prefix up to step {divergence_step_idx}. "
                f"The first divergence appears at:\n\"{divergence_step_text}\""
            )
        elif divergence_step_idx == 0:
            divergence_hint = (
                "The two trajectories diverge from the very first step. "
                "Focus on the fundamental approach difference."
            )
        else:
            divergence_hint = ""

        prompt = STAGE1_DIFF_PROMPT.format(
            question=question,
            chosen_path=chosen_path,
            rejected_path=rejected_path,
            divergence_hint=divergence_hint,
        )

        for _ in range(max(1, len(self.urls))):
            url = self.router.next()
            try:
                response = await async_llm_call(
                    prompt=prompt, model=self.model, url=url,
                    api_key=self.api_key, temperature=0.0, max_tokens=4096,
                    timeout=self.timeout,
                )
                if response:
                    self.router.report_success(url)
                    return self._parse_diff_response(response)
                self.router.report_failure(url, "empty diff response")
            except Exception as e:
                self.router.report_failure(url, str(e))
        return {"status": "failed", "differences": []}

    @staticmethod
    def _parse_diff_response(response: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response.strip())
            return {"status": "success", "differences": parsed.get("differences", [])}
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return {"status": "success", "differences": parsed.get("differences", [])}
                except json.JSONDecodeError:
                    pass
            return {"status": "parse_error", "raw": response[:500]}

    async def process_sample(
        self, sample: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> List[Dict[str, Any]]:
        """处理单个 learning-zone 样本,产出 learning packet(s)。"""
        pairs = mine_pairs(sample)
        if not pairs:
            return []

        question = sample.get("question", "")
        expected = sample.get("expected_answer", "")

        packets = []
        for pair in pairs:
            async with semaphore:
                diff_result = await self._diff_analysis(
                    question=question,
                    chosen_path=pair["anchor"],
                    rejected_path=pair["wrong_path"],
                    divergence_step_idx=pair["divergence_step_idx"],
                    divergence_step_text=pair["divergence_step_text"],
                )

            packets.append({
                "idx": sample.get("idx"),
                "question": question,
                "expected_answer": expected,
                "assigned_level": sample.get("assigned_level"),
                "success_rate": sample.get("sr"),
                "chosen_path": pair["anchor"],
                "rejected_path": pair["wrong_path"],
                "divergence_step_idx": pair["divergence_step_idx"],
                "divergence_step_text": pair["divergence_step_text"],
                "similarity": pair["similarity"],
                "diff": diff_result,
                "diff_success": diff_result.get("status") == "success",
            })

        return packets

    async def run(
        self,
        level_data: Dict[str, List[Dict[str, Any]]],
        concurrency: int = 10,
    ) -> Dict[str, Any]:
        """对全部 learning-zone 样本执行 mining + diff + 打包。

        Args:
            level_data: {"1": [...], "2": [...], "3": [...], "4": [...]}
            concurrency: diff 分析并发数

        Returns:
            {"level1": [packet, ...], "level2": [...], ...}
        """
        semaphore = asyncio.Semaphore(concurrency)
        all_packets: Dict[str, List[Dict]] = {"level1": [], "level2": [], "level3": [], "level4": []}
        stats = {"total_samples": 0, "total_pairs_found": 0, "diff_success": 0}

        for lv in ("1", "2", "3", "4"):
            samples = level_data.get(lv, [])
            if not samples:
                continue

            stats["total_samples"] += len(samples)

            async def process_one(sample: Dict) -> List[Dict]:
                return await self.process_sample(sample, semaphore)

            coros = [process_one(s) for s in samples]

            level_packets = []
            for coro in async_tqdm(
                asyncio.as_completed(coros), total=len(coros),
                desc=f"Mining Level {lv}",
            ):
                packets = await coro
                level_packets.extend(packets)

            stats["total_pairs_found"] += len(level_packets)
            stats["diff_success"] += sum(1 for p in level_packets if p.get("diff_success"))

            all_packets[f"level{lv}"] = level_packets
            logger.info(
                "  Level %s: %d samples → %d pairs (diff_success=%d)",
                lv, len(samples), len(level_packets),
                sum(1 for p in level_packets if p.get("diff_success")),
            )

        logger.info(
            "Stage 1.2 完成: %d samples → %d pairs (diff_ok=%d)",
            stats["total_samples"], stats["total_pairs_found"], stats["diff_success"],
        )

        return {"packets": all_packets, "stats": stats}

    @staticmethod
    def save(packets: Dict[str, List[Dict]], output_dir: Path) -> Dict[str, Path]:
        """保存各 level 的 learning packets。"""
        paths = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, items in packets.items():
            path = output_dir / f"{key}_learning_packets.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            paths[key] = path
            logger.info("Learning packets 已保存: %s (%d 条)", path, len(items))
        return paths
