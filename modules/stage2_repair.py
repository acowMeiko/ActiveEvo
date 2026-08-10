"""
modules/stage2_repair.py — Recursive Principle Repair Protocol

Stage 2 原则修复协议:
  1. 邻域验证: 在 3-5 个逻辑同构邻域任务上测试原则
  2. 通过 (pass_ratio >= threshold) → 原则合格, 进入存储
  3. 失败 → 诊断 (vague / over_general / misdirected) → 修补 → 重新验证
  4. 递归上限 MAX_DEPTH 防止死循环

用法:
    protocol = RecursiveRepairProtocol(plan_module, max_depth=3, pass_threshold=0.6)
    result = await protocol.run(principle, neighbors)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RecursiveRepairProtocol:
    """递归原则修复协议。

    在邻域任务上验证原则 → 失败时诊断并修补 → 重新验证 (最多 max_depth 轮)。
    """

    MAX_DEPTH = 3
    PASS_THRESHOLD = 0.6  # 邻域通过率阈值

    def __init__(
        self,
        plan_module: Any,
        max_depth: int = MAX_DEPTH,
        pass_threshold: float = PASS_THRESHOLD,
        validation_concurrency: int = 4,
    ):
        self.plan = plan_module
        self.max_depth = max(1, int(max_depth))
        self.pass_threshold = float(pass_threshold)
        self.validation_concurrency = max(1, int(validation_concurrency))

    @staticmethod
    def _safe_str(val: Any) -> str:
        if isinstance(val, list):
            return " ".join(str(v).strip() for v in val if str(v).strip())
        return str(val or "").strip()

    def _principle_to_text(self, principle: Dict[str, Any]) -> str:
        parts = []
        trigger = self._safe_str(principle.get("trigger_condition", ""))
        evidence = self._safe_str(principle.get("required_evidence", ""))
        exclusion = self._safe_str(principle.get("exclusion_condition", ""))
        action = self._safe_str(principle.get("action_guidance", ""))
        verification = self._safe_str(principle.get("verification_step", ""))
        scope = self._safe_str(principle.get("scope", ""))
        if trigger:
            parts.append(f"[Trigger] {trigger}")
        if evidence:
            parts.append(f"[Required evidence] {evidence}")
        if exclusion:
            parts.append(f"[Reject when] {exclusion}")
        if action:
            parts.append(f"[Action] {action}")
        if verification:
            parts.append(f"[Verification] {verification}")
        if scope:
            parts.append(f"[Scope] {scope}")
        text = "\n".join(parts)
        p = self._safe_str(principle.get("principle", ""))
        if p and p not in text:
            text = f"{text}\n[Summary] {p}"
        return text.strip()

    async def _validate_on_neighbors(
        self,
        principle: Dict[str, str],
        neighbors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """在所有邻域任务上测试原则, 返回详细结果。"""
        semaphore = asyncio.Semaphore(self.validation_concurrency)
        principle_text = self._principle_to_text(principle)
        if not principle_text:
            return {"pass_count": 0, "total": len(neighbors), "results": [], "pass_ratio": 0.0}

        async def test_one(nbr: Dict[str, Any]) -> Dict[str, Any]:
            question = nbr.get("question_text", nbr.get("question", ""))
            expected = nbr.get("expected_answer", "")
            async with semaphore:
                result = await self.plan.crucible_test(
                    question=question,
                    principle={"principle": principle_text},
                    expected_answer=expected,
                    max_attempts=2,
                )
            result["neighbor_id"] = nbr.get("question_id", "")
            result["neighbor_question"] = question
            return result

        tasks = [test_one(n) for n in neighbors]
        results = await asyncio.gather(*tasks)

        pass_count = sum(1 for r in results if r.get("is_correct", False))
        total = len(results)
        return {
            "pass_count": pass_count,
            "total": total,
            "pass_ratio": pass_count / total if total else 0.0,
            "results": list(results),
        }

    async def run(
        self,
        principle: Dict[str, str],
        neighbors: List[Dict[str, Any]],
        source_dsd: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """运行修复协议。

        Args:
            principle: 待验证的原则 dict (来自 self_cpe_extract)
            neighbors: 邻域任务列表 (来自 DSD registry)
            source_dsd: 源问题的 DSD 语义 (用于修复时的上下文)

        Returns:
            {"status": "validated"|"discarded",
             "principle": {...},  # 最终通过的原则 (status=validated 时)
             "repair_history": [...]}  # 每轮修复记录
        """
        repair_history: List[Dict[str, Any]] = []
        current_principle = dict(principle)

        for depth in range(self.max_depth + 1):
            # ── 验证 ──
            validation = await self._validate_on_neighbors(current_principle, neighbors)
            pass_ratio = validation["pass_ratio"]
            pass_count = validation["pass_count"]
            total = validation["total"]

            if depth == 0:
                logger.info(
                    "Repair round 0 (initial): pass %d/%d = %.2f (threshold=%.2f)",
                    pass_count, total, pass_ratio, self.pass_threshold,
                )

            if pass_ratio >= self.pass_threshold:
                logger.info(
                    "Principle validated at depth %d: pass %d/%d = %.2f",
                    depth, pass_count, total, pass_ratio,
                )
                return {
                    "status": "validated",
                    "principle": current_principle,
                    "repair_history": repair_history,
                    "final_validation": validation,
                }

            # ── 已达最大深度, 放弃 ──
            if depth >= self.max_depth:
                logger.warning(
                    "Max repair depth (%d) reached, discarding principle: pass %d/%d = %.2f",
                    self.max_depth, pass_count, total, pass_ratio,
                )
                return {
                    "status": "discarded",
                    "principle": current_principle,
                    "repair_history": repair_history,
                    "final_validation": validation,
                }

            # ── 收集失败案例 ──
            failed = [
                r for r in validation["results"]
                if not r.get("is_correct", False)
            ]

            # ── 诊断 ──
            source_ctx = ""
            if source_dsd:
                source_ctx = (
                    f"Core Skill: {source_dsd.get('core_skill', '')}. "
                    f"Constraints: {source_dsd.get('context_constraints', source_dsd.get('condition', ''))}. "
                    f"Objective: {source_dsd.get('task_objective', source_dsd.get('target', ''))}."
                )
            diagnosis = await self.plan.repair_diagnose(
                principle=current_principle,
                failed_validations=failed,
                source_context=source_ctx,
            )
            logger.info(
                "Repair round %d: diagnosis=%s reason=%s",
                depth + 1,
                diagnosis.get("diagnosis"),
                diagnosis.get("reason", "")[:120],
            )

            # ── 修补 ──
            patch_result = await self.plan.repair_patch(
                principle=current_principle,
                diagnosis_result=diagnosis,
                failed_example=failed[0] if failed else {},
                source_dsd=source_dsd or {},
            )

            repair_history.append({
                "depth": depth + 1,
                "pre_patch_validation": validation,
                "diagnosis": diagnosis,
                "patch_result": patch_result,
            })

            if patch_result.get("status") != "success":
                logger.warning("Repair patch failed at depth %d", depth + 1)
                continue

            patched = patch_result.get("principle", {})
            if patched.get("repair_status") == "unfixable":
                logger.warning("Principle declared unfixable at depth %d", depth + 1)
                return {
                    "status": "discarded",
                    "principle": current_principle,
                    "repair_history": repair_history,
                    "final_validation": validation,
                }

            current_principle = patched

        return {
            "status": "discarded",
            "principle": current_principle,
            "repair_history": repair_history,
        }
