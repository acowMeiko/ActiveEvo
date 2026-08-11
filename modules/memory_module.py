"""
modules/memory_module.py — MemoryModule 类

Agent 的「记忆」模块，负责：
  - 经验入库 (从 learning packets / principle records 提取并导入)
  - 三维 FAISS 向量索引构建
  - 三维门控检索
  - 按轮次快照
  - 统计信息

内部委托给 modules/memory.py 的 Memory 类。
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import Config
from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
from modules.principle_schema import (
    SCHEMA_VERSION,
    build_memory_views,
    principle_content_from_item,
    validate_principle_content,
)

logger = logging.getLogger(__name__)


def _clean(val: Any) -> str:
    return str(val).strip() if val else ""


class MemoryModule:
    """Agent 的记忆管理模块。

    使用方式:
        mem = MemoryModule(config, memory_dir="memory")
        mem.load()

        # Stage 2: 入库
        mem.ingest_from_learning_packets(packets, round_id=1, dataset="dpo_level1")
        mem.build_indices()
        mem.save()
        mem.snapshot(round_id=1)

        # Stage 3: 检索
        results = mem.retrieve(features, top_k=5, threshold=0.55)
    """

    def __init__(self, config: Config, memory_dir: str = "memory/v3"):
        self.config = config
        self.memory_dir = memory_dir
        self._memory = None  # Lazy init

    @property
    def memory(self):
        """内部 Memory 实例 (lazy)。"""
        if self._memory is None:
            from modules.memory import Memory
            self._memory = Memory(self.memory_dir)
        return self._memory

    # ================================================================
    # 持久化
    # ================================================================

    def load(self) -> None:
        """从磁盘加载记忆。"""
        self.memory.load()
        logger.info("MemoryModule loaded: %d principles", len(self.memory.principles))

    def save(self) -> None:
        """保存记忆到磁盘。"""
        self.memory.save()

    def snapshot(self, round_id: int) -> None:
        """按轮次保存快照。"""
        self.memory.snapshot(round_id)

    def stats(self) -> Dict[str, Any]:
        """返回记忆库统计信息。"""
        return self.memory.stats()

    # ================================================================
    # 入库
    # ================================================================

    def ingest_from_learning_packets(
        self,
        packets: List[Dict[str, Any]],
        round_id: int,
        dataset: str,
    ) -> Dict[str, int]:
        """从 learning packets 批量提取 principle 并入库。

        Args:
            packets: Stage 1 产出的 learning_packets 列表
            round_id: 当前课程轮次 (1-4)
            dataset: 数据集名称 (e.g. "dpo_level1")

        Returns:
            {"added": N, "updated": N, "duplicated": N, "skipped": N}
        """
        counters = {"added": 0, "updated": 0, "duplicated": 0, "skipped": 0}
        for packet in packets:
            item = self._extract_from_learning_packet(packet, round_id, dataset)
            if item is None:
                counters["skipped"] += 1
                continue
            mid, action = self.memory.ingest_principle(item)
            if action == "KEEP_NEW":
                counters["added"] += 1
            elif action == "UPDATE_EXISTING":
                counters["updated"] += 1
            elif action == "DEPRECATE":
                counters["duplicated"] += 1
        logger.info(
            "Ingested %d packets: added=%d, updated=%d, dup=%d, skipped=%d",
            len(packets), counters["added"], counters["updated"],
            counters["duplicated"], counters["skipped"],
        )
        return counters

    def ingest_from_principle_records(
        self,
        records: List[Dict[str, Any]],
        round_id: int,
        dataset: str,
    ) -> Dict[str, int]:
        """从 principle DPO records (旧流程产物) 批量入库。"""
        counters = {"added": 0, "updated": 0, "duplicated": 0, "skipped": 0}
        for record in records:
            item = self._extract_from_principle_record(record, round_id, dataset)
            if item is None:
                counters["skipped"] += 1
                continue
            mid, action = self.memory.ingest_principle(item)
            if action == "KEEP_NEW":
                counters["added"] += 1
            elif action == "UPDATE_EXISTING":
                counters["updated"] += 1
            elif action == "DEPRECATE":
                counters["duplicated"] += 1
        return counters

    def ingest_validated_principles(
        self,
        records: List[Dict[str, Any]],
        round_id: int,
        dataset: str,
        auto_dedup: bool = False,
    ) -> Dict[str, int]:
        """从 Stage 2 产出的 validated principles 批量入库。"""
        counters = {"added": 0, "updated": 0, "duplicated": 0, "skipped": 0}
        for record in records:
            item = self._extract_from_validated_principle(record, round_id, dataset)
            if item is None:
                counters["skipped"] += 1
                continue
            mid, action = self.memory.ingest_principle(item, auto_dedup=auto_dedup)
            if action == "KEEP_NEW":
                counters["added"] += 1
            elif action == "UPDATE_EXISTING":
                counters["updated"] += 1
            elif action == "DEPRECATE":
                counters["duplicated"] += 1
        logger.info(
            "Validated principles ingested: added=%d, updated=%d, dup=%d, skipped=%d",
            counters["added"], counters["updated"], counters["duplicated"], counters["skipped"],
        )
        return counters

    def commit_stage2_candidate(
        self,
        record: Dict[str, Any],
        round_id: int,
        dataset: str,
    ) -> Dict[str, Any]:
        """Commit one v3 candidate with conservative version reconciliation."""
        item = self._extract_from_validated_principle(record, round_id, dataset)
        if item is None:
            return {"operation": "reject", "memory_id": "", "reason": "invalid_candidate"}

        matches = []
        if item.get("status") == "active" and self.memory.principles:
            if not self.is_indexed:
                self.build_indices()
            matches = self.retrieve_strict_semantic_matches(item.get("source_dsd", {}), facet_threshold=0.95, top_k=1)

        if not matches:
            memory_id = self.memory.add(item)
            if item.get("status") == "active":
                self.memory.index_added_principle(item)
            return {"operation": "add", "memory_id": memory_id, "version_id": item.get("version_id")}

        existing = matches[0]["item"]
        existing_content = existing.get("content", {}) if isinstance(existing.get("content"), dict) else {}
        new_content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
        compatible = all(
            " ".join(str(existing_content.get(field, "")).lower().split())
            == " ".join(str(new_content.get(field, "")).lower().split())
            for field in ("trigger_condition", "exclusion_condition")
        )
        old_gain = int((existing.get("quality_metrics") or {}).get("holdout_net_gain", 0) or 0)
        new_gain = int((item.get("quality_metrics") or {}).get("holdout_net_gain", 0) or 0)
        if compatible and new_gain > old_gain:
            memory_id = str(existing.get("memory_id", ""))
            self.memory.replace_version(memory_id, item, reason="stronger paired holdout evidence")
            self.memory.build_indices()
            current = next((p for p in self.memory.principles if p.get("memory_id") == memory_id), {})
            return {"operation": "replace", "memory_id": memory_id, "version_id": current.get("version_id")}

        # A merge is never performed without a new paired holdout run of the
        # merged text. The candidate already remains in Stage-2 audit output;
        # do not leak a provisional merge into the active-only live Store.
        if compatible:
            return {
                "operation": "merge_deferred",
                "memory_id": str(existing.get("memory_id", "")),
                "version_id": existing.get("version_id"),
                "reason": "merge_requires_revalidation",
            }
        memory_id = self.memory.add(item)
        if item.get("status") == "active":
            self.memory.index_added_principle(item)
        return {
            "operation": "add",
            "memory_id": memory_id,
            "version_id": item.get("version_id"),
        }

    def import_legacy_as_provisional(self, legacy_dir: str | Path) -> Dict[str, int]:
        """Copy legacy records into v3 as inactive, auditable compatibility items."""
        import json

        source = Path(legacy_dir) / "principles" / "principles.json"
        counters = {"imported": 0, "skipped": 0, "already_present": 0}
        if not source.exists():
            return counters
        with source.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            return counters
        for legacy in records:
            if not isinstance(legacy, dict):
                counters["skipped"] += 1
                continue
            content, schema_source = principle_content_from_item(legacy)
            if not content:
                counters["skipped"] += 1
                continue
            legacy_id = str(legacy.get("memory_id", ""))
            if legacy_id and legacy_id in self.memory._id_set:
                counters["already_present"] += 1
                continue
            source_dsd = legacy.get("source_dsd", {}) if isinstance(legacy.get("source_dsd"), dict) else {
                CORE_FIELD: get_field(legacy, CORE_FIELD),
                CONDITION_FIELD: get_field(legacy, CONDITION_FIELD),
                TARGET_FIELD: get_field(legacy, TARGET_FIELD),
            }
            views = build_memory_views(
                content,
                source_dsd=source_dsd,
                validation={"legacy_revalidation_required": True},
                audit={"legacy_schema_source": schema_source, "legacy_source_path": str(source)},
            )
            item = {
                **legacy,
                **content,
                **views,
                "content": content,
                "source_dsd": source_dsd,
                "principle": content["action_guidance"],
                "principle_text": content["action_guidance"],
                "status": "provisional",
                "legacy_status": "provisional_legacy",
                "schema_version": SCHEMA_VERSION,
                "operation": "legacy_import",
                "admission": {"reasons": ["legacy_requires_v3_revalidation"]},
            }
            before = len(self.memory.principles)
            self.memory.add(item)
            counters["imported" if len(self.memory.principles) > before else "already_present"] += 1
        return counters

    # ================================================================
    # 检索
    # ================================================================

    def retrieve(
        self,
        features: Dict[str, Any],
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """三维门控检索。

        Args:
            features: {core_skill, context_constraints, task_objective}
            top_k: 最大返回数
            threshold: 三维都必须 >= 此值才通过门控

        Returns:
            [{memory_id, item, dimension_similarities, score}, ...]
        """
        return self.memory.retrieve(features, top_k=top_k, threshold=threshold)

    def retrieve_single_vector(
        self,
        description_text: str,
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Single-vector retrieval: holistic description, KNN search.

        Args:
            description_text: LLM-generated holistic task description of the query.
        """
        return self.memory.retrieve_single_vector(description_text, top_k=top_k, threshold=threshold)

    def retrieve_high_recall(
        self,
        features: Dict[str, Any],
        *,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Threshold-free candidate generation for cross-encoder reranking."""
        return self.memory.retrieve_high_recall(features, top_k=top_k)

    def retrieve_strict_semantic_matches(
        self,
        features: Dict[str, Any],
        *,
        facet_threshold: float = 0.95,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Near-identical DSD matches used for replacement/merge decisions."""
        return self.memory.retrieve_strict_semantic_matches(
            features,
            facet_threshold=facet_threshold,
            top_k=top_k,
        )

    # ================================================================
    # 索引
    # ================================================================

    def build_indices(self) -> None:
        """(重新)构建 FAISS 三维向量索引。"""
        self.memory.build_indices()

    def save_indices(self) -> None:
        """保存 FAISS 索引到磁盘。"""
        self.memory.save_indices()

    # ================================================================
    # 检查
    # ================================================================

    @property
    def is_indexed(self) -> bool:
        """是否已构建向量索引。"""
        return self.memory._core_embeddings is not None

    @property
    def principle_count(self) -> int:
        return len(self.memory.principles)

    # ================================================================
    # Internal
    # ================================================================

    @staticmethod
    def _extract_from_learning_packet(
        packet: Dict[str, Any],
        round_id: int,
        dataset: str,
    ) -> Optional[Dict[str, Any]]:
        """从 learning packet 提取 principle_item。"""
        question = _clean(packet.get("question", ""))
        if not question:
            return None

        metadata = packet.get("metadata", {})
        stage2_candidate = bool(packet.get("stage2_candidate", metadata.get("stage2_candidate", False)))
        if not stage2_candidate:
            contrast = packet.get("contrast", {}) if isinstance(packet.get("contrast", {}), dict) else {}
            diff_ok = contrast.get("status") == "success"
            has_pair = all([
                question,
                _clean(packet.get("chosen_trajectory", "")) or _clean(packet.get("chosen_path", "")),
                _clean(packet.get("hard_negative_trajectory", "")) or _clean(packet.get("rejected_path", "")),
            ])
            stage2_candidate = diff_ok and has_pair
        if not stage2_candidate:
            return None

        semantics = packet.get("problem_semantics", {})
        core_skill = get_field(semantics, CORE_FIELD)
        ctx_constraints = get_field(semantics, CONDITION_FIELD)
        task_obj = get_field(semantics, TARGET_FIELD)

        if not (core_skill or ctx_constraints or task_obj):
            return None

        # 提取 principle: 从 contrast.differences 中拼接 CorrectThinking
        contrast = packet.get("contrast", {})
        principle = _clean(contrast.get("principle", ""))
        if not principle:
            differences = contrast.get("differences", [])
            parts = [d.get("CorrectThinking", "") for d in differences if d.get("CorrectThinking")]
            principle = "; ".join(parts)

        if not principle:
            return None

        return {
            "question": question,
            "core_skill": core_skill,
            "context_constraints": ctx_constraints,
            "task_objective": task_obj,
            "principle": principle,
            "source_round": round_id,
            "source_dataset": dataset,
            "quality_metrics": {},
        }

    @staticmethod
    def _extract_from_principle_record(
        record: Dict[str, Any],
        round_id: int,
        dataset: str,
    ) -> Optional[Dict[str, Any]]:
        """从 principle DPO record (旧流程) 提取 principle_item。"""
        import re

        # 从 input 字段提取 question
        input_text = _clean(record.get("input", "")) or _clean(record.get("instruction", ""))
        question = ""
        for pattern in [
            r"\[Question\]:\s*\n?(.*?)(?=\n\n\[|$)",
            r"\[Question\]:\s*(.+?)(?:\n\n|$)",
        ]:
            m = re.search(pattern, input_text, re.DOTALL)
            if m:
                question = m.group(1).strip()
                break

        if not question:
            return None

        core_skill = get_field(record, CORE_FIELD)
        ctx_constraints = get_field(record, CONDITION_FIELD)
        task_obj = get_field(record, TARGET_FIELD)
        principle = _clean(record.get("chosen", ""))

        if not (core_skill or ctx_constraints or task_obj):
            return None
        if not principle:
            return None

        return {
            "question": question,
            "core_skill": core_skill,
            "context_constraints": ctx_constraints,
            "task_objective": task_obj,
            "principle": principle,
            "source_round": round_id,
            "source_dataset": dataset,
            "quality_metrics": {},
        }

    @staticmethod
    def _extract_from_validated_principle(
        record: Dict[str, Any],
        round_id: int,
        dataset: str,
    ) -> Optional[Dict[str, Any]]:
        """从 Stage 2 validated principle record 提取 principle_item。"""
        question = _clean(record.get("source_question_text", record.get("question", "")))
        if not question:
            return None

        source_dsd = record.get("source_dsd", {}) if isinstance(record.get("source_dsd", {}), dict) else {}
        core_skill = get_field(source_dsd, CORE_FIELD)
        ctx_constraints = get_field(source_dsd, CONDITION_FIELD)
        task_obj = get_field(source_dsd, TARGET_FIELD)

        raw_content = record.get("content") if isinstance(record.get("content"), dict) else {
            "introspection": record.get("introspection"),
            "trigger_condition": record.get("trigger_condition"),
            "required_evidence": record.get("required_evidence"),
            "exclusion_condition": record.get("exclusion_condition"),
            "action_guidance": record.get("action_guidance"),
            "verification_step": record.get("verification_step"),
            "generalization_level": record.get("generalization_level"),
            "scope": record.get("scope"),
            "failure_mode": record.get("failure_mode"),
        }
        content, schema_errors = validate_principle_content(raw_content, require_extended=True)
        if schema_errors:
            logger.warning(
                "Skipping malformed validated principle %s: %s",
                record.get("source_question_id", "unknown"), schema_errors,
            )
            return None
        principle = content["action_guidance"]

        neighbor_cases = record.get("neighbor_cases", record.get("validated_neighbors", []))
        if not isinstance(neighbor_cases, list):
            neighbor_cases = []
        neighbor_pass_count = int(
            record.get("neighbor_pass_count", sum(1 for case in neighbor_cases if isinstance(case, dict) and case.get("passed")))
            or 0
        )
        neighbor_total_count = int(record.get("neighbor_total_count", len(neighbor_cases)) or 0)
        neighbor_pass_ratio = (
            float(record.get("neighbor_pass_ratio"))
            if record.get("neighbor_pass_ratio") is not None
            else neighbor_pass_count / max(1, neighbor_total_count)
        )

        quality_metrics = {
            "validation_protocol": "paired_neighbor_validation",
            "neighbor_pass_count": neighbor_pass_count,
            "neighbor_total_count": neighbor_total_count,
            "neighbor_pass_ratio": neighbor_pass_ratio,
            "neighbor_gate_pass": bool(record.get("neighbor_gate_pass", False)),
            "qualified_for_memory": bool(record.get("qualified_for_memory", False)),
            "rank_score": float(record.get("rank_score", 0.0) or 0.0),
            "holdout_net_gain": int((record.get("holdout_summary") or {}).get("net_gain", 0) or 0),
            "holdout_gain_count": int((record.get("holdout_summary") or {}).get("gain_count", 0) or 0),
            "holdout_loss_count": int((record.get("holdout_summary") or {}).get("loss_count", 0) or 0),
            "historical_gain_rate": float((record.get("holdout_summary") or {}).get("historical_gain_rate", 0.0) or 0.0),
        }

        validation_view = record.get("validation_view", {}) if isinstance(record.get("validation_view"), dict) else {}
        audit_view = record.get("audit_view", {}) if isinstance(record.get("audit_view"), dict) else {}
        views = build_memory_views(
            content,
            source_dsd=source_dsd,
            validation=validation_view,
            audit=audit_view,
        )

        return {
            "question": question,
            "core_skill": core_skill,
            "context_constraints": ctx_constraints,
            "task_objective": task_obj,
            "principle": principle,
            "principle_text": principle,
            "content": content,
            **content,
            "source_round": round_id,
            "source_dataset": dataset,
            "source_dsd": source_dsd,
            "validated_neighbors": neighbor_cases,
            "quality_metrics": quality_metrics,
            "status": record.get("status", "active") or "active",
            "schema_version": record.get("schema_version", SCHEMA_VERSION),
            "version_id": record.get("version_id", ""),
            "parent_version_id": record.get("parent_version_id"),
            "operation": record.get("operation", "add"),
            "admission": record.get("admission", {}),
            **views,
        }
