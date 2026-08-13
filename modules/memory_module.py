"""
modules/memory_module.py — MemoryModule 类

Agent 的「记忆」模块，负责：
  - 三维 FAISS 向量索引构建
  - 三维门控检索
  - 按轮次快照
  - 统计信息

内部委托给 modules/memory.py 的 Memory 类。
"""
import logging
from typing import Any, Dict, List

from core.config import Config

logger = logging.getLogger(__name__)


class MemoryModule:
    """Agent 的记忆管理模块。

    使用方式:
        mem = MemoryModule(config, memory_dir="memory")
        mem.load()

        # Stage 2 commits through modules.stage2.factory.
        mem.build_indices()
        mem.save()
        mem.snapshot(round_id=1)

        # Stage 3: 检索
        results = mem.retrieve(features, top_k=5, threshold=0.55)
    """

    def __init__(self, config: Config, memory_dir: str = "memory/store"):
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
