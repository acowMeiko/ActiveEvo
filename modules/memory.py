"""
modules/memory.py — 统一外部记忆管理器 (Memory)

职责：
  - 管理结构化经验 (principle) 的持久化存储
  - 三维语义向量索引构建与检索
  - 相似经验检测与去重决策
  - 按轮次快照保存
  - 增量记忆演化

存储布局：
  memory/
  ├── principles/
  │   └── principles.json      # 所有经验条目
  ├── indices/
  │   ├── core_skill.index
  │   ├── context_constraints.index
  │   └── task_objective.index
  ├── metadata/
  │   ├── memory_data.json     # ids, embeddings, texts
  │   └── stats.json           # 统计
  └── snapshots/
      ├── round_1/
      ├── round_2/
      └── ...
"""
import json
import hashlib
import logging
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.embedding_utils import DEFAULT_EMBEDDING_MODEL, allow_hash_fallback, load_sentence_transformer_with_fallback

logger = logging.getLogger(__name__)

# ── field names ──────────────────────────────────────────────────
CORE_FIELD = "core_skill"
CONDITION_FIELD = "context_constraints"
TARGET_FIELD = "task_objective"

# ── similarity thresholds ────────────────────────────────────────
DUPLICATE_THRESHOLD = 0.9    # sim > this → DEPRECATE (near-duplicate)
UPDATE_THRESHOLD = 0.75      # sim > this → UPDATE_EXISTING
# else → KEEP_NEW


def _get_field(d: dict, field: str) -> str:
    """Get field with backward-compat fallback."""
    _LEGACY = {
        "core_skill": "core_math_concept",
        "context_constraints": "conditions_structure",
        "task_objective": "solution_target",
    }
    val = d.get(field, "")
    if not val:
        val = d.get(_LEGACY.get(field, ""), "")
    return str(val).strip() if val else ""


def _clean(val: Any) -> str:
    return str(val).strip() if val else ""


class Memory:
    """统一的外部记忆管理器。

    每条 principle_item 的结构：
    {
        "memory_id":  str,             # 唯一 ID
        "question":   str,             # 来源问题
        "core_skill": str,             # 三维语义
        "context_constraints": str,
        "task_objective": str,
        "principle":  str,             # 核心经验文本
        "source_round": int,           # 来源轮次 (1/2/3/4)
        "source_dataset": str,         # e.g. "dpo_level1"
        "quality_metrics": dict,       # optional
        "status": "active" | "deprecated" | "linked",
    }
    """

    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.principles: List[Dict[str, Any]] = []
        self._id_set: set = set()
        self.store_version = 0
        self._pending_events: List[Dict[str, Any]] = []
        self._torch_runtime_warmed = False
        # Default to strict mode: retrieval must use a real embedding model.
        self._force_embedding = os.environ.get("MEMUA_FORCE_EMBEDDING", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }

        # Embedding caches (populated by build_indices)
        self._embedding_model = None
        self._core_embeddings: Optional[np.ndarray] = None
        self._condition_embeddings: Optional[np.ndarray] = None
        self._target_embeddings: Optional[np.ndarray] = None
        self._faiss_core = None
        self._faiss_condition = None
        self._faiss_target = None
        # Single-vector ablation: holistic embedding (no 3D decomposition)
        self._single_embeddings: Optional[np.ndarray] = None
        self._faiss_single = None

    # ================================================================
    # Persistence
    # ================================================================

    def load(self, memory_dir: Optional[str] = None) -> None:
        """从磁盘加载已有记忆。"""
        d = Path(memory_dir) if memory_dir else self.memory_dir
        self.memory_dir = d
        principles_file = d / "principles" / "principles.json"

        if principles_file.exists():
            with open(principles_file, "r", encoding="utf-8") as f:
                self.principles = json.load(f)
            self._id_set = {p["memory_id"] for p in self.principles if "memory_id" in p}
            logger.info("Memory loaded: %d principles from %s", len(self.principles), principles_file)
        else:
            self.principles = []
            self._id_set = set()
            logger.info("No existing memory at %s, starting fresh", d)

        manifest_file = d / "metadata" / "store_manifest.json"
        if manifest_file.exists():
            try:
                with manifest_file.open("r", encoding="utf-8") as handle:
                    self.store_version = int(json.load(handle).get("store_version", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.store_version = 0
        for item in self.principles:
            item.setdefault("item_version", 1)
            item.setdefault("lineage", {"parent_memory_ids": [], "supersedes": []})

        # Warm torch-related runtime before importing faiss.
        self._warm_torch_runtime()

        # Try loading FAISS indices
        self._load_indices(d)

    def save(self, memory_dir: Optional[str] = None) -> None:
        """保存记忆到磁盘。"""
        d = Path(memory_dir) if memory_dir else self.memory_dir
        principles_dir = d / "principles"
        principles_dir.mkdir(parents=True, exist_ok=True)

        principles_path = principles_dir / "principles.json"
        principles_temp = principles_path.with_suffix(".json.tmp")
        with open(principles_temp, "w", encoding="utf-8") as f:
            json.dump(self.principles, f, ensure_ascii=False, indent=2)
        os.replace(principles_temp, principles_path)

        # Save stats
        metadata_dir = d / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        stats = self.stats()
        with open(metadata_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        if self._pending_events:
            self.store_version += 1
            committed_at = datetime.now(timezone.utc).isoformat()
            events_file = metadata_dir / "memory_events.jsonl"
            with events_file.open("a", encoding="utf-8") as handle:
                for event in self._pending_events:
                    event["store_version"] = self.store_version
                    event["committed_at"] = committed_at
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._pending_events.clear()
        manifest = {
            "store_version": self.store_version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "active_count": sum(1 for item in self.principles if item.get("status") == "active"),
            "item_count": len(self.principles),
        }
        with (metadata_dir / "store_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        self._export_readable_views(d)

        logger.info("Memory saved: %d principles to %s", len(self.principles), d)

    def snapshot(self, round_id: int, memory_dir: Optional[str] = None) -> None:
        """按轮次保存快照到 snapshots/round_{id}/"""
        d = Path(memory_dir) if memory_dir else self.memory_dir
        snapshot_dir = d / "snapshots" / f"round_{round_id}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save principles snapshot
        principles_dir = snapshot_dir / "principles"
        principles_dir.mkdir(parents=True, exist_ok=True)
        with open(principles_dir / "principles.json", "w", encoding="utf-8") as f:
            json.dump(self.principles, f, ensure_ascii=False, indent=2)

        # Copy indices if they exist
        indices_src = d / "indices"
        if indices_src.exists():
            indices_dst = snapshot_dir / "indices"
            if indices_dst.exists():
                shutil.rmtree(indices_dst)
            shutil.copytree(indices_src, indices_dst)

        # Save metadata
        metadata_dir = snapshot_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(self.stats(), f, ensure_ascii=False, indent=2)

        logger.info("Snapshot saved: round_%d (%d principles) → %s", round_id, len(self.principles), snapshot_dir)

    # ================================================================
    # Memory Write Operations
    # ================================================================

    def _generate_id(self, item: Dict[str, Any]) -> str:
        """生成唯一 memory_id。"""
        content = f"{item.get('question', '')}|{item.get('principle', '')}|{item.get('source_round', '')}"
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"mem_{item.get('source_dataset', 'unknown')}_{h}"

    def _record_event(
        self,
        operation: str,
        memory_id: str,
        *,
        item_version: int,
        changed_fields: Optional[List[str]] = None,
        reason: str = "",
    ) -> None:
        self._pending_events.append({
            "operation": operation,
            "memory_id": memory_id,
            "item_version": item_version,
            "changed_fields": changed_fields or [],
            "reason": reason,
        })

    def _export_readable_views(self, root: Path) -> None:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        with (exports / "memory_items.jsonl").open("w", encoding="utf-8") as handle:
            for item in self.principles:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (exports / "memory_cards.md").open("w", encoding="utf-8") as handle:
            handle.write(f"# Memory Cards (store version {self.store_version})\n\n")
            for item in self.principles:
                content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
                handle.write(
                    f"## {item.get('memory_id', 'unknown')} · v{item.get('item_version', 1)} · {item.get('status', 'active')}\n\n"
                    f"- Source question: {item.get('question', '')}\n"
                    f"- Trigger: {content.get('trigger_condition', item.get('trigger_condition', ''))}\n"
                    f"- Action: {content.get('action_guidance', item.get('principle', ''))}\n"
                    f"- Scope: {content.get('scope', item.get('scope', ''))}\n"
                    f"- Failure mode: {content.get('failure_mode', item.get('failure_mode', ''))}\n"
                    f"- Introspection: {content.get('introspection', item.get('introspection', ''))}\n\n"
                )

    def add(self, principle_item: Dict[str, Any]) -> str:
        """添加一条新的结构化经验，返回 memory_id。"""
        if "memory_id" not in principle_item:
            principle_item["memory_id"] = self._generate_id(principle_item)
        if "status" not in principle_item:
            principle_item["status"] = "active"
        principle_item.setdefault("item_version", 1)
        principle_item.setdefault("schema_version", "3.0")
        if not principle_item.get("version_id"):
            principle_item["version_id"] = f"{principle_item['memory_id']}_v{principle_item['item_version']}"
        principle_item.setdefault("parent_version_id", None)
        principle_item.setdefault("operation", "add")
        principle_item.setdefault("version_history", [])
        principle_item.setdefault("lineage", {"parent_memory_ids": [], "supersedes": []})

        mid = principle_item["memory_id"]
        if mid in self._id_set:
            # Avoid true duplicate IDs
            logger.debug("Skipping duplicate memory_id: %s", mid)
            return mid

        self.principles.append(principle_item)
        self._id_set.add(mid)
        self._record_event("ADD", mid, item_version=int(principle_item["item_version"]))
        return mid

    def update(self, memory_id: str, updated_fields: Dict[str, Any]) -> None:
        """更新已有经验的指定字段。"""
        for p in self.principles:
            if p.get("memory_id") == memory_id:
                previous_version = int(p.get("item_version", 1) or 1)
                previous_snapshot = deepcopy({key: value for key, value in p.items() if key != "version_history"})
                previous_snapshot["status"] = "deprecated"
                operation = str(updated_fields.get("operation", "UPDATE")).upper()
                reason = str(updated_fields.get("update_reason", ""))
                persisted_fields = {
                    key: value for key, value in updated_fields.items()
                    if key not in {"operation", "update_reason"}
                }
                p.update(persisted_fields)
                history = p.setdefault("version_history", [])
                history.append(previous_snapshot)
                p["item_version"] = previous_version + 1
                p["parent_version_id"] = previous_snapshot.get("version_id") or f"{memory_id}_v{previous_version}"
                p["version_id"] = f"{memory_id}_v{p['item_version']}"
                p["schema_version"] = "3.0"
                p["operation"] = operation.lower()
                self._record_event(
                    operation,
                    memory_id,
                    item_version=p["item_version"],
                    changed_fields=sorted(persisted_fields),
                    reason=reason,
                )
                return
        logger.warning("Memory ID not found for update: %s", memory_id)

    def replace_version(self, memory_id: str, replacement: Dict[str, Any], *, reason: str = "") -> None:
        """Append an auditable replacement version while keeping a stable ID."""
        payload = deepcopy(replacement)
        payload.pop("memory_id", None)
        payload["status"] = payload.get("status", "active")
        payload["operation"] = "REPLACE"
        payload["update_reason"] = reason
        self.update(memory_id, payload)

    def deprecate(self, memory_id: str) -> None:
        """将经验标记为 deprecated。"""
        self.update(memory_id, {"status": "deprecated", "operation": "DEPRECATE"})

    def detect_similar(
        self,
        principle_item: Dict[str, Any],
        threshold: float = UPDATE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """检测与已有经验的相似项。

        Returns:
            List of {"memory_id": str, "similarity": float, "item": dict}
            按相似度降序。
        """
        if self._core_embeddings is None or len(self.principles) == 0:
            return []
        if not self._ensure_index_alignment():
            return []

        query_core, query_cond, query_target = self._encode_item(principle_item)
        if query_core is None:
            return []

        # Compute cosine similarities (embeddings are already L2-normalized)
        core_sims = self._core_embeddings @ query_core
        cond_sims = self._condition_embeddings @ query_cond
        target_sims = self._target_embeddings @ query_target

        # Average similarity across three facets
        avg_sims = (core_sims + cond_sims + target_sims) / 3.0

        results = []
        active_principles = [p for p in self.principles if p.get("status") == "active"]
        for i, sim in enumerate(avg_sims):
            if i >= len(active_principles):
                break
            if float(sim) >= threshold:
                results.append({
                    "memory_id": active_principles[i].get("memory_id", ""),
                    "similarity": float(sim),
                    "item": active_principles[i],
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def decide_action(
        self,
        new_item: Dict[str, Any],
        similar_items: List[Dict[str, Any]],
    ) -> str:
        """决策: KEEP_NEW / UPDATE_EXISTING / DEPRECATE

        - 最相似项 sim > 0.9: DEPRECATE (near-duplicate, don't add)
        - 最相似项 sim > 0.75: UPDATE_EXISTING (merge/update existing)
        - else: KEEP_NEW (add as new entry)
        """
        if not similar_items:
            return "KEEP_NEW"

        top_sim = similar_items[0]["similarity"]
        if top_sim > DUPLICATE_THRESHOLD:
            return "DEPRECATE"
        if top_sim > UPDATE_THRESHOLD:
            return "UPDATE_EXISTING"
        return "KEEP_NEW"

    def ingest_principle(
        self,
        principle_item: Dict[str, Any],
        auto_dedup: bool = True,
    ) -> Tuple[str, str]:
        """高级接口：检测相似 → 决策 → 执行。

        Returns:
            (memory_id, action): action ∈ {"KEEP_NEW", "UPDATE_EXISTING", "DEPRECATE"}
        """
        if auto_dedup and self._core_embeddings is not None:
            similar = self.detect_similar(principle_item)
            action = self.decide_action(principle_item, similar)
        else:
            action = "KEEP_NEW"

        if action == "KEEP_NEW":
            before_count = len(self.principles)
            mid = self.add(principle_item)
            if len(self.principles) > before_count:
                if self._core_embeddings is None:
                    self.build_indices()
                else:
                    self._append_to_indices(principle_item)
        elif action == "UPDATE_EXISTING":
            # Update the most similar item with new principle text (if longer/better)
            target_id = similar[0]["memory_id"]
            existing = similar[0]["item"]
            # Keep the longer principle text
            if len(principle_item.get("principle", "")) > len(existing.get("principle", "")):
                self.update(target_id, {
                    "principle": principle_item.get("principle", ""),
                    "quality_metrics": principle_item.get("quality_metrics", {}),
                })
            mid = target_id
        else:  # DEPRECATE
            mid = similar[0]["memory_id"]
            logger.debug("Principle too similar to %s (sim=%.3f), skipping", mid, similar[0]["similarity"])

        return mid, action

    # ================================================================
    # Retrieval
    # ================================================================

    def retrieve(
        self,
        query_features: Dict[str, Any],
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Default retrieval entrypoint: reasoning-aligned experience association."""
        return self.retrieve_reasoning_aligned(
            query_features=query_features,
            top_k=top_k,
            threshold=threshold,
        )

    def retrieve_hard_gated(
        self,
        query_features: Dict[str, Any],
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """基于三维语义检索相似经验。

        Args:
            query_features: {core_skill, context_constraints, task_objective}
            top_k: 最大返回数
            threshold: 三维都必须 >= 此值才通过门控

        Returns:
            所有通过门控的经验，按综合相关度排序。
            每项: {memory_id, item, dimension_similarities, score}
        """
        if self._core_embeddings is None or len(self.principles) == 0:
            return []
        if not self._ensure_index_alignment():
            return []

        query_core, query_cond, query_target = self._encode_features(query_features)
        if query_core is None:
            return []

        active_principles = [p for p in self.principles if p.get("status") == "active"]
        if not active_principles:
            return []

        # Compute per-dimension similarities
        core_sims = self._core_embeddings @ query_core
        cond_sims = self._condition_embeddings @ query_cond
        target_sims = self._target_embeddings @ query_target

        results = []
        for j, item in enumerate(active_principles):
            cs = float(core_sims[j])
            co = float(cond_sims[j])
            ts = float(target_sims[j])

            # Hard gate: all three dimensions must pass
            if cs >= threshold and co >= threshold and ts >= threshold:
                results.append({
                    "memory_id": item.get("memory_id", ""),
                    "item": item,
                    "dimension_similarities": {
                        CORE_FIELD: cs,
                        CONDITION_FIELD: co,
                        TARGET_FIELD: ts,
                    },
                    "score": (cs + co + ts) / 3.0,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve_strict_semantic_matches(
        self,
        query_features: Dict[str, Any],
        *,
        facet_threshold: float = 0.95,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return only near-identical semantic matches in active memory.

        This is used by Stage 2 memory reconciliation after a principle has
        already passed source + neighbor gates. A candidate is considered for
        replacement/merge only when all three semantic facets are above the
        provided threshold.
        """
        if self._core_embeddings is None or len(self.principles) == 0:
            return []
        if not self._ensure_index_alignment():
            return []

        query_core, query_cond, query_target = self._encode_features(query_features)
        if query_core is None:
            return []

        active_principles = [p for p in self.principles if p.get("status") == "active"]
        if not active_principles:
            return []

        core_sims = self._core_embeddings @ query_core
        cond_sims = self._condition_embeddings @ query_cond
        target_sims = self._target_embeddings @ query_target

        results = []
        for j, item in enumerate(active_principles):
            cs = float(core_sims[j])
            co = float(cond_sims[j])
            ts = float(target_sims[j])
            if cs >= facet_threshold and co >= facet_threshold and ts >= facet_threshold:
                results.append({
                    "memory_id": item.get("memory_id", ""),
                    "item": item,
                    "dimension_similarities": {
                        CORE_FIELD: cs,
                        CONDITION_FIELD: co,
                        TARGET_FIELD: ts,
                    },
                    "score": (cs + co + ts) / 3.0,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve_reasoning_aligned(
        self,
        query_features: Dict[str, Any],
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories through multi-facet recall and quality-aware reranking."""
        if self._core_embeddings is None or len(self.principles) == 0:
            return []
        if not self._ensure_index_alignment():
            return []

        query_core, query_cond, query_target = self._encode_features(query_features)
        if query_core is None:
            return []

        active_principles = [p for p in self.principles if p.get("status") == "active"]
        if not active_principles:
            return []

        core_sims = self._core_embeddings @ query_core
        cond_sims = self._condition_embeddings @ query_cond
        target_sims = self._target_embeddings @ query_target

        default_candidate_pool = max(top_k * 6, 24)
        candidate_pool_size = min(
            len(active_principles),
            max(
                top_k,
                int(os.environ.get("MEMUA_RETRIEVAL_CANDIDATE_POOL", str(default_candidate_pool))),
            ),
        )
        candidate_sources = self._collect_candidate_sources(
            query_core=query_core,
            query_cond=query_cond,
            query_target=query_target,
            pool_size=candidate_pool_size,
        )
        if not candidate_sources:
            candidate_sources = {i: [] for i in range(len(active_principles))}

        min_supported_facets = max(
            1,
            min(3, int(os.environ.get("MEMUA_RETRIEVAL_MIN_SUPPORTED_FACETS", "2"))),
        )
        weak_facet_floor = max(
            0.0,
            threshold - float(os.environ.get("MEMUA_RETRIEVAL_WEAK_FACET_MARGIN", "0.18")),
        )
        structure_floor = max(
            0.0,
            threshold - float(os.environ.get("MEMUA_RETRIEVAL_STRUCTURE_MARGIN", "0.05")),
        )
        structure_weight = float(os.environ.get("MEMUA_RETRIEVAL_STRUCTURE_WEIGHT", "0.72"))
        quality_weight = float(os.environ.get("MEMUA_RETRIEVAL_QUALITY_WEIGHT", "0.28"))
        total_weight = structure_weight + quality_weight
        if total_weight <= 0:
            structure_weight, quality_weight = 0.72, 0.28
            total_weight = 1.0
        structure_weight /= total_weight
        quality_weight /= total_weight

        results = []
        for j in sorted(candidate_sources.keys()):
            item = active_principles[j]
            cs = float(core_sims[j])
            co = float(cond_sims[j])
            ts = float(target_sims[j])
            facet_scores = {
                CORE_FIELD: cs,
                CONDITION_FIELD: co,
                TARGET_FIELD: ts,
            }
            matched_facets = [name for name, value in facet_scores.items() if value >= threshold]
            weakest_facet = min(facet_scores, key=facet_scores.get)
            weakest_value = facet_scores[weakest_facet]
            structure_score = self._compute_structure_score(facet_scores, threshold)
            quality_score = self._compute_quality_score(item)
            relevance_score = (cs + co + ts) / 3.0
            final_score = structure_weight * structure_score + quality_weight * quality_score

            if len(matched_facets) < min_supported_facets:
                continue
            if weakest_value < weak_facet_floor:
                continue
            if structure_score < structure_floor:
                continue

            results.append({
                "memory_id": item.get("memory_id", ""),
                "item": item,
                "dimension_similarities": facet_scores,
                "matched_facets": matched_facets,
                "candidate_facets": candidate_sources.get(j, []),
                "supported_facet_count": len(matched_facets),
                "candidate_source_count": len(candidate_sources.get(j, [])),
                "weakest_facet": weakest_facet,
                "structure_score": structure_score,
                "quality_score": quality_score,
                "relevance_score": relevance_score,
                "score": final_score,
            })

        results.sort(
            key=lambda x: (x["score"], x["structure_score"], x["relevance_score"]),
            reverse=True,
        )
        return results[:top_k]

    def retrieve_single_vector(
        self,
        description_text: str,
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Single-vector retrieval: encode holistic description, KNN search.

        DSD ablation — instead of decomposing into 3 facets with gating,
        encode one holistic task description as a single embedding.

        Args:
            description_text: LLM-generated holistic description of the query problem.
            top_k: max results to return.
            threshold: minimum cosine similarity.
        """
        if self._single_embeddings is None or self._faiss_single is None:
            if self._core_embeddings is not None:
                self._build_single_index()
            else:
                return []
        if not self._ensure_index_alignment():
            return []

        model = self._get_embedding_model()
        query_vec = self._encode_texts(model, [description_text])
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)

        active_principles = [p for p in self.principles if p.get("status") == "active"]
        if not active_principles:
            return []

        # Cosine similarity via inner product (embeddings are L2-normalized)
        scores = (self._single_embeddings @ query_vec.T).flatten()

        results = []
        for j in range(len(active_principles)):
            score = float(scores[j])
            if score >= threshold:
                results.append({
                    "memory_id": active_principles[j].get("memory_id", ""),
                    "item": active_principles[j],
                    "score": score,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve_high_recall(
        self,
        features: Dict[str, Any],
        *,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return a threshold-free union of the strongest three-facet matches.

        This is the candidate-generation stage for a downstream cross encoder.
        It deliberately optimizes recall: hard trigger/scope decisions belong to
        the reranker, not to cosine thresholds over decomposed text.
        """
        if not self._ensure_index_alignment():
            return []
        active = [item for item in self.principles if item.get("status") == "active"]
        if not active:
            return []
        model = self._get_embedding_model()
        query_core = self._encode_texts(model, [_get_field(features, CORE_FIELD) or "unknown"])[0]
        query_cond = self._encode_texts(model, [_get_field(features, CONDITION_FIELD) or "unknown"])[0]
        query_target = self._encode_texts(model, [_get_field(features, TARGET_FIELD) or "unknown"])[0]
        core_scores = self._core_embeddings @ query_core
        condition_scores = self._condition_embeddings @ query_cond
        target_scores = self._target_embeddings @ query_target

        pool_per_facet = min(len(active), max(top_k * 3, 30))
        candidate_indices = set()
        for scores in (core_scores, condition_scores, target_scores):
            candidate_indices.update(np.argsort(scores)[-pool_per_facet:].tolist())

        results = []
        for idx in candidate_indices:
            facets = {
                CORE_FIELD: float(core_scores[idx]),
                CONDITION_FIELD: float(condition_scores[idx]),
                TARGET_FIELD: float(target_scores[idx]),
            }
            ordered = sorted(facets.values(), reverse=True)
            # Two-facet mean rewards structural agreement while the union keeps
            # a candidate alive when one semantic decomposition is imperfect.
            score = float(sum(ordered[:2]) / 2.0)
            results.append({
                "memory_id": active[idx].get("memory_id", ""),
                "item": active[idx],
                "dimension_similarities": facets,
                "retrieval_score": score,
                "score": score,
                "retrieval_method": "three_facet_high_recall_union",
            })
        results.sort(key=lambda entry: entry["retrieval_score"], reverse=True)
        return results[:top_k]

    def get_new_principles(self, since_round: int) -> List[Dict[str, Any]]:
        """获取指定轮次之后新增的经验。"""
        return [
            p for p in self.principles
            if p.get("status") == "active" and p.get("source_round", 0) > since_round
        ]

    # ================================================================
    # Index Building
    # ================================================================

    def _build_single_index(self) -> None:
        """Build only the single-vector FAISS index from existing principles.

        Used for lazy initialization when the DSD indices exist but the
        single-vector index was never built (e.g., memory created before
        the single-vector ablation code was deployed).
        """
        active = [p for p in self.principles if p.get("status") == "active"]
        if not active:
            logger.warning("No active principles for single-vector index")
            return

        model = self._get_embedding_model()
        single_texts = [
            p.get("principle", "") or f"{p.get('question', '')}" for p in active
        ]
        self._single_embeddings = self._encode_texts(model, single_texts)

        import faiss
        dim = self._single_embeddings.shape[1]
        self._faiss_single = faiss.IndexFlatIP(dim)
        self._faiss_single.add(np.ascontiguousarray(self._single_embeddings.astype(np.float32)))
        self._index_alignment_checked = True
        logger.info("Single-vector FAISS index lazily built: %d vectors × %d dims", len(active), dim)

    def build_indices(self) -> None:
        """(重新)构建 FAISS 三维向量索引。"""
        active = [p for p in self.principles if p.get("status") == "active"]
        if not active:
            logger.warning("No active principles to index")
            return

        model = self._get_embedding_model()

        core_texts = [_get_field(p, CORE_FIELD) or "unknown" for p in active]
        cond_texts = [_get_field(p, CONDITION_FIELD) or "unknown" for p in active]
        target_texts = [_get_field(p, TARGET_FIELD) or "unknown" for p in active]

        logger.info("Encoding %d principles for three-facet indexing…", len(active))
        self._core_embeddings = self._encode_texts(model, core_texts)
        self._condition_embeddings = self._encode_texts(model, cond_texts)
        self._target_embeddings = self._encode_texts(model, target_texts)

        # Single-vector ablation: holistic description (principle text)
        single_texts = [
            p.get("principle", "") or f"{p.get('question', '')}" for p in active
        ]
        logger.info("Encoding %d principles for single-vector indexing…", len(active))
        self._single_embeddings = self._encode_texts(model, single_texts)

        self._build_faiss_indices()
        logger.info("FAISS indices built: %d vectors × %d dims", len(active), self._core_embeddings.shape[1])

    def save_indices(self, memory_dir: Optional[str] = None) -> None:
        """保存 FAISS 索引和元数据到磁盘。"""
        d = Path(memory_dir) if memory_dir else self.memory_dir

        if self._faiss_core is None:
            logger.warning("No indices to save — call build_indices() first")
            return

        import faiss

        indices_dir = d / "indices"
        indices_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_core, str(indices_dir / "core_skill.index"))
        faiss.write_index(self._faiss_condition, str(indices_dir / "context_constraints.index"))
        faiss.write_index(self._faiss_target, str(indices_dir / "task_objective.index"))
        if self._faiss_single is not None:
            faiss.write_index(self._faiss_single, str(indices_dir / "single_vector.index"))

        # Save metadata for compatibility with retrieve_principles.py
        metadata_dir = d / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        active = [p for p in self.principles if p.get("status") == "active"]
        memory_data = {
            "ids": [p.get("memory_id", f"mem_{i}") for i, p in enumerate(active)],
            "metadatas": [
                {
                    "question": p.get("question", ""),
                    CORE_FIELD: _get_field(p, CORE_FIELD),
                    CONDITION_FIELD: _get_field(p, CONDITION_FIELD),
                    TARGET_FIELD: _get_field(p, TARGET_FIELD),
                    "principle": p.get("principle", ""),
                    "source": p.get("source_dataset", ""),
                    "source_round": p.get("source_round", 0),
                    "source_dsd": p.get("source_dsd", {}),
                    "validated_neighbors": p.get("validated_neighbors", []),
                    "quality_metrics": p.get("quality_metrics", {}),
                }
                for p in active
            ],
            "concept_embeddings": self._core_embeddings.tolist(),
            "condition_embeddings": self._condition_embeddings.tolist(),
            "target_embeddings": self._target_embeddings.tolist(),
            "concept_texts": [_get_field(p, CORE_FIELD) for p in active],
            "condition_texts": [_get_field(p, CONDITION_FIELD) for p in active],
            "target_texts": [_get_field(p, TARGET_FIELD) for p in active],
        }
        with open(metadata_dir / "memory_data.json", "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

        logger.info("Indices saved to %s", indices_dir)

    def _load_indices(self, memory_dir: Path) -> None:
        """尝试从磁盘加载预建索引。"""
        indices_dir = memory_dir / "indices"
        core_path = indices_dir / "core_skill.index"
        if not core_path.exists():
            return

        try:
            import faiss
            self._faiss_core = faiss.read_index(str(core_path))
            self._faiss_condition = faiss.read_index(str(indices_dir / "context_constraints.index"))
            self._faiss_target = faiss.read_index(str(indices_dir / "task_objective.index"))

            # Reconstruct embeddings from FAISS for similarity computation
            n = self._faiss_core.ntotal
            dim = self._faiss_core.d
            self._core_embeddings = np.zeros((n, dim), dtype=np.float32)
            self._condition_embeddings = np.zeros((n, dim), dtype=np.float32)
            self._target_embeddings = np.zeros((n, dim), dtype=np.float32)
            for i in range(n):
                self._core_embeddings[i] = self._faiss_core.reconstruct(i)
                self._condition_embeddings[i] = self._faiss_condition.reconstruct(i)
                self._target_embeddings[i] = self._faiss_target.reconstruct(i)

            # Load single-vector index (optional — back compat with older indices)
            single_path = indices_dir / "single_vector.index"
            if single_path.exists():
                self._faiss_single = faiss.read_index(str(single_path))
                sn = self._faiss_single.ntotal
                sd = self._faiss_single.d
                self._single_embeddings = np.zeros((sn, sd), dtype=np.float32)
                for i in range(sn):
                    self._single_embeddings[i] = self._faiss_single.reconstruct(i)
                logger.info("Loaded single-vector index: %d vectors × %d dims", sn, sd)

            logger.info("Loaded FAISS indices: %d vectors × %d dims", n, dim)
        except Exception as e:
            logger.warning("Failed to load FAISS indices: %s", e)

    # ================================================================
    # Stats
    # ================================================================

    def stats(self) -> Dict[str, Any]:
        """返回记忆库统计。"""
        active = [p for p in self.principles if p.get("status") == "active"]
        deprecated = [p for p in self.principles if p.get("status") == "deprecated"]

        by_round: Dict[int, int] = {}
        for p in active:
            r = p.get("source_round", 0)
            by_round[r] = by_round.get(r, 0) + 1

        return {
            "total": len(self.principles),
            "active": len(active),
            "deprecated": len(deprecated),
            "by_round": by_round,
            "indexed": self._core_embeddings is not None,
            "index_size": self._faiss_core.ntotal if self._faiss_core else 0,
        }

    # ================================================================
    # Internal Helpers
    # ================================================================

    def _get_embedding_model(self):
        """Lazy-load embedding model (BAAI/bge-m3)."""
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            self._warm_torch_runtime()
        except Exception as exc:
            if self._force_embedding:
                raise RuntimeError(
                    "Embedding model warmup failed and strict mode is enabled. "
                    "Set MEMUA_FORCE_EMBEDDING=0 only if you intentionally allow hash fallback."
                ) from exc
            logger.warning("Embedding model warmup failed (%s) — using hash fallback", exc)
            self._embedding_model = _HashEmbeddingModel()

        if self._embedding_model is None:
            if self._force_embedding:
                raise RuntimeError(
                    "Embedding model unavailable after warmup in strict mode. "
                    f"Please ensure {DEFAULT_EMBEDDING_MODEL} can be loaded."
                )
            if not allow_hash_fallback():
                raise RuntimeError(
                    "Embedding model unavailable and hash fallback is disabled. "
                    "Set MEMUA_ALLOW_HASH_FALLBACK=1 only for debugging."
                )
            logger.warning("Embedding model unavailable after warmup — using hash fallback")
            self._embedding_model = _HashEmbeddingModel()

        return self._embedding_model

    def _encode_texts(self, model, texts: List[str]) -> np.ndarray:
        """Encode texts and L2-normalize."""
        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
        return embeddings

    def _encode_item(self, item: Dict[str, Any]) -> Tuple:
        """Encode a single principle item's three fields."""
        model = self._get_embedding_model()
        core = _get_field(item, CORE_FIELD) or "unknown"
        cond = _get_field(item, CONDITION_FIELD) or "unknown"
        target = _get_field(item, TARGET_FIELD) or "unknown"
        vecs = self._encode_texts(model, [core, cond, target])
        return vecs[0], vecs[1], vecs[2]

    def _encode_features(self, features: Dict[str, Any]) -> Tuple:
        """Encode query features (same as _encode_item)."""
        return self._encode_item(features)

    def _build_faiss_indices(self) -> None:
        """Build FAISS IndexFlatIP from cached embeddings (3D DSD + single-vector)."""
        import faiss
        dim = self._core_embeddings.shape[1]

        self._faiss_core = faiss.IndexFlatIP(dim)
        self._faiss_condition = faiss.IndexFlatIP(dim)
        self._faiss_target = faiss.IndexFlatIP(dim)

        self._faiss_core.add(self._core_embeddings)
        self._faiss_condition.add(self._condition_embeddings)
        self._faiss_target.add(self._target_embeddings)

        # Single-vector index
        single_dim = self._single_embeddings.shape[1]
        self._faiss_single = faiss.IndexFlatIP(single_dim)
        self._faiss_single.add(self._single_embeddings)

    def _append_to_indices(self, item: Dict[str, Any]) -> None:
        """Append one newly added active principle to cached embeddings and FAISS."""
        if item.get("status") != "active":
            return

        core_vec, cond_vec, target_vec = self._encode_item(item)
        core_vec = np.asarray(core_vec, dtype=np.float32).reshape(1, -1)
        cond_vec = np.asarray(cond_vec, dtype=np.float32).reshape(1, -1)
        target_vec = np.asarray(target_vec, dtype=np.float32).reshape(1, -1)

        self._core_embeddings = np.vstack([self._core_embeddings, core_vec])
        self._condition_embeddings = np.vstack([self._condition_embeddings, cond_vec])
        self._target_embeddings = np.vstack([self._target_embeddings, target_vec])

        if self._faiss_core is None:
            self._build_faiss_indices()
            return

        self._faiss_core.add(core_vec)
        self._faiss_condition.add(cond_vec)
        self._faiss_target.add(target_vec)

        # Single-vector append
        model = self._get_embedding_model()
        single_text = item.get("principle", "") or item.get("question", "")
        single_vec = self._encode_texts(model, [single_text])
        self._single_embeddings = np.vstack([self._single_embeddings, single_vec])
        if self._faiss_single is not None:
            self._faiss_single.add(single_vec)

    def _ensure_index_alignment(self) -> bool:
        """Ensure cached embeddings cover exactly the active principles."""
        active_count = sum(1 for p in self.principles if p.get("status") == "active")
        if active_count == 0:
            return False
        if self._core_embeddings is None:
            self.build_indices()
        elif self._core_embeddings.shape[0] != active_count:
            logger.info(
                "Rebuilding memory indices because active principle count changed: index=%d active=%d",
                self._core_embeddings.shape[0],
                active_count,
            )
            self.build_indices()
        return self._core_embeddings is not None and self._core_embeddings.shape[0] == active_count

    def _collect_candidate_sources(
        self,
        *,
        query_core: np.ndarray,
        query_cond: np.ndarray,
        query_target: np.ndarray,
        pool_size: int,
    ) -> Dict[int, List[str]]:
        """Recall a candidate pool from each semantic facet independently."""
        candidate_sources: Dict[int, List[str]] = {}
        searches = [
            (CORE_FIELD, self._core_embeddings, self._faiss_core, query_core),
            (CONDITION_FIELD, self._condition_embeddings, self._faiss_condition, query_cond),
            (TARGET_FIELD, self._target_embeddings, self._faiss_target, query_target),
        ]
        for facet_name, embeddings, faiss_index, query_vec in searches:
            for idx in self._search_top_indices(embeddings, faiss_index, query_vec, pool_size):
                candidate_sources.setdefault(idx, [])
                if facet_name not in candidate_sources[idx]:
                    candidate_sources[idx].append(facet_name)
        return candidate_sources

    def _search_top_indices(
        self,
        embeddings: np.ndarray,
        faiss_index: Any,
        query_vec: np.ndarray,
        k: int,
    ) -> List[int]:
        """Return top-k candidate indices for one facet."""
        total = int(embeddings.shape[0]) if embeddings is not None else 0
        if total <= 0:
            return []
        k = max(1, min(int(k), total))
        query_arr = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if faiss_index is not None:
            scores, indices = faiss_index.search(query_arr, k)
            return [int(i) for i in indices[0] if int(i) >= 0]

        sims = embeddings @ query_vec
        if total <= k:
            order = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, k - 1)[:k]
            order = top[np.argsort(-sims[top])]
        return [int(i) for i in order]

    def _compute_structure_score(
        self,
        facet_scores: Dict[str, float],
        threshold: float,
    ) -> float:
        """Score how consistently a memory item aligns across facets."""
        values = np.asarray(list(facet_scores.values()), dtype=np.float32)
        facet_min = float(values.min())
        facet_mean = float(values.mean())
        facet_std = float(values.std())
        facet_balance = max(0.0, 1.0 - facet_std)
        support_ratio = sum(v >= threshold for v in values) / max(len(values), 1)
        return (
            0.45 * facet_min
            + 0.25 * facet_mean
            + 0.15 * facet_balance
            + 0.15 * support_ratio
        )

    def _compute_quality_score(self, item: Dict[str, Any]) -> float:
        """Estimate how trustworthy a memory item is for reuse."""
        metrics = item.get("quality_metrics", {}) or {}
        if not isinstance(metrics, dict) or not metrics:
            return 0.5

        overall_ratio = float(metrics.get("overall_pass_ratio", 0.0) or 0.0)
        overall_total = int(metrics.get("overall_total_count", 0) or 0)
        overall_pass = int(metrics.get("overall_pass_count", 0) or 0)
        if overall_ratio <= 0.0 and overall_total > 0:
            overall_ratio = overall_pass / max(overall_total, 1)

        neighbor_total = int(metrics.get("neighbor_total_count", 0) or 0)
        neighbor_pass = int(metrics.get("neighbor_pass_count", 0) or 0)
        neighbor_ratio = neighbor_pass / max(neighbor_total, 1) if neighbor_total > 0 else overall_ratio

        source_pass = 1.0 if bool(metrics.get("source_pass", False)) else 0.0
        qualified = 1.0 if bool(metrics.get("qualified_for_memory", False)) else 0.0
        rank_score = float(metrics.get("rank_score", 0.0) or 0.0)
        rank_score = max(0.0, min(1.0, rank_score))

        available = []
        if overall_ratio > 0.0 or overall_total > 0:
            available.append((overall_ratio, 0.40))
        if neighbor_total > 0:
            available.append((neighbor_ratio, 0.30))
        available.append((source_pass, 0.15))
        available.append((qualified, 0.10))
        available.append((rank_score, 0.05))

        total_weight = sum(weight for _, weight in available)
        if total_weight <= 0:
            return 0.5
        return sum(value * weight for value, weight in available) / total_weight

    def _warm_torch_runtime(self) -> None:
        """Instantiate SentenceTransformer before faiss to avoid loader conflicts."""
        if self._embedding_model is not None:
            self._torch_runtime_warmed = True
            return
        device_pref = os.environ.get("MEMUA_EMBEDDING_DEVICE", "auto").strip().lower()
        if device_pref == "cpu" or device_pref.startswith("cuda"):
            devices = [device_pref]
        else:
            devices = ["cuda", "cpu"] if os.system("nvidia-smi > /dev/null 2>&1") == 0 else ["cpu"]
        cache_folder_text = os.environ.get("MEMUA_HF_CACHE_DIR", "").strip()
        cache_folder = Path(cache_folder_text).expanduser() if cache_folder_text else None
        if cache_folder is not None:
            cache_folder.mkdir(parents=True, exist_ok=True)
        cache_kwargs = {"cache_folder": str(cache_folder)} if cache_folder is not None else {}

        attempts = [
            {
                "name": "local cache only",
                "kwargs": {
                    "local_files_only": True,
                    **cache_kwargs,
                },
            },
            {
                "name": "online download",
                "kwargs": {
                    "local_files_only": False,
                    **cache_kwargs,
                },
            },
        ]

        last_exc: Optional[Exception] = None
        for device in devices:
            for attempt in attempts:
                try:
                    self._embedding_model, loaded_from = load_sentence_transformer_with_fallback(
                        model_name=DEFAULT_EMBEDDING_MODEL,
                        device=device,
                        **attempt["kwargs"],
                    )
                    self._torch_runtime_warmed = True
                    logger.info(
                        "Preloaded embedding model before FAISS operations: %s on %s (%s)",
                        loaded_from,
                        device,
                        attempt["name"],
                    )
                    return
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc).lower()
                    logger.warning(
                        "Failed to preload embedding model (%s, device=%s): %s",
                        attempt["name"],
                        device,
                        exc,
                    )
                    # CUDA OOM → skip remaining attempts on this device
                    if device == "cuda" and ("out of memory" in err_str or "oom" in err_str):
                        logger.warning("CUDA OOM detected; skipping remaining candidates on CUDA")
                        break
            if device == "cuda" and len(devices) > 1:
                logger.warning("CUDA embedding load failed; will retry on CPU.")

        if self._force_embedding:
            raise RuntimeError(
                f"Unable to load embedding model {DEFAULT_EMBEDDING_MODEL} after local and online attempts. "
                "Check MEMUA_EMBEDDING_MODEL_PATH or the optional MEMUA_HF_CACHE_DIR."
            ) from last_exc

        if not allow_hash_fallback():
            raise RuntimeError(
                "Embedding model unavailable and hash fallback is disabled. "
                "Set MEMUA_ALLOW_HASH_FALLBACK=1 only for debugging."
            ) from last_exc
        logger.warning("Embedding model unavailable after warmup — using hash fallback")


class _HashEmbeddingModel:
    """Hash-based pseudo-embedding for testing (no GPU required)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts, show_progress_bar: bool = False):
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256((text or "").encode("utf-8")).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.stack(vectors, axis=0)
