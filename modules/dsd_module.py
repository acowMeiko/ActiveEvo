"""modules/dsd_module.py — dataset-level DSD registry.

This registry is a Stage 2 working store, not memory.
It keeps one semantic signature per source question, then provides
stable neighborhood lookup for principle validation.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm.asyncio import tqdm as async_tqdm

from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field, semantic_fields_complete
from retrieve_principles import SemanticFeatureEncoder
from output_paths import (
    stage2_dsd_condition_index_file,
    stage2_dsd_core_index_file,
    stage2_dsd_registry_dir,
    stage2_dsd_registry_file,
    stage2_dsd_target_index_file,
)
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


def stage1_packet_question_id(packet: Dict[str, Any], index: int) -> str:
    """Return the stable source-question identity used throughout Stage 2."""
    return str(packet.get("question_id") or packet.get("idx") or index)


def stage1_packet_is_candidate(packet: Dict[str, Any]) -> bool:
    """Mirror the Stage-2 eligibility rule without invoking a model."""
    metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
    explicit = packet.get("stage2_candidate", metadata.get("stage2_candidate"))
    if explicit is not None:
        return bool(explicit)
    question = packet.get("question_text") or packet.get("question")
    answer = packet.get("expected_answer") or packet.get("answer")
    chosen = packet.get("chosen_trajectory") or packet.get("chosen_path")
    rejected = packet.get("hard_negative_trajectory") or packet.get("rejected_path")
    return all((question, answer, chosen, rejected))


def _packet_preference(packet: Dict[str, Any]) -> tuple:
    """Choose one deterministic, information-rich packet per source question."""
    diff = packet.get("contrast") or packet.get("diff") or {}
    divergence = packet.get("divergence_step_text") or (packet.get("metadata") or {}).get("divergence_step_text", "")
    stable_payload = json.dumps(packet, ensure_ascii=False, sort_keys=True, default=str)
    return (
        int(stage1_packet_is_candidate(packet)),
        int(packet.get("diff_success") is True),
        int(bool(diff)),
        int(bool(str(divergence).strip())),
        hashlib.sha256(stable_payload.encode("utf-8")).hexdigest(),
    )


def canonicalize_stage1_packets(
    packets: Sequence[Dict[str, Any]],
    selected_question_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Collapse repeated trajectories to one canonical packet per question.

    Stage 1 can emit several contrastive packets for the same question.  Stage 2
    memory IDs and DSD neighborhoods are question-scoped, so allowing duplicate
    rows here silently creates duplicate candidate IDs and invalid Pilot counts.
    """
    selected = {str(value) for value in selected_question_ids or []}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for index, packet in enumerate(packets):
        if not isinstance(packet, dict):
            continue
        question_id = stage1_packet_question_id(packet, index)
        if selected and question_id not in selected:
            continue
        normalized = dict(packet)
        normalized["question_id"] = question_id
        grouped.setdefault(question_id, []).append(normalized)
    canonical = [max(values, key=_packet_preference) for values in grouped.values()]
    canonical.sort(key=lambda item: str(item.get("question_id", "")))
    return canonical


@dataclass
class DSDRecord:
    question_id: str
    question_text: str
    expected_answer: str
    source_round: int
    source_dataset: str
    source_metadata: Dict[str, Any]
    source_dsd: Dict[str, str]
    core_embedding: List[float]
    context_embedding: List[float]
    target_embedding: List[float]


class DSDRegistry:
    """Dataset-level DSD registry for Stage 2 neighborhood validation."""

    def __init__(
        self,
        dataset: str,
        round_id: int,
        output_root: Path,
        embedding_model_name: str = "BAAI/bge-m3",
        backbone: str = "",
        cache_dir: Optional[Path] = None,
        rebuild_cache: bool = False,
    ):
        self.dataset = dataset
        self.round_id = round_id
        self.output_root = Path(output_root)
        self.backbone = backbone
        self.embedding_model_name = embedding_model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rebuild_cache = bool(rebuild_cache)
        self.cache_hit = False
        # Delay the heavy BGE load until an actual cache miss. Both NV arms can
        # therefore consume a cache without even constructing the encoder.
        self.encoder: Optional[SemanticFeatureEncoder] = None
        self.records: List[Dict[str, Any]] = []
        self._by_question_id: Dict[str, Dict[str, Any]] = {}
        self._core_embeddings: Optional[np.ndarray] = None
        self._context_embeddings: Optional[np.ndarray] = None
        self._target_embeddings: Optional[np.ndarray] = None
        self._neighbor_cache: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def registry_dir(self) -> Path:
        return stage2_dsd_registry_dir(self.output_root, self.dataset, self.backbone)

    def _normalize_record(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question_id = str(
            item.get("question_id")
            or item.get("idx")
            or idx
        )
        question_text = str(item.get("question_text") or item.get("question") or "").strip()
        expected_answer = str(item.get("expected_answer") or item.get("answer") or "").strip()
        metadata = dict(item.get("metadata") or item.get("source_metadata") or {})
        source_round = int(item.get("source_round") or metadata.get("source_round") or self.round_id or 0)
        source_dataset = str(item.get("source_dataset") or metadata.get("dataset") or self.dataset).strip()
        source_dsd = item.get("source_dsd") or item.get("problem_semantics") or {}
        if not isinstance(source_dsd, dict):
            source_dsd = {}
        source_dsd = {
            CORE_FIELD: get_field(source_dsd, CORE_FIELD),
            CONDITION_FIELD: get_field(source_dsd, CONDITION_FIELD),
            TARGET_FIELD: get_field(source_dsd, TARGET_FIELD),
        }
        return {
            "question_id": question_id,
            "question_text": question_text,
            "expected_answer": expected_answer,
            "source_round": source_round,
            "source_dataset": source_dataset,
            "source_metadata": metadata,
            "source_dsd": source_dsd,
            "chosen_trajectory": str(
                item.get("chosen_trajectory")
                or item.get("chosen_path")
                or ""
            ).strip(),
            "hard_negative_trajectory": str(
                item.get("hard_negative_trajectory")
                or item.get("rejected_path")
                or ""
            ).strip(),
            "contrast": item.get("contrast") or item.get("diff") or {},
        }

    async def build(self, packets: Sequence[Dict[str, Any]], plan_module: Any, concurrency: int = 8) -> Dict[str, Any]:
        """Extract DSD signatures for all Stage 1 source questions."""
        packets = canonicalize_stage1_packets(packets)
        cache_key = hashlib.sha256(json.dumps({
            "schema": "dsd-cache-v1",
            "dataset": self.dataset,
            "round_id": self.round_id,
            "embedding_model": self.embedding_model_name,
            "packets": list(packets),
        }, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if self.cache_dir and not self.rebuild_cache and self._load_cache(cache_key):
            self.cache_hit = True
            logger.info("Stage 2 DSD cache hit: %s", self.cache_dir)
            return self.summary()
        self.encoder = SemanticFeatureEncoder(model_name=self.embedding_model_name)
        semaphore = asyncio.Semaphore(max(1, concurrency))
        normalized_records: List[Dict[str, Any]] = []

        async def build_one(idx: int, packet: Dict[str, Any]) -> Dict[str, Any]:
            base = self._normalize_record(packet, idx)
            question_text = base["question_text"]
            if not question_text:
                return {**base, "status": "skipped", "reason": "empty_question"}

            metadata = packet.get("metadata", {}) if isinstance(packet.get("metadata"), dict) else {}
            explicit_candidate = packet.get("stage2_candidate", metadata.get("stage2_candidate"))
            has_pair = all([
                base["question_text"],
                base["expected_answer"],
                base["chosen_trajectory"],
                base["hard_negative_trajectory"],
            ])
            # Stage 1 supplies contrastive trajectories; Stage 2 owns diff
            # generation.  Preserve an explicit false zoning decision.
            stage2_candidate = bool(explicit_candidate) if explicit_candidate is not None else has_pair
            base["stage2_candidate"] = stage2_candidate
            # Compatibility alias for older readers.
            base["is_usable"] = stage2_candidate

            existing_dsd = {
                CORE_FIELD: get_field(base["source_dsd"], CORE_FIELD),
                CONDITION_FIELD: get_field(base["source_dsd"], CONDITION_FIELD),
                TARGET_FIELD: get_field(base["source_dsd"], TARGET_FIELD),
            }
            if semantic_fields_complete(existing_dsd):
                base["source_dsd"] = existing_dsd
                base["semantic_source"] = "stage1"
                base["zone"] = packet.get("zone") or packet.get("metadata", {}).get("zone", "learning")
                base["divergence_step_idx"] = packet.get("divergence_step_idx", packet.get("metadata", {}).get("divergence_step_idx", 0))
                base["divergence_step_text"] = packet.get("divergence_step_text", packet.get("metadata", {}).get("divergence_step_text", ""))
                base["trajectory_similarity"] = packet.get("trajectory_similarity", packet.get("similarity", packet.get("metadata", {}).get("trajectory_similarity", 0.0)))
                return base

            # DSD 缺失 → 调 LLM 提取语义 (所有 packet 都需要, 用于邻域检索)
            async with semaphore:
                semantics = await plan_module.extract_semantics(question_text)
            source_dsd = {
                CORE_FIELD: get_field(semantics, CORE_FIELD),
                CONDITION_FIELD: get_field(semantics, CONDITION_FIELD),
                TARGET_FIELD: get_field(semantics, TARGET_FIELD),
            }
            base["source_dsd"] = source_dsd
            base["semantic_source"] = semantics.get("_semantics_source", "unknown")
            base["zone"] = packet.get("zone") or packet.get("metadata", {}).get("zone", "learning")
            base["divergence_step_idx"] = packet.get("divergence_step_idx", packet.get("metadata", {}).get("divergence_step_idx", 0))
            base["divergence_step_text"] = packet.get("divergence_step_text", packet.get("metadata", {}).get("divergence_step_text", ""))
            base["trajectory_similarity"] = packet.get("trajectory_similarity", packet.get("similarity", packet.get("metadata", {}).get("trajectory_similarity", 0.0)))
            return base

        tasks = [build_one(idx, packet) for idx, packet in enumerate(packets)]
        for coro in async_tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Stage 2.1 DSD semantics",
        ):
            normalized_records.append(await coro)

        normalized_records.sort(key=lambda item: item.get("question_id", ""))
        self.records = [item for item in normalized_records if item.get("status") != "skipped"]
        self._by_question_id = {record["question_id"]: record for record in self.records}
        logger.info(
            "Stage 2.2: batch encoding %d DSD routing facets for neighborhood lookup "
            "(core_skill, context_constraints, task_objective; these are not Memory solver/validation/audit views)",
            len(self.records),
        )
        self._build_embeddings()
        self._precompute_neighbors()
        if self.cache_dir:
            self._save_cache(cache_key)
        summary = self.summary()
        return summary

    def _load_cache(self, cache_key: str) -> bool:
        manifest_path = self.cache_dir / "manifest.json"
        records_path = self.cache_dir / "records.json"
        neighbors_path = self.cache_dir / "neighbors.json"
        arrays = [self.cache_dir / name for name in ("core_embeddings.npy", "context_embeddings.npy", "target_embeddings.npy")]
        if not all(path.exists() for path in [manifest_path, records_path, neighbors_path, *arrays]):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_key") != cache_key:
                logger.info("Stage 2 DSD cache miss (input/config hash changed): %s", self.cache_dir)
                return False
            self.records = json.loads(records_path.read_text(encoding="utf-8"))
            self._neighbor_cache = json.loads(neighbors_path.read_text(encoding="utf-8"))
            self._core_embeddings = np.load(arrays[0])
            self._context_embeddings = np.load(arrays[1])
            self._target_embeddings = np.load(arrays[2])
            self._by_question_id = {str(record["question_id"]): record for record in self.records}
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unusable DSD cache %s: %s", self.cache_dir, exc)
            return False

    def _save_cache(self, cache_key: str) -> None:
        if self._core_embeddings is None or self._context_embeddings is None or self._target_embeddings is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lightweight = []
        for record in self.records:
            item = {key: value for key, value in record.items() if not key.endswith("_embedding")}
            lightweight.append(item)
        save_json(lightweight, str(self.cache_dir / "records.json"))
        save_json(self._neighbor_cache, str(self.cache_dir / "neighbors.json"))
        np.save(self.cache_dir / "core_embeddings.npy", self._core_embeddings)
        np.save(self.cache_dir / "context_embeddings.npy", self._context_embeddings)
        np.save(self.cache_dir / "target_embeddings.npy", self._target_embeddings)
        save_json({
            "schema": "dsd-cache-v1",
            "cache_key": cache_key,
            "dataset": self.dataset,
            "round_id": self.round_id,
            "embedding_model": self.embedding_model_name,
            "record_count": len(self.records),
            "dimensions": {
                "core_skill": int(self._core_embeddings.shape[1]),
                "context_constraints": int(self._context_embeddings.shape[1]),
                "task_objective": int(self._target_embeddings.shape[1]),
            },
            "neighbor_policy": {"core_min": 0.35, "context_min": 0.15, "target_min": 0.25, "cache_top_k": 16},
        }, str(self.cache_dir / "manifest.json"))

    def _build_embeddings(self, batch_size: int = 256) -> None:
        if not self.records:
            self._core_embeddings = None
            self._context_embeddings = None
            self._target_embeddings = None
            return

        features = [record["source_dsd"] for record in self.records]
        logger.info("Encoding DSD core/context/target facets in batches (batch_size=%d)", batch_size)
        if self.encoder is None:
            self.encoder = SemanticFeatureEncoder(model_name=self.embedding_model_name)
        self._core_embeddings, self._context_embeddings, self._target_embeddings = self.encoder.encode_feature_batch(
            features,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        for record, core_vec, context_vec, target_vec in zip(self.records, self._core_embeddings, self._context_embeddings, self._target_embeddings):
            record["core_embedding"] = core_vec.tolist()
            record["context_embedding"] = context_vec.tolist()
            record["target_embedding"] = target_vec.tolist()

    def _precompute_neighbors(self, cache_top_k: int = 16) -> None:
        self._neighbor_cache = {}
        if not self.records or self._core_embeddings is None:
            return

        logger.info("Stage 2.3: precomputing neighbor cache for %d DSD records", len(self.records))
        core_sims = self._core_embeddings @ self._core_embeddings.T
        context_sims = self._context_embeddings @ self._context_embeddings.T
        target_sims = self._target_embeddings @ self._target_embeddings.T
        combined = (core_sims + context_sims + target_sims) / 3.0

        question_texts = [record.get("question_text", "").strip() for record in self.records]
        max_neighbors = min(max(1, cache_top_k), max(1, len(self.records) - 1))

        for source_idx, source in enumerate(self.records):
            candidates: List[Dict[str, Any]] = []
            for idx, record in enumerate(self.records):
                if idx == source_idx:
                    continue
                if question_texts[idx] == question_texts[source_idx]:
                    continue

                core_sim = float(core_sims[source_idx, idx])
                context_sim = float(context_sims[source_idx, idx])
                target_sim = float(target_sims[source_idx, idx])
                score = float(combined[source_idx, idx])
                payload = {
                    "question_id": record.get("question_id", ""),
                    "question_text": record.get("question_text", ""),
                    "expected_answer": record.get("expected_answer", ""),
                    "source_round": record.get("source_round", 0),
                    "source_dataset": record.get("source_dataset", ""),
                    "source_dsd": record.get("source_dsd", {}),
                    "dimension_similarities": {
                        CORE_FIELD: core_sim,
                        CONDITION_FIELD: context_sim,
                        TARGET_FIELD: target_sim,
                    },
                    "score": score,
                }
                if core_sim >= 0.35 and target_sim >= 0.25 and context_sim >= 0.15:
                    candidates.append(payload)

            candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            # Validation evidence must remain structurally eligible.  Returning
            # weak nearest neighbors merely to fill top-k confounds memory
            # quality with neighborhood quality.
            self._neighbor_cache[source.get("question_id", "")] = candidates[:max_neighbors]

    def get_record(self, question_id: str) -> Optional[Dict[str, Any]]:
        return self._by_question_id.get(str(question_id))

    def get_neighbors(self, question_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if question_id in self._neighbor_cache:
            return self._neighbor_cache[question_id][:top_k]

        if not self.records or self._core_embeddings is None:
            return []

        source = self.get_record(question_id)
        if not source:
            return []

        source_idx = None
        for idx, record in enumerate(self.records):
            if record.get("question_id") == str(question_id):
                source_idx = idx
                break
        if source_idx is None:
            return []

        source_core = self._core_embeddings[source_idx]
        source_context = self._context_embeddings[source_idx]
        source_target = self._target_embeddings[source_idx]
        neighbors: List[Dict[str, Any]] = []
        for idx, record in enumerate(self.records):
            if idx == source_idx:
                continue
            if record.get("question_text", "").strip() == source.get("question_text", "").strip():
                continue

            core_sim = float(self._core_embeddings[idx] @ source_core)
            context_sim = float(self._context_embeddings[idx] @ source_context)
            target_sim = float(self._target_embeddings[idx] @ source_target)
            if core_sim < 0.35 or target_sim < 0.25:
                continue
            if context_sim < 0.15:
                continue
            score = (core_sim + context_sim + target_sim) / 3.0
            neighbors.append({
                "question_id": record.get("question_id", ""),
                "question_text": record.get("question_text", ""),
                "expected_answer": record.get("expected_answer", ""),
                "source_round": record.get("source_round", 0),
                "source_dataset": record.get("source_dataset", ""),
                "source_dsd": record.get("source_dsd", {}),
                "dimension_similarities": {
                    CORE_FIELD: core_sim,
                    CONDITION_FIELD: context_sim,
                    TARGET_FIELD: target_sim,
                },
                "score": score,
            })

        neighbors.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return neighbors[:top_k]

    def save(self) -> Dict[str, Any]:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        registry_payload = {
            "dataset": self.dataset,
            "round_id": self.round_id,
            "summary": summary,
            "records": self.records,
        }
        save_json(registry_payload, str(stage2_dsd_registry_file(self.output_root, self.dataset, self.backbone)))
        self._save_dimension_index(stage2_dsd_core_index_file(self.output_root, self.dataset, self.backbone), "core_skill", self._core_embeddings)
        self._save_dimension_index(stage2_dsd_condition_index_file(self.output_root, self.dataset, self.backbone), "context_constraints", self._context_embeddings)
        self._save_dimension_index(stage2_dsd_target_index_file(self.output_root, self.dataset, self.backbone), "task_objective", self._target_embeddings)
        return summary

    def _save_dimension_index(self, path: Path, dimension: str, embeddings: Optional[np.ndarray]) -> None:
        payload = {
            "dataset": self.dataset,
            "round_id": self.round_id,
            "dimension": dimension,
            "records": [],
        }
        if embeddings is not None:
            for record, vector in zip(self.records, embeddings):
                payload["records"].append({
                    "question_id": record.get("question_id", ""),
                    "question_text": record.get("question_text", ""),
                    "score": float(np.linalg.norm(vector)),
                    "source_dsd": record.get("source_dsd", {}),
                })
        save_json(payload, str(path))

    def summary(self) -> Dict[str, Any]:
        usable = [record for record in self.records if record.get("stage2_candidate", record.get("is_usable", True))]
        return {
            "dataset": self.dataset,
            "round_id": self.round_id,
            "total_records": len(self.records),
            "usable_records": len(usable),
            "reused_stage1_semantics": sum(1 for record in self.records if record.get("semantic_source") == "stage1"),
            "semantic_source_counts": self._semantic_source_counts(),
            "cache_hit": self.cache_hit,
            "cache_dir": str(self.cache_dir) if self.cache_dir else "",
        }

    def _semantic_source_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            key = str(record.get("semantic_source", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts
