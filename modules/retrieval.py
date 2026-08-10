#!/usr/bin/env python
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import Config
from utils.embedding_utils import DEFAULT_EMBEDDING_MODEL, allow_hash_fallback, load_sentence_transformer_with_fallback
from utils.io_utils import load_json

logger = logging.getLogger(__name__)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return np.asarray(vec, dtype=np.float32)


def _to_numpy(embeddings: Any) -> np.ndarray:
    arr = embeddings
    if getattr(arr, "ndim", None) == 1:
        arr = arr.reshape(1, -1)
    cpu_method = getattr(arr, "cpu", None)
    if callable(cpu_method):
        arr = cpu_method()
    numpy_method = getattr(arr, "numpy", None)
    if callable(numpy_method):
        arr = numpy_method()
    return np.asarray(arr, dtype=np.float32)


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_empty_principle(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        for nested in value.values():
            if isinstance(nested, dict):
                if any(_clean_text(v) for v in nested.values()):
                    return False
            elif _clean_text(nested):
                return False
        return True
    return False


def _coerce_principle_package(metadata: Dict[str, Any], record: Dict[str, Any]) -> Any:
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
    candidates = []
    if isinstance(record, dict):
        candidates.extend([
            record.get("golden_principle"),
            record.get("principle"),
        ])
    candidates.extend([
        metadata.get("golden_principle"),
        metadata.get("principle"),
        metadata.get("golden_principle_main"),
    ])

    for candidate in candidates:
        if not _is_empty_principle(candidate):
            if isinstance(candidate, str):
                return {
                    "problem_definition": {
                        "core_skill": get_field(metadata, CORE_FIELD),
                        "context_constraints": get_field(metadata, CONDITION_FIELD),
                        "task_objective": get_field(metadata, TARGET_FIELD),
                        "usage_scenario": _clean_text(metadata.get("golden_principle_usage_scenario", "")),
                    },
                    "experience_principle": {
                        "principle": candidate.strip(),
                        "explanation": _clean_text(metadata.get("golden_principle_explanation", "")),
                    },
                }
            return candidate

    return {
        "problem_definition": {
            "core_skill": get_field(metadata, CORE_FIELD),
            "context_constraints": get_field(metadata, CONDITION_FIELD),
            "task_objective": get_field(metadata, TARGET_FIELD),
            "usage_scenario": _clean_text(metadata.get("golden_principle_usage_scenario", "")),
        },
        "experience_principle": {
            "principle": "",
            "explanation": _clean_text(metadata.get("golden_principle_explanation", "")),
        },
    }


def _top_indices_from_scores(scores: np.ndarray, k: int) -> List[int]:
    total = int(scores.shape[0]) if scores is not None else 0
    if total <= 0:
        return []
    k = max(1, min(int(k), total))
    if total <= k:
        order = np.argsort(-scores)
    else:
        top = np.argpartition(-scores, k - 1)[:k]
        order = top[np.argsort(-scores[top])]
    return [int(i) for i in order]


def _compute_structure_score(facet_scores: Dict[str, float], threshold: float) -> float:
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


def _compute_quality_score(metadata: Dict[str, Any], record: Dict[str, Any]) -> float:
    metrics = (
        metadata.get("quality_metrics", {})
        or record.get("quality_metrics", {})
        or {}
    )
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


class _HashEmbeddingModel:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(
        self,
        texts: List[str],
        show_progress_bar: bool = False,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256((text or "").encode("utf-8")).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            vectors.append(_normalize(rng.standard_normal(self.dim).astype(np.float32)))
        return np.stack(vectors, axis=0)


class SemanticFeatureEncoder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.model = self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed; using hash embedding fallback")
            return _HashEmbeddingModel()

        device_pref = os.environ.get("MEMUA_EMBEDDING_DEVICE", "auto").strip().lower()
        if device_pref == "cpu" or device_pref.startswith("cuda"):
            device = device_pref
        else:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        logger.info("Loading retrieval embedding model: %s on %s", self.model_name, device)
        try:
            model, loaded_from = load_sentence_transformer_with_fallback(
                model_name=self.model_name,
                device=device,
                local_files_only=True,
            )
            logger.info("Loaded retrieval embedding model from %s", loaded_from)
            return model
        except Exception as exc:
            # Retry on CPU when CUDA fails (e.g., GPU OOM from vLLM)
            if device != "cpu":
                logger.warning(
                    "Failed to load %s on %s: %s; retrying on cpu",
                    self.model_name, device, exc,
                )
                try:
                    model, loaded_from = load_sentence_transformer_with_fallback(
                        model_name=self.model_name,
                        device="cpu",
                        local_files_only=True,
                    )
                    logger.info("Loaded retrieval embedding model from %s (cpu fallback)", loaded_from)
                    return model
                except Exception as cpu_exc:
                    logger.warning("CPU fallback also failed: %s", cpu_exc)
                    exc = cpu_exc

            if not allow_hash_fallback():
                raise RuntimeError(
                    "Retrieval embedding model failed to load and hash fallback is disabled. "
                    "Set MEMUA_ALLOW_HASH_FALLBACK=1 only for debugging."
                ) from exc
            logger.warning(
                "Failed to load retrieval embedding model %s: %s; falling back to hash embeddings",
                self.model_name,
                exc,
            )
            return _HashEmbeddingModel()

    @staticmethod
    def _feature_text(features: Dict[str, Any], primary_key: str, legacy_key: str) -> str:
        return str(features.get(primary_key, features.get(legacy_key, "")) or "")

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = _to_numpy(
            self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                batch_size=batch_size or self.batch_size,
            )
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms

    def encode_feature_batch(
        self,
        features_list: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not features_list:
            empty = np.zeros((0, 0), dtype=np.float32)
            return empty, empty, empty

        core_texts = [self._feature_text(features, "core_skill", "core_math_concept") for features in features_list]
        context_texts = [self._feature_text(features, "context_constraints", "conditions_structure") for features in features_list]
        target_texts = [self._feature_text(features, "task_objective", "solution_target") for features in features_list]

        core_embeddings = self._encode_texts(core_texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        context_embeddings = self._encode_texts(context_texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        target_embeddings = self._encode_texts(target_texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        return core_embeddings, context_embeddings, target_embeddings

    def encode_features(self, features: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        core_embeddings, context_embeddings, target_embeddings = self.encode_feature_batch([features], show_progress_bar=False)
        return core_embeddings[0], context_embeddings[0], target_embeddings[0]


class OnlinePrincipleRetriever:
    def __init__(
        self,
        config: Config,
        vector_db_path: str,
        top_k: int = 5,
        threshold: float = 0.55,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        prefer_chroma: bool = True,
        semantic_memory_bank_path: Optional[str] = None,
    ):
        self.config = config
        self.vector_db_path = Path(vector_db_path)
        self.top_k = top_k
        self.threshold = threshold
        self.embedding_model_name = embedding_model_name
        self.prefer_chroma = prefer_chroma
        self.semantic_memory_bank_path = Path(semantic_memory_bank_path) if semantic_memory_bank_path else None

        self.encoder = SemanticFeatureEncoder(model_name=embedding_model_name)
        self.db = self._load_db()
        self.semantic_lookup = self._load_semantic_lookup()

    def _load_db(self) -> Dict[str, Any]:
        data_path = self.vector_db_path / "memory_data.json"
        if data_path.exists():
            data = load_json(str(data_path))
            if isinstance(data, dict):
                return data
        return {}

    def _load_semantic_lookup(self) -> Dict[str, Dict[Any, Dict[str, Any]]]:
        lookup = {"by_question_source": {}, "by_question": {}}
        if not self.semantic_memory_bank_path or not self.semantic_memory_bank_path.exists():
            return lookup

        data = load_json(str(self.semantic_memory_bank_path))
        records = data.get("records", []) if isinstance(data, dict) else []
        for record in records:
            if not isinstance(record, dict):
                continue
            question = _clean_text(record.get("question", ""))
            source_dataset = _clean_text(record.get("source_dataset", ""))
            if question:
                lookup["by_question"][question] = record
                lookup["by_question_source"][(question, source_dataset)] = record
        return lookup

    def _enrich_from_semantic_bank(self, metadata: Dict[str, Any], record: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        question = _clean_text(metadata.get("question", "")) or _clean_text(record.get("question", ""))
        source_dataset = (
            _clean_text(record.get("source_dataset", ""))
            or _clean_text(metadata.get("source_dataset", ""))
            or _clean_text(metadata.get("source", ""))
        )

        bank_record = (
            self.semantic_lookup["by_question_source"].get((question, source_dataset))
            or self.semantic_lookup["by_question"].get(question)
        )
        if not isinstance(bank_record, dict):
            return metadata, record

        enriched_metadata = dict(metadata)
        enriched_record = dict(record) if isinstance(record, dict) else {}

        for key in ["core_skill", "context_constraints", "task_objective", "core_math_concept", "conditions_structure", "solution_target", "source_dataset"]:
            if not _clean_text(enriched_metadata.get(key, "")):
                enriched_metadata[key] = bank_record.get(key, "")
            if not _clean_text(enriched_record.get(key, "")):
                enriched_record[key] = bank_record.get(key, "")

        if _is_empty_principle(enriched_metadata.get("principle")):
            enriched_metadata["principle"] = bank_record.get("principle", "")
        if _is_empty_principle(enriched_record.get("principle")):
            enriched_record["principle"] = bank_record.get("principle", "")

        if not _clean_text(enriched_record.get("question", "")):
            enriched_record["question"] = bank_record.get("question", "")
        if not _clean_text(enriched_record.get("source_dataset", "")):
            enriched_record["source_dataset"] = bank_record.get("source_dataset", "")
        return enriched_metadata, enriched_record

    def retrieve(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.db:
            return []

        ids = self.db.get("ids", [])
        metadatas = self.db.get("metadatas", [])
        records = self.db.get("records", [])
        concept_embeddings = np.asarray(self.db.get("concept_embeddings", []), dtype=np.float32)
        condition_embeddings = np.asarray(self.db.get("condition_embeddings", []), dtype=np.float32)
        target_embeddings = np.asarray(self.db.get("target_embeddings", []), dtype=np.float32)

        if concept_embeddings.size == 0:
            return []

        query_concept, query_condition, query_target = self.encoder.encode_features(features)
        concept_scores = concept_embeddings @ query_concept
        condition_scores = condition_embeddings @ query_condition if condition_embeddings.size else np.zeros_like(concept_scores)
        target_scores = target_embeddings @ query_target if target_embeddings.size else np.zeros_like(concept_scores)

        default_candidate_pool = max(self.top_k * 6, 24)
        candidate_pool_size = min(
            len(metadatas),
            max(
                self.top_k,
                int(os.environ.get("MEMUA_RETRIEVAL_CANDIDATE_POOL", str(default_candidate_pool))),
            ),
        )
        candidate_indices = set()
        for scores in (concept_scores, condition_scores, target_scores):
            for idx in _top_indices_from_scores(scores, candidate_pool_size):
                candidate_indices.add(idx)

        min_supported_facets = max(
            1,
            min(3, int(os.environ.get("MEMUA_RETRIEVAL_MIN_SUPPORTED_FACETS", "2"))),
        )
        threshold = float(self.threshold)
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

        payload = []
        for idx in sorted(candidate_indices):
            if idx >= len(metadatas):
                continue

            metadata = metadatas[idx]
            record = records[idx] if idx < len(records) else {}
            metadata, record = self._enrich_from_semantic_bank(metadata, record if isinstance(record, dict) else {})
            concept_score = float(concept_scores[idx])
            condition_score = float(condition_scores[idx]) if idx < len(condition_scores) else 0.0
            target_score = float(target_scores[idx]) if idx < len(target_scores) else 0.0
            facet_scores = {
                "core_skill": concept_score,
                "context_constraints": condition_score,
                "task_objective": target_score,
            }
            matched_facets = [name for name, value in facet_scores.items() if value >= threshold]
            weakest_facet = min(facet_scores, key=facet_scores.get)
            weakest_value = facet_scores[weakest_facet]
            structure_score = _compute_structure_score(facet_scores, threshold)
            quality_score = _compute_quality_score(metadata, record if isinstance(record, dict) else {})
            relevance_score = (concept_score + condition_score + target_score) / 3
            final_score = structure_weight * structure_score + quality_weight * quality_score

            if len(matched_facets) < min_supported_facets:
                continue
            if weakest_value < weak_facet_floor:
                continue
            if structure_score < structure_floor:
                continue

            principle = _coerce_principle_package(metadata, record if isinstance(record, dict) else {})
            principle_text = ""
            if isinstance(principle, dict):
                principle_text = _clean_text((principle.get("experience_principle") or {}).get("principle", ""))
            elif isinstance(principle, str):
                principle_text = _clean_text(principle)

            payload.append({
                "matched_memory": {
                    "question": metadata.get("question", ""),
                    "golden_principle": principle,
                    "source_dataset": (
                        record.get("source_dataset")
                        or metadata.get("source_dataset")
                        or metadata.get("source")
                        or ""
                    ) if isinstance(record, dict) else (metadata.get("source_dataset") or metadata.get("source") or ""),
                },
                "dimension_similarities": {
                    "core_skill": concept_score,
                    "context_constraints": condition_score,
                    "task_objective": target_score,
                },
                "score": final_score,
                "structure_score": structure_score,
                "quality_score": quality_score,
                "relevance_score": relevance_score,
                "matched_facets": matched_facets,
                "supported_facet_count": len(matched_facets),
                "weakest_facet": weakest_facet,
                "memory_id": ids[idx] if idx < len(ids) else f"mem_{idx}",
                "principle_text": principle_text,
            })

        non_empty = [item for item in payload if _clean_text(item.get("principle_text", ""))]
        if non_empty:
            payload = non_empty
        payload.sort(
            key=lambda item: (
                item.get("score", 0.0),
                item.get("structure_score", 0.0),
                item.get("relevance_score", 0.0),
            ),
            reverse=True,
        )
        return payload[: self.top_k]
