"""Retrieval protocols used by the Stage-3 precision experiment.

The production retriever remains unchanged.  This module exposes auditable
alternatives that all operate on the same frozen ``Memory`` snapshot.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field


PROTOCOLS = (
    "r0_single_vector",
    "r1_current_three_facet",
    "r2_strict_three_facet",
    "r3_hierarchical_three_facet",
    "r4_draft_verified",
    "r5_dual_retrieval",
)

SEMANTIC_HYBRID_PROTOCOLS = (
    "e0_r3_current",
    "e1_r3_source_dedup",
    "e2_three_facet_high_recall",
    "e3_cross_encoder",
    "h1_dense_sparse",
    "h2_dense_sparse_multivector",
    "h3_draft_soft_rerank",
)


def _active(memory: Any) -> List[Dict[str, Any]]:
    return [item for item in memory.principles if item.get("status") == "active"]


def _top_indices(scores: np.ndarray, count: int) -> List[int]:
    if len(scores) == 0:
        return []
    count = min(max(1, int(count)), len(scores))
    return [int(i) for i in np.argsort(-scores)[:count]]


def _entry(
    item: Dict[str, Any],
    scores: Mapping[str, float],
    score: float,
    protocol: str,
    *,
    candidate_facets: Sequence[str] = (),
) -> Dict[str, Any]:
    values = {str(key): float(value) for key, value in scores.items()}
    return {
        "memory_id": str(item.get("memory_id", "")),
        "item": item,
        "dimension_similarities": values,
        "score": float(score),
        "retrieval_score": float(score),
        "retrieval_method": protocol,
        "candidate_facets": list(candidate_facets),
        "weakest_facet": min(values, key=values.get) if values else "",
    }


def facet_score_rows(memory: Any, features: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return all active memories with their three cosine similarities."""
    if not memory._ensure_index_alignment():
        return []
    items = _active(memory)
    if not items:
        return []
    query_core, query_condition, query_target = memory._encode_features(dict(features))
    core = memory._core_embeddings @ query_core
    condition = memory._condition_embeddings @ query_condition
    target = memory._target_embeddings @ query_target
    rows = []
    for idx, item in enumerate(items):
        rows.append(
            {
                "index": idx,
                "item": item,
                "memory_id": str(item.get("memory_id", "")),
                "scores": {
                    CORE_FIELD: float(core[idx]),
                    CONDITION_FIELD: float(condition[idx]),
                    TARGET_FIELD: float(target[idx]),
                },
            }
        )
    return rows


def retrieve_current(memory: Any, features: Mapping[str, Any], *, top_k: int, threshold: float) -> List[Dict[str, Any]]:
    return memory.retrieve_reasoning_aligned(dict(features), top_k=top_k, threshold=threshold)


def retrieve_strict(memory: Any, features: Mapping[str, Any], *, top_k: int, threshold: float) -> List[Dict[str, Any]]:
    return memory.retrieve_hard_gated(dict(features), top_k=top_k, threshold=threshold)


def retrieve_hierarchical(
    memory: Any,
    features: Mapping[str, Any],
    *,
    top_k: int = 12,
    pool_per_facet: int = 30,
    core_threshold: float = 0.55,
    condition_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """Use the three facets asymmetrically: recall, veto, then rerank."""
    rows = facet_score_rows(memory, features)
    if not rows:
        return []
    arrays = {
        field: np.asarray([row["scores"][field] for row in rows], dtype=np.float32)
        for field in (CORE_FIELD, CONDITION_FIELD, TARGET_FIELD)
    }
    source_by_index: Dict[int, List[str]] = {}
    for field, values in arrays.items():
        for idx in _top_indices(values, pool_per_facet):
            source_by_index.setdefault(idx, []).append(field)

    result = []
    for idx, source_facets in source_by_index.items():
        row = rows[idx]
        scores = row["scores"]
        if scores[CORE_FIELD] < core_threshold:
            continue
        if scores[CONDITION_FIELD] < condition_threshold:
            continue
        score = (
            0.45 * scores[CORE_FIELD]
            + 0.35 * scores[CONDITION_FIELD]
            + 0.20 * scores[TARGET_FIELD]
        )
        result.append(
            _entry(
                row["item"], scores, score, "r3_hierarchical_three_facet",
                candidate_facets=source_facets,
            )
        )
    result.sort(key=lambda value: (value["score"], min(value["dimension_similarities"].values())), reverse=True)
    return result[:top_k]


def source_group_key(entry: Mapping[str, Any]) -> str:
    """Return the immutable source-question group for one Memory candidate."""
    item = entry.get("item", {}) if isinstance(entry.get("item"), Mapping) else {}
    question_id = str(item.get("question_id", "")).strip()
    return question_id or str(entry.get("memory_id", "")).strip()


def deduplicate_source_groups(
    entries: Sequence[Mapping[str, Any]],
    *,
    top_k: int | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep the highest-ranked candidate from each source question.

    The input order is meaningful and is preserved.  The second return value is
    an audit trail for candidates removed from a duplicated source group.
    """
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    first_by_group: Dict[str, str] = {}
    for rank, raw in enumerate(entries, 1):
        entry = dict(raw)
        group = source_group_key(entry)
        entry["source_group"] = group
        if group in first_by_group:
            removed.append(
                {
                    "memory_id": str(entry.get("memory_id", "")),
                    "source_group": group,
                    "original_rank": rank,
                    "kept_memory_id": first_by_group[group],
                }
            )
            continue
        first_by_group[group] = str(entry.get("memory_id", ""))
        kept.append(entry)
        if top_k is not None and len(kept) >= max(0, int(top_k)):
            break
    return kept, removed


def retrieve_three_facet_union(
    memory: Any,
    features: Mapping[str, Any],
    *,
    top_k: int = 24,
    pool_per_facet: int = 30,
    weights: Mapping[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """Threshold-free three-facet recall with asymmetric semantic roles."""
    rows = facet_score_rows(memory, features)
    if not rows:
        return []
    weights = dict(weights or {CORE_FIELD: 0.45, CONDITION_FIELD: 0.35, TARGET_FIELD: 0.20})
    arrays = {
        field: np.asarray([row["scores"][field] for row in rows], dtype=np.float32)
        for field in (CORE_FIELD, CONDITION_FIELD, TARGET_FIELD)
    }
    source_by_index: Dict[int, List[str]] = {}
    ranks_by_index: Dict[int, Dict[str, int]] = {}
    for field, values in arrays.items():
        for rank, idx in enumerate(_top_indices(values, pool_per_facet), 1):
            source_by_index.setdefault(idx, []).append(field)
            ranks_by_index.setdefault(idx, {})[field] = rank

    result: List[Dict[str, Any]] = []
    for idx, source_facets in source_by_index.items():
        row = rows[idx]
        scores = row["scores"]
        score = sum(float(weights.get(field, 0.0)) * scores[field] for field in weights)
        entry = _entry(
            row["item"], scores, score, "three_facet_high_recall_union",
            candidate_facets=source_facets,
        )
        entry["facet_ranks"] = ranks_by_index[idx]
        result.append(entry)
    result.sort(
        key=lambda value: (
            value["score"],
            value["dimension_similarities"].get(CORE_FIELD, 0.0),
            value["dimension_similarities"].get(CONDITION_FIELD, 0.0),
        ),
        reverse=True,
    )
    return result[:top_k]


def retrieve_three_facet_joint(
    memory: Any,
    features: Mapping[str, Any],
    *,
    top_k: int = 24,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rank memories by simultaneous agreement on all three semantic facets.

    The weakest facet receives half of the score, so a topic-level match cannot
    hide a constraint or objective mismatch.  Source-question duplicates are
    removed before truncation.  The second result is the deduplication trace.
    """
    rows = facet_score_rows(memory, features)
    ranked: List[Dict[str, Any]] = []
    for row in rows:
        scores = row["scores"]
        values = [scores[CORE_FIELD], scores[CONDITION_FIELD], scores[TARGET_FIELD]]
        weakest = min(values)
        mean = sum(values) / 3.0
        joint = 0.5 * weakest + 0.5 * mean
        entry = _entry(row["item"], scores, joint, "three_facet_joint")
        entry["joint_score"] = float(joint)
        entry["facet_mean"] = float(mean)
        entry["facet_min"] = float(weakest)
        ranked.append(entry)
    ranked.sort(
        key=lambda value: (
            value["joint_score"], value["facet_min"], value["facet_mean"],
            str(value["memory_id"]),
        ),
        reverse=True,
    )
    return deduplicate_source_groups(ranked, top_k=top_k)


@dataclass
class DenseTextIndex:
    """Small in-memory cosine index sharing the Memory embedding model."""

    ids: List[str]
    texts: List[str]
    embeddings: np.ndarray

    @classmethod
    def build(cls, memory: Any, ids: Iterable[str], texts: Iterable[str]) -> "DenseTextIndex":
        id_list = [str(value) for value in ids]
        text_list = [str(value).strip() or "unknown" for value in texts]
        if len(id_list) != len(text_list):
            raise ValueError("DenseTextIndex IDs and texts must have the same length")
        model = memory._get_embedding_model()
        embeddings = memory._encode_texts(model, text_list)
        return cls(id_list, text_list, embeddings)

    def search(self, memory: Any, text: str, *, top_k: int) -> List[Dict[str, Any]]:
        if not self.ids:
            return []
        model = memory._get_embedding_model()
        query = memory._encode_texts(model, [str(text).strip() or "unknown"])[0]
        scores = self.embeddings @ query
        return [
            {
                "memory_id": self.ids[idx],
                "score": float(scores[idx]),
                "rank": rank,
                "indexed_text": self.texts[idx],
            }
            for rank, idx in enumerate(_top_indices(scores, top_k), 1)
        ]


class BGEHybridFacetIndex:
    """BGE-M3 dense/sparse/multi-vector index over the three semantic facets.

    ``FlagEmbedding`` is imported lazily so dense-only Stage-3 workflows retain
    their existing dependency and startup behavior.
    """

    def __init__(
        self,
        model_path: str,
        items: Sequence[Mapping[str, Any]],
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 1024,
        cache_path: str | Path | None = None,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "H1/H2/H3 require FlagEmbedding. Install requirements.txt in the active environment."
            ) from exc
        use_fp16 = str(device).startswith("cuda")
        # FlagEmbedding uses the plural ``devices`` argument. Passing an
        # unknown singular ``device`` is silently accepted through **kwargs
        # and then auto-selects every visible GPU, which would collide with
        # the vLLM solver replicas.
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16, devices=[device])
        self.items = [dict(item) for item in items]
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.fields = (CORE_FIELD, CONDITION_FIELD, TARGET_FIELD)
        facet_texts = {
            field: [str(get_field(item, field, "") or "unknown") for item in self.items]
            for field in self.fields
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {"model": str(Path(model_path).resolve()), "max_length": self.max_length, "texts": facet_texts},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache = Path(cache_path) if cache_path else None
        cached = None
        if cache and cache.is_file():
            with cache.open("rb") as handle:
                value = pickle.load(handle)
            if isinstance(value, dict) and value.get("fingerprint") == fingerprint:
                cached = value.get("encoded")
        if isinstance(cached, dict) and set(cached) == set(self.fields):
            self.encoded = cached
        else:
            self.encoded = {field: self._encode(texts) for field, texts in facet_texts.items()}
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(cache.suffix + ".tmp")
                with temporary.open("wb") as handle:
                    pickle.dump({"fingerprint": fingerprint, "encoded": self.encoded}, handle)
                temporary.replace(cache)

    def _encode(self, texts: Sequence[str]) -> Dict[str, Any]:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    @staticmethod
    def _sparse_dot(left: Mapping[Any, Any], right: Mapping[Any, Any]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return float(sum(float(value) * float(right.get(key, 0.0)) for key, value in left.items()))

    @staticmethod
    def _colbert_score(query: Any, document: Any) -> float:
        q = np.asarray(query, dtype=np.float32)
        d = np.asarray(document, dtype=np.float32)
        if q.ndim != 2 or d.ndim != 2 or q.size == 0 or d.size == 0:
            return 0.0
        return float((q @ d.T).max(axis=1).mean())

    def retrieve(
        self,
        features: Mapping[str, Any],
        *,
        modalities: Sequence[str],
        pool_per_facet: int,
        top_k: int,
        rank_constant: int = 60,
        facet_weights: Mapping[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        allowed = {"dense", "sparse", "multivector"}
        requested = tuple(dict.fromkeys(str(value) for value in modalities))
        if not requested or set(requested) - allowed:
            raise ValueError(f"Unsupported BGE-M3 modalities: {requested}")
        facet_weights = dict(
            facet_weights or {CORE_FIELD: 0.45, CONDITION_FIELD: 0.35, TARGET_FIELD: 0.20}
        )
        query_encoded = {
            field: self._encode([str(get_field(features, field, "") or "unknown")])
            for field in self.fields
        }
        per_item: Dict[int, Dict[str, Any]] = {}
        for field in self.fields:
            stored = self.encoded[field]
            query = query_encoded[field]
            modality_scores: Dict[str, np.ndarray] = {}
            if "dense" in requested:
                modality_scores["dense"] = np.asarray(stored["dense_vecs"], dtype=np.float32) @ np.asarray(
                    query["dense_vecs"][0], dtype=np.float32
                )
            if "sparse" in requested:
                q_sparse = query["lexical_weights"][0]
                modality_scores["sparse"] = np.asarray(
                    [self._sparse_dot(q_sparse, value) for value in stored["lexical_weights"]],
                    dtype=np.float32,
                )
            if "multivector" in requested:
                q_colbert = query["colbert_vecs"][0]
                modality_scores["multivector"] = np.asarray(
                    [self._colbert_score(q_colbert, value) for value in stored["colbert_vecs"]],
                    dtype=np.float32,
                )

            for modality, values in modality_scores.items():
                for rank, idx in enumerate(_top_indices(values, pool_per_facet), 1):
                    row = per_item.setdefault(
                        idx,
                        {"facet_rrf": {}, "modality_ranks": [], "raw_modality_scores": {}},
                    )
                    row["facet_rrf"][field] = row["facet_rrf"].get(field, 0.0) + 1.0 / (
                        rank_constant + rank
                    )
                    row["modality_ranks"].append(
                        {"facet": field, "modality": modality, "rank": rank}
                    )
                    row["raw_modality_scores"].setdefault(field, {})[modality] = float(values[idx])

        result: List[Dict[str, Any]] = []
        for idx, trace in per_item.items():
            facet_rrf = trace["facet_rrf"]
            score = sum(float(facet_weights.get(field, 0.0)) * facet_rrf.get(field, 0.0) for field in self.fields)
            result.append(
                {
                    "memory_id": str(self.items[idx].get("memory_id", "")),
                    "item": self.items[idx],
                    "score": float(score),
                    "retrieval_score": float(score),
                    "retrieval_method": "bge_m3_" + "_".join(requested),
                    "facet_rrf_scores": facet_rrf,
                    "modality_ranks": trace["modality_ranks"],
                    "raw_modality_scores": trace["raw_modality_scores"],
                }
            )
        result.sort(key=lambda row: (row["score"], str(row["memory_id"])), reverse=True)
        return result[: max(0, int(top_k))]


def correction_text(item: Mapping[str, Any]) -> str:
    content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
    signals = content.get("failure_signals", [])
    if isinstance(signals, str):
        signals = [signals]
    return "Failure mode: {mode}\nFailure signals: {signals}".format(
        mode=str(content.get("failure_mode", "")).strip(),
        signals="; ".join(str(value).strip() for value in signals if str(value).strip()),
    ).strip()


def reciprocal_rank_fusion(
    structure: Sequence[Mapping[str, Any]],
    correction: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 12,
    rank_constant: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse only memories present in both rankings."""
    structure_by_id = {str(row.get("memory_id")): (rank, row) for rank, row in enumerate(structure, 1)}
    correction_by_id = {str(row.get("memory_id")): (rank, row) for rank, row in enumerate(correction, 1)}
    shared = set(structure_by_id) & set(correction_by_id)
    result = []
    for memory_id in shared:
        structure_rank, structure_row = structure_by_id[memory_id]
        correction_rank, correction_row = correction_by_id[memory_id]
        score = 1.0 / (rank_constant + structure_rank) + 1.0 / (rank_constant + correction_rank)
        merged = dict(structure_row)
        merged.update(
            {
                "score": score,
                "retrieval_score": score,
                "retrieval_method": "r5_dual_retrieval",
                "structure_rank": structure_rank,
                "correction_rank": correction_rank,
                "correction_similarity": float(correction_row.get("score", 0.0)),
            }
        )
        result.append(merged)
    result.sort(key=lambda row: (row["score"], -row["structure_rank"]), reverse=True)
    return result[:top_k]


def reciprocal_rank_fusion_union(
    rankings: Sequence[Sequence[Mapping[str, Any]]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse the union of rankings; absent candidates receive no contribution."""
    scores: Dict[str, float] = {}
    rows: Dict[str, Dict[str, Any]] = {}
    rank_trace: Dict[str, List[Dict[str, int]]] = {}
    for ranking_index, ranking in enumerate(rankings):
        for rank, raw in enumerate(ranking, 1):
            memory_id = str(raw.get("memory_id", ""))
            if not memory_id:
                continue
            rows.setdefault(memory_id, dict(raw))
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (rank_constant + rank)
            rank_trace.setdefault(memory_id, []).append({"ranking": ranking_index, "rank": rank})
    result = []
    for memory_id, score in scores.items():
        row = dict(rows[memory_id])
        row.update(
            {
                "score": float(score),
                "retrieval_score": float(score),
                "retrieval_method": "rrf_union",
                "rrf_ranks": rank_trace[memory_id],
            }
        )
        result.append(row)
    result.sort(key=lambda row: (row["score"], str(row["memory_id"])), reverse=True)
    return result[:top_k]


def soft_draft_rerank(
    entries: Sequence[Mapping[str, Any]],
    correction_scores: Mapping[str, float],
    *,
    structure_weight: float = 0.70,
    error_weight: float = 0.30,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    """Blend structural relevance with draft-error similarity without hard intersection."""
    if structure_weight < 0 or error_weight < 0 or structure_weight + error_weight <= 0:
        raise ValueError("soft rerank weights must be non-negative and have a positive sum")
    total = structure_weight + error_weight
    structure_weight /= total
    error_weight /= total
    structure_values = [float(row.get("rerank_score", row.get("score", 0.0))) for row in entries]
    error_values = [float(correction_scores.get(str(row.get("memory_id", "")), 0.0)) for row in entries]

    def normalise(values: Sequence[float]) -> List[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high <= low:
            return [1.0 if high > 0 else 0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    structure_norm = normalise(structure_values)
    error_norm = normalise(error_values)
    result: List[Dict[str, Any]] = []
    for raw, structure, error, structure_raw, error_raw in zip(
        entries, structure_norm, error_norm, structure_values, error_values,
    ):
        row = dict(raw)
        row.update(
            {
                "structure_score_raw": structure_raw,
                "draft_error_score_raw": error_raw,
                "structure_score_normalized": structure,
                "draft_error_score_normalized": error,
                "score": structure_weight * structure + error_weight * error,
                "retrieval_method": "draft_error_soft_rerank",
            }
        )
        row["retrieval_score"] = row["score"]
        result.append(row)
    result.sort(key=lambda row: (row["score"], row["structure_score_raw"]), reverse=True)
    return result[:top_k]


def serializable_entries(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Remove the full memory object while preserving every retrieval score."""
    payload = []
    for value in entries:
        row = {key: item for key, item in dict(value).items() if key != "item"}
        payload.append(row)
    return payload
