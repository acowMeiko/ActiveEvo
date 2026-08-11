#!/usr/bin/env python
"""
Stage 2 - merged memory-bank construction and vector-db building.

This script:
1. Reads Stage 1 DPO data directly.
2. Builds the flattened semantic memory bank.
3. Saves a pre-embedding preview for human inspection.
4. Builds the three-facet vector database.
"""
import os
import sys
import re
import json
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.config import Config
from embedding_utils import DEFAULT_EMBEDDING_MODEL, allow_hash_fallback, load_sentence_transformer_with_fallback
from utils.io_utils import load_json, save_json
from output_paths import stage1_principle_file, stage1_learning_packets_file, stage2_memory_file, stage2_vector_dir
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DATASET_ORDER = ["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"]


def get_embedding_function():
    try:
        device = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        logger.info("Loading sentence-transformers model: %s", DEFAULT_EMBEDDING_MODEL)
        model, loaded_from = load_sentence_transformer_with_fallback(
            model_name=DEFAULT_EMBEDDING_MODEL,
            device=device,
            local_files_only=True,
        )
        logger.info("Loaded vector-db embedding model from %s", loaded_from)
        return model
    except Exception:
        if not allow_hash_fallback():
            raise
        logger.warning("Real embedding model unavailable; using hash embedding fallback for testing only")

        class HashEmbeddingModel:
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

        return HashEmbeddingModel()


def _to_numpy(embeddings):
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    cpu_method = getattr(embeddings, "cpu", None)
    if callable(cpu_method):
        cpu_result = cpu_method()
        numpy_method = getattr(cpu_result, "numpy", None)
        embeddings = numpy_method() if callable(numpy_method) else cpu_result
    return np.asarray(embeddings, dtype=np.float32)


def clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def extract_question(input_text: str) -> str:
    if not input_text:
        return ""
    patterns = [
        r'\[Question\]:\s*\n?(.*?)(?=\n\n\[|$)',
        r'\[Question\]:\s*(.+?)(?:\n\n|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, input_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def build_record_from_learning_packet(item: Dict[str, Any], idx: int, dataset_name: str) -> Dict[str, Any]:
    """从 learning packet 构建向量库记录。"""
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
    question = clean_text(item.get("question", ""))
    if not question:
        return {"id": idx, "status": "no_question_found", "source_dataset": dataset_name}

    semantics = item.get("problem_semantics", {})
    core_skill = get_field(semantics, CORE_FIELD)
    context_constraints = get_field(semantics, CONDITION_FIELD)
    task_objective = get_field(semantics, TARGET_FIELD)

    if not (core_skill or context_constraints or task_objective):
        return {"id": idx, "status": "missing_semantics", "source_dataset": dataset_name, "question": question}

    # 从 contrast.differences 提炼 principle 摘要（取 CorrectThinking 列表拼接）
    differences = item.get("contrast", {}).get("differences", [])
    correct_thinking_parts = [
        d.get("CorrectThinking", "") for d in differences if d.get("CorrectThinking")
    ]
    principle = "; ".join(correct_thinking_parts) if correct_thinking_parts else ""

    return {
        "id": idx,
        "status": "success",
        "question": question,
        CORE_FIELD: core_skill,
        CONDITION_FIELD: context_constraints,
        TARGET_FIELD: task_objective,
        # backward compat aliases
        "core_math_concept": core_skill,
        "conditions_structure": context_constraints,
        "solution_target": task_objective,
        "principle": principle,
        "source_dataset": dataset_name,
    }


def build_record_from_principle_record(item: Dict[str, Any], idx: int, dataset_name: str) -> Dict[str, Any]:
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
    question = extract_question(clean_text(item.get("input", ""))) or extract_question(clean_text(item.get("instruction", "")))
    if not question:
        return {"id": idx, "status": "no_question_found", "source_dataset": dataset_name}

    core_skill = get_field(item, CORE_FIELD)
    context_constraints = get_field(item, CONDITION_FIELD)
    task_objective = get_field(item, TARGET_FIELD)
    principle = clean_text(item.get("chosen", ""))

    if not (core_skill or context_constraints or task_objective):
        return {"id": idx, "status": "missing_semantics", "source_dataset": dataset_name, "question": question}

    return {
        "id": idx,
        "status": "success",
        "question": question,
        CORE_FIELD: core_skill,
        CONDITION_FIELD: context_constraints,
        TARGET_FIELD: task_objective,
        # backward compat aliases
        "core_math_concept": core_skill,
        "conditions_structure": context_constraints,
        "solution_target": task_objective,
        "principle": principle,
        "source_dataset": dataset_name,
    }


def dataset_paths(config: Config, dataset_name: str) -> Tuple[List[str], Path, Path, Path]:
    if dataset_name == "all":
        source_datasets = DATASET_ORDER
        memory_path = config.paths.output_dir / "stage2" / "global_memory" / "01_memory_bank" / "semantic_memory_bank.json"
        vector_dir = config.paths.output_dir / "stage2" / "global_memory" / "02_vector_db"
        preview_path = config.paths.output_dir / "stage2" / "global_memory" / "00_pre_embedding_preview" / "pre_embedding_preview.json"
    else:
        source_datasets = [dataset_name]
        memory_path = stage2_memory_file(config.paths.output_dir, dataset_name)
        vector_dir = stage2_vector_dir(config.paths.output_dir, dataset_name)
        preview_path = config.paths.output_dir / "stage2" / dataset_name / "00_pre_embedding_preview" / "pre_embedding_preview.json"
    return source_datasets, memory_path, vector_dir, preview_path


def build_memory_bank_from_stage1(dataset_name: str, config: Config) -> Tuple[Dict[str, Any], List[str], Path]:
    source_datasets, memory_path, _, _ = dataset_paths(config, dataset_name)
    records: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for ds in source_datasets:
        # 优先从 learning_packets.json 读取（新流程）
        lp_path = stage1_learning_packets_file(config.paths.output_dir, ds)
        principle_path = stage1_principle_file(config.paths.output_dir, ds)

        if lp_path.exists():
            data = load_json(str(lp_path))
            if not isinstance(data, list):
                logger.warning("Invalid learning_packets format, skipped: %s", lp_path)
                continue
            logger.info("Stage2 reading from learning_packets: %s (%d items)", lp_path, len(data))
            for idx, item in enumerate(data):
                metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                stage2_candidate = bool(item.get("stage2_candidate", metadata.get("stage2_candidate", False)))
                if not stage2_candidate:
                    contrast = item.get("contrast", {}) if isinstance(item.get("contrast", {}), dict) else {}
                    diff_ok = contrast.get("status") == "success"
                    has_pair = all([
                        clean_text(item.get("question", "")),
                        clean_text(item.get("chosen_trajectory", "")) or clean_text(item.get("chosen_path", "")),
                        clean_text(item.get("hard_negative_trajectory", "")) or clean_text(item.get("rejected_path", "")),
                    ])
                    stage2_candidate = diff_ok and has_pair
                if not stage2_candidate:
                    failed.append({"id": idx, "status": "not_usable", "source_dataset": ds})
                    continue
                parsed = build_record_from_learning_packet(item, idx, ds)
                if parsed.get("status") == "success":
                    records.append(parsed)
                else:
                    failed.append(parsed)
        elif principle_path.exists():
            # 备用：如果 learning_packets.json 不存在，回退到旧版 principle 文件
            logger.warning("learning_packets not found, falling back to principle file: %s", principle_path)
            data = load_json(str(principle_path))
            if not isinstance(data, list):
                logger.warning("Invalid principle data format, skipped: %s", principle_path)
                continue
            for idx, item in enumerate(data):
                parsed = build_record_from_principle_record(item, idx, ds)
                if parsed.get("status") == "success":
                    records.append(parsed)
                else:
                    failed.append(parsed)
        else:
            logger.warning("No input found for dataset %s (checked %s and %s)", ds, lp_path, principle_path)

    memory_bank = {
        "source_datasets": source_datasets,
        "total": len(records) + len(failed),
        "success": len(records),
        "failed": len(failed),
        "records": records,
        "failed_records": failed,
    }

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(memory_bank, str(memory_path), indent=4)
    logger.info("Semantic memory bank saved: %s", memory_path)
    if failed:
        status_counts: Dict[str, int] = {}
        for item in failed:
            status = item.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        logger.warning("Memory-bank failures: %s", status_counts)
    return memory_bank, source_datasets, memory_path


def save_pre_embedding_preview(records: List[Dict[str, Any]], preview_path: Path, limit: int = 10) -> None:
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
    preview_records = []
    for record in records:
        preview_records.append({
            "id": record.get("id"),
            "source_dataset": record.get("source_dataset", "unknown"),
            "question": clean_text(record.get("question", "")),
            CORE_FIELD: get_field(record, CORE_FIELD),
            CONDITION_FIELD: get_field(record, CONDITION_FIELD),
            TARGET_FIELD: get_field(record, TARGET_FIELD),
            "principle": clean_text(record.get("principle", "")),
        })

    payload = {
        "total_records": len(preview_records),
        "preview_limit": limit,
        "preview_examples": preview_records[:limit],
        "all_records": preview_records,
    }
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, str(preview_path), indent=4)

    logger.info("Pre-embedding preview saved: %s", preview_path)
    for sample in preview_records[:min(limit, 3)]:
        logger.info(
            "Preview | id=%s | source=%s | skill=%s | constraints=%s | objective=%s",
            sample.get("id"),
            sample.get("source_dataset"),
            sample.get(CORE_FIELD, ""),
            sample.get(CONDITION_FIELD, ""),
            sample.get(TARGET_FIELD, ""),
        )


def build_vector_db(records: List[Dict[str, Any]], source_datasets: List[str], output_dir: str, use_chroma: bool = False):
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
    if not records:
        logger.error("No records available for embedding")
        return

    logger.info("Embedding source datasets: %s", source_datasets)
    logger.info("Preparing %s records for three-facet embedding", len(records))
    embedding_fn = get_embedding_function()

    concept_texts: List[str] = []
    condition_texts: List[str] = []
    target_texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for i, record in enumerate(records):
        question = clean_text(record.get("question", ""))
        if not question:
            continue

        core_skill = get_field(record, CORE_FIELD)
        context_constraints = get_field(record, CONDITION_FIELD)
        task_objective = get_field(record, TARGET_FIELD)
        principle = clean_text(record.get("principle", ""))

        concept_texts.append(core_skill)
        condition_texts.append(context_constraints)
        target_texts.append(task_objective)
        metadatas.append({
            "question": question,
            CORE_FIELD: core_skill,
            CONDITION_FIELD: context_constraints,
            TARGET_FIELD: task_objective,
            # backward compat aliases
            "core_math_concept": core_skill,
            "conditions_structure": context_constraints,
            "solution_target": task_objective,
            "principle": principle,
            "source": record.get("source_dataset", "unknown"),
        })
        ids.append(f"mem_{record.get('source_dataset', 'unknown')}_{i}")

    concept_embeddings = _to_numpy(embedding_fn.encode(concept_texts, show_progress_bar=True))
    condition_embeddings = _to_numpy(embedding_fn.encode(condition_texts, show_progress_bar=True))
    target_embeddings = _to_numpy(embedding_fn.encode(target_texts, show_progress_bar=True))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_chroma:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=output_dir)
            for name, texts, embeddings in [
                ("semantic_concept", concept_texts, concept_embeddings),
                ("semantic_condition", condition_texts, condition_embeddings),
                ("semantic_target", target_texts, target_embeddings),
            ]:
                collection = client.get_or_create_collection(name=name)
                collection.upsert(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
            logger.info("ChromaDB vector DB updated at %s", output_dir)
            return
        except ImportError:
            logger.warning("ChromaDB not installed; falling back to FAISS")

    try:
        import faiss

        faiss.normalize_L2(concept_embeddings)
        faiss.normalize_L2(condition_embeddings)
        faiss.normalize_L2(target_embeddings)

        concept_index = faiss.IndexFlatIP(concept_embeddings.shape[1])
        condition_index = faiss.IndexFlatIP(condition_embeddings.shape[1])
        target_index = faiss.IndexFlatIP(target_embeddings.shape[1])

        concept_index.add(concept_embeddings)
        condition_index.add(condition_embeddings)
        target_index.add(target_embeddings)

        faiss.write_index(concept_index, str(output_path / "concept.index"))
        faiss.write_index(condition_index, str(output_path / "condition.index"))
        faiss.write_index(target_index, str(output_path / "target.index"))

        db_data = {
            "source_datasets": source_datasets,
            "ids": ids,
            "metadatas": metadatas,
            "concept_texts": concept_texts,
            "condition_texts": condition_texts,
            "target_texts": target_texts,
            "concept_embeddings": concept_embeddings.tolist(),
            "condition_embeddings": condition_embeddings.tolist(),
            "target_embeddings": target_embeddings.tolist(),
        }
        save_json(db_data, str(output_path / "memory_data.json"), indent=4)
        logger.info("FAISS vector DB built at %s", output_dir)
        logger.info(
            "Index stats -> concept: %s | condition: %s | target: %s",
            concept_index.ntotal,
            condition_index.ntotal,
            target_index.ntotal,
        )
    except ImportError:
        logger.error("FAISS is not installed. Please install faiss-cpu or faiss-gpu.")


def main():
    parser = argparse.ArgumentParser(description="Merged Stage 2 memory-bank and vector-db builder")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=DATASET_ORDER + ["all"])
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--use-chroma", action="store_true", default=False)
    parser.add_argument("--preview-only", action="store_true", default=False, help="Only build memory bank and preview, skip embedding")
    parser.add_argument("--preview-limit", type=int, default=10, help="Number of preview rows saved and sampled in logs")
    args = parser.parse_args()

    config = Config.from_core_file()
    source_datasets, _, default_vector_dir, preview_path = dataset_paths(config, args.dataset)
    memory_bank, source_datasets, _ = build_memory_bank_from_stage1(args.dataset, config)
    records = memory_bank.get("records", [])

    if not records:
        logger.error("No valid records found in source datasets: %s", source_datasets)
        return

    save_pre_embedding_preview(records, preview_path, limit=args.preview_limit)
    if args.preview_only:
        logger.info("Preview-only mode enabled; embedding skipped.")
        return

    output_dir = args.output or str(default_vector_dir)
    build_vector_db(records, source_datasets, output_dir, args.use_chroma)


if __name__ == "__main__":
    main()
