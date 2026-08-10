#!/usr/bin/env python
"""
Stage 2 - Step 2: 向量化与多维索引建库 (极简扁平化 + 全局融合版)
将极简扁平化的三层语义分别向量化，构建独立维度索引。
支持单级别建库，也支持 --dataset all 全局级别融合建库。
"""
import os
import sys
import argparse
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import Config
from utils.embedding_utils import DEFAULT_EMBEDDING_MODEL, allow_hash_fallback, load_sentence_transformer_with_fallback
from utils.io_utils import load_json, save_json
from utils.output_paths import stage2_memory_file, stage2_vector_dir
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_function():
    try:
        device = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        logger.info("加载 sentence-transformers (%s)...", DEFAULT_EMBEDDING_MODEL)
        model, loaded_from = load_sentence_transformer_with_fallback(
            model_name=DEFAULT_EMBEDDING_MODEL,
            device=device,
            local_files_only=True,
        )
        logger.info("从 %s 加载 embedding 模型成功", loaded_from)
        return model
    except Exception:
        if not allow_hash_fallback():
            raise
        logger.warning("真实 embedding 模型不可用，使用哈希向量兜底 (仅供测试)")

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


def load_records(dataset_name: str, config: Config):
    """
    智能加载器：支持加载单一数据集，或融合所有数据集 (dataset_name == 'all')
    """
    root = config.paths.output_dir
    
    if dataset_name == "all":
        datasets = ["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"]
    else:
        datasets = [dataset_name]
        
    merged_records = []
    
    for ds in datasets:
        file_path = stage2_memory_file(root, ds)
        if not file_path.exists():
            logger.warning(f"跳过缺失的语义库: {file_path}")
            continue
            
        data = load_json(str(file_path))
        records = data.get("records", []) if isinstance(data, dict) else []
        
        for record in records:
            if isinstance(record, dict):
                item = dict(record)
                item["source_dataset"] = ds
                merged_records.append(item)
                
    return merged_records, datasets


def build_vector_db(records: list, source_datasets: list, output_dir: str, use_chroma: bool = False):
    if not records:
        logger.error("没有数据可处理，请检查输入文件。")
        return

    logger.info(f"纳入数据集: {source_datasets}")
    logger.info(f"准备处理总数据量: {len(records)} 条")
    
    embedding_fn = get_embedding_function()

    concept_texts = []
    condition_texts = []
    target_texts = []
    metadatas = []
    ids = []

    def clean_text(value) -> str:
        return str(value).strip() if value is not None else ""

    for i, record in enumerate(records):
        question = clean_text(record.get("question", ""))
        if not question:
            continue

        core_concept = clean_text(record.get("core_math_concept", ""))
        conditions = clean_text(record.get("conditions_structure", ""))
        target = clean_text(record.get("solution_target", ""))
        principle = clean_text(record.get("principle", ""))

        concept_texts.append(core_concept)
        condition_texts.append(conditions)
        target_texts.append(target)
        
        metadatas.append({
            "question": question,
            "core_math_concept": core_concept,
            "conditions_structure": conditions,
            "solution_target": target,
            "principle": principle,
            "source": record.get("source_dataset", "unknown")
        })
        
        source_dataset = record.get("source_dataset", "unknown")
        ids.append(f"mem_{source_dataset}_{i}")

    logger.info(f"准备分别向量化 {len(ids)} 条语义记录...")
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
            logger.info(f"ChromaDB 多维向量库已更新至: {output_dir}")
            return
        except ImportError:
            logger.warning("ChromaDB 未安装，自动降级为 FAISS 方案")

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

        logger.info(f"FAISS 多维向量库已构建完成: {output_dir}")
        logger.info(f"统计 -> 概念索引: {concept_index.ntotal}, 条件索引: {condition_index.ntotal}, 目标索引: {target_index.ntotal}")
    except ImportError:
        logger.error("FAISS 库未安装，无法建库！请执行: pip install faiss-cpu")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Step 2: 向量数据库建库 (正交三维)")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        choices=["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4", "all"],
                        help="数据集名称，输入 'all' 会融合构建全局知识库")
    parser.add_argument("--output", "-o", type=str, default=None, help="自定义输出目录")
    parser.add_argument("--use-chroma", action="store_true", default=False, help="开启使用 ChromaDB (默认使用 FAISS)")
    args = parser.parse_args()

    config = Config.from_core_file()
    
    records, source_datasets = load_records(args.dataset, config)
    
    if not records:
        logger.error(f"未在 {source_datasets} 中找到任何有效数据，请先运行 parse_memory.py！")
        return

    if args.output:
        output_dir = args.output
    elif args.dataset == "all":
        output_dir = str(config.paths.output_dir / "stage2" / "global_memory" / "02_vector_db")
    else:
        output_dir = str(stage2_vector_dir(config.paths.output_dir, args.dataset))

    build_vector_db(records, source_datasets, output_dir, args.use_chroma)


if __name__ == "__main__":
    main()
