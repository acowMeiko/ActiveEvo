#!/usr/bin/env python
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding_utils import DEFAULT_EMBEDDING_MODEL, load_sentence_transformer_with_fallback
from utils.io_utils import save_json


def clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def principle_from_record(record: Dict[str, Any]) -> str:
    contrast = record.get("contrast", {}) or {}
    differences = contrast.get("differences", []) or []
    parts: List[str] = []
    for diff in differences:
        text = clean_text(diff.get("CorrectThinking", ""))
        if text:
            parts.append(text)
    return "; ".join(parts)


def hash_embed(text: str, dim: int = 1024) -> List[float]:
    seed = int.from_bytes(hashlib.sha256((text or "").encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def model_embed_batch(texts: List[str], device: str = "cpu") -> List[List[float]]:
    model, loaded_from = load_sentence_transformer_with_fallback(
        model_name=DEFAULT_EMBEDDING_MODEL,
        device=device,
        local_files_only=True,
    )
    print(f"Loaded embedding model from {loaded_from}")
    emb = model.encode(texts, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb = emb / norms
    return emb.tolist()


def convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
    dsd = record.get("source_dsd", {}) or {}
    principle = principle_from_record(record)
    return {
        "question": clean_text(record.get("question_text", "")),
        "core_skill": clean_text(dsd.get("core_skill", "")),
        "context_constraints": clean_text(dsd.get("context_constraints", "")),
        "task_objective": clean_text(dsd.get("task_objective", "")),
        "core_math_concept": clean_text(dsd.get("core_skill", "")),
        "conditions_structure": clean_text(dsd.get("context_constraints", "")),
        "solution_target": clean_text(dsd.get("task_objective", "")),
        "principle": principle,
        "source_dataset": clean_text(record.get("source_dataset", "")),
        "question_id": clean_text(record.get("question_id", "")),
        "expected_answer": clean_text(record.get("expected_answer", "")),
        "stage2_candidate": bool(record.get("stage2_candidate", record.get("is_usable"))),
        "is_usable": bool(record.get("stage2_candidate", record.get("is_usable"))),
        "core_embedding": record.get("core_embedding", []),
        "context_embedding": record.get("context_embedding", []),
        "target_embedding": record.get("target_embedding", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild stage2 retriever files from question_dsd_registry.json")
    parser.add_argument("--registry", required=True, help="Path to question_dsd_registry.json")
    parser.add_argument("--semantic-bank-out", required=True, help="Output semantic_memory_bank.json path")
    parser.add_argument("--vector-db-out", required=True, help="Output directory for memory_data.json")
    parser.add_argument("--only-usable", action="store_true", default=True, help="Keep only records marked as stage2 candidates")
    parser.add_argument("--embedding-source", choices=["registry", "hash", "model"], default="registry")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    data = json.load(registry_path.open("r", encoding="utf-8"))
    records = data.get("records", [])

    converted = [convert_record(r) for r in records]
    if args.only_usable:
        converted = [r for r in converted if r["stage2_candidate"]]

    usable_records = []
    ids = []
    metadatas = []
    concept_texts = []
    condition_texts = []
    target_texts = []
    concept_embeddings = []
    condition_embeddings = []
    target_embeddings = []

    pending_records = []
    for idx, record in enumerate(converted):
        if not (record["question"] and record["principle"]):
            continue
        if args.embedding_source == "registry" and not (record["core_embedding"] and record["context_embedding"] and record["target_embedding"]):
            continue
        pending_records.append((idx, record))

    if args.embedding_source == "model":
        core_vectors = model_embed_batch([r["core_skill"] for _, r in pending_records], device=args.device)
        condition_vectors = model_embed_batch([r["context_constraints"] for _, r in pending_records], device=args.device)
        target_vectors = model_embed_batch([r["task_objective"] for _, r in pending_records], device=args.device)
    else:
        core_vectors = condition_vectors = target_vectors = None

    for pos, (idx, record) in enumerate(pending_records):
        usable_records.append(record)
        ids.append(f"mem_{record['source_dataset']}_{record['question_id'] or idx}")
        concept_texts.append(record["core_skill"])
        condition_texts.append(record["context_constraints"])
        target_texts.append(record["task_objective"])
        metadatas.append({
            "question": record["question"],
            "core_skill": record["core_skill"],
            "context_constraints": record["context_constraints"],
            "task_objective": record["task_objective"],
            "core_math_concept": record["core_math_concept"],
            "conditions_structure": record["conditions_structure"],
            "solution_target": record["solution_target"],
            "principle": record["principle"],
            "source": record["source_dataset"],
            "source_dataset": record["source_dataset"],
        })
        if args.embedding_source == "hash":
            concept_embeddings.append(hash_embed(record["core_skill"]))
            condition_embeddings.append(hash_embed(record["context_constraints"]))
            target_embeddings.append(hash_embed(record["task_objective"]))
        elif args.embedding_source == "model":
            concept_embeddings.append(core_vectors[pos])
            condition_embeddings.append(condition_vectors[pos])
            target_embeddings.append(target_vectors[pos])
        else:
            concept_embeddings.append(record["core_embedding"])
            condition_embeddings.append(record["context_embedding"])
            target_embeddings.append(record["target_embedding"])

    semantic_bank_payload = {
        "source": str(registry_path),
        "total_records": len(usable_records),
        "records": usable_records,
    }
    semantic_bank_out = Path(args.semantic_bank_out)
    semantic_bank_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(semantic_bank_payload, str(semantic_bank_out), indent=2)

    vector_dir = Path(args.vector_db_out)
    vector_dir.mkdir(parents=True, exist_ok=True)
    db_payload = {
        "source_datasets": sorted({r["source_dataset"] for r in usable_records}),
        "ids": ids,
        "metadatas": metadatas,
        "records": usable_records,
        "concept_texts": concept_texts,
        "condition_texts": condition_texts,
        "target_texts": target_texts,
        "concept_embeddings": concept_embeddings,
        "condition_embeddings": condition_embeddings,
        "target_embeddings": target_embeddings,
    }
    save_json(db_payload, str(vector_dir / "memory_data.json"), indent=2)

    print(json.dumps({
        "registry": str(registry_path),
        "usable_records": len(usable_records),
        "embedding_source": args.embedding_source,
        "semantic_bank_out": str(semantic_bank_out),
        "vector_db_out": str(vector_dir / "memory_data.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
