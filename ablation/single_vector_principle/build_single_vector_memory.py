#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from embedding_utils import DEFAULT_EMBEDDING_MODEL, load_sentence_transformer_with_fallback


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_matrix(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return arr / norms


def principle_text(record: Dict[str, Any]) -> str:
    value = record.get("principle")
    if isinstance(value, dict):
        parts: List[str] = []
        experience = value.get("experience_principle")
        if isinstance(experience, dict):
            parts.append(clean_text(experience.get("principle")))
        parts.append(clean_text(value.get("principle")))
        return "\n".join(part for part in parts if part)
    return clean_text(value)


def coerce_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("status", "active") != "active":
            continue
        text = principle_text(item)
        if not text:
            continue
        key = (clean_text(item.get("memory_id")), text)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "memory_id": clean_text(item.get("memory_id")) or f"single_vec_{len(records):06d}",
                "question": clean_text(item.get("question")),
                "core_skill": clean_text(item.get("core_skill") or item.get("core_math_concept")),
                "context_constraints": clean_text(item.get("context_constraints") or item.get("conditions_structure")),
                "task_objective": clean_text(item.get("task_objective") or item.get("solution_target")),
                "principle": text,
                "source_round": item.get("source_round"),
                "source_dataset": clean_text(item.get("source_dataset")),
            }
        )
    return records


def build_one(source_path: Path, output_dir: Path, model_name: str, device: str, batch_size: int) -> Dict[str, Any]:
    source_items = load_json(source_path)
    if not isinstance(source_items, list):
        raise ValueError(f"Principle source must be a JSON list: {source_path}")

    records = coerce_records(source_items)
    if not records:
        raise ValueError(f"No usable principle records found in {source_path}")

    model, loaded_from = load_sentence_transformer_with_fallback(
        model_name=model_name,
        device=device,
        local_files_only=True,
    )
    texts = [record["principle"] for record in records]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = normalize_matrix(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "single_vector_principles.json", records)
    np.savez_compressed(output_dir / "single_vector_principles.npz", embeddings=embeddings)

    manifest = {
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "embedding_model": model_name,
        "loaded_embedding_model": loaded_from,
        "embedding_device": device,
        "vector_count": len(records),
        "vector_definition": "one normalized embedding per principle text only",
    }
    save_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-vector principle memory for ablation.")
    parser.add_argument("--source", type=Path, help="principles.json source for one memory view")
    parser.add_argument("--source-root", type=Path, help="root containing cumulative_level*/principles/principles.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--embedding-model", default=os.environ.get("MEMUA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--device", default=os.environ.get("MEMUA_EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    manifests = {}
    if args.source_root:
        for level in args.levels:
            source = args.source_root / f"cumulative_level{level}" / "principles" / "principles.json"
            if not source.exists():
                raise FileNotFoundError(source)
            out = args.output_dir / f"cumulative_level{level}"
            manifests[f"cumulative_level{level}"] = build_one(source, out, args.embedding_model, args.device, args.batch_size)
    elif args.source:
        manifests["single"] = build_one(args.source, args.output_dir, args.embedding_model, args.device, args.batch_size)
    else:
        raise SystemExit("Provide either --source or --source-root.")

    save_json(args.output_dir / "build_summary.json", manifests)
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
