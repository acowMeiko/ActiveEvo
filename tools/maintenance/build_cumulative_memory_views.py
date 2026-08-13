#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


LEVELS = ["level1", "level2", "level3", "level4"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def level_bank_path(root: Path, level: str) -> Path:
    return root / "stage2" / level / "01_memory_bank" / "semantic_memory_bank_model.json"


def level_vector_path(root: Path, level: str) -> Path:
    return root / "stage2" / level / "02_vector_db_model" / "memory_data.json"


def cumulative_name(idx: int) -> str:
    return f"cumulative_level{idx + 1}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cumulative memory views from per-level Stage 2 outputs")
    parser.add_argument("--output-root", required=True, help="Root output dir, e.g. output_qwen_20260414_105343")
    parser.add_argument(
        "--out-subdir",
        default="stage2_cumulative_model",
        help="Subdir under output-root to store cumulative memory views",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    out_root = root / args.out_subdir

    cumulative_records: List[Dict[str, Any]] = []
    cumulative_ids: List[str] = []
    cumulative_metadatas: List[Dict[str, Any]] = []
    cumulative_concept_texts: List[str] = []
    cumulative_condition_texts: List[str] = []
    cumulative_target_texts: List[str] = []
    cumulative_concept_embeddings: List[List[float]] = []
    cumulative_condition_embeddings: List[List[float]] = []
    cumulative_target_embeddings: List[List[float]] = []
    cumulative_sources: List[str] = []

    report: Dict[str, Any] = {"output_root": str(root), "views": {}}

    for idx, level in enumerate(LEVELS):
        bank = load_json(level_bank_path(root, level))
        vector = load_json(level_vector_path(root, level))

        cumulative_records.extend(bank.get("records", []))
        cumulative_ids.extend(vector.get("ids", []))
        cumulative_metadatas.extend(vector.get("metadatas", []))
        cumulative_concept_texts.extend(vector.get("concept_texts", []))
        cumulative_condition_texts.extend(vector.get("condition_texts", []))
        cumulative_target_texts.extend(vector.get("target_texts", []))
        cumulative_concept_embeddings.extend(vector.get("concept_embeddings", []))
        cumulative_condition_embeddings.extend(vector.get("condition_embeddings", []))
        cumulative_target_embeddings.extend(vector.get("target_embeddings", []))
        cumulative_sources.extend(vector.get("source_datasets", []))

        view_name = cumulative_name(idx)
        view_root = out_root / view_name

        bank_payload = {
            "source": f"cumulative:{','.join(LEVELS[: idx + 1])}",
            "total_records": len(cumulative_records),
            "records": cumulative_records,
        }
        vector_payload = {
            "source_datasets": sorted(set(cumulative_sources)),
            "ids": cumulative_ids,
            "metadatas": cumulative_metadatas,
            "records": cumulative_records,
            "concept_texts": cumulative_concept_texts,
            "condition_texts": cumulative_condition_texts,
            "target_texts": cumulative_target_texts,
            "concept_embeddings": cumulative_concept_embeddings,
            "condition_embeddings": cumulative_condition_embeddings,
            "target_embeddings": cumulative_target_embeddings,
        }

        save_json(view_root / "01_memory_bank" / "semantic_memory_bank_model.json", bank_payload)
        save_json(view_root / "02_vector_db_model" / "memory_data.json", vector_payload)
        report["views"][view_name] = {
            "levels": LEVELS[: idx + 1],
            "records": len(cumulative_records),
            "path": str(view_root),
        }

    save_json(out_root / "cumulative_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
