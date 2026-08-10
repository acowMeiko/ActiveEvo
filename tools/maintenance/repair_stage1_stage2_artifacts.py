#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DATASETS = ["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def has_success_failure_pair(item: Dict[str, Any]) -> bool:
    return all(
        [
            clean_text(item.get("question") or item.get("question_text")),
            clean_text(item.get("expected_answer")),
            clean_text(item.get("chosen_trajectory") or item.get("chosen_path")),
            clean_text(item.get("hard_negative_trajectory") or item.get("rejected_path")),
        ]
    )


def diff_ok(item: Dict[str, Any]) -> bool:
    contrast = item.get("contrast", {}) if isinstance(item.get("contrast", {}), dict) else {}
    return contrast.get("status") == "success"


def should_be_stage2_candidate(item: Dict[str, Any]) -> bool:
    return diff_ok(item) and has_success_failure_pair(item)


def patch_learning_packets(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    changed = 0
    candidates = 0
    for item in items:
        new_value = should_be_stage2_candidate(item)
        old_top = bool(item.get("stage2_candidate", item.get("is_usable", False)))
        old_alias = bool(item.get("is_usable", False))
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            item["metadata"] = metadata
        old_meta = bool(metadata.get("stage2_candidate", metadata.get("is_usable", False)))

        item["stage2_candidate"] = new_value
        item["is_usable"] = new_value
        metadata["stage2_candidate"] = new_value
        metadata["is_usable"] = new_value

        if new_value:
            candidates += 1
        if old_top != new_value or old_alias != new_value or old_meta != new_value:
            changed += 1

    return items, {
        "total": len(items),
        "stage2_candidates": candidates,
        "changed_items": changed,
    }


def patch_registry(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    records = data.get("records", [])
    changed = 0
    candidates = 0
    for record in records:
        new_value = should_be_stage2_candidate(
            {
                "question_text": record.get("question_text"),
                "expected_answer": record.get("expected_answer"),
                "chosen_trajectory": record.get("chosen_trajectory"),
                "hard_negative_trajectory": record.get("hard_negative_trajectory"),
                "contrast": record.get("contrast"),
            }
        )
        old_top = bool(record.get("stage2_candidate", record.get("is_usable", False)))
        old_alias = bool(record.get("is_usable", False))
        source_metadata = record.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = {}
            record["source_metadata"] = source_metadata
        old_meta = bool(source_metadata.get("stage2_candidate", source_metadata.get("is_usable", False)))

        record["stage2_candidate"] = new_value
        record["is_usable"] = new_value
        source_metadata["stage2_candidate"] = new_value
        source_metadata["is_usable"] = new_value

        if new_value:
            candidates += 1
        if old_top != new_value or old_alias != new_value or old_meta != new_value:
            changed += 1

    summary = data.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        data["summary"] = summary
    summary["usable_records"] = candidates
    summary["stage2_candidates"] = candidates
    summary["total_records"] = len(records)
    return data, {
        "total": len(records),
        "stage2_candidates": candidates,
        "changed_items": changed,
    }


def maybe_backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline repair for Stage 1 learning_packets and Stage 2 registry")
    parser.add_argument("--output-root", required=True, help="Root output dir, e.g. output_qwen_20260414_105343")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Dataset names to patch")
    parser.add_argument("--write", action="store_true", help="Write changes in place")
    parser.add_argument("--backup", action="store_true", help="Create .bak files before first write")
    parser.add_argument("--report-out", help="Optional JSON report path")
    args = parser.parse_args()

    root = Path(args.output_root)
    report: Dict[str, Any] = {"output_root": str(root), "datasets": {}}

    for dataset in args.datasets:
        lp_path = root / "stage1" / dataset / "04_learning_packets" / "learning_packets.json"
        reg_path = root / "stage2" / dataset / "00_dsd_registry" / "question_dsd_registry.json"
        dataset_report: Dict[str, Any] = {}

        if lp_path.exists():
            lp_data = load_json(lp_path)
            if not isinstance(lp_data, list):
                raise ValueError(f"Unexpected learning_packets format: {lp_path}")
            patched, stats = patch_learning_packets(lp_data)
            dataset_report["learning_packets"] = {"path": str(lp_path), **stats}
            if args.write:
                if args.backup:
                    maybe_backup(lp_path)
                save_json(lp_path, patched)
        else:
            dataset_report["learning_packets"] = {"path": str(lp_path), "missing": True}

        if reg_path.exists():
            reg_data = load_json(reg_path)
            if not isinstance(reg_data, dict) or "records" not in reg_data:
                raise ValueError(f"Unexpected registry format: {reg_path}")
            patched, stats = patch_registry(reg_data)
            dataset_report["registry"] = {"path": str(reg_path), **stats}
            if args.write:
                if args.backup:
                    maybe_backup(reg_path)
                save_json(reg_path, patched)
        else:
            dataset_report["registry"] = {"path": str(reg_path), "missing": True}

        report["datasets"][dataset] = dataset_report

    if args.report_out:
        save_json(Path(args.report_out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
