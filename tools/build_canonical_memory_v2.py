#!/usr/bin/env python3
"""Build a read-only-derived canonical store without modifying legacy memory."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.principle_schema import build_memory_views, validate_principle_content


def clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_action(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(clean(entry) for entry in value if clean(entry))
    return clean(value)


def canonical_content(principle: Dict[str, Any], final_action: str) -> Dict[str, str]:
    return {
        "introspection": clean(principle.get("introspection")),
        "trigger_condition": clean(principle.get("trigger_condition")),
        "action_guidance": final_action,
        "scope": clean(principle.get("scope")),
        "failure_mode": clean(principle.get("failure_mode")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-memory", required=True)
    parser.add_argument("--stage2-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-action-match", type=float, default=0.75)
    args = parser.parse_args()

    by_question: Dict[str, list] = defaultdict(list)
    for level in range(1, 5):
        path = Path(args.stage2_root) / f"dpo_level{level}" / "01_memory_bank" / "llama_semantic_memory_bank.json"
        for record in json.loads(path.read_text(encoding="utf-8")):
            validated = record.get("validated_principle")
            if not isinstance(validated, dict):
                continue
            principle = validated.get("principle") if isinstance(validated.get("principle"), dict) else validated
            if not isinstance(principle, dict):
                continue
            question = clean(principle.get("source_question_text") or record.get("source_question_text"))
            by_question[question].append((normalize_action(principle.get("action_guidance")), principle, level))

    legacy = json.loads(Path(args.legacy_memory).read_text(encoding="utf-8"))
    output_items = []
    rejected = []
    for item in legacy:
        question = clean(item.get("question"))
        final_action = clean(item.get("principle"))
        options = by_question.get(question, [])
        ranked = sorted(
            ((difflib.SequenceMatcher(None, final_action, action).ratio(), principle, level) for action, principle, level in options),
            key=lambda row: row[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] < args.min_action_match:
            rejected.append({"memory_id": item.get("memory_id"), "question": question, "reason": "no canonical stage2 match", "best_action_similarity": ranked[0][0] if ranked else 0.0})
            continue
        similarity, principle, level = ranked[0]
        content = canonical_content(principle, final_action)
        normalized, errors = validate_principle_content(content)
        if errors:
            rejected.append({"memory_id": item.get("memory_id"), "question": question, "reason": "; ".join(errors), "best_action_similarity": similarity})
            continue
        rebuilt = {
            "memory_id": item.get("memory_id"),
            "question": question,
            "core_skill": clean(item.get("core_skill")),
            "context_constraints": clean(item.get("context_constraints")),
            "task_objective": clean(item.get("task_objective")),
            "principle": final_action,
            "content": normalized,
            "views": build_memory_views(normalized),
            "source_round": item.get("source_round"),
            "source_dataset": item.get("source_dataset"),
            "quality_metrics": item.get("quality_metrics", {}),
            "status": item.get("status", "active"),
            "item_version": item.get("item_version", 1),
            "lineage": item.get("lineage", {"parent_memory_ids": [], "supersedes": []}),
            "schema_version": "memory.v2",
            "schema_source": "canonical_stage2",
            "canonical_provenance": {"stage2_level": level, "action_match_similarity": similarity},
        }
        output_items.append(rebuilt)

    root = Path(args.output)
    (root / "principles").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "principles" / "principles.json").write_text(json.dumps(output_items, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "schema_version": "memory.v2",
        "source_count": len(legacy),
        "canonical_count": len(output_items),
        "rejected_count": len(rejected),
        "minimum_action_match": args.min_action_match,
        "legacy_source_was_modified": False,
        "rejected": rejected,
    }
    (root / "metadata" / "migration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejected"}, indent=2))


if __name__ == "__main__":
    main()
