#!/usr/bin/env python3
"""Freeze structural Application-Gate decisions and raw/adapted memory texts."""
from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import Config
from modules.plan_module import PlanModule
from modules.principle_schema import principle_content_from_item, render_solver_memory


def load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_adapted_memory(adaptation: dict[str, Any]) -> str:
    lines = ["[Adapted Past Experience 1]"]
    lesson = str(adaptation.get("adapted_lesson", "")).strip()
    action = str(adaptation.get("adapted_action", "")).strip()
    if lesson:
        lines.append(f"- Lesson learned: {lesson}")
    if action:
        lines.append(f"- Recommended action: {action}")
    return "\n".join(lines) if action else ""


def selected_retrieval(reference: dict[str, Any]) -> dict[str, Any] | None:
    selected_ids = set((reference.get("retrieval_trace") or {}).get("selected_ids", []))
    for retrieved in reference.get("retrieved_principles", []) or []:
        memory_id = retrieved.get("memory_id") or (retrieved.get("item") or {}).get("memory_id", "")
        if not selected_ids or memory_id in selected_ids:
            return retrieved
    return None


def build_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 structural Application Gate",
        "",
        f"- Candidates: {summary['candidate_count']}",
        f"- Structurally accepted: {summary['gate_accepted_count']}",
        f"- Rejected by gate: {summary['gate_rejected_count']}",
        f"- Eligible raw/adapted comparisons: {summary['experiment_eligible_count']}",
        f"- Adapter rejected after gate acceptance: {summary['adapter_rejected_count']}",
        "",
        "The solver sees neither source questions nor gate metadata. Raw and adapted conditions use the same accepted candidate set.",
        "",
    ]
    for record in records:
        if not record.get("candidate_present"):
            continue
        judgment = record["gate_judgment"]
        adaptation = record["adaptation"]
        lines.extend([
            f"## idx {record['idx']} — {record['memory_id']}",
            "",
            f"Gate: **{'ACCEPT' if judgment.get('accepted') else 'REJECT'}**, "
            f"confidence={judgment.get('confidence', 0.0):.3f}",
            "",
            f"Gate parse stage: `{record.get('gate_parse_stage', 'not_run')}`; "
            f"strict protocol compliant: `{record.get('gate_protocol_compliant', False)}`",
            "",
            f"Protocol errors: {record.get('gate_protocol_errors', [])}",
            "",
            f"Reason: {judgment.get('reason', '')}",
            "",
            f"Violated conditions: {judgment.get('violated_conditions', [])}",
            "",
            f"Source-specific elements: {judgment.get('source_specific_elements', [])}",
            "",
            f"Adaptation status: `{adaptation.get('status', 'reject')}`",
            "",
            f"Adapted verification (audit only; not sent to Solver): {adaptation.get('adapted_verification', '')}",
            "",
            f"Removed source-specific elements: {adaptation.get('removed_source_specific_elements', [])}",
            "",
            f"Declared changes: {adaptation.get('changes', [])}",
            "",
            "Source question:",
            "",
            record.get("source_question", ""),
            "",
            "Current question:",
            "",
            record.get("question", ""),
            "",
            "Raw solver text:",
            "",
            "```text",
            record.get("raw_memory_text", "") or "(not injected)",
            "```",
            "",
            "Adapted solver text:",
            "",
            "```text",
            record.get("adapted_memory_text", "") or "(not injected)",
            "```",
            "",
            "Unified diff:",
            "",
            "```diff",
            record.get("raw_vs_adapted_diff", "") or "(no adapted comparison)",
            "```",
            "",
        ])
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    items = load_list(args.test_file)
    reference = load_list(args.reference_guided)
    reference_by_idx = {int(row["idx"]): row for row in reference}
    selected_indices = list(range(min(len(items), args.limit or len(items))))
    ports = list(args.ports)
    urls = {port: f"http://127.0.0.1:{port}/v1/chat/completions" for port in ports}
    config = Config.from_core_file()
    plans = {
        port: PlanModule(config, urls=[urls[port]], model_override=args.model, timeout=args.timeout, seed=args.seed)
        for port in ports
    }
    locks = {port: asyncio.Lock() for port in ports}

    async def prepare_one(idx: int) -> dict[str, Any]:
        item = items[idx]
        question = str(item.get("question") or item.get("problem") or "")
        frozen = reference_by_idx[idx]
        retrieved = selected_retrieval(frozen)
        candidate_present = bool(frozen.get("retrieval_used") and retrieved)
        record: dict[str, Any] = {
            "idx": idx,
            "question": question,
            "candidate_present": candidate_present,
            "memory_id": "",
            "source_question": "",
            "source_content": {},
            "schema_source": "none",
            "gate_judgment": {
                "accepted": False,
                "confidence": 0.0,
                "reason": "no_frozen_candidate",
                "violated_conditions": [],
                "source_specific_elements": [],
            },
            "adaptation": {"status": "reject", "reason": "no_frozen_candidate"},
            "experiment_eligible": False,
            "raw_memory_text": "",
            "adapted_memory_text": "",
            "raw_vs_adapted_diff": "",
        }
        if not candidate_present or retrieved is None:
            return record
        source_item = retrieved.get("item", {}) if isinstance(retrieved.get("item"), dict) else {}
        content, schema_source = principle_content_from_item(source_item)
        if not content:
            record["gate_judgment"]["reason"] = "invalid_memory_schema"
            return record
        memory_id = str(retrieved.get("memory_id") or source_item.get("memory_id", ""))
        candidate = {
            "memory_id": memory_id,
            "content": content,
            "source_item": source_item,
            "retrieval": retrieved,
        }
        port = ports[idx % len(ports)]
        async with locks[port]:
            gate = await plans[port].judge_memory_applicability(
                question, [candidate], confidence_threshold=args.confidence_threshold
            )
            judgment = (gate.get("judgments") or [{}])[0]
            adaptation = await plans[port].adapt_memory_for_question(question, candidate, judgment)
        raw_text = render_solver_memory(content)
        adapted_text = render_adapted_memory(adaptation)
        eligible = bool(judgment.get("accepted") and adaptation.get("status") in {"adapted", "unchanged"} and adapted_text)
        if not eligible:
            raw_for_experiment = ""
            adapted_for_experiment = ""
        else:
            raw_for_experiment = raw_text
            adapted_for_experiment = adapted_text
        diff = "\n".join(difflib.unified_diff(
            raw_text.splitlines(), adapted_text.splitlines(),
            fromfile="raw_memory", tofile="adapted_memory", lineterm="",
        )) if adapted_text else ""
        record.update({
            "memory_id": memory_id,
            "source_question": str(source_item.get("question", "")),
            "source_structure": {
                "core_skill": source_item.get("core_skill", ""),
                "context_constraints": source_item.get("context_constraints", ""),
                "task_objective": source_item.get("task_objective", ""),
            },
            "source_content": content,
            "schema_source": schema_source,
            "gate_judgment": judgment,
            "gate_raw_response": gate.get("raw_response", ""),
            "gate_parse_stage": gate.get("parse_stage", "parse_failed"),
            "gate_format_parsed": gate.get("format_parsed", False),
            "gate_protocol_compliant": gate.get("protocol_compliant", False),
            "gate_protocol_errors": gate.get("protocol_errors", []),
            "adaptation": adaptation,
            "experiment_eligible": eligible,
            "raw_memory_text": raw_text,
            "adapted_memory_text": adapted_text,
            "raw_memory_text_for_experiment": raw_for_experiment,
            "adapted_memory_text_for_experiment": adapted_for_experiment,
            "raw_vs_adapted_diff": diff,
            "port": port,
        })
        return record

    tasks = [asyncio.create_task(prepare_one(idx)) for idx in selected_indices]
    records: list[dict[str, Any]] = []
    for task in asyncio.as_completed(tasks):
        record = await task
        records.append(record)
        if len(records) == 1 or len(records) == len(tasks) or len(records) % 5 == 0:
            print(f"gate completed {len(records)}/{len(tasks)} idx={record['idx']}", flush=True)
    records.sort(key=lambda row: row["idx"])
    candidates = [row for row in records if row["candidate_present"]]
    accepted = [row for row in candidates if row["gate_judgment"].get("accepted")]
    eligible = [row for row in candidates if row["experiment_eligible"]]
    summary = {
        "stage": 3,
        "protocol": "source_current_structural_gate_v1",
        "test_file": str(args.test_file),
        "test_file_sha256": file_sha256(args.test_file),
        "reference_guided": str(args.reference_guided),
        "reference_guided_sha256": file_sha256(args.reference_guided),
        "model": args.model,
        "ports": ports,
        "seed": args.seed,
        "confidence_threshold": args.confidence_threshold,
        "item_count": len(records),
        "candidate_count": len(candidates),
        "gate_accepted_count": len(accepted),
        "gate_rejected_count": len(candidates) - len(accepted),
        "experiment_eligible_count": len(eligible),
        "adapter_rejected_count": len(accepted) - len(eligible),
        "gate_format_parsed_count": sum(bool(row.get("gate_format_parsed")) for row in candidates),
        "gate_protocol_compliant_count": sum(
            bool(row.get("gate_protocol_compliant")) for row in candidates
        ),
        "gate_parse_stage_counts": {
            stage: sum(row.get("gate_parse_stage") == stage for row in candidates)
            for stage in sorted({str(row.get("gate_parse_stage", "not_run")) for row in candidates})
        },
        "accepted_indices": [row["idx"] for row in accepted],
        "rejected_indices": [row["idx"] for row in candidates if not row["gate_judgment"].get("accepted")],
        "eligible_indices": [row["idx"] for row in eligible],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gate_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "text_comparison.md").write_text(
        build_report(records, summary), encoding="utf-8"
    )
    print(
        f"gate done candidates={len(candidates)} accepted={len(accepted)} "
        f"eligible={len(eligible)} output={args.output}", flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--reference-guided", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", default=[8004, 8006, 8007])
    parser.add_argument("--model", default="llama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
