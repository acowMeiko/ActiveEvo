#!/usr/bin/env python3
"""Prepare Top-10 recall -> Top-5 rerank -> batch Gate -> adaptation traces."""
from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import Config
from modules.memory_module import MemoryModule
from modules.plan_module import PlanModule
from modules.principle_schema import principle_content_from_item, render_solver_memory
from modules.reranker_module import QwenApplicabilityReranker


def load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_adapted_memory(adaptation: dict[str, Any], index: int = 1) -> str:
    lesson = str(adaptation.get("adapted_lesson", "")).strip()
    action = str(adaptation.get("adapted_action", "")).strip()
    if not action:
        return ""
    lines = [f"[Adapted Past Experience {index}]"]
    if lesson:
        lines.append(f"- Lesson learned: {lesson}")
    lines.append(f"- Recommended action: {action}")
    return "\n".join(lines)


def compact_retrieval(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": result.get("memory_id") or (result.get("item") or {}).get("memory_id", ""),
        "retrieval_score": float(result.get("retrieval_score", result.get("score", 0.0)) or 0.0),
        "dimension_similarities": result.get("dimension_similarities", {}),
        "retrieval_method": result.get("retrieval_method", ""),
    }


def audit_gate_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    """Persist the blind-label inputs for every Gate candidate, not only selections."""
    source_item = candidate.get("source_item", {})
    source_dsd = source_item.get("source_dsd", {}) if isinstance(source_item.get("source_dsd"), dict) else {}
    return {
        "memory_id": candidate["memory_id"],
        "rerank_rank": rank,
        "rerank_score": float(candidate["rerank_score"]),
        "retrieval_score": float(candidate["retrieval"].get("score", 0.0)),
        "schema_source": candidate["schema_source"],
        "source_question": str(source_item.get("question", source_item.get("source_question_text", ""))),
        "source_structure": {
            "core_skill": source_dsd.get("core_skill", source_item.get("core_skill", "")),
            "context_constraints": source_dsd.get(
                "context_constraints", source_item.get("context_constraints", "")
            ),
            "task_objective": source_dsd.get("task_objective", source_item.get("task_objective", "")),
        },
        "principle": dict(candidate["content"]),
    }


def build_text_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 Top-K routing and adaptation",
        "",
        f"- Items: {summary['item_count']}",
        f"- Items with Top-K recall: {summary['items_with_recall']}",
        f"- Recalled candidate pairs: {summary['recalled_candidate_count']}",
        f"- Candidate pairs sent to Gate: {summary['gate_candidate_count']}",
        f"- Gate-accepted candidate pairs: {summary['gate_accepted_candidate_count']}",
        f"- Items with a selected memory: {summary['items_with_gate_accept']}",
        f"- Raw/Adapted eligible items: {summary['experiment_eligible_count']}",
        "",
        "Reranker scores are relevance signals only. Gate acceptance is computed from structural fields.",
        "",
    ]
    for record in records:
        if not record.get("selected_memories"):
            continue
        lines.extend([f"## idx {record['idx']}", "", f"Current question: {record['question']}", ""])
        for selected in record["selected_memories"]:
            lines.extend([
                f"### {selected['memory_id']}",
                "",
                f"Reranker score: {selected['rerank_score']:.6f}",
                "",
                f"Gate judgment: `{json.dumps(selected['gate_judgment'], ensure_ascii=False)}`",
                "",
                f"Adaptation status: `{selected['adaptation'].get('status', 'reject')}`",
                "",
                f"Source question: {selected['source_question']}",
                "",
                "Raw solver text:", "", "```text", selected["raw_memory_text"], "```", "",
                "Adapted solver text:", "", "```text", selected["adapted_memory_text"], "```", "",
                "Unified diff:", "", "```diff", selected["raw_vs_adapted_diff"] or "(no diff)", "```", "",
                f"Adapted verification (audit only): {selected['adaptation'].get('adapted_verification', '')}",
                "",
                f"Removed elements: {selected['adaptation'].get('removed_source_specific_elements', [])}",
                "",
            ])
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    items = load_list(args.test_file)
    selected_indices = list(range(min(len(items), args.limit or len(items))))
    ports = list(args.ports)
    urls = {port: f"http://127.0.0.1:{port}/v1/chat/completions" for port in ports}
    config = Config.from_core_file()
    plans = {
        port: PlanModule(config, urls=[urls[port]], model_override=args.model, timeout=args.timeout, seed=args.seed)
        for port in ports
    }
    locks = {port: asyncio.Lock() for port in ports}

    memory = MemoryModule(config, memory_dir=str(args.memory_dir))
    memory.load()
    if not memory.is_indexed:
        raise RuntimeError(f"Memory indices are unavailable: {args.memory_dir}")
    reranker = QwenApplicabilityReranker(
        str(args.reranker_model),
        device=args.reranker_device,
        batch_size=args.reranker_batch_size,
    )

    async def extract_one(idx: int) -> tuple[int, dict[str, str], int]:
        question = str(items[idx].get("question") or items[idx].get("problem") or "")
        port = ports[idx % len(ports)]
        async with locks[port]:
            features = await plans[port].extract_semantics(question)
        print(f"semantics completed idx={idx} port={port}", flush=True)
        return idx, features, port

    feature_rows = await asyncio.gather(*(extract_one(idx) for idx in selected_indices))
    prepared: dict[int, dict[str, Any]] = {}
    for idx, features, port in sorted(feature_rows):
        question = str(items[idx].get("question") or items[idx].get("problem") or "")
        recalled = memory.retrieve_high_recall(features, top_k=args.recall_top_k)
        candidates = []
        for result in recalled:
            source_item = result.get("item", {}) if isinstance(result.get("item"), dict) else {}
            content, schema_source = principle_content_from_item(source_item)
            if not content:
                continue
            candidates.append({
                "memory_id": result.get("memory_id") or source_item.get("memory_id", ""),
                "content": content,
                "schema_source": schema_source,
                "source_item": source_item,
                "retrieval": result,
            })
        ranked = reranker.rank(question, candidates)
        gate_batch = ranked[:args.gate_top_k]
        prepared[idx] = {
            "idx": idx,
            "question": question,
            "features": features,
            "port": port,
            "recalled": recalled,
            "ranked": ranked,
            "gate_batch": gate_batch,
        }
        print(
            f"routing prepared idx={idx} recalled={len(recalled)} gate_batch={len(gate_batch)}",
            flush=True,
        )

    async def gate_one(idx: int) -> dict[str, Any]:
        row = prepared[idx]
        question = row["question"]
        port = row["port"]
        batch = row["gate_batch"]
        base_record: dict[str, Any] = {
            "idx": idx,
            "question": question,
            "features": row["features"],
            "port": port,
            "candidate_present": bool(batch),
            "memory_id": "",
            "source_question": "",
            "gate_judgment": {"accepted": False, "reason": "no_gate_selection"},
            "adaptation": {"status": "reject", "reason": "no_gate_selection"},
            "experiment_eligible": False,
            "raw_memory_text": "",
            "adapted_memory_text": "",
            "raw_memory_text_for_experiment": "",
            "adapted_memory_text_for_experiment": "",
            "retrieval_trace": {
                "protocol": "top10_rerank5_batch_gate_max1",
                "recalled_candidates": [compact_retrieval(result) for result in row["recalled"]],
                "reranked_candidates": [
                    {
                        **compact_retrieval(candidate["retrieval"]),
                        "rerank_score": float(candidate["rerank_score"]),
                        "sent_to_gate": rank < len(batch),
                    }
                    for rank, candidate in enumerate(row["ranked"])
                ],
                "gate_candidate_ids": [candidate["memory_id"] for candidate in batch],
                "gate_candidates": [
                    audit_gate_candidate(candidate, rank)
                    for rank, candidate in enumerate(batch, 1)
                ],
                "gate_result": None,
                "selected_ids": [],
            },
            "selected_memories": [],
        }
        if not batch:
            return base_record
        async with locks[port]:
            gate = await plans[port].judge_memory_applicability(
                question,
                batch,
                confidence_threshold=args.confidence_threshold,
                max_tokens=args.gate_max_tokens,
            )
            accepted_ids = set(gate.get("selected_ids", []))
            selected = [candidate for candidate in batch if candidate["memory_id"] in accepted_ids][
                :args.max_selected
            ]
            judgment_by_id = {
                judgment.get("memory_id", ""): judgment for judgment in gate.get("judgments", [])
            }
            selected_memories = []
            for position, candidate in enumerate(selected, 1):
                judgment = judgment_by_id.get(candidate["memory_id"], {})
                adaptation = await plans[port].adapt_memory_for_question(question, candidate, judgment)
                raw_text = render_solver_memory(candidate["content"], position)
                adapted_text = render_adapted_memory(adaptation, position)
                diff = "\n".join(difflib.unified_diff(
                    raw_text.splitlines(), adapted_text.splitlines(),
                    fromfile="raw_memory", tofile="adapted_memory", lineterm="",
                )) if adapted_text else ""
                selected_memories.append({
                    "memory_id": candidate["memory_id"],
                    "rerank_score": float(candidate["rerank_score"]),
                    "retrieval_score": float(candidate["retrieval"].get("score", 0.0)),
                    "schema_source": candidate["schema_source"],
                    "source_question": str(candidate["source_item"].get("question", "")),
                    "source_content": candidate["content"],
                    "gate_judgment": judgment,
                    "adaptation": adaptation,
                    "raw_memory_text": raw_text,
                    "adapted_memory_text": adapted_text,
                    "raw_vs_adapted_diff": diff,
                })
        eligible_selected = [
            selected for selected in selected_memories
            if selected["adaptation"].get("status") in {"adapted", "unchanged"}
            and selected["adapted_memory_text"]
        ]
        raw_combined = "\n\n".join(selected["raw_memory_text"] for selected in eligible_selected)
        adapted_combined = "\n\n".join(selected["adapted_memory_text"] for selected in eligible_selected)
        base_record["retrieval_trace"]["gate_result"] = gate
        base_record["retrieval_trace"]["selected_ids"] = [
            selected["memory_id"] for selected in eligible_selected
        ]
        representative = selected_memories[0] if selected_memories else None
        base_record.update({
            "memory_id": eligible_selected[0]["memory_id"] if eligible_selected else "",
            "source_question": eligible_selected[0]["source_question"] if eligible_selected else "",
            "gate_judgment": (
                representative["gate_judgment"]
                if representative else {"accepted": False, "reason": "no_gate_selection"}
            ),
            "adaptation": (
                representative["adaptation"]
                if representative else {"status": "reject", "reason": "no_gate_selection"}
            ),
            "experiment_eligible": bool(eligible_selected),
            "raw_memory_text": raw_combined,
            "adapted_memory_text": adapted_combined,
            "raw_memory_text_for_experiment": raw_combined,
            "adapted_memory_text_for_experiment": adapted_combined,
            "selected_memories": selected_memories,
            "gate_parse_stage": gate.get("parse_stage", "parse_failed"),
            "gate_format_parsed": gate.get("format_parsed", False),
            "gate_protocol_compliant": gate.get("protocol_compliant", False),
            "gate_protocol_errors": gate.get("protocol_errors", []),
        })
        return base_record

    records = await asyncio.gather(*(gate_one(idx) for idx in selected_indices))
    records.sort(key=lambda record: record["idx"])
    gate_calls = [record for record in records if record["candidate_present"]]
    all_judgments = [
        judgment
        for record in gate_calls
        for judgment in ((record["retrieval_trace"].get("gate_result") or {}).get("judgments", []))
    ]
    accepted_judgments = [judgment for judgment in all_judgments if judgment.get("accepted")]
    gate_candidate_count = sum(
        len(record["retrieval_trace"]["gate_candidate_ids"]) for record in gate_calls
    )
    eligible = [record for record in records if record["experiment_eligible"]]
    parse_counts = Counter(record.get("gate_parse_stage", "not_run") for record in gate_calls)
    summary = {
        "stage": 3,
        "protocol": "top10_rerank5_batch_gate_max1_v1",
        "test_file": str(args.test_file),
        "test_file_sha256": file_sha256(args.test_file),
        "memory_dir": str(args.memory_dir),
        "memory_principle_count": memory.principle_count,
        "reranker_model": str(args.reranker_model),
        "reranker_device": args.reranker_device,
        "model": args.model,
        "ports": ports,
        "seed": args.seed,
        "recall_top_k": args.recall_top_k,
        "gate_top_k": args.gate_top_k,
        "max_selected": args.max_selected,
        "confidence_threshold": args.confidence_threshold,
        "item_count": len(records),
        "items_with_recall": sum(bool(record["retrieval_trace"]["recalled_candidates"]) for record in records),
        "recalled_candidate_count": sum(
            len(record["retrieval_trace"]["recalled_candidates"]) for record in records
        ),
        "gate_item_count": len(gate_calls),
        "gate_candidate_count": gate_candidate_count,
        "gate_judgment_count": len(all_judgments),
        "gate_accepted_candidate_count": len(accepted_judgments),
        "items_with_gate_accept": sum(
            bool((record["retrieval_trace"].get("gate_result") or {}).get("selected_ids"))
            for record in records
        ),
        "experiment_eligible_count": len(eligible),
        "candidate_count": gate_candidate_count,
        "gate_accepted_count": len(accepted_judgments),
        "gate_rejected_count": gate_candidate_count - len(accepted_judgments),
        "adapter_rejected_count": sum(
            selected["adaptation"].get("status") == "reject"
            for record in records for selected in record["selected_memories"]
        ),
        "gate_format_parsed_count": sum(bool(record.get("gate_format_parsed")) for record in gate_calls),
        "gate_protocol_compliant_count": sum(
            bool(record.get("gate_protocol_compliant")) for record in gate_calls
        ),
        "gate_parse_stage_counts": dict(sorted(parse_counts.items())),
        "accepted_indices": [record["idx"] for record in records if record["experiment_eligible"]],
        "rejected_indices": [record["idx"] for record in records if not record["experiment_eligible"]],
        "eligible_indices": [record["idx"] for record in eligible],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gate_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "text_comparison.md").write_text(
        build_text_report(records, summary), encoding="utf-8"
    )
    print(
        f"top-k gate done gate_pairs={gate_candidate_count} accepted_pairs={len(accepted_judgments)} "
        f"eligible_items={len(eligible)} output={args.output}", flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", default=[8004, 8006, 8007])
    parser.add_argument("--model", default="llama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--recall-top-k", type=int, default=10)
    parser.add_argument("--gate-top-k", type=int, default=5)
    parser.add_argument("--max-selected", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--gate-max-tokens", type=int, default=3072)
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.gate_top_k > args.recall_top_k:
        parser.error("--gate-top-k must be <= --recall-top-k")
    if args.max_selected > args.gate_top_k:
        parser.error("--max-selected must be <= --gate-top-k")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
