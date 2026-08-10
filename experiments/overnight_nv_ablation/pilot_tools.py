#!/usr/bin/env python3
"""Build and summarize the reproducible 4x25 Stage-2 format pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LEVELS = (1, 2, 3, 4)


def load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_manifest(args: argparse.Namespace) -> int:
    from modules.dsd_module import canonicalize_stage1_packets, stage1_packet_is_candidate

    run_root = Path(args.run_root)
    entries = []
    for level in LEVELS:
        source = ROOT / "output" / "stage1" / "02_learning_packets" / args.model / f"level{level}_learning_packets.json"
        packets = load(source, [])
        if not isinstance(packets, list):
            raise ValueError(f"Learning packet file is not a list: {source}")
        canonical = canonicalize_stage1_packets(packets)
        candidates = [
            str(packet.get("question_id", "")) for packet in canonical if stage1_packet_is_candidate(packet)
        ]
        ranked = sorted(
            candidates,
            key=lambda question_id: hashlib.sha256(f"{question_id}:{args.seed}".encode("utf-8")).hexdigest(),
        )
        selected = ranked[: args.per_level]
        if len(selected) != args.per_level:
            raise ValueError(f"Level {level} has only {len(selected)} eligible candidates")
        selection_path = run_root / "selection" / f"level{level}.json"
        save(selection_path, {
            "level": level,
            "seed": args.seed,
            "expected_count": args.per_level,
            "question_ids": selected,
        })
        entries.append({
            "level": level,
            "source": str(source),
            "source_hash": sha256_file(source),
            "eligible_count": len(candidates),
            "selected_count": len(selected),
            "question_ids": selected,
            "selection_file": str(selection_path),
        })
    manifest = {
        "schema": "format-pilot-manifest-v2",
        "seed": args.seed,
        "per_level": args.per_level,
        "total": sum(entry["selected_count"] for entry in entries),
        "levels": entries,
    }
    save(run_root / "sample_manifest.json", manifest)
    print(json.dumps({"sample_manifest": str(run_root / "sample_manifest.json"), "total": manifest["total"]}))
    return 0


def command_bootstrap_cache(args: argparse.Namespace) -> int:
    """Convert the completed deterministic DSD registry into reusable cache files."""
    import numpy as np
    from modules.dsd_module import DSDRegistry

    run_root = Path(args.run_root)
    converted = []
    for level in LEVELS:
        destination = run_root / "dsd_cache" / f"dpo_level{level}"
        if (destination / "manifest.json").exists():
            converted.append({"level": level, "status": "already_cached"})
            continue
        source = ROOT / "memory" / f"dpo_level{level}" / "00_dsd_registry" / "question_dsd_registry.json"
        packet_path = ROOT / "output" / "stage1" / "02_learning_packets" / args.model / f"level{level}_learning_packets.json"
        payload = load(source, {})
        packets = load(packet_path, [])
        records = payload.get("records", []) if isinstance(payload, dict) else []
        if not records or not isinstance(packets, list):
            converted.append({"level": level, "status": "source_unavailable"})
            continue
        required_arrays = ("core_embedding", "context_embedding", "target_embedding")
        if any(any(name not in record for name in required_arrays) for record in records):
            converted.append({"level": level, "status": "embeddings_missing"})
            continue
        registry = DSDRegistry(
            dataset=f"dpo_level{level}", round_id=level, output_root=run_root,
            backbone=args.model, cache_dir=destination,
        )
        registry.records = records
        registry._by_question_id = {str(record["question_id"]): record for record in records}
        registry._core_embeddings = np.asarray([record["core_embedding"] for record in records], dtype=np.float32)
        registry._context_embeddings = np.asarray([record["context_embedding"] for record in records], dtype=np.float32)
        registry._target_embeddings = np.asarray([record["target_embedding"] for record in records], dtype=np.float32)
        registry._precompute_neighbors()
        cache_key = hashlib.sha256(json.dumps({
            "schema": "dsd-cache-v1",
            "dataset": f"dpo_level{level}",
            "round_id": level,
            "embedding_model": registry.embedding_model_name,
            "packets": packets,
        }, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        registry._save_cache(cache_key)
        converted.append({"level": level, "status": "converted", "record_count": len(records)})
        del registry, records, payload
    save(run_root / "dsd_cache" / "bootstrap_summary.json", converted)
    print(json.dumps({"cache_bootstrap": converted}))
    return 0


def iter_units(run_root: Path, section: str) -> Iterable[Dict[str, Any]]:
    for level in LEVELS:
        path = run_root / section / f"level{level}" / "01_candidates" / "candidates.json"
        values = load(path, [])
        if isinstance(values, list):
            yield from (value for value in values if isinstance(value, dict))


def command_report(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    manifest = load(run_root / "sample_manifest.json", {})
    bank_units: List[Dict[str, Any]] = []
    for level in LEVELS:
        payload = load(run_root / "candidate_bank" / f"dpo_level{level}.json", {})
        values = payload.get("candidates", []) if isinstance(payload, dict) else []
        bank_units.extend(value for value in values if isinstance(value, dict))
    gate_units = list(iter_units(run_root, "gate_results"))
    management_units = list(iter_units(run_root, "management_results"))

    decisions: List[Dict[str, Any]] = []
    for level in LEVELS:
        values = load(run_root / "management_results" / f"level{level}" / "09_store_commit" / "store_commit.json", [])
        if isinstance(values, list):
            decisions.extend(value for value in values if isinstance(value, dict))
    decision_by_candidate = {str(item.get("candidate_id", "")): item for item in decisions}

    schema_valid = sum((unit.get("schema_check") or {}).get("valid") is True for unit in bank_units)
    repair_used = sum(
        ((unit.get("generation_result") or {}).get("generation_trace") or {}).get("repair_used") is True
        for unit in bank_units
    )
    principle_parse_stages = Counter(
        str((unit.get("generation_result") or {}).get("parse_stage", "missing")) for unit in bank_units
    )
    gate_attempts = [unit for unit in gate_units if unit.get("gate_result")]
    gate_compliant = sum((unit.get("gate_result") or {}).get("protocol_compliant") is True for unit in gate_attempts)
    gate_id_complete = sum(
        not any(
            str(error).startswith(("missing_neighbor_ids:", "unexpected_neighbor_ids:", "duplicate_neighbor_id"))
            for error in (unit.get("gate_result") or {}).get("protocol_errors", [])
        )
        for unit in gate_attempts
    )
    gate_parse_stages = Counter(str((unit.get("gate_result") or {}).get("parse_stage", "missing")) for unit in gate_attempts)

    action_counts: Counter[str] = Counter()
    management_records = []
    for unit in bank_units:
        candidate_id = str(unit.get("candidate_id", ""))
        if (unit.get("schema_check") or {}).get("valid") is not True:
            action = "reject"
            reason = "schema_invalid"
        elif unit.get("status") != "candidate_ready":
            action = "reject"
            reason = ",".join((unit.get("admission") or {}).get("reasons", [])) or "quality_not_ready"
        else:
            decision = decision_by_candidate.get(candidate_id, {})
            operation = str(decision.get("operation", "reject"))
            action = {
                "add": "add",
                "replace": "update",
                "merge": "merge",
                "merge_deferred": "covered",
            }.get(operation, "reject")
            reason = str(decision.get("reason", operation))
        action_counts[action] += 1
        management_records.append({
            "candidate_id": candidate_id,
            "question_id": unit.get("question_id", ""),
            "action": action,
            "reason": reason,
            "target_memory_id": decision_by_candidate.get(candidate_id, {}).get("memory_id", ""),
        })
    save(run_root / "management_results.json", management_records)

    denominator = schema_valid
    managed = sum(action_counts[name] for name in ("add", "merge", "update", "covered"))
    format_misclassified = sum(
        (unit.get("gate_result") or {}).get("status") == "parse_failed"
        and "principle_quality_failed" in (unit.get("admission") or {}).get("reasons", [])
        for unit in gate_units
    )
    format_failures = []
    for unit in bank_units:
        trace = (unit.get("generation_result") or {}).get("generation_trace") or {}
        if trace.get("repair_used") or (unit.get("schema_check") or {}).get("valid") is not True:
            format_failures.append({
                "candidate_id": unit.get("candidate_id", ""),
                "question_id": unit.get("question_id", ""),
                "stage": "principle",
                "schema_check": unit.get("schema_check", {}),
                "generation_trace": trace,
            })
    for unit in gate_attempts:
        gate = unit.get("gate_result") or {}
        if not gate.get("protocol_compliant") or gate.get("repair_used"):
            format_failures.append({
                "candidate_id": unit.get("candidate_id", ""),
                "question_id": unit.get("question_id", ""),
                "stage": "gate",
                "gate_result": gate,
            })
    save(run_root / "format_failures.json", format_failures)

    total = len(bank_units)
    rates = {
        "schema_valid_rate": schema_valid / total if total else 0.0,
        "principle_no_format_repair_rate": (total - repair_used) / total if total else 0.0,
        "gate_parse_rate": gate_compliant / len(gate_attempts) if gate_attempts else 0.0,
        "neighbor_id_complete_rate": gate_id_complete / len(gate_attempts) if gate_attempts else 0.0,
        "effective_management_coverage": managed / denominator if denominator else 0.0,
    }
    checks = {
        "sample_total_100": total == 100 and manifest.get("total") == 100,
        "schema_valid_100pct": rates["schema_valid_rate"] == 1.0,
        "principle_no_format_repair_at_least_95pct": rates["principle_no_format_repair_rate"] >= 0.95,
        "gate_parse_at_least_95pct": rates["gate_parse_rate"] >= 0.95,
        "neighbor_id_complete_at_least_95pct": rates["neighbor_id_complete_rate"] >= 0.95,
        "format_failures_not_quality_failures": format_misclassified == 0,
        "effective_management_at_least_70pct": rates["effective_management_coverage"] >= 0.70,
    }
    summary = {
        "status": "pass" if all(checks.values()) else "failed",
        "sample_distribution": {str(item["level"]): item["selected_count"] for item in manifest.get("levels", [])},
        "total": total,
        "schema_valid_count": schema_valid,
        "principle_format_repair_count": repair_used,
        "principle_parse_stage_counts": dict(principle_parse_stages),
        "gate_attempt_count": len(gate_attempts),
        "gate_protocol_compliant_count": gate_compliant,
        "gate_parse_stage_counts": dict(gate_parse_stages),
        "neighbor_id_complete_count": gate_id_complete,
        "management_action_counts": dict(action_counts),
        "managed_count": managed,
        "management_denominator": denominator,
        "format_failure_misclassified_as_quality": format_misclassified,
        "rates": rates,
        "checks": checks,
    }
    save(run_root / "pilot_summary.json", summary)
    lines = [
        "# Format and Memory-Management Pilot",
        "",
        f"Status: **{summary['status'].upper()}**",
        "",
        "| Metric | Value | Required |",
        "|---|---:|---:|",
        f"| Samples | {total} | 100 |",
        f"| Schema valid | {schema_valid}/{total} ({rates['schema_valid_rate']:.1%}) | 100% |",
        f"| No Principle format repair | {total - repair_used}/{total} ({rates['principle_no_format_repair_rate']:.1%}) | >=95% |",
        f"| Gate parse | {gate_compliant}/{len(gate_attempts)} ({rates['gate_parse_rate']:.1%}) | >=95% |",
        f"| Neighbor ID complete | {gate_id_complete}/{len(gate_attempts)} ({rates['neighbor_id_complete_rate']:.1%}) | >=95% |",
        f"| Effectively managed | {managed}/{denominator} ({rates['effective_management_coverage']:.1%}) | >=70% |",
        "",
        "## Management actions",
        "",
    ]
    for name in ("add", "merge", "update", "covered", "reject"):
        lines.append(f"- {name}: {action_counts[name]}")
    lines.extend(["", "## Parse stages", "", f"- Principle: {dict(principle_parse_stages)}", f"- Gate: {dict(gate_parse_stages)}", "", "## Checks", ""])
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    (run_root / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(run_root / "pilot_report.md"), "status": summary["status"], "rates": rates}))
    return 0 if summary["status"] == "pass" else 10


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--run-root", required=True)
    manifest.add_argument("--model", default="llama")
    manifest.add_argument("--per-level", type=int, default=25)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.set_defaults(handler=command_manifest)
    bootstrap = subparsers.add_parser("bootstrap-cache")
    bootstrap.add_argument("--run-root", required=True)
    bootstrap.add_argument("--model", default="llama")
    bootstrap.set_defaults(handler=command_bootstrap_cache)
    report = subparsers.add_parser("report")
    report.add_argument("--run-root", required=True)
    report.set_defaults(handler=command_report)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
