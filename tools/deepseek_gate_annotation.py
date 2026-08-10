#!/usr/bin/env python3
"""Blind DeepSeek labeling and Gate precision/recall evaluation.

The API key is read only from an environment variable. It is never accepted as
a command-line value and is never written to an artifact.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = {"applicable", "reject", "uncertain"}
BOOLEAN_FIELDS = (
    "trigger_match",
    "required_evidence_match",
    "structural_match",
    "scope_match",
    "contraindicated",
)

SYSTEM_PROMPT = """You are an independent expert annotator for a mathematical reasoning memory system.
Judge whether ONE past reasoning principle is safely applicable to the CURRENT problem.

You are blind to retrieval scores and to another model's decision. Do not solve the full current problem.
Compare the source/current mathematical objects, operators, recurrences or invariants, constraints, objective, and the prerequisites of the recommended action.

Rules:
1. Topic similarity alone is insufficient.
2. Different constants, variable names, wording, or a reversed unknown do not force rejection when the same operation remains executable after rebinding roles.
3. `required_evidence_match=true` only if every required fact is observable or derivable in the current problem.
4. `structural_match=false` when the action transfers an incompatible recurrence, object, constraint, theorem, or source-only formula.
5. `contraindicated=true` when an exclusion condition holds or the advice could mislead the solver.
6. Use `uncertain` only when the supplied information genuinely cannot decide safe applicability.
7. Check elementary mathematical claims before relying on them.
8. Output one raw JSON object only, with no markdown or extra explanation.

Required JSON schema:
{
  "label": "applicable | reject | uncertain",
  "trigger_match": true,
  "required_evidence_match": true,
  "structural_match": true,
  "scope_match": true,
  "contraindicated": false,
  "confidence": 0.0,
  "matched_evidence": ["short current-problem evidence"],
  "violated_conditions": ["short mismatch or missing prerequisite"],
  "source_specific_risk": ["source-only constant, formula, object, or conclusion"],
  "reason": "two or three concise sentences"
}

For `applicable`, all four match fields must be true and contraindicated must be false.
For `reject`, at least one required match must be false or contraindicated must be true.
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_api_url(base: str) -> str:
    value = base.strip().rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""


def parse_annotation(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    candidates = [text.strip(), balanced_json_object(text)]
    parsed = None
    for candidate in candidates:
        if not candidate:
            continue
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate.strip())
        candidate = re.sub(r"\s*```$", "", candidate)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        return None, ["response_is_not_a_json_object"]

    label = str(parsed.get("label", "")).strip().lower()
    if label not in LABELS:
        errors.append("invalid_label")
    normalized: dict[str, Any] = {"label": label}
    for field in BOOLEAN_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, bool):
            errors.append(f"{field}_must_be_boolean")
        normalized[field] = value
    confidence = parsed.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("confidence_must_be_number_0_to_1")
        normalized["confidence"] = 0.0
    else:
        normalized["confidence"] = float(confidence)
    for field in ("matched_evidence", "violated_conditions", "source_specific_risk"):
        value = parsed.get(field, [])
        if not isinstance(value, list):
            errors.append(f"{field}_must_be_array")
            value = [str(value)] if value else []
        normalized[field] = [str(item) for item in value]
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason_must_be_nonempty_string")
    normalized["reason"] = str(reason or "").strip()

    match_values = [normalized.get(field) is True for field in BOOLEAN_FIELDS[:-1]]
    structurally_applicable = all(match_values) and normalized.get("contraindicated") is False
    if label == "applicable" and not structurally_applicable:
        errors.append("applicable_label_conflicts_with_fields")
    if label == "reject" and structurally_applicable:
        errors.append("reject_label_conflicts_with_fields")
    return normalized, errors


def extract_pairs(gate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for record in gate_records:
        trace = record.get("retrieval_trace", {})
        candidates = trace.get("gate_candidates", [])
        if not isinstance(candidates, list):
            continue
        gate_result = trace.get("gate_result", {}) or {}
        judgments = gate_result.get("judgments", []) if isinstance(gate_result, dict) else []
        judgment_by_id = {
            str(judgment.get("memory_id", "")): judgment
            for judgment in judgments if isinstance(judgment, dict)
        }
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            memory_id = str(candidate.get("memory_id", ""))
            if not memory_id:
                continue
            llama_judgment = judgment_by_id.get(memory_id, {})
            accepted = bool(llama_judgment.get("accepted"))
            rank = int(candidate.get("rerank_rank", 999) or 999)
            stratum = "llama_accepted" if accepted else (
                "high_rank_rejected" if rank <= 2 else "other_rejected"
            )
            pairs.append({
                "pair_id": f"item_{int(record['idx']):05d}__{memory_id}",
                "item_idx": int(record["idx"]),
                "memory_id": memory_id,
                "current_question": str(record.get("question", "")),
                "source_question": str(candidate.get("source_question", "")),
                "source_structure": candidate.get("source_structure", {}),
                "principle": candidate.get("principle", {}),
                "sampling_stratum": stratum,
                "hidden_reference": {
                    "llama_accepted": accepted,
                    "llama_judgment": llama_judgment,
                    "gate_parse_stage": gate_result.get("parse_stage", ""),
                    "rerank_rank": rank,
                    "rerank_score": candidate.get("rerank_score"),
                    "retrieval_score": candidate.get("retrieval_score"),
                },
            })
    return pairs


def stratified_sample(
    pairs: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    accepted_cap: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups = {
        name: [pair for pair in pairs if pair["sampling_stratum"] == name]
        for name in ("llama_accepted", "high_rank_rejected", "other_rejected")
    }
    for values in groups.values():
        rng.shuffle(values)
    target = min(sample_size, len(pairs))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def take(values: list[dict[str, Any]], count: int) -> None:
        for pair in values:
            if len(selected) >= target or count <= 0:
                break
            if pair["pair_id"] in selected_ids:
                continue
            selected.append(pair)
            selected_ids.add(pair["pair_id"])
            count -= 1

    accepted_target = target if accepted_cap is None else min(target, max(0, accepted_cap))
    take(groups["llama_accepted"], accepted_target)
    remaining = target - len(selected)
    take(groups["high_rank_rejected"], math.ceil(remaining * 0.60))
    remaining = target - len(selected)
    take(groups["other_rejected"], remaining)
    remaining = target - len(selected)
    if remaining:
        leftovers = [pair for pair in pairs if pair["pair_id"] not in selected_ids]
        rng.shuffle(leftovers)
        take(leftovers, remaining)
    rng.shuffle(selected)
    return selected


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    """Return exactly what DeepSeek may see; hidden Gate fields are excluded."""
    return {
        "pair_id": task["pair_id"],
        "current_question": task["current_question"],
        "source_question": task["source_question"],
        "source_structure": task["source_structure"],
        "principle": task["principle"],
    }


def build_user_prompt(task: dict[str, Any]) -> str:
    return "Label this question-memory pair independently:\n" + json.dumps(
        public_task(task), ensure_ascii=False, indent=2
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


async def annotate(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit(
            "aiohttp is required only for the `label` command; install it in the runtime environment"
        ) from exc
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    api_url = normalize_api_url(args.api_base)
    existing = read_jsonl(args.labels)
    completed_ids = {
        row.get("pair_id") for row in existing
        if row.get("status") == "success" and row.get("pair_id")
    }
    pending = [task for task in tasks if task["pair_id"] not in completed_ids]
    print(f"annotation resume: completed={len(completed_ids)} pending={len(pending)}", flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=min(30, args.timeout))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def one(task: dict[str, Any]) -> dict[str, Any]:
            prompt = build_user_prompt(task)
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
            }
            last_error = ""
            raw_response = ""
            usage = None
            result: dict[str, Any] | None = None
            attempts_used = 0
            async with semaphore:
                for attempt in range(1, args.retries + 1):
                    attempts_used = attempt
                    try:
                        async with session.post(api_url, headers=headers, json=payload) as response:
                            body = await response.text()
                            if response.status == 200:
                                data = json.loads(body)
                                choices = data.get("choices", []) if isinstance(data, dict) else []
                                message = choices[0].get("message", {}) if choices else {}
                                raw_response = str(message.get("content", ""))
                                usage = data.get("usage") if isinstance(data, dict) else None
                                annotation, errors = parse_annotation(raw_response)
                                if annotation is not None and not errors:
                                    result = {
                                        "pair_id": task["pair_id"],
                                        "status": "success",
                                        "annotation": annotation,
                                        "schema_errors": [],
                                        "raw_response": raw_response,
                                        "model": args.model,
                                        "attempt": attempt,
                                        "usage": usage,
                                        "annotated_at": datetime.now(timezone.utc).isoformat(),
                                    }
                                    break
                                last_error = "schema:" + ",".join(errors)
                                if attempt == args.retries:
                                    result = {
                                        "pair_id": task["pair_id"],
                                        "status": "schema_failed",
                                        "annotation": annotation,
                                        "schema_errors": errors,
                                        "raw_response": raw_response,
                                        "model": args.model,
                                        "attempt": attempt,
                                        "usage": usage,
                                        "annotated_at": datetime.now(timezone.utc).isoformat(),
                                    }
                                    break
                                payload["messages"] = [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": raw_response},
                                    {
                                        "role": "user",
                                        "content": (
                                            "The previous response violated the required schema: "
                                            + ", ".join(errors)
                                            + ". Return one corrected raw JSON object only."
                                        ),
                                    },
                                ]
                                continue
                            last_error = f"http_{response.status}:{body[:500]}"
                            if response.status not in {429, 500, 502, 503, 504}:
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
                        last_error = f"{type(exc).__name__}:{exc}"
                    if attempt < args.retries:
                        await asyncio.sleep(min(30.0, args.retry_backoff * (2 ** (attempt - 1))))
                if result is None:
                    result = {
                        "pair_id": task["pair_id"],
                        "status": "api_failed",
                        "annotation": None,
                        "schema_errors": [],
                        "raw_response": raw_response,
                        "error": last_error,
                        "model": args.model,
                        "attempt": attempts_used,
                        "usage": usage,
                        "annotated_at": datetime.now(timezone.utc).isoformat(),
                    }
                if args.request_interval > 0:
                    await asyncio.sleep(args.request_interval)
            async with write_lock:
                with args.labels.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"labeled {task['pair_id']} status={result['status']}", flush=True)
            return result

        await asyncio.gather(*(one(task) for task in pending))


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate(tasks: list[dict[str, Any]], labels_path: Path, output_dir: Path) -> dict[str, Any]:
    task_by_id = {task["pair_id"]: task for task in tasks}
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(labels_path):
        pair_id = str(row.get("pair_id", ""))
        if pair_id in task_by_id:
            latest[pair_id] = row
    usable = []
    uncertain = []
    failures = []
    for pair_id, task in task_by_id.items():
        row = latest.get(pair_id)
        if not row or row.get("status") != "success" or not isinstance(row.get("annotation"), dict):
            failures.append({"task": task, "label_record": row})
            continue
        label = row["annotation"].get("label")
        combined = {"task": task, "label_record": row}
        if label == "uncertain":
            uncertain.append(combined)
        elif label in {"applicable", "reject"}:
            usable.append(combined)
        else:
            failures.append(combined)

    tp = fp = tn = fn = 0
    disagreements = []
    applicable_pairs = []
    for combined in usable:
        task = combined["task"]
        deepseek_positive = combined["label_record"]["annotation"]["label"] == "applicable"
        llama_positive = bool(task["hidden_reference"].get("llama_accepted"))
        if llama_positive and deepseek_positive:
            tp += 1
        elif llama_positive and not deepseek_positive:
            fp += 1
        elif not llama_positive and deepseek_positive:
            fn += 1
        else:
            tn += 1
        if deepseek_positive:
            applicable_pairs.append({
                **public_task(task),
                "deepseek_annotation": combined["label_record"]["annotation"],
                "llama_gate": task["hidden_reference"],
            })
        if llama_positive != deepseek_positive:
            disagreements.append(combined)

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    accuracy = safe_ratio(tp + tn, tp + fp + tn + fn)
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    metrics = {
        "sample_size_requested": len(tasks),
        "label_records_found": len(latest),
        "usable_binary_labels": len(usable),
        "uncertain_labels": len(uncertain),
        "failed_or_missing_labels": len(failures),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "precision_wilson_95": wilson_interval(tp, tp + fp),
        "recall": recall,
        "recall_wilson_95": wilson_interval(tp, tp + fn),
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "deepseek_applicable_count": len(applicable_pairs),
        "llama_accepted_count": sum(
            bool(task["hidden_reference"].get("llama_accepted")) for task in tasks
        ),
        "disagreement_count": len(disagreements),
        "interpretation": "DeepSeek labels are independent silver labels; human review is required for paper claims.",
    }
    write_json(output_dir / "gate_metrics.json", metrics)
    write_json(output_dir / "deepseek_applicable_pairs.json", applicable_pairs)
    write_json(output_dir / "uncertain_and_failed.json", {"uncertain": uncertain, "failures": failures})

    with (output_dir / "annotation_adjudication.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "pair_id", "item_idx", "memory_id", "sampling_stratum", "llama_label",
            "deepseek_label", "deepseek_confidence", "deepseek_reason",
            "current_question", "source_question", "human_label", "human_notes",
        ])
        writer.writeheader()
        ordered = disagreements + [combined for combined in usable if combined not in disagreements]
        for combined in ordered:
            task = combined["task"]
            annotation = combined["label_record"]["annotation"]
            writer.writerow({
                "pair_id": task["pair_id"],
                "item_idx": task["item_idx"],
                "memory_id": task["memory_id"],
                "sampling_stratum": task["sampling_stratum"],
                "llama_label": "applicable" if task["hidden_reference"].get("llama_accepted") else "reject",
                "deepseek_label": annotation.get("label", ""),
                "deepseek_confidence": annotation.get("confidence", ""),
                "deepseek_reason": annotation.get("reason", ""),
                "current_question": task["current_question"],
                "source_question": task["source_question"],
                "human_label": "",
                "human_notes": "",
            })

    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    report = "\n".join([
        "# DeepSeek silver-label Gate audit",
        "",
        f"- Requested pairs: {len(tasks)}",
        f"- Usable binary labels: {len(usable)}",
        f"- Uncertain: {len(uncertain)}",
        f"- Failed/missing: {len(failures)}",
        f"- Llama accepted: {metrics['llama_accepted_count']}",
        f"- DeepSeek applicable: {len(applicable_pairs)}",
        f"- Disagreements: {len(disagreements)}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| TP | {tp} |",
        f"| FP | {fp} |",
        f"| TN | {tn} |",
        f"| FN | {fn} |",
        f"| Precision | {fmt(precision)} |",
        f"| Recall | {fmt(recall)} |",
        f"| Specificity | {fmt(specificity)} |",
        f"| Accuracy | {fmt(accuracy)} |",
        f"| F1 | {fmt(f1)} |",
        "",
        "DeepSeek provides independent silver labels, not human gold labels. Review disagreements first in `annotation_adjudication.csv`, then audit a random subset of agreements before using metrics in the paper.",
        "",
    ])
    (output_dir / "gate_evaluation_report.md").write_text(report, encoding="utf-8")
    return metrics


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load_json(args.gate_records)
    if not isinstance(records, list):
        raise SystemExit("--gate-records must contain a JSON list")
    pairs = extract_pairs(records)
    if not pairs:
        raise SystemExit(
            "No full Gate candidates found. Re-run tools/prepare_stage3_topk_gate.py "
            "with the current code so retrieval_trace.gate_candidates is saved."
        )
    sampled = stratified_sample(pairs, args.sample_size, args.seed, args.accepted_cap)
    metadata = {
        "gate_records": str(args.gate_records),
        "candidate_pair_count": len(pairs),
        "sample_size_requested": args.sample_size,
        "sample_size_actual": len(sampled),
        "seed": args.seed,
        "accepted_cap": args.accepted_cap,
        "stratum_counts": dict(Counter(task["sampling_stratum"] for task in sampled)),
        "blind_fields_sent_to_api": list(public_task(sampled[0]).keys()) if sampled else [],
        "hidden_fields_not_sent": ["hidden_reference", "sampling_stratum", "item_idx", "memory_id"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "annotation_tasks.json", sampled)
    write_json(args.output / "annotation_manifest.json", metadata)
    print(
        f"prepared {len(sampled)}/{len(pairs)} candidate pairs at {args.output / 'annotation_tasks.json'}",
        flush=True,
    )
    return sampled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Stratify and freeze blind annotation tasks")
    prepare_parser.add_argument("--gate-records", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--sample-size", type=int, default=100)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument(
        "--accepted-cap",
        type=int,
        default=None,
        help="Optional cap for accepted pairs in a balanced smoke test; omit for the 100-pair audit",
    )

    label_parser = subparsers.add_parser("label", help="Call an OpenAI-compatible DeepSeek API with resume")
    label_parser.add_argument("--tasks", type=Path, required=True)
    label_parser.add_argument("--labels", type=Path, required=True)
    label_parser.add_argument("--api-base", required=True)
    label_parser.add_argument("--model", required=True)
    label_parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    label_parser.add_argument("--concurrency", type=int, default=4)
    label_parser.add_argument("--timeout", type=int, default=180)
    label_parser.add_argument("--retries", type=int, default=4)
    label_parser.add_argument("--retry-backoff", type=float, default=2.0)
    label_parser.add_argument("--request-interval", type=float, default=0.0)
    label_parser.add_argument("--max-tokens", type=int, default=1200)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare DeepSeek silver labels with Llama Gate")
    evaluate_parser.add_argument("--tasks", type=Path, required=True)
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "label":
        tasks = load_json(args.tasks)
        if not isinstance(tasks, list):
            raise SystemExit("--tasks must contain a JSON list")
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        asyncio.run(annotate(args, tasks))
        print(f"labeling finished in {time.time() - started:.1f}s: {args.labels}", flush=True)
    elif args.command == "evaluate":
        tasks = load_json(args.tasks)
        if not isinstance(tasks, list):
            raise SystemExit("--tasks must contain a JSON list")
        metrics = evaluate(tasks, args.labels, args.output)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
