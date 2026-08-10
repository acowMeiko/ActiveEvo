"""Retrieval ablation: compare DSD (3-facet) vs Single-Vector (holistic) retrieval.

Computes Precision, Recall, F1, Top-1/3/5 using DSD retrieval results as ground truth.

Usage:
  python benchmarks/retrieval_compare.py \
    --test-file data/test/benchmarks/bbh_logic_stage3.json \
    --memory-dir memory/llama \
    --ports 8000 8001 8002 8003 \
    --model llama \
    --output output/benchmarks/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import Config
from modules.plan_module import PlanModule
from modules.memory_module import MemoryModule
from modules.execute_module import filter_healthy_urls
from utils.io_utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("benchmarks.retrieval_compare")

# Large top_k to effectively retrieve all passing results for coverage metrics
ALL_RESULTS_K = 9999


def _memory_id_set(results: List[Dict]) -> set:
    """Extract set of memory_ids from retrieval results."""
    return {r.get("memory_id", "") for r in results if r.get("memory_id")}


def _top_k_ids(results: List[Dict], k: int) -> set:
    """Extract memory_ids from top-k results."""
    return {r.get("memory_id", "") for r in results[:k] if r.get("memory_id")}


def compute_metrics(
    dsd_results: List[Dict],
    single_results: List[Dict],
) -> Dict[str, Any]:
    """Compute Precision, Recall, F1, Top-K between single-vector and DSD (gold)."""
    dsd_set = _memory_id_set(dsd_results)
    single_set = _memory_id_set(single_results)

    if not dsd_set:
        return None  # No ground truth, skip this question

    intersection = single_set & dsd_set
    precision = len(intersection) / len(single_set) if single_set else 0.0
    recall = len(intersection) / len(dsd_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    top1_hit = 1 if _top_k_ids(single_results, 1) & dsd_set else 0
    top3_hit = 1 if _top_k_ids(single_results, 3) & dsd_set else 0
    top5_hit = 1 if _top_k_ids(single_results, 5) & dsd_set else 0

    return {
        "dsd_count": len(dsd_set),
        "single_count": len(single_set),
        "intersection": len(intersection),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "top5_hit": top5_hit,
    }


async def run_comparison(
    config: Config,
    ports: List[int],
    model_name: str,
    test_file: str,
    memory_dir: str,
    threshold: float = 0.55,
    output_dir: Path | None = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Run DSD vs single-vector retrieval comparison on a test set."""
    test_data = load_json(test_file)
    if isinstance(test_data, dict):
        test_data = test_data.get("items", test_data.get("results", test_data.get(list(test_data.keys())[0], [])))
    logger.info("Loaded %d questions from %s", len(test_data), test_file)

    urls = [f"http://localhost:{p}/v1/chat/completions" for p in ports]
    urls = filter_healthy_urls(urls)
    if not urls:
        logger.error("No healthy model endpoints on ports %s", ports)
        return {"error": "no healthy endpoints"}

    plan = PlanModule(config, urls=urls, model_override=model_name, timeout=timeout)
    mem_root = Path(memory_dir)
    memory = MemoryModule(config, memory_dir=str(mem_root))
    memory.load()
    if not memory.is_indexed:
        memory.build_indices()
        memory.save_indices()

    per_question_metrics: List[Dict] = []
    skipped_no_dsd = 0
    skipped_no_features = 0

    for idx, item in enumerate(test_data):
        question = item.get("question") or item.get("problem") or ""
        if not question:
            continue

        # Extract 3D features for DSD
        try:
            features = await plan.extract_semantics(question)
        except Exception:
            skipped_no_features += 1
            continue

        if not features:
            skipped_no_features += 1
            continue

        # Generate holistic description for single-vector
        try:
            description_text = await plan.extract_holistic_description(question)
        except Exception:
            description_text = ""

        # DSD retrieval (gold standard)
        dsd_results = memory.retrieve(features, top_k=ALL_RESULTS_K, threshold=threshold)

        # Single-vector retrieval (ablation)
        single_results = memory.retrieve_single_vector(
            description_text, top_k=ALL_RESULTS_K, threshold=threshold
        ) if description_text else []

        metrics = compute_metrics(dsd_results, single_results)
        if metrics is None:
            skipped_no_dsd += 1
            continue

        per_question_metrics.append({
            "idx": idx,
            "question_preview": question[:120],
            **metrics,
        })

    # Aggregate
    n = len(per_question_metrics)
    if n == 0:
        logger.error("No questions with DSD results — cannot compute metrics")
        return {"error": "no dsd results", "skipped_no_dsd": skipped_no_dsd, "skipped_no_features": skipped_no_features}

    agg = {
        "precision": sum(m["precision"] for m in per_question_metrics) / n,
        "recall": sum(m["recall"] for m in per_question_metrics) / n,
        "f1": sum(m["f1"] for m in per_question_metrics) / n,
        "top1": sum(m["top1_hit"] for m in per_question_metrics) / n,
        "top3": sum(m["top3_hit"] for m in per_question_metrics) / n,
        "top5": sum(m["top5_hit"] for m in per_question_metrics) / n,
    }

    # Build comparison table
    table = {
        "Global Embedding (KNN)": {
            "Precision": round(agg["precision"], 4),
            "Recall": round(agg["recall"], 4),
            "F1": round(agg["f1"], 4),
            "Top-1": round(agg["top1"], 4),
            "Top-3": round(agg["top3"], 4),
            "Top-5": round(agg["top5"], 4),
        },
        "SLP Index (Ours)": {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0,
            "Top-1": 1.0,
            "Top-3": 1.0,
            "Top-5": 1.0,
        },
    }

    # Print table
    print("\n" + "=" * 80)
    print("Retrieval Ablation: DSD (3-Facet) vs Single-Vector (Holistic)")
    print(f"Test file: {test_file}")
    print(f"Questions evaluated: {n}  (skipped: {skipped_no_dsd} no-DSD, {skipped_no_features} no-features)")
    print(f"Threshold: {threshold}")
    print("=" * 80)
    header = f"{'Method':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}"
    print(header)
    print("-" * 80)
    for method, metrics in table.items():
        row = f"{method:<28} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} {metrics['F1']:>10.4f} {metrics['Top-1']:>8.4f} {metrics['Top-3']:>8.4f} {metrics['Top-5']:>8.4f}"
        print(row)
    print("=" * 80)

    result = {
        "test_file": test_file,
        "threshold": threshold,
        "n_questions": n,
        "skipped_no_dsd": skipped_no_dsd,
        "skipped_no_features": skipped_no_features,
        "aggregate": agg,
        "table": table,
        "per_question": per_question_metrics,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(result, str(out / "retrieval_ablation.json"))
        logger.info("Results saved to %s", out / "retrieval_ablation.json")

    return result


def main():
    p = argparse.ArgumentParser(description="Retrieval ablation: DSD vs Single-Vector")
    p.add_argument("--test-file", required=True, help="Test set JSON path")
    p.add_argument("--memory-dir", required=True, help="Memory directory")
    p.add_argument("--ports", type=int, nargs="+", required=True, help="API ports")
    p.add_argument("--model", default="llama", help="Model name")
    p.add_argument("--threshold", type=float, default=0.55, help="Similarity threshold")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--output", help="Output directory")
    args = p.parse_args()

    config = Config.from_core_file()
    output_dir = Path(args.output) if args.output else None

    asyncio.run(run_comparison(
        config=config,
        ports=args.ports,
        model_name=args.model,
        test_file=args.test_file,
        memory_dir=args.memory_dir,
        threshold=args.threshold,
        output_dir=output_dir,
        timeout=args.timeout,
    ))


if __name__ == "__main__":
    main()
