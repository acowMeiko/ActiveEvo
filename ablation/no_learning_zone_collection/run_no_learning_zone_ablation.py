#!/usr/bin/env python
"""Ablation: remove learning-zone collection — send ALL training samples downstream.

Stage 1 runs ONCE (generates packets for all 4 levels).
Stage 2 iterates per level to build memory incrementally.
Stage 3 evaluates per level on a test set.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger("no_learning_zone")

from core.agent_pipeline import AgentPipeline
from core.config import Config
from output_paths import (
    stage2_summary_file,
    stage3_guided_results_file,
    stage3_guided_summary_file,
    stage3_progress_state_file,
)
from utils.io_utils import load_json, save_json


def test_scope_key(path: Optional[Path]) -> str:
    if not path:
        return "no_learning_zone_default"
    parent = path.parent.name or "root"
    return f"no_learning_zone_{parent}_{path.stem}"


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: remove learning-zone collection and send all training samples through the downstream pipeline."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--test-input", type=Path, default=None)
    parser.add_argument("--ports", type=int, nargs="+", default=[8000, 8001, 8002, 8003])
    parser.add_argument("--model", default=None)
    parser.add_argument("--stage1-concurrency", type=int, default=10)
    parser.add_argument("--stage1-diff-concurrency", type=int, default=10)
    parser.add_argument("--stage1-k-trials", type=int, default=10)
    parser.add_argument("--stage2-concurrency", type=int, default=8)
    parser.add_argument("--stage3-concurrency", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retry-on-wrong", type=int, default=2)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--stage1-sampling-source", type=Path, default=None,
                        help="Reuse existing unified_sampling_results.json (skip K-trial resampling)")
    args = parser.parse_args()

    os.environ["MEMUA_OUTPUT_DIR"] = str(args.output_dir.resolve())
    config = Config.from_core_file()
    config.ensure_dirs()
    config.setup_logging()

    memory_dir = args.memory_dir or (args.output_dir / "memory")
    backbone = args.model or ""
    pipeline = AgentPipeline(
        config=config,
        ports=args.ports,
        model_override=args.model,
        memory_dir=str(memory_dir),
    )

    test_data = None
    if args.test_input and not args.skip_stage3:
        test_data = load_json(str(args.test_input))
        if not isinstance(test_data, list):
            raise ValueError(f"test input must be a JSON list: {args.test_input}")

    run_summary: Dict[str, Any] = {
        "ablation": "no_learning_zone_collection",
        "definition": "Stage1 collects ALL samples (master+learning+panic) instead of only learning-zone.",
        "output_dir": str(args.output_dir),
        "memory_dir": str(memory_dir),
        "levels": args.levels,
        "test_input": str(args.test_input) if args.test_input else None,
        "levels_summary": {},
    }

    scope = test_scope_key(args.test_input)

    # ── Stage 1: sampling + mining (runs ONCE, generates packets for all 4 levels) ──
    if not args.skip_stage1:
        # If reusing existing sampling data, copy it to the expected location first
        if args.stage1_sampling_source:
            from output_paths import stage1_unified_sampling_file
            target_path = stage1_unified_sampling_file(config.paths.output_dir, backbone)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(args.stage1_sampling_source), str(target_path))
            logger.info("Copied sampling source: %s → %s", args.stage1_sampling_source, target_path)

        dataset_path = str(ROOT_DIR / "data" / "deepscaler_merged_6000.json")
        packet_path = await pipeline.stage1_learn(
            dataset_path=dataset_path,
            concurrency=args.stage1_concurrency,
            diff_concurrency=args.stage1_diff_concurrency,
            k_trials=args.stage1_k_trials,
            backbone=backbone,
            collect_all_samples=True,
        )
        run_summary["stage1_packet_path"] = str(packet_path) if packet_path else None

    # ── Stage 2 + Stage 3: iterate per level ──
    for level in args.levels:
        dataset = f"dpo_level{level}"
        level_payload: Dict[str, Any] = {"dataset": dataset}

        if not args.skip_stage2:
            stage2_summary = await pipeline.stage2_memorize(
                dataset=dataset,
                round_id=level,
                validation_concurrency=args.stage2_concurrency,
                backbone=backbone,
            )
            level_payload["stage2"] = stage2_summary
        else:
            summary_path = stage2_summary_file(config.paths.output_dir, dataset)
            if summary_path.exists():
                level_payload["stage2"] = load_json(str(summary_path))

        if test_data is not None and not args.skip_stage3:
            result = await pipeline.stage3_evaluate(
                test_data,
                threshold=args.threshold,
                top_k=args.top_k,
                concurrency=args.stage3_concurrency,
                mode="guided",
                retry_on_wrong=args.retry_on_wrong,
                eval_level=level,
                dataset=dataset,
                test_scope_key=scope,
                backbone=backbone,
                progress_state_path=stage3_progress_state_file(config.paths.output_dir, scope),
            )
            save_json(result["results"], str(stage3_guided_results_file(config.paths.output_dir, dataset)))
            save_json(result["summary"], str(stage3_guided_summary_file(config.paths.output_dir, dataset)))
            level_payload["stage3"] = result["summary"]

        run_summary["levels_summary"][f"level{level}"] = level_payload
        save_json(run_summary, args.output_dir / "no_learning_zone_run_summary.json")

    save_json(run_summary, args.output_dir / "no_learning_zone_run_summary.json")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
