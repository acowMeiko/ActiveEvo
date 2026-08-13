#!/usr/bin/env python3
"""ActiveEvo project entry point.

The pipeline is the user-facing controller. It keeps deployment separate in
``api_call/`` and delegates actual Stage 1/2/3 work to core modules.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from types import SimpleNamespace
from pathlib import Path

from core.agent_pipeline import AgentPipeline
from core.config import Config
from modules.memory_module import MemoryModule
from modules.stage2 import build_level_memory

logger = logging.getLogger(__name__)


LEVEL_PATTERN = re.compile(r"level([1-4])$")


def _level_number(level: str | int) -> int:
    text = str(level).strip().lower()
    if text.isdigit() and 1 <= int(text) <= 4:
        return int(text)
    match = LEVEL_PATTERN.fullmatch(text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Unsupported level {level!r}; use level1, level2, level3, or level4."
        )
    return int(match.group(1))


def _public_level(level: str | int) -> str:
    return f"level{_level_number(level)}"


def _default_output_dir(config: Config, variant: str) -> Path:
    return config.paths.output_dir / "stage3" / variant


def _python_executable() -> str:
    return sys.executable or "python"


def apply_pipeline_environment(config: Config) -> None:
    defaults = config.pipeline
    os.environ.setdefault("MEMUA_EMBEDDING_DEVICE", defaults.embedding_device)
    os.environ.setdefault("MEMUA_EMBEDDING_MODEL", defaults.embedding_model)
    os.environ.setdefault("MEMUA_EMBEDDING_MODEL_PATH", str(config.paths.embedding_model_dir))
    os.environ.setdefault("MEMUA_FORCE_EMBEDDING", defaults.force_embedding)
    os.environ.setdefault("MEMUA_ALLOW_HASH_FALLBACK", defaults.allow_hash_fallback)


def subprocess_environment(config: Config) -> dict[str, str]:
    env = os.environ.copy()
    defaults = config.pipeline
    env.setdefault("MEMUA_EMBEDDING_DEVICE", defaults.embedding_device)
    env.setdefault("MEMUA_EMBEDDING_MODEL", defaults.embedding_model)
    env.setdefault("MEMUA_EMBEDDING_MODEL_PATH", str(config.paths.embedding_model_dir))
    env.setdefault("MEMUA_FORCE_EMBEDDING", defaults.force_embedding)
    env.setdefault("MEMUA_ALLOW_HASH_FALLBACK", defaults.allow_hash_fallback)
    return env


def _add_common_runtime_args(parser: argparse.ArgumentParser, config: Config) -> None:
    defaults = config.pipeline
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=list(defaults.default_ports),
        help="OpenAI-compatible local endpoint ports. Deployment is managed by api_call/.",
    )
    parser.add_argument("--model", default=defaults.default_model, help="Served model name.")
    parser.add_argument("--timeout", type=int, default=defaults.request_timeout)


def build_parser(config: Config) -> argparse.ArgumentParser:
    defaults = config.pipeline
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="ActiveEvo three-stage controller.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser("stage1", help="Build capability-signal learning packets.")
    stage1.add_argument("--dataset", default=defaults.stage1_dataset_path, required=not bool(defaults.stage1_dataset_path))
    stage1.add_argument("--expected-dataset-sha256", default=defaults.stage1_expected_dataset_sha256)
    stage1.add_argument("--expected-dataset-size", type=int, default=defaults.stage1_expected_dataset_size)
    stage1.add_argument("--k-trials", type=int, default=defaults.stage1_k_trials)
    stage1.add_argument("--experiment-seed", type=int, default=defaults.stage1_experiment_seed)
    stage1.add_argument("--temperature", type=float, default=defaults.stage1_temperature)
    stage1.add_argument("--top-p", type=float, default=defaults.stage1_top_p)
    stage1.add_argument("--top-logprobs", type=int, default=defaults.stage1_top_logprobs)
    stage1.add_argument("--entropy-histogram-bins", type=int, default=defaults.stage1_entropy_histogram_bins)
    stage1.add_argument("--max-tokens", type=int, default=defaults.stage1_max_tokens)
    stage1.add_argument("--backbone", default=defaults.default_backbone)
    stage1.add_argument("--concurrency", type=int, default=defaults.stage1_concurrency)
    stage1.add_argument("--task-batch-size", type=int, default=defaults.stage1_task_batch_size)
    stage1.add_argument("--skip-sampling", action="store_true")
    stage1.add_argument("--resume", action="store_true")
    _add_common_runtime_args(stage1, config)

    stage2 = subparsers.add_parser("stage2", help="Generate and quality-audit memories.")
    stage2.add_argument("--level", default="level1", type=_public_level, choices=list(defaults.public_levels))
    stage2.add_argument("--memory-dir", default=defaults.default_memory_dir)
    stage2.add_argument("--backbone", default=defaults.default_backbone)
    stage2.add_argument("--model-concurrency", type=int, default=defaults.stage2_concurrency)
    stage2.add_argument("--limit", type=int, default=0, help="Maximum candidate count; 0 means all.")
    stage2.add_argument("--run-tag", default="", help="Optional artifact isolation tag.")
    stage2.add_argument("--prepare-candidates-only", action="store_true")
    stage2.add_argument("--candidate-bank", default="", help="Frozen candidate bank JSON.")
    stage2.add_argument("--selection-file", default="", help="Stable sampled question IDs.")
    stage2.add_argument("--output-dir", default="", help="Stage 2 audit artifact directory.")

    stage3 = subparsers.add_parser("stage3", help="Evaluate baseline or memory-guided solving.")
    stage3.add_argument("--variant", choices=["baseline", "guided"], default="guided")
    stage3.add_argument("--test-file", default=defaults.default_test_file)
    stage3.add_argument("--memory-dir", default=defaults.default_memory_dir)
    stage3.add_argument("--output-dir", default="")
    stage3.add_argument("--concurrency", type=int, default=defaults.stage3_concurrency)
    stage3.add_argument("--temperature", type=float, default=defaults.default_temperature)
    stage3.add_argument("--threshold", type=float, default=defaults.stage3_threshold)
    stage3.add_argument("--candidate-top-k", type=int, default=defaults.stage3_candidate_top_k)
    stage3.add_argument("--gate-top-k", type=int, default=defaults.stage3_gate_top_k)
    stage3.add_argument("--max-injected", type=int, default=defaults.stage3_max_injected)
    stage3.add_argument("--limit", type=int, default=0, help="Smoke-test prefix size.")
    _add_common_runtime_args(stage3, config)

    full = subparsers.add_parser("all", help="Run Stage 1/2/3 over multiple levels.")
    full.add_argument("--levels", nargs="+", default=list(defaults.public_levels), type=_public_level)
    full.add_argument("--dataset", default=defaults.stage1_dataset_path)
    full.add_argument("--memory-dir", default=defaults.default_memory_dir)
    full.add_argument("--test-file", default=defaults.default_test_file)
    full.add_argument("--concurrency", type=int, default=defaults.stage1_concurrency)
    full.add_argument("--stage2-concurrency", type=int, default=defaults.stage2_concurrency)
    full.add_argument("--eval-concurrency", type=int, default=defaults.stage3_concurrency)
    full.add_argument("--threshold", type=float, default=defaults.stage3_threshold)
    full.add_argument("--top-k", type=int, default=5)
    full.add_argument("--skip-sampling", action="store_true")
    _add_common_runtime_args(full, config)

    return parser


def build_agent_pipeline(args: argparse.Namespace, config: Config) -> AgentPipeline:
    return AgentPipeline(
        config=config,
        ports=getattr(args, "ports", None),
        model_override=getattr(args, "model", None),
        memory_dir=getattr(args, "memory_dir", config.pipeline.default_memory_dir),
        timeout=getattr(args, "timeout", config.pipeline.request_timeout),
    )


def validate_stage1_inputs(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    if not dataset.is_file():
        raise FileNotFoundError(f"Stage 1 dataset not found: {dataset}")
    actual_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if args.expected_dataset_sha256 and actual_hash != args.expected_dataset_sha256:
        raise ValueError(
            f"Dataset SHA-256 mismatch: expected {args.expected_dataset_sha256}, got {actual_hash}"
        )
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != args.expected_dataset_size:
        raise ValueError(
            f"Stage 1 requires {args.expected_dataset_size} records; got "
            f"{len(payload) if isinstance(payload, list) else type(payload).__name__}"
        )
    missing = [index for index, item in enumerate(payload) if not isinstance(item, dict) or any(
        not str(item.get(field, "")).strip() for field in ("question", "answer", "difficulty_level")
    )]
    if missing:
        logger.warning(
            "Stage 1 will retain %d invalid input record(s) outside Master/Learning/Panic: %s",
            len(missing), missing[:10],
        )
    if args.skip_sampling:
        return
    if len(args.ports) != 4:
        raise ValueError(f"Stage 1 expects exactly four actor endpoints; got ports={args.ports}")
    from modules.execute_module import is_url_reachable
    dead = [port for port in args.ports if not is_url_reachable(f"http://127.0.0.1:{port}")]
    if dead:
        raise RuntimeError(f"Stage 1 actor endpoints are not reachable: {dead}")


async def run_stage1(args: argparse.Namespace, config: Config) -> None:
    validate_stage1_inputs(args)
    pipeline = build_agent_pipeline(args, config)
    logger.info("Stage 1 started: sampling once and producing level1-level4 learning bundles")
    await pipeline.stage1_learn(
        dataset_path=args.dataset,
        concurrency=args.concurrency,
        k_trials=args.k_trials,
        experiment_seed=args.experiment_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.top_logprobs,
        entropy_histogram_bins=args.entropy_histogram_bins,
        max_tokens=args.max_tokens,
        task_batch_size=args.task_batch_size,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_dataset_size=args.expected_dataset_size,
        skip_sampling=args.skip_sampling,
        backbone=args.backbone,
        resume=args.resume,
    )
    logger.info("Stage 1 completed: level1-level4")


async def run_stage2(args: argparse.Namespace, config: Config) -> None:
    runtime = SimpleNamespace(
        config=config,
        memory=MemoryModule(config, memory_dir=args.memory_dir),
    )
    level = _public_level(args.level)
    round_id = _level_number(level)
    dataset = level
    logger.info("Stage 2 Memory Factory started: %s", level)
    await build_level_memory(
        runtime,
        level=dataset,
        round_id=round_id,
        model_concurrency=args.model_concurrency,
        backbone=args.backbone,
        max_records=args.limit,
        run_tag=args.run_tag,
        prepare_candidates_only=args.prepare_candidates_only,
        candidate_bank=args.candidate_bank,
        selection_file=args.selection_file,
        artifact_dir=args.output_dir,
    )
    logger.info("Stage 2 completed: %s", level)


async def run_stage3(args: argparse.Namespace, config: Config) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(config, args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        _python_executable(),
        "scripts/stage3/evaluate.py",
        "--variant",
        args.variant,
        "--test-file",
        args.test_file,
        "--output-dir",
        str(output_dir),
        "--ports",
        *[str(port) for port in args.ports],
        "--model",
        args.model,
        "--concurrency",
        str(args.concurrency),
        "--temperature",
        str(args.temperature),
        "--threshold",
        str(args.threshold),
        "--candidate-top-k",
        str(args.candidate_top_k),
        "--gate-top-k",
        str(args.gate_top_k),
        "--max-injected",
        str(args.max_injected),
    ]
    if args.variant == "guided":
        command.extend(["--memory-dir", args.memory_dir])
    if args.limit:
        command.extend(["--limit", str(args.limit)])

    logger.info("Stage 3 started: %s", args.variant)
    process = await asyncio.create_subprocess_exec(*command, env=subprocess_environment(config))
    return_code = await process.wait()
    if return_code != 0:
        raise RuntimeError(f"Stage 3 evaluator failed with exit code {return_code}")
    logger.info("Stage 3 completed: %s -> %s", args.variant, output_dir)


async def run_all(args: argparse.Namespace, config: Config) -> None:
    levels = [_public_level(level) for level in args.levels]
    if not args.skip_sampling:
        if not args.dataset:
            raise ValueError("--dataset is required unless --skip-sampling is used")
        stage1_args = argparse.Namespace(
            command="stage1",
            dataset=args.dataset,
            expected_dataset_sha256=config.pipeline.stage1_expected_dataset_sha256,
            expected_dataset_size=config.pipeline.stage1_expected_dataset_size,
            k_trials=config.pipeline.stage1_k_trials,
            experiment_seed=config.pipeline.stage1_experiment_seed,
            temperature=config.pipeline.stage1_temperature,
            top_p=config.pipeline.stage1_top_p,
            top_logprobs=config.pipeline.stage1_top_logprobs,
            entropy_histogram_bins=config.pipeline.stage1_entropy_histogram_bins,
            max_tokens=config.pipeline.stage1_max_tokens,
            concurrency=args.concurrency,
            task_batch_size=config.pipeline.stage1_task_batch_size,
            skip_sampling=False,
            resume=False,
            backbone=config.pipeline.default_backbone,
            ports=args.ports,
            model=args.model,
            timeout=args.timeout,
        )
        await run_stage1(stage1_args, config)

    for level in levels:
        stage2_args = argparse.Namespace(
            command="stage2",
            level=level,
            memory_dir=args.memory_dir,
            backbone=config.pipeline.default_backbone,
            model_concurrency=args.stage2_concurrency,
            limit=0,
            run_tag="",
            prepare_candidates_only=False,
            candidate_bank="",
            selection_file="",
            output_dir="",
        )
        await run_stage2(stage2_args, config)

    stage3_args = argparse.Namespace(
        command="stage3",
        variant="guided",
        test_file=args.test_file,
        memory_dir=args.memory_dir,
        output_dir="",
        concurrency=args.eval_concurrency,
        temperature=config.pipeline.default_temperature,
        threshold=args.threshold,
        candidate_top_k=args.top_k,
        gate_top_k=config.pipeline.stage3_gate_top_k,
        max_injected=config.pipeline.stage3_max_injected,
        limit=0,
        ports=args.ports,
        model=args.model,
        timeout=args.timeout,
    )
    await run_stage3(stage3_args, config)
    logger.info("All stages completed: %s", ", ".join(levels))


async def main() -> None:
    config = Config.from_core_file()
    args = build_parser(config).parse_args()
    apply_pipeline_environment(config)
    config.ensure_dirs()
    config.setup_logging()

    if args.command == "stage1":
        await run_stage1(args, config)
    elif args.command == "stage2":
        await run_stage2(args, config)
    elif args.command == "stage3":
        await run_stage3(args, config)
    elif args.command == "all":
        await run_all(args, config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())
