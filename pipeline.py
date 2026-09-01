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
from modules.stage3_snapshot import prepare_transfer_snapshot
from modules.stage2 import (
    build_level_memory,
    compare_pilot_branches,
    compare_generators,
    evaluate_dev8_gate,
    pilot_status,
    prepare_pilot_manifest,
    run_qwen_protocol_probe,
    run_generator_pilot,
    validate_source_principles,
    freeze_stage2_round,
    incremental_registry_status,
    prepare_incremental_target,
    run_incremental_candidate_round,
    audit_and_store_production,
    generate_production_candidates,
    prepare_production_target,
    production_status,
    run_production_complete,
    validate_production_candidates,
    build_principle_test_report,
    generate_raw_candidates,
    generate_variants,
    prepare_principle_test,
    principle_test_status,
    require_variant_approvals,
    run_audit_ablation,
    run_principle_test_smoke,
    run_provider_probe,
    run_robustness_effect_test,
    run_source_effect_test,
    run_transfer_effect_test,
    run_kimi_synthesis_recovery,
    run_role_compare,
    run_adaptive_tradeoff_compare,
    prepare_l23_protocol,
    run_forced_timing_compare,
    run_l23_protocol_calibration,
    run_intervention_compare,
)

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
    stage1.add_argument(
        "--panic-max-correct", type=int, default=defaults.stage1_panic_max_correct,
        help="Largest correct@K count assigned to Panic Zone.",
    )
    stage1.add_argument(
        "--master-min-correct", type=int, default=defaults.stage1_master_min_correct,
        help="Smallest correct@K count assigned to Master Zone.",
    )
    stage1.add_argument("--experiment-seed", type=int, default=defaults.stage1_experiment_seed)
    stage1.add_argument("--temperature", type=float, default=defaults.stage1_temperature)
    stage1.add_argument("--top-p", type=float, default=defaults.stage1_top_p)
    stage1.add_argument("--top-logprobs", type=int, default=defaults.stage1_top_logprobs)
    stage1.add_argument("--entropy-histogram-bins", type=int, default=defaults.stage1_entropy_histogram_bins)
    stage1.add_argument("--max-tokens", type=int, default=defaults.stage1_max_tokens)
    stage1.add_argument(
        "--finalizer-max-tokens", type=int, default=defaults.stage1_finalizer_max_tokens,
    )
    stage1.add_argument(
        "--finalizer-timeout", type=int, default=defaults.stage1_finalizer_timeout,
    )
    stage1.add_argument("--backbone", default=defaults.default_backbone)
    stage1.add_argument("--concurrency", type=int, default=defaults.stage1_concurrency)
    stage1.add_argument("--task-batch-size", type=int, default=defaults.stage1_task_batch_size)
    stage1.add_argument("--skip-sampling", action="store_true")
    stage1.add_argument("--resume", action="store_true")
    _add_common_runtime_args(stage1, config)

    stage2 = subparsers.add_parser("stage2", help="Generate and quality-audit memories.")
    stage2.add_argument(
        "--mode", choices=[
            "build", "validate-source", "probe-qwen", "compare-generators",
            "prepare-pilot", "pilot-run", "pilot-status", "pilot-compare",
            "principle-test",
            "incremental",
        ], default="build",
        help="Build Memory candidates or run source-question Principle validation.",
    )
    stage2.add_argument("--level", default="level1", type=_public_level, choices=list(defaults.public_levels))
    stage2.add_argument(
        "--generator", choices=["deepseek", "qwen"], default="deepseek",
        help="Stage-2 Principle generator. The auditor remains independently configured.",
    )
    stage2.add_argument("--memory-dir", default=defaults.default_memory_dir)
    stage2.add_argument("--backbone", default=defaults.default_backbone)
    stage2.add_argument("--model-concurrency", type=int, default=defaults.stage2_concurrency)
    stage2.add_argument("--limit", type=int, default=0, help="Maximum candidate count; 0 means all.")
    stage2.add_argument("--run-tag", default="", help="Optional artifact isolation tag.")
    stage2.add_argument("--prepare-candidates-only", action="store_true")
    stage2.add_argument("--candidate-bank", default="", help="Frozen candidate bank JSON.")
    stage2.add_argument("--selection-file", default="", help="Stable sampled question IDs.")
    stage2.add_argument("--output-dir", default="", help="Stage 2 audit artifact directory.")
    stage2.add_argument(
        "--candidate-artifact-root", default="",
        help="Stage 2 artifact root containing candidate_banks/ and per-level learning bundles.",
    )
    stage2.add_argument(
        "--validation-output-dir", default="",
        help="Read-only source-validation output directory.",
    )
    stage2.add_argument(
        "--validation-concurrency", type=int,
        default=defaults.stage2_source_validation_concurrency,
    )
    stage2.add_argument(
        "--validation-seeds", type=int, nargs="+",
        default=list(defaults.stage2_source_validation_seeds),
    )
    stage2.add_argument(
        "--validation-temperatures", type=float, nargs="+",
        default=list(defaults.stage2_source_validation_temperatures),
    )
    stage2.add_argument("--expected-candidates", type=int, default=0)
    stage2.add_argument("--expected-questions", type=int, default=0)
    stage2.add_argument("--deepseek-artifact-root", default="")
    stage2.add_argument("--qwen-artifact-root", default="")
    stage2.add_argument("--deepseek-validation-dir", default="")
    stage2.add_argument("--qwen-validation-dir", default="")
    stage2.add_argument(
        "--pilot-root",
        default="output/stage2/generator_comparison/pilot100_v2",
        help="Shared root for the recoverable DeepSeek/Qwen 100-question pilot.",
    )
    stage2.add_argument(
        "--pilot-dev-manifest",
        default="output/stage2/generator_comparison/manifests/dev8.json",
        help="Frozen two-question-per-level development manifest.",
    )
    stage2.add_argument(
        "--pilot-phase", choices=["dev8", "remaining92", "all100"], default="dev8",
    )
    stage2.add_argument("--pilot-concurrency", type=int, default=2)
    stage2.add_argument("--pilot-seed", type=int, default=42)
    stage2.add_argument("--retry-terminal", action="store_true")
    stage2.add_argument(
        "--test-phase",
        choices=[
            "prepare", "probe", "smoke", "variants", "generate", "audit",
            "source", "transfer", "robustness", "report", "status",
            "role-compare", "kimi-synthesis-recovery", "protocol-prepare", "protocol-run",
            "intervention-compare", "forced-timing-compare", "adaptive-tradeoff-compare",
        ],
        default="status",
        help="Phase of the isolated 40-source/40-variant Principle test.",
    )
    stage2.add_argument(
        "--test-root", default="output/stage2/principle_test_80_v1",
        help="Artifact root for the isolated Principle test; never a Memory Store.",
    )
    stage2.add_argument(
        "--protocol-exclude-root", action="append", default=[],
        help=(
            "Earlier protocol-calibration root whose frozen source questions "
            "must be excluded. Repeat this option for multiple prior rounds."
        ),
    )
    stage2.add_argument(
        "--protocol-source-root", action="append", default=[],
        help=(
            "Completed protocol-calibration root to merge for intervention timing "
            "comparison. Repeat this option for multiple roots."
        ),
    )
    stage2.add_argument(
        "--comparison-cache-root", default="",
        help=(
            "Completed intervention-compare root containing frozen plans, "
            "plan-only controls, and generic self-check controls."
        ),
    )
    stage2.add_argument(
        "--timing-smoke-only", action="store_true",
        help="Run only two Principle candidates at seeds 42-44; full --resume reuses them.",
    )
    stage2.add_argument(
        "--adaptive-smoke-only", action="store_true",
        help="Run only two Principle candidates at seeds 42-44 for the adaptive trade-off.",
    )
    stage2.add_argument("--resume", action="store_true", help="Reuse hash-compatible completed artifacts.")
    stage2.add_argument(
        "--incremental-phase",
        choices=["prepare", "generate", "validate", "audit-store", "complete", "status"],
        default="status",
        help="Run one recoverable phase of the 1200-question Stage-2 production flow.",
    )
    stage2.add_argument(
        "--target-size", default="40",
        help="Nested Stage-2 production target: 300 or 1200 source questions.",
    )
    stage2.add_argument(
        "--registry-root", default="output/stage2/candidate_registry",
        help="Global per-question Candidate Registry.",
    )
    stage2.add_argument(
        "--incremental-workspace", default="output/stage2/incremental",
        help="Target manifests and cumulative Candidate Banks.",
    )
    stage2.add_argument(
        "--production-root", default="output/stage2/production_1200_v1",
        help="Frozen 1200-question manifests, production batches, validation, and Memory outputs.",
    )
    stage2.add_argument(
        "--kimi-max-calls", type=int, default=300,
        help="Hard total Kimi budget, including helpful audits, reaudits, and 30 diagnostics.",
    )
    stage2.add_argument(
        "--stage2-state-root", default="state/stage2",
        help="Durable protocol locks and immutable release manifests.",
    )
    stage2.add_argument(
        "--incremental-target-root", default="",
        help="Explicit completed round root used by incremental freeze.",
    )
    stage2.add_argument(
        "--quality-decision", default="",
        help="Round quality JSON with approved=true; calibration report is used when omitted.",
    )
    _add_common_runtime_args(stage2, config)

    stage3 = subparsers.add_parser("stage3", help="Evaluate baseline or memory-guided solving.")
    stage3.add_argument("--variant", choices=["baseline", "guided"], default="guided")
    stage3.add_argument("--test-file", default=defaults.default_test_file)
    stage3.add_argument("--memory-dir", default=defaults.default_memory_dir)
    stage3.add_argument(
        "--pending-memory-store", default="",
        help=(
            "Stage-2 pending_transfer_validation JSONL. For guided transfer evaluation, "
            "an isolated active snapshot is built under --memory-dir."
        ),
    )
    stage3.add_argument(
        "--generation-records-dir", default="",
        help="Stage-2 raw_candidates/questions directory containing routing_semantics.",
    )
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
    if not args.ports:
        raise ValueError("Stage 1 requires at least one actor endpoint")
    from modules.execute_module import is_url_reachable
    dead = [port for port in args.ports if not is_url_reachable(f"http://127.0.0.1:{port}")]
    if dead:
        raise RuntimeError(f"Stage 1 actor endpoints are not reachable: {dead}")


async def run_stage1(args: argparse.Namespace, config: Config) -> None:
    validate_stage1_inputs(args)
    if not 0 <= args.panic_max_correct < args.master_min_correct <= args.k_trials:
        raise ValueError(
            "Stage 1 zone thresholds must satisfy "
            f"0 <= panic_max_correct < master_min_correct <= k_trials; got "
            f"{args.panic_max_correct}, {args.master_min_correct}, {args.k_trials}"
        )
    pipeline = build_agent_pipeline(args, config)
    logger.info("Stage 1 started: sampling once and producing level1-level4 learning bundles")
    await pipeline.stage1_learn(
        dataset_path=args.dataset,
        concurrency=args.concurrency,
        k_trials=args.k_trials,
        panic_max_correct=args.panic_max_correct,
        master_min_correct=args.master_min_correct,
        experiment_seed=args.experiment_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.top_logprobs,
        entropy_histogram_bins=args.entropy_histogram_bins,
        max_tokens=args.max_tokens,
        finalizer_max_tokens=args.finalizer_max_tokens,
        finalizer_timeout=args.finalizer_timeout,
        task_batch_size=args.task_batch_size,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_dataset_size=args.expected_dataset_size,
        skip_sampling=args.skip_sampling,
        backbone=args.backbone,
        resume=args.resume,
    )
    logger.info("Stage 1 completed: level1-level4")


async def run_stage2(args: argparse.Namespace, config: Config) -> None:
    if args.mode == "incremental":
        target = int(args.target_size)
        if target not in {300, 1200}:
            raise ValueError("Stage 2 production target must be 300 or 1200")
        production_root = Path(args.production_root)
        target_root = production_root / f"batch_{target:04d}"
        phase = args.incremental_phase
        incremental_phase_zh = {
            "prepare": "冻结1200题生产池和当前批次",
            "generate": "DeepSeek生成并执行代码格式检查",
            "validate": "本地Llama以Forced-C进行来源题验证",
            "audit-store": "Kimi限额审查、修复重验并分流Memory",
            "complete": "顺序完成生成、验证、审查和Memory分流",
            "status": "查看生产批次状态",
        }.get(phase, phase)
        logger.info(
            "Stage 2生产流程：操作=%s（%s），目标规模=%s，backbone=%s",
            incremental_phase_zh, phase, target, args.backbone,
        )
        if phase == "status":
            result = production_status(
                production_root=production_root, target_size=target,
            )
        elif phase == "prepare":
            result = prepare_production_target(
                config, production_root=production_root,
                target_size=target, backbone=args.backbone,
            )
        elif phase == "generate":
            prepare_production_target(
                config, production_root=production_root,
                target_size=target, backbone=args.backbone,
            )
            result = await generate_production_candidates(
                config, target_root=target_root, concurrency=args.model_concurrency,
            )
        elif phase == "validate":
            result = await validate_production_candidates(
                config, target_root=target_root, ports=args.ports,
                model=args.model, concurrency=args.validation_concurrency,
                timeout=args.timeout,
            )
        elif phase == "audit-store":
            result = await audit_and_store_production(
                config, target_root=target_root, ports=args.ports,
                model=args.model, validation_concurrency=args.validation_concurrency,
                audit_concurrency=args.model_concurrency, timeout=args.timeout,
                kimi_max_calls=args.kimi_max_calls,
            )
        else:
            result = await run_production_complete(
                config, production_root=production_root,
                target_size=target, backbone=args.backbone,
                ports=args.ports, model=args.model,
                generation_concurrency=args.model_concurrency,
                validation_concurrency=args.validation_concurrency,
                audit_concurrency=args.model_concurrency,
                timeout=args.timeout, kimi_max_calls=args.kimi_max_calls,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.mode == "principle-test":
        root = Path(args.test_root)
        phase = args.test_phase
        phase_zh = {
            "prepare": "准备并冻结40道来源题清单",
            "probe": "远程模型接口探针",
            "smoke": "8题最小流程测试",
            "variants": "生成并验算40道结构变式题",
            "generate": "生成40道来源题的Principle",
            "audit": "比较三种质量审查协议",
            "source": "来源题纠偏效果与介入方式测试",
            "transfer": "结构变式题迁移测试",
            "robustness": "温度稳健性测试",
            "report": "生成最终验收报告",
            "status": "查看当前测试状态",
            "role-compare": "8题双生成器交叉审查与Llama配对验证",
            "kimi-synthesis-recovery": "仅恢复4道Kimi合成截断题",
            "protocol-prepare": "冻结全新Level 2/3共20道协议校准题",
            "protocol-run": "DeepSeek生成、Kimi六维审查与Llama配对验证",
            "intervention-compare": "40题Principle介入时机对比",
            "forced-timing-compare": "40题Principle固定注入时机对比",
            "adaptive-tradeoff-compare": "草稿感知的B/C自适应介入对比",
        }.get(phase, phase)
        logger.info(
            "Stage 2 Principle测试：阶段=%s（%s），结果目录=%s，backbone=%s，模型=%s，"
            "端口=%s，生成并发=%d，验证并发=%d",
            phase_zh, phase, root, args.backbone, args.model, ",".join(map(str, args.ports)),
            args.model_concurrency, args.validation_concurrency,
        )
        if phase == "prepare":
            result = prepare_principle_test(
                config, test_root=root, backbone=args.backbone,
                sources_per_level=config.stage2.principle_test_sources_per_level,
                reserve_per_level=config.stage2.principle_test_reserve_per_level,
                seed=config.stage2.principle_test_seed,
            )
        elif phase == "probe":
            result = await run_provider_probe(config, test_root=root, repeats=3)
        elif phase == "smoke":
            result = await run_principle_test_smoke(
                config, test_root=root, backbone=args.backbone,
                ports=args.ports, model=args.model,
                concurrency=args.validation_concurrency, timeout=args.timeout,
            )
        elif phase == "variants":
            result = await generate_variants(
                config, test_root=root, concurrency=args.model_concurrency,
            )
        elif phase == "generate":
            require_variant_approvals(root)
            result = await generate_raw_candidates(
                config, test_root=root, concurrency=args.model_concurrency,
            )
        elif phase == "audit":
            result = await run_audit_ablation(
                config, test_root=root, concurrency=args.model_concurrency,
            )
        elif phase == "source":
            result = await run_source_effect_test(
                config, test_root=root, ports=args.ports, model=args.model,
                concurrency=args.validation_concurrency, timeout=args.timeout,
            )
        elif phase == "transfer":
            result = await run_transfer_effect_test(
                config, test_root=root, ports=args.ports, model=args.model,
                concurrency=args.validation_concurrency, timeout=args.timeout,
            )
        elif phase == "robustness":
            result = await run_robustness_effect_test(
                config, test_root=root, ports=args.ports, model=args.model,
                concurrency=args.validation_concurrency, timeout=args.timeout,
            )
        elif phase == "role-compare":
            result = await run_role_compare(
                config, test_root=root, ports=args.ports, model=args.model,
                model_concurrency=args.model_concurrency,
                validation_concurrency=args.validation_concurrency,
                timeout=args.timeout, resume=args.resume,
            )
        elif phase == "kimi-synthesis-recovery":
            result = await run_kimi_synthesis_recovery(
                config, test_root=root, resume=args.resume,
            )
        elif phase == "protocol-prepare":
            result = prepare_l23_protocol(
                config, test_root=root, backbone=args.backbone,
                exclude_protocol_roots=[
                    Path(value) for value in args.protocol_exclude_root
                ],
            )
        elif phase == "protocol-run":
            result = await run_l23_protocol_calibration(
                config, test_root=root, ports=args.ports, model=args.model,
                generation_concurrency=args.model_concurrency,
                validation_concurrency=args.validation_concurrency,
                timeout=args.timeout,
            )
        elif phase == "intervention-compare":
            if not args.protocol_source_root:
                raise ValueError(
                    "intervention-compare requires at least one --protocol-source-root"
                )
            result = await run_intervention_compare(
                config,
                test_root=root,
                source_roots=[Path(value) for value in args.protocol_source_root],
                ports=args.ports,
                model=args.model,
                validation_concurrency=args.validation_concurrency,
                timeout=args.timeout,
            )
        elif phase == "forced-timing-compare":
            if not args.protocol_source_root:
                raise ValueError(
                    "forced-timing-compare requires at least one --protocol-source-root"
                )
            if not args.comparison_cache_root:
                raise ValueError(
                    "forced-timing-compare requires --comparison-cache-root"
                )
            result = await run_forced_timing_compare(
                config,
                test_root=root,
                source_roots=[Path(value) for value in args.protocol_source_root],
                comparison_cache_root=Path(args.comparison_cache_root),
                ports=args.ports,
                model=args.model,
                validation_concurrency=args.validation_concurrency,
                timeout=args.timeout,
                smoke_only=args.timing_smoke_only,
            )
        elif phase == "adaptive-tradeoff-compare":
            if not args.protocol_source_root:
                raise ValueError(
                    "adaptive-tradeoff-compare requires at least one --protocol-source-root"
                )
            if not args.comparison_cache_root:
                raise ValueError(
                    "adaptive-tradeoff-compare requires --comparison-cache-root"
                )
            result = await run_adaptive_tradeoff_compare(
                config,
                test_root=root,
                source_roots=[Path(value) for value in args.protocol_source_root],
                comparison_cache_root=Path(args.comparison_cache_root),
                ports=args.ports,
                model=args.model,
                validation_concurrency=args.validation_concurrency,
                timeout=args.timeout,
                smoke_only=args.adaptive_smoke_only,
            )
        elif phase == "report":
            result = build_principle_test_report(root)
        else:
            result = principle_test_status(root)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.mode in {"prepare-pilot", "pilot-run", "pilot-status", "pilot-compare"}:
        pilot_root = Path(args.pilot_root)
        if args.mode == "prepare-pilot":
            manifest = prepare_pilot_manifest(
                config,
                output_root=pilot_root,
                dev_manifest_path=Path(args.pilot_dev_manifest),
                backbone=args.backbone,
                per_level=25,
                seed=args.pilot_seed,
            )
            logger.info(
                "Frozen pilot manifest created: questions=%d hash=%s path=%s",
                manifest["actual_count"], manifest["manifest_hash"], pilot_root,
            )
            return
        if not (pilot_root / "sample_manifest.json").exists():
            prepare_pilot_manifest(
                config,
                output_root=pilot_root,
                dev_manifest_path=Path(args.pilot_dev_manifest),
                backbone=args.backbone,
                per_level=25,
                seed=args.pilot_seed,
            )
        if args.mode == "pilot-status":
            print(json.dumps(pilot_status(pilot_root), ensure_ascii=False, indent=2))
            return
        if args.mode == "pilot-compare":
            result = compare_pilot_branches(pilot_root)
            logger.info("Paired generator report written: %s", pilot_root / "comparison")
            print(json.dumps({
                "manifest_hash": result["manifest_hash"],
                "winner_selected": result["winner_selected"],
            }, ensure_ascii=False, indent=2))
            return
        if args.generator == "qwen" and args.pilot_phase == "dev8":
            probe_dir = pilot_root / "qwen_protocol_probe"
            probe = await run_qwen_protocol_probe(config, output_dir=probe_dir, repeats=3)
            if not probe["all_passed"]:
                raise RuntimeError(f"Qwen thinking protocol probe failed: {probe}")
        result = await run_generator_pilot(
            config,
            output_root=pilot_root,
            generator_id=args.generator,
            phase=args.pilot_phase,
            concurrency=args.pilot_concurrency,
            retry_terminal=args.retry_terminal,
        )
        logger.info(
            "Generator pilot phase completed: generator=%s phase=%s terminal=%s/%s",
            args.generator, args.pilot_phase,
            result["phase_terminal_count"], result["phase_question_count"],
        )
        if args.pilot_phase == "dev8":
            gate = evaluate_dev8_gate(pilot_root, args.generator)
            print(json.dumps(gate, ensure_ascii=False, indent=2))
            if not gate["passed"]:
                raise RuntimeError(f"Development gate failed: {gate['reasons']}")
        return
    if args.mode == "probe-qwen":
        output_dir = Path(args.output_dir or "output/stage2/generator_comparison/qwen_protocol_probe")
        summary = await run_qwen_protocol_probe(config, output_dir=output_dir, repeats=3)
        if not summary["all_passed"]:
            raise RuntimeError(f"Qwen thinking protocol probe failed: {summary}")
        logger.info("Qwen thinking protocol probe passed: %s", output_dir)
        return
    if args.mode == "compare-generators":
        required = {
            "--deepseek-artifact-root": args.deepseek_artifact_root,
            "--qwen-artifact-root": args.qwen_artifact_root,
            "--output-dir": args.output_dir,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"stage2 compare-generators missing: {', '.join(missing)}")
        result = compare_generators(
            deepseek_root=Path(args.deepseek_artifact_root),
            qwen_root=Path(args.qwen_artifact_root),
            output_dir=Path(args.output_dir),
            deepseek_validation=(
                Path(args.deepseek_validation_dir) if args.deepseek_validation_dir else None
            ),
            qwen_validation=(
                Path(args.qwen_validation_dir) if args.qwen_validation_dir else None
            ),
        )
        logger.info(
            "Stage 2 generator comparison completed: winner=%s output=%s",
            result.get("winner") or "not_selected",
            args.output_dir,
        )
        return
    if args.mode == "validate-source":
        if not args.candidate_artifact_root:
            raise ValueError("stage2 --mode validate-source requires --candidate-artifact-root")
        artifact_root = Path(args.candidate_artifact_root)
        validation_output = (
            Path(args.validation_output_dir)
            if args.validation_output_dir else
            artifact_root / "source_validation_llama10"
        )
        await validate_source_principles(
            config,
            artifact_root=artifact_root,
            output_dir=validation_output,
            ports=args.ports,
            model=args.model,
            seeds=tuple(args.validation_seeds),
            temperatures=tuple(args.validation_temperatures),
            concurrency=args.validation_concurrency,
            timeout=args.timeout,
            expected_candidates=args.expected_candidates,
            expected_questions=args.expected_questions,
        )
        logger.info("Stage 2 source validation completed: %s", validation_output)
        return
    runtime = SimpleNamespace(
        config=config,
        memory=MemoryModule(config, memory_dir=args.memory_dir),
    )
    level = _public_level(args.level)
    round_id = _level_number(level)
    dataset = level
    logger.info("Stage 2 Memory Factory started: %s", level)
    summary = await build_level_memory(
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
        generator_id=args.generator,
    )
    if summary.get("status") == "empty_stage1":
        raise FileNotFoundError(
            "Stage 2 found no Stage-1 packets. "
            f"Resolved path: {summary.get('packet_path')}. "
            "Set --backbone to the Stage-1 output subdirectory (for example: --backbone llama)."
        )
    logger.info("Stage 2 completed: %s", level)


async def run_stage3(args: argparse.Namespace, config: Config) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(config, args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.variant == "guided" and args.pending_memory_store:
        if not args.generation_records_dir:
            raise ValueError("--generation-records-dir is required with --pending-memory-store")
        snapshot = prepare_transfer_snapshot(
            pending_store=args.pending_memory_store,
            generation_dir=args.generation_records_dir,
            snapshot_dir=args.memory_dir,
        )
        logger.info(
            "Stage 3迁移评测快照已准备：%d条Memory，原Store保持pending状态，快照=%s",
            snapshot["memory_count"], args.memory_dir,
        )
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
            mode="build",
            generator="deepseek",
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
