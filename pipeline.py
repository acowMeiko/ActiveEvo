#!/usr/bin/env python
"""
MemMetaV3 — 经验驱动的自进化推理框架
====================================

统一入口文件。清晰展示三阶段 Agent 工作流：

  Round N (N = 1, 2, 3, 4):
  ┌─── Stage 1: 学习区挖掘 ────────────────────────────────────┐
  │  execute.sample_unstable()    — 多次推理，发现 learning-zone  │
  │  execute.mine_hard_negatives() — 挖掘困难负样本               │
  │  plan.extract_semantics()      — 三维语义特征提取             │
  │  plan.diff_analysis()          — 正确/错误路径差异分析         │
  └───────────────────────────────────────────────────────────────┘
                                  ↓
  ┌─── Stage 2: 记忆构建 ─────────────────────────────────────────┐
  │  memory.ingest()              — 经验入库 + 去重               │
  │  memory.build_indices()       — FAISS 三维向量索引            │
  │  memory.save() + snapshot()   — 持久化 + 快照                │
  └───────────────────────────────────────────────────────────────┘
                                  ↓
  ┌─── Stage 3: 检索增强评估 ─────────────────────────────────────┐
  │  plan.extract_semantics()     — 提取测试题语义特征            │
  │  memory.retrieve()            — 三维门控检索匹配经验          │
  │  execute.answer_guided()      — 经验引导答题                 │
  │  execute.check_answer()       — 答案校验                    │
  └───────────────────────────────────────────────────────────────┘

四轮课程学习: Round 1 → Round 2 → Round 3 → Round 4
记忆跨轮增量保留: Round N 的记忆 = Round 1..N-1 全量 + Round N 新增

用法:
    # 全流程四轮 (4卡服务器部署)
    python pipeline.py --levels 1 2 3 4 --ports 8100 8101 8102 8103

  # 全流程四轮 (单卡本地测试)
  python pipeline.py --levels 1 2 3 4 --local

  # 仅跑指定 Stage
  python pipeline.py --stage 1 -d dpo_level1 --ports 8100 8101
  python pipeline.py --stage 2 -d dpo_level1
  python pipeline.py --stage 3 -d dpo_level1 --ports 8100

  # 跳过采样和挖掘 (复用已有 Stage 1 中间结果)
  python pipeline.py --levels 1 --skip-sampling --skip-mining
"""
import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

from core.config import Config
from core.agent_pipeline import AgentPipeline
from utils.output_paths import stage3_progress_state_file
from utils.io_utils import load_json, save_json

logger = logging.getLogger(__name__)


def _dataset_level(dataset: str) -> int:
    text = str(dataset or "")
    match = re.search(r"dpo_level(\d+)", text)
    if match:
        return int(match.group(1))
    return 1


def _stage3_test_scope_key(test_input: Optional[str]) -> str:
    if not test_input:
        return "default"
    path = Path(str(test_input))
    parent = path.parent.name if path.parent.name else "root"
    stem = path.stem or "dataset"
    return f"{parent}_{stem}"


# ════════════════════════════════════════════════════════════════
# CLI 参数解析
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="MemMetaV3 — 经验驱动的自进化推理框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline.py --levels 1 2 3 4 --ports 8100 8101 8102 8103
  python pipeline.py --levels 1 2 3 4 --local
  python pipeline.py --stage 1 -d dpo_level1 --ports 8100 8101
  python pipeline.py --stage 3 -d dpo_level1
""",
    )

    # 运行模式
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="课程学习轮次 (默认: 1 2 3 4)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="仅运行指定 Stage (需配合 -d)")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="指定数据集 (如 dpo_level1, 与 --stage 配合)")

    # 部署模式
    parser.add_argument("--local", action="store_true",
                        help="单卡本地 vLLM 测试模式 (端口 8000)")
    parser.add_argument("--ports", type=int, nargs="+", default=None,
                        help="多卡部署端口 (如: 8000 8001 8002 ...)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="覆盖模型名称")
    parser.add_argument("--backbone", type=str, default=os.environ.get("MEMUA_BACKBONE", ""),
                        help="Stage 1/2 产物标签，例如 llama")

    # 测试与记忆
    parser.add_argument("--test-input", type=str, default=None,
                        help="测试集路径")
    parser.add_argument("--memory-dir", type=str, default="memory/v3",
                        help="外部记忆目录")

    # Stage 1 参数
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--diff-concurrency", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=15)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-mining", action="store_true")

    # Stage 3 参数
    parser.add_argument("--eval-concurrency", type=int, default=10)
    parser.add_argument("--stage2-concurrency", type=int, default=8,
                        help="Stage 2 验证并发数")
    parser.add_argument("--stage2-candidate-concurrency", type=int, default=0,
                        help="Stage 2 端到端候选 worker 数；0 使用环境配置或 generation 并发")
    parser.add_argument("--stage2-limit", type=int, default=0,
                        help="Stage 2 最多处理多少个候选；0 表示全部")
    parser.add_argument("--stage2-run-tag", type=str, default="",
                        help="Stage 2 输出隔离标签，例如 smoke10 或 pilot30")
    parser.add_argument("--dsd-cache-dir", type=str, default="",
                        help="可复用的 DSD/BGE 缓存目录")
    parser.add_argument("--rebuild-dsd", action="store_true",
                        help="忽略现有 DSD 缓存并重建")
    parser.add_argument("--prepare-dsd-only", action="store_true",
                        help="只构建/检查 DSD 缓存")
    parser.add_argument("--prepare-candidates-only", action="store_true",
                        help="只生成并冻结 Quality-pass 候选")
    parser.add_argument("--candidate-bank", type=str, default="",
                        help="冻结候选银行 JSON 路径")
    parser.add_argument("--neighbor-validation", choices=["on", "off", "gate-only"], default="on",
                        help="Stage 2 Neighbor Validation 消融开关")
    parser.add_argument("--stage2-selection-file", type=str, default="",
                        help="只处理 JSON 中列出的 question_id；用于稳定分层 Pilot")
    parser.add_argument("--stage2-output-dir", type=str, default="",
                        help="Stage 2 审计产物隔离目录")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=5)

    # 其他
    parser.add_argument("--timeout", type=int, default=600)

    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
# 核心编排
# ════════════════════════════════════════════════════════════════

async def run_single_stage(pipeline: AgentPipeline, args, config: Config):
    """单 Stage 模式。"""
    dataset = args.dataset or "dpo_level1"
    level = _dataset_level(dataset)

    if args.stage == 1:
        logger.info("═══ Stage 1: 学习区挖掘 (统一数据集) ═══")
        await pipeline.stage1_learn(
            concurrency=args.concurrency,
            skip_sampling=args.skip_sampling,
        )

    elif args.stage == 2:
        logger.info("═══ Stage 2: 记忆构建 (%s, round=%d) ═══", dataset, level)
        await pipeline.stage2_memorize(
            dataset=dataset,
            round_id=level,
            validation_concurrency=args.stage2_concurrency,
            backbone=args.backbone,
            max_records=args.stage2_limit,
            run_tag=args.stage2_run_tag,
            dsd_cache_dir=args.dsd_cache_dir,
            rebuild_dsd=args.rebuild_dsd,
            prepare_dsd_only=args.prepare_dsd_only,
            prepare_candidates_only=args.prepare_candidates_only,
            candidate_bank=args.candidate_bank,
            neighbor_validation=args.neighbor_validation != "off",
            gate_only=args.neighbor_validation == "gate-only",
            selection_file=args.stage2_selection_file,
            artifact_dir=args.stage2_output_dir,
            candidate_concurrency=args.stage2_candidate_concurrency,
        )

    elif args.stage == 3:
        logger.info("═══ Stage 3: 检索增强评估 (%s) ═══", dataset)
        test_path = args.test_input or str(config.paths.test_dir / "deepscaler.json")
        test_data = load_json(test_path)
        test_scope_key = _stage3_test_scope_key(test_path)
        result = await pipeline.stage3_evaluate(
            test_data=test_data,
            threshold=args.threshold,
            top_k=args.top_k,
            concurrency=args.eval_concurrency,
            eval_level=level,
            dataset=dataset,
            progress_state_path=stage3_progress_state_file(config.paths.output_dir, test_scope_key),
            test_scope_key=test_scope_key,
        )
        from utils.output_paths import stage3_guided_results_file, stage3_guided_summary_file
        out_root = config.paths.output_dir
        save_json(result["results"], str(stage3_guided_results_file(out_root, dataset)))
        save_json(result["summary"], str(stage3_guided_summary_file(out_root, dataset)))


async def run_full_curriculum(pipeline: AgentPipeline, args, config: Config):
    """全流程四轮课程学习。

    每轮流程:
      Stage 1: 能力感知的样本分区与正负对构造
        1.1 execute.sample_unstable()     — K次推理，三分区 (Learning/Master/Panic)
        1.2 execute.mine_hard_negatives() — 逐步前缀分叉分数挖掘正负样本对
        1.3 封装 learning_pairs           — 含 zone 标签 + divergence info

      Stage 2: 记忆构建
        2.1 memory.load()                 — 加载前轮记忆 (增量)
        2.2 memory.ingest()               — 提取经验入库 + 去重
        2.3 memory.build_indices()        — 三维 FAISS 索引
        2.4 memory.save() + snapshot()    — 持久化 + 快照

      Stage 3: 检索增强评估 + 学习闭环
        3.1 plan.extract_semantics()      — 提取测试题三维语义
        3.2 memory.retrieve()             — 三维门控检索 (importance_score 加权)
        3.3 execute.answer_guided()       — 经验引导答题
        3.4 record_feedback()             — 实时更新 principle 重要性
        3.5 execute.check_answer()        — 答案校验 + 汇总
    """
    summaries = await pipeline.run_full_curriculum(
        levels=args.levels,
        test_input=args.test_input,
        concurrency=args.concurrency,
        stage2_concurrency=args.stage2_concurrency,
        eval_concurrency=args.eval_concurrency,
        threshold=args.threshold,
        top_k=args.top_k,
        skip_sampling=args.skip_sampling,
    )
    return summaries


# ════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════

async def main():
    args = parse_args()

    # 初始化配置
    config = Config.from_core_file()
    config.ensure_dirs()
    config.setup_logging()

    # 确定部署模式
    ports = args.ports
    if args.local and not ports:
        ports = [8000]  # 单卡本地 vLLM 端口
        logger.info("单卡本地测试模式 (port=8000)")

    # 构建 Agent Pipeline
    pipeline = AgentPipeline(
        config=config,
        ports=ports,
        model_override=args.model,
        memory_dir=args.memory_dir,
        timeout=args.timeout,
    )

    # 分发
    if args.stage is not None:
        await run_single_stage(pipeline, args, config)
    else:
        summaries = await run_full_curriculum(pipeline, args, config)
        logger.info("全部完成! 各轮摘要: %s", summaries)


if __name__ == "__main__":
    asyncio.run(main())
