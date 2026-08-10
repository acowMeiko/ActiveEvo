#!/usr/bin/env python
"""
Stage 1 - Learning Pair Builder

串联三阶段流程，产出 learning_pairs.json（仅含正负对 + 分区信息，无 diff/principle）：
  1. 多次推理采样，识别 Learning/Master/Panic Zone  (sample_unstable_questions)
  2. Hard negative 挖掘，逐步分叉分数选取最优正/负轨迹  (hard_negative_mining)
  3. 封装 learning_pairs（含 zone 标签 + divergence info，不含 diff/principle）

Principle 生成已移至 Stage 2。

Learning Pair 格式：
  {
    "idx":                  int,
    "question":             str,
    "expected_answer":      str,
    "zone":                 str,   # "learning" | "master" | "panic"
    "chosen_trajectory":    str,
    "rejected_trajectory":  str,
    "divergence_score":     float,
    "divergence_step_idx":  int,
    "divergence_step_text": str,
    "metadata":             dict
  }

用法示例：
  # 全流程
  python scripts/stage1/build_learning_packets.py --dataset dpo_level1 --ports 8000 8001

  # 跳过采样（复用已有 sampling_results.json）
  python scripts/stage1/build_learning_packets.py --dataset dpo_level1 --skip-sampling --ports 8000 8001

  # 跳过采样 + 挖掘（复用已有 hard_negative_pairs.json）
  python scripts/stage1/build_learning_packets.py --dataset dpo_level1 --skip-sampling --skip-mining --ports 8000 8001
"""
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import Config
from utils.io_utils import load_json, save_json
from utils.output_paths import (
    stage1_learning_pairs_file,
    stage1_mining_file,
    stage1_sampling_file,
)

# 复用已有模块中的核心类，不重复实现逻辑
from scripts.stage1.sample_unstable_questions import ModelTester, build_urls
from scripts.stage1.hard_negative_mining import HardNegativeMiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage A: 多次推理采样
# ---------------------------------------------------------------------------

async def run_sampling(
    config: Config,
    dataset_name: str,
    urls: List[str],
    concurrency: int,
    batch_size: int,
    max_trials: int,
    temperature: float,
) -> Optional[Path]:
    """运行推理采样，写入 sampling_results.json，返回输出路径。"""
    output_path = stage1_sampling_file(config.paths.output_dir, dataset_name)

    tester = ModelTester(config, dataset_name, urls=urls)
    tester.temperature = temperature
    tester.batch_size = batch_size
    tester.max_trials = max_trials

    logger.info("=" * 60)
    logger.info("Stage A: 认知边界采样")
    logger.info(f"并发: {concurrency} | 单题批次: {batch_size} | 最大试验: {max_trials}")
    logger.info("=" * 60)

    summary = await tester.run_test(concurrency=concurrency)
    if "error" in summary:
        logger.error(f"采样失败: {summary['error']}")
        return None

    logger.info(
        f"[Stage A 完成] 总样本: {summary['total']} | "
        f"learning-zone: {summary['usable_samples']} | "
        f"输出: {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Stage B: Hard Negative 挖掘
# ---------------------------------------------------------------------------

async def run_mining(
    config: Config,
    dataset_name: str,
    sampling_path: Path,
) -> Optional[Path]:
    """从 sampling_results.json 挖掘 hard negative pairs，写入 hard_negative_pairs.json。"""
    output_path = stage1_mining_file(config.paths.output_dir, dataset_name)

    logger.info("=" * 60)
    logger.info("Stage B: Hard Negative 挖掘")
    logger.info("=" * 60)

    miner = HardNegativeMiner(config, dataset_name)

    # run_from_file 是单线程版本，避免在 asyncio 事件循环内 fork 进程导致的潜在问题
    stats = await asyncio.to_thread(
        miner.run_from_file,
        str(sampling_path),
        str(output_path),
    )

    if "error" in stats:
        logger.error(f"挖掘失败: {stats['error']}")
        return None

    logger.info(
        f"[Stage B 完成] 有效样本: {stats.get('total_valid_samples', 0)} | "
        f"divergent: {stats.get('divergent_questions', 0)} | "
        f"pairs: {stats.get('total_pairs_generated', 0)} | "
        f"输出: {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Stage C: 封装 Learning Pairs（纯数据，无 LLM，含分区标签）
# ---------------------------------------------------------------------------

def build_learning_pairs(
    config: Config,
    dataset_name: str,
    pairs_path: Path,
) -> Optional[Path]:
    """从 hard_negative_pairs.json + sampling_results.json 封装 learning_pairs.json。

    - 读取 sampling_results 以获取 zone 标签 (learning/master/panic)
    - 直接读取 hard_negative_mining 产出的 divergence 信息
    - 无 LLM 调用，无 diff/principle 生成
    """
    output_path = stage1_learning_pairs_file(config.paths.output_dir, dataset_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: List[Dict[str, Any]] = load_json(str(pairs_path))
    if not isinstance(pairs, list) or not pairs:
        logger.error("pairs 文件为空或格式不正确")
        return None

    # 构建采样状态映射 (idx → success_rate, zone)
    sampling_path = stage1_sampling_file(config.paths.output_dir, dataset_name)
    success_rate_map: Dict[int, Optional[float]] = {}
    status_map: Dict[int, str] = {}
    if sampling_path.exists():
        try:
            raw = load_json(str(sampling_path))
            for item in raw.get("results", []):
                idx = item.get("idx")
                if idx is not None:
                    success_rate_map[idx] = item.get("success_rate", 0.0)
                    status_map[idx] = item.get("status", "unknown")
        except Exception as exc:
            logger.warning(f"无法读取 sampling_results: {exc}")

    _ZONE_MAP = {
        "early_stop": "learning",
        "all_correct": "master",
        "all_wrong": "panic",
    }

    logger.info("=" * 60)
    logger.info("Stage C: Learning Pair 封装")
    logger.info(f"共 {len(pairs)} 对正负样本")
    logger.info("=" * 60)

    learning_pairs: List[Dict[str, Any]] = []
    skipped = 0
    for item in pairs:
        question = str(item.get("question", "")).strip()
        expected = str(item.get("expected_answer", "")).strip()
        chosen = str(item.get("chosen_path", "")).strip()
        rejected = str(item.get("rejected_path", "")).strip()
        item_idx = item.get("idx", 0)

        if not all([question, expected, chosen, rejected]):
            skipped += 1
            continue

        divergence_score = item.get("similarity", 0.0)
        sampling_status = status_map.get(item_idx, "unknown")
        zone = _ZONE_MAP.get(sampling_status, "learning")

        learning_pairs.append({
            "idx": item_idx,
            "question": question,
            "expected_answer": expected,
            "zone": zone,
            "chosen_trajectory": chosen,
            "rejected_trajectory": rejected,
            "divergence_score": divergence_score,
            "divergence_step_idx": item.get("divergence_step_idx", 0),
            "divergence_step_text": item.get("divergence_step_text", ""),
            "metadata": {
                "dataset": dataset_name,
                "success_rate": success_rate_map.get(item_idx),
                "sampling_status": sampling_status,
                "is_usable": zone == "learning" and divergence_score > 0.3,
            },
        })

    learning_pairs.sort(key=lambda x: x.get("idx", 0))
    save_json(learning_pairs, str(output_path))

    learning_n = sum(1 for p in learning_pairs if p.get("zone") == "learning")
    logger.info(
        f"[Stage C 完成] pairs: {len(learning_pairs)} (learning={learning_n}, skipped={skipped}) | "
        f"输出: {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 Learning Pair Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"],
        help="数据集名称",
    )
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=None,
        help="多实例本地 API 端口列表，例如 --ports 8000 8001 8002",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=120,
        help="Stage A 并行题目数 (默认 120)",
    )
    parser.add_argument(
        "--diff-concurrency",
        type=int,
        default=20,
        help="已废弃：Stage C 不再做 diff 分析，保留此参数仅为兼容旧脚本调用",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Stage A 每题单轮生成数 (默认 5)",
    )
    parser.add_argument(
        "--max-trials", "-t",
        type=int,
        default=15,
        help="Stage A 每题最大推理次数 (默认 15)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Stage A 推理温度 (默认 0.7)",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="跳过 Stage A，复用已有 sampling_results.json",
    )
    parser.add_argument(
        "--skip-mining",
        action="store_true",
        help="跳过 Stage B，复用已有 hard_negative_pairs.json（自动隐含 --skip-sampling）",
    )

    args = parser.parse_args()

    # --skip-mining 隐含 --skip-sampling
    if args.skip_mining:
        args.skip_sampling = True

    config = Config.from_core_file()
    config.setup_logging()

    base_url = config.hyperparams.weak_model_url
    urls = build_urls(base_url, args.ports)

    logger.info("=" * 70)
    logger.info("MetaEvo Stage 1 - Learning Pair Builder")
    logger.info(f"数据集: {args.dataset} | 端口池: {[u.split(':')[-1][:6] for u in urls]}")
    logger.info("=" * 70)

    # ── Stage A ──────────────────────────────────────────────────────────────
    sampling_path = stage1_sampling_file(config.paths.output_dir, args.dataset)

    if args.skip_sampling:
        if not sampling_path.exists():
            logger.error(f"--skip-sampling 指定但文件不存在: {sampling_path}")
            raise SystemExit(1)
        logger.info(f"[Stage A 已跳过] 使用已有文件: {sampling_path}")
    else:
        result = await run_sampling(
            config=config,
            dataset_name=args.dataset,
            urls=urls,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_trials=args.max_trials,
            temperature=args.temperature,
        )
        if result is None:
            logger.error("Stage A 失败，流程终止")
            raise SystemExit(1)
        sampling_path = result

    # ── Stage B ──────────────────────────────────────────────────────────────
    pairs_path = stage1_mining_file(config.paths.output_dir, args.dataset)

    if args.skip_mining:
        if not pairs_path.exists():
            logger.error(f"--skip-mining 指定但文件不存在: {pairs_path}")
            raise SystemExit(1)
        logger.info(f"[Stage B 已跳过] 使用已有文件: {pairs_path}")
    else:
        result = await run_mining(
            config=config,
            dataset_name=args.dataset,
            sampling_path=sampling_path,
        )
        if result is None:
            logger.error("Stage B 失败，流程终止")
            raise SystemExit(1)
        pairs_path = result

    # ── Stage C ──────────────────────────────────────────────────────────────
    output = build_learning_pairs(
        config=config,
        dataset_name=args.dataset,
        pairs_path=pairs_path,
    )
    if output is None:
        logger.error("Stage C 失败，流程终止")
        raise SystemExit(1)

    logger.info("=" * 70)
    logger.info(f"全流程完成！Learning Pairs 保存于: {output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
