#!/usr/bin/env python
"""
Stage 1 双 Backbone 并行运行器

同时使用两个模型对同一批 6000 题执行采样+挖掘:
  - Backbone A (Qwen2.5-14B):  GPU 0-3, Ports 8100-8103
  - Backbone B (Llama-3.1-8B): GPU 4-7, Ports 8104-8107

输出分离:
  output/stage1/
    01_sampling/
      qwen_unified_sampling_results.json
      llama_unified_sampling_results.json
    02_learning_packets/
      qwen/  level{1..4}_learning_packets.json
      llama/ level{1..4}_learning_packets.json

用法:
  python scripts/stage1/run_stage1_dual.py --concurrency 64
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import Config
from core.agent_pipeline import AgentPipeline
from modules.execute_module import filter_healthy_urls, URLRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dual_stage1")


def build_backbone_configs():
    """从环境变量读取两个 backbone 的配置。"""
    # Backbone A
    ports_a = [int(p) for p in os.environ.get("PORTS", "8100 8101 8102 8103").split()]
    model_a = os.environ.get("SERVED_MODEL_NAME", "qwen")

    # Backbone B
    ports_b = [int(p) for p in os.environ.get("PORTS2", "8004 8005 8006 8007").split()]
    model_b = os.environ.get("SERVED_MODEL2_NAME", "llama")

    host = os.environ.get("HOST", "0.0.0.0")

    url_a = os.environ.get("MODEL_URL", f"http://{host}:8000/v1")
    url_b = os.environ.get("MODEL2_URL", f"http://{host}:8004/v1")

    urls_a = URLRouter.build_urls(url_a, ports_a)
    urls_b = URLRouter.build_urls(url_b, ports_b)

    # 健康检查
    urls_a = filter_healthy_urls(urls_a)
    urls_b = filter_healthy_urls(urls_b)

    if not urls_a:
        raise RuntimeError("Backbone A (Qwen) 无可用端点")
    if not urls_b:
        raise RuntimeError("Backbone B (Llama) 无可用端点")

    return {
        "qwen": {"urls": urls_a, "model": model_a, "label": "qwen"},
        "llama": {"urls": urls_b, "model": model_b, "label": "llama"},
    }


async def run_dual_stage1(sample_concurrency: int = 128, diff_concurrency: int = 96):
    """并行运行两个 backbone 的 Stage 1。"""
    config = Config.from_core_file()
    config.ensure_dirs()
    config.setup_logging()

    backbones = build_backbone_configs()

    # 只需要一个 pipeline 实例作为配置容器
    pipeline = AgentPipeline(config=config, ports=None)

    async def run_one(bb: dict) -> dict:
        label = bb["label"]
        n_urls = len(bb["urls"])
        # 按副本数缩放并发: 4卡→200%, 3卡→150%
        scale = n_urls / 4 * 2
        sc = max(32, int(sample_concurrency * scale))
        dc = max(16, int(diff_concurrency * scale))

        logger.info("=" * 60)
        logger.info("启动 Stage 1 [%s]: model=%s replicas=%d sample_conc=%d diff_conc=%d",
                     label, bb["model"], n_urls, sc, dc)
        logger.info("=" * 60)

        # diff 分析也在 miner.run 用同一个 concurrency, 这里仅控制采样
        result_dir = await pipeline.stage1_learn(
            concurrency=sc,
            diff_concurrency=dc,
            backbone=label,
            urls_override=bb["urls"],
            model_override=bb["model"],
        )
        return {"backbone": label, "replicas": n_urls, "result_dir": str(result_dir) if result_dir else None}

    # 并行执行
    results = await asyncio.gather(
        run_one(backbones["qwen"]),
        run_one(backbones["llama"]),
    )

    logger.info("=" * 60)
    logger.info("双 Backbone Stage 1 全部完成!")
    for r in results:
        status = "OK" if r["result_dir"] else "FAILED"
        logger.info("  [%s] %d卡 %s → %s", r["backbone"], r["replicas"], status, r["result_dir"])
    logger.info("=" * 60)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 双 Backbone 并行运行")
    parser.add_argument("--sample-concurrency", type=int, default=128,
                        help="每 4 卡的采样并发基准 (默认 128, 实际按副本数缩放)")
    parser.add_argument("--diff-concurrency", type=int, default=96,
                        help="每 4 卡的 diff 并发基准 (默认 96)")
    args = parser.parse_args()
    asyncio.run(run_dual_stage1(
        sample_concurrency=args.sample_concurrency,
        diff_concurrency=args.diff_concurrency,
    ))


if __name__ == "__main__":
    main()
