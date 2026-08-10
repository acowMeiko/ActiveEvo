#!/usr/bin/env python
"""
Stage 1 单 Backbone (Llama) — Qwen 已跑完，只补 Llama 侧。
用法:
  python run_stage1_llama.py --sample-concurrency 128 --diff-concurrency 96
  python run_stage1_llama.py --resume   # 中断后继续
"""
import asyncio, logging, os, sys
from pathlib import Path

from core.config import Config
from core.agent_pipeline import AgentPipeline
from modules.execute_module import filter_healthy_urls, URLRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("stage1_llama")


def build_llama_config():
    ports = [int(p) for p in os.environ.get("PORTS", "8100 8101 8102 8103 8104 8105 8106 8107").split()]
    model = os.environ.get("SERVED_MODEL_NAME", "llama")
    host = os.environ.get("HOST", "0.0.0.0")
    base_url = f"http://{host}:8000/v1"
    urls = URLRouter.build_urls(base_url, ports)
    urls = filter_healthy_urls(urls)
    if not urls:
        raise RuntimeError("Llama 无可用端点")
    return {"urls": urls, "model": model, "label": "llama"}


async def run(sample_concurrency: int = 128, diff_concurrency: int = 96, resume: bool = False):
    config = Config.from_core_file()
    config.ensure_dirs()
    config.setup_logging()

    bb = build_llama_config()
    n_urls = len(bb["urls"])
    scale = n_urls / 4 * 2
    sc = max(32, int(sample_concurrency * scale))
    dc = max(16, int(diff_concurrency * scale))

    action = "续跑" if resume else "启动"
    logger.info("%s Stage 1 [llama]: replicas=%d sample_conc=%d diff_conc=%d",
                 action, n_urls, sc, dc)

    pipeline = AgentPipeline(config=config, ports=None, memory_dir="memory/llama")
    result_dir = await pipeline.stage1_learn(
        concurrency=sc,
        diff_concurrency=dc,
        backbone="llama",
        urls_override=bb["urls"],
        model_override=bb["model"],
        resume=resume,
    )
    logger.info("Llama Stage 1 完成: %s", result_dir)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample-concurrency", type=int, default=128)
    p.add_argument("--diff-concurrency", type=int, default=96)
    p.add_argument("--resume", action="store_true",
                   help="断点续跑: 从 checkpoint 恢复已完成的采样, 只跑剩余题目")
    args = p.parse_args()
    asyncio.run(run(args.sample_concurrency, args.diff_concurrency, resume=args.resume))


if __name__ == "__main__":
    main()
