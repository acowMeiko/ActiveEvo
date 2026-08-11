#!/usr/bin/env python
"""
scripts/rediff_failed.py — 对 diff_success=false 的 packet 重新跑 diff 分析

用法:
  # 单个文件
  python scripts/rediff_failed.py \
    --input output/stage1/02_learning_packets/qwen/level1_learning_packets.json \
    --ports 8000 8001 8002 8003 --model qwen

  # 批量: qwen 四个 level
  python scripts/rediff_failed.py \
    --backbone qwen --levels 1 2 3 4 \
    --ports 8000 8001 8002 8003 --model qwen

  # 批量: llama 四个 level (用 qwen 跑 diff, 通过 --diff-model 指定)
  python scripts/rediff_failed.py \
    --backbone llama --levels 1 2 3 4 \
    --ports 8000 8001 8002 8003 --model qwen
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from tqdm.asyncio import tqdm as async_tqdm

from modules.execute_module import URLRouter, filter_healthy_urls
from modules.stage1_miner import Stage1Miner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rediff")


async def rediff_one(
    input_path: Path,
    urls: List[str],
    model: str,
    api_key: str = "EMPTY",
    timeout: int = 600,
    concurrency: int = 8,
) -> Dict[str, int]:
    """重跑单个文件的失败 diff, 返回 {total, failed, fixed}."""
    packets = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(packets, list):
        logger.error("不是 JSON 数组: %s", input_path)
        return {"total": 0, "failed": 0, "fixed": 0}

    failed = [p for p in packets if not p.get("diff_success")]
    if not failed:
        logger.info("[%s] 全部 %d 条 diff 已成功, 跳过", input_path.parent.name, len(packets))
        return {"total": len(packets), "failed": 0, "fixed": 0}

    logger.info("[%s] %d/%d 条 diff 失败, 开始重跑", input_path.stem, len(failed), len(packets))

    miner = Stage1Miner(urls=urls, model=model, api_key=api_key, timeout=timeout)
    semaphore = asyncio.Semaphore(concurrency)

    async def redo_one(packet: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            diff_result = await miner._diff_analysis(
                question=packet["question"],
                chosen_path=packet["chosen_path"],
                rejected_path=packet["rejected_path"],
                divergence_step_idx=packet.get("divergence_step_idx", 0),
                divergence_step_text=packet.get("divergence_step_text", ""),
            )
        packet["diff"] = diff_result
        packet["diff_success"] = diff_result.get("status") == "success"
        return packet

    tasks = [redo_one(p) for p in failed]
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc=f"Re-diff {input_path.stem}"):
        await coro

    # 合并修复结果
    by_idx = {p["idx"]: p for p in failed}
    fixed = 0
    for i, p in enumerate(packets):
        if not p.get("diff_success") and p["idx"] in by_idx:
            updated = by_idx[p["idx"]]
            packets[i] = updated
            if updated.get("diff_success"):
                fixed += 1

    # 备份 + 覆盖
    backup = input_path.with_suffix(".json.bak")
    input_path.rename(backup)
    input_path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("[%s] 完成: %d/%d 修复成功, 备份 → %s", input_path.stem, fixed, len(failed), backup)
    return {"total": len(packets), "failed": len(failed), "fixed": fixed}


async def rediff_batch(
    backbone: str,
    levels: List[int],
    urls: List[str],
    model: str,
    output_root: Optional[Path] = None,
    api_key: str = "EMPTY",
    timeout: int = 600,
    concurrency: int = 8,
) -> None:
    """批量重跑 backbone 下所有 level 的失败 diff."""
    root = output_root or ROOT_DIR / "output"
    base_dir = root / "stage1" / "02_learning_packets" / backbone

    if not base_dir.exists():
        logger.error("目录不存在: %s", base_dir)
        return

    total_stats = {"total": 0, "failed": 0, "fixed": 0}
    for lv in levels:
        path = base_dir / f"level{lv}_learning_packets.json"
        if not path.exists():
            logger.warning("文件不存在, 跳过: %s", path)
            continue
        stats = await rediff_one(path, urls, model, api_key, timeout, concurrency)
        for k in total_stats:
            total_stats[k] += stats[k]

    logger.info(
        "批量完成 [%s]: total=%d, failed=%d, fixed=%d",
        backbone, total_stats["total"], total_stats["failed"], total_stats["fixed"],
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description="重跑失败的 diff 分析")
    p.add_argument("--input", type=Path, default=None, help="单个 learning packet JSON 路径")
    p.add_argument("--backbone", type=str, default=None, help="批量模式: qwen / llama")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="批量模式要处理的 level (默认 1 2 3 4)")
    p.add_argument("--output-root", type=Path, default=None, help="output 目录")
    p.add_argument("--ports", type=int, nargs="+", default=[8000, 8001, 8002, 8003])
    p.add_argument("--host", default="localhost")
    p.add_argument("--model", default="qwen")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    urls = [f"http://{args.host}:{port}/v1/chat/completions" for port in args.ports]
    urls = filter_healthy_urls(urls)
    if not urls:
        logger.error("无可用端点")
        sys.exit(1)

    if args.input:
        if not args.input.exists():
            logger.error("文件不存在: %s", args.input)
            sys.exit(1)
        asyncio.run(rediff_one(
            input_path=args.input,
            urls=urls, model=args.model,
            api_key=args.api_key, timeout=args.timeout,
            concurrency=args.concurrency,
        ))
    elif args.backbone:
        asyncio.run(rediff_batch(
            backbone=args.backbone,
            levels=args.levels,
            urls=urls, model=args.model,
            output_root=args.output_root,
            api_key=args.api_key, timeout=args.timeout,
            concurrency=args.concurrency,
        ))
    else:
        logger.error("需要 --input 或 --backbone")
        sys.exit(1)


if __name__ == "__main__":
    main()
