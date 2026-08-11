#!/usr/bin/env python
"""
补救脚本: 从已有采样 JSON 按论文 SR 阈值重分 Level，重跑 mining。

只会重跑 Stage 1.2 (mining + diff)，不会重跑昂贵的 Stage 1.1 (K-trial 采样)。

用法:
  # Qwen
  python scripts/remediate_levels.py \
    --sampling output/stage1/01_sampling/qwen_unified_sampling_results.json \
    --backbone qwen --ports 8000 8001 8002 8003 --model qwen

  # Llama
  python scripts/remediate_levels.py \
    --sampling output/stage1/01_sampling/llama_unified_sampling_results.json \
    --backbone llama --ports 8004 8005 8006 8007 --model llama

  # 仅预览分布，不执行 mining
  python scripts/remediate_levels.py --sampling ... --backbone qwen --dry-run
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from modules.execute_module import URLRouter, filter_healthy_urls
from modules.stage1_miner import Stage1Miner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("remediate")


# ── Level 分配 (论文定义) ──

def reassign_levels_by_sr(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按论文 SR 阈值重新分配 Level。

    Level 1: SR >= 0.75  (近掌握)
    Level 2: 0.50 <= SR < 0.75
    Level 3: 0.25 <= SR < 0.50
    Level 4: SR < 0.25   (近恐慌)

    只处理 learning zone (0 < SR < 1) 的样本。
    """
    learning = [r for r in results if r.get("zone") == "learning"]
    level_data: Dict[str, List[Dict]] = {"1": [], "2": [], "3": [], "4": []}
    counts = {"1": 0, "2": 0, "3": 0, "4": 0}

    for r in learning:
        sr = r.get("sr", 0.0)
        if sr >= 0.75:
            lv = "1"
        elif sr >= 0.50:
            lv = "2"
        elif sr >= 0.25:
            lv = "3"
        else:
            lv = "4"
        r["assigned_level"] = int(lv)
        level_data[lv].append(r)
        counts[lv] += 1

    return level_data


def print_distribution(results: List[Dict[str, Any]], tag: str) -> None:
    """打印各 zone 和 level 分布。"""
    master = sum(1 for r in results if r.get("zone") == "master")
    learning = sum(1 for r in results if r.get("zone") == "learning")
    panic = sum(1 for r in results if r.get("zone") == "panic")
    no_data = sum(1 for r in results if r.get("zone") == "no_data")

    print(f"\n{'='*60}")
    print(f"  {tag} — Zone 分布")
    print(f"{'='*60}")
    print(f"  Master:   {master:>5}")
    print(f"  Learning: {learning:>5}")
    print(f"  Panic:    {panic:>5}")
    print(f"  No data:  {no_data:>5}")
    print(f"  Total:    {len(results):>5}")

    learning_results = [r for r in results if r.get("zone") == "learning"]
    if not learning_results:
        return

    level_data = reassign_levels_by_sr(results)
    print(f"\n  Level 分配 (按论文 SR 阈值):")
    print(f"  {'Level':<10} {'样本数':<8} {'SR 范围':<20} {'correct_count 范围'}")
    print(f"  {'-'*55}")
    for lv in ("1", "2", "3", "4"):
        items = level_data[lv]
        if items:
            srs = [r["sr"] for r in items]
            ccs = [r["correct_count"] for r in items]
            print(f"  Level {lv:<5} {len(items):<8} [{min(srs):.2f}, {max(srs):.2f}]{'':<10} [{min(ccs)}, {max(ccs)}]")
        else:
            print(f"  Level {lv:<5} {0:<8} (空)")

    # 旧 vs 新对比
    print(f"\n  [对比] 旧等分 vs 新 SR 阈值:")
    print(f"  {'Level':<10} {'旧样本数':<12} {'新样本数':<12}")
    old_sizes = {"1": 730, "2": 730, "3": 730, "4": learning - 2190}  # 近似
    for lv in ("1", "2", "3", "4"):
        new_size = len(level_data[lv])
        print(f"  Level {lv:<5} {old_sizes[lv]:<12} {new_size:<12}")


def build_urls_from_ports(
    ports: Sequence[int], host: str = "0.0.0.0"
) -> List[str]:
    return [f"http://{host}:{port}/v1/chat/completions" for port in ports]


async def run_remediate(
    sampling_path: Path,
    backbone: str,
    urls: List[str],
    model: str,
    api_key: str,
    timeout: int,
    diff_concurrency: int,
    output_root: Path,
    dry_run: bool = False,
) -> None:
    # ── 加载采样数据 ──
    logger.info("加载采样数据: %s", sampling_path)
    data = json.loads(sampling_path.read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = data.get("results", [])
    if not results:
        logger.error("采样文件无 results")
        return

    # ── 打印分布 ──
    print_distribution(results, f"{backbone} 当前采样")

    # 检查轨迹数据完整性
    learning = [r for r in results if r.get("zone") == "learning"]
    has_traj = sum(
        1 for r in learning
        if r.get("correct_trajectories") and r.get("wrong_trajectories")
    )
    logger.info(
        "Learning zone %d 样本, %d 有完整正负轨迹",
        len(learning), has_traj,
    )

    if dry_run:
        logger.info("[dry-run] 跳过 mining，仅预览分布")
        return

    # ── 按论文 SR 重分 Level ──
    level_data = reassign_levels_by_sr(results)
    for lv in ("1", "2", "3", "4"):
        logger.info("  新 Level %s: %d 样本", lv, len(level_data[lv]))

    # ── 过滤健康 URL ──
    healthy = filter_healthy_urls(urls)
    if not healthy:
        logger.error("无可用 vLLM 端点")
        return
    logger.info("使用 %d 个端点: %s", len(healthy), healthy[:3])

    # ── 重跑 Mining ──
    miner = Stage1Miner(
        urls=healthy,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    logger.info(
        "开始 mining (diff_concurrency=%d)...", diff_concurrency
    )
    mining_result = await miner.run(level_data, concurrency=diff_concurrency)

    # ── 保存 ──
    packets_dir = output_root / "stage1" / "02_learning_packets" / backbone
    packets_dir.mkdir(parents=True, exist_ok=True)

    packets = mining_result.get("packets", {})
    stats = mining_result.get("stats", {})

    Stage1Miner.save(packets, packets_dir)

    # 保存统计
    summary_path = packets_dir / "stage1_summary.json"
    old_summary = data.get("summary", {})
    summary_path.write_text(
        json.dumps(
            {
                "backbone": backbone,
                "model": model,
                "remediated": True,
                "old_level_assignment": "equal_quartile_ascending",
                "new_level_assignment": "sr_thresholds_per_paper",
                "sampling": old_summary,
                "mining": stats,
                "level_samples": {
                    lv: len(level_data[lv])
                    for lv in ("1", "2", "3", "4")
                },
                "level_packets": {
                    lv: len(packets.get(f"level{lv}", []))
                    for lv in ("1", "2", "3", "4")
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("补救完成: %s", packets_dir)
    logger.info("  Stats: %s", json.dumps(stats, ensure_ascii=False))


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="补救: 按论文 SR 阈值重分 Level 并重跑 mining"
    )
    p.add_argument(
        "--sampling", type=Path, required=True,
        help="已有采样 JSON 路径",
    )
    p.add_argument(
        "--backbone", type=str, required=True,
        help="模型标识 (qwen/llama)",
    )
    p.add_argument(
        "--ports", type=int, nargs="+", default=[8000, 8001, 8002, 8003],
        help="vLLM 端口列表",
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--model", default=None, help="served model name")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--diff-concurrency", type=int, default=96)
    p.add_argument(
        "--output-root", type=Path,
        default=ROOT_DIR / "output",
        help="输出根目录",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="仅预览分布，不执行 mining",
    )
    args = p.parse_args()

    if not args.sampling.exists():
        logger.error("采样文件不存在: %s", args.sampling)
        sys.exit(1)

    model = args.model or args.backbone
    urls = build_urls_from_ports(args.ports, args.host)

    logger.info(
        "补救启动: backbone=%s model=%s ports=%s",
        args.backbone, model, args.ports,
    )

    asyncio.run(
        run_remediate(
            sampling_path=args.sampling,
            backbone=args.backbone,
            urls=urls,
            model=model,
            api_key=args.api_key,
            timeout=args.timeout,
            diff_concurrency=args.diff_concurrency,
            output_root=args.output_root,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
