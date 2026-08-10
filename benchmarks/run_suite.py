"""Benchmark 批量运行器 — 对所有 benchmark 和模型运行评估。

用法:
  python benchmarks/run_suite.py \
    --models llama qwen \
    --mode guided \
    --concurrency 64 \
    --output output/benchmarks

模型端口分配 (来自 benchmarks/config.py):
  llama: ports 8000-8003 (GPUs 0-3)
  qwen:  ports 8004-8007 (GPUs 4-7)
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

from benchmarks.config import MODELS, BENCHMARKS
from benchmarks.evaluator import evaluate_single
from core.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("benchmarks.run_suite")


def _find_memory_dir(model_key: str) -> Path:
    backbone = MODELS[model_key]["backbone"]
    mem_dir = Path(f"memory/{backbone}")
    if mem_dir.exists():
        return mem_dir
    return Path("memory")


async def run_all(
    models: List[str],
    mode: str = "guided",
    threshold: float = 0.55,
    top_k: int = 5,
    concurrency: int = 64,
    retry_on_wrong: int = 2,
    timeout: int = 600,
    output_root: Path = Path("output/benchmarks"),
    skip: List[str] | None = None,
    only: List[str] | None = None,
    retrieval_method: str = "dsd",  # "dsd" | "single"
    resume: bool = False,
) -> Dict[str, Any]:
    """运行一个 model 的所有 benchmark。"""
    config = Config.from_core_file()
    skip_set = set(skip or [])
    only_set = set(only or [])

    all_summaries: Dict[str, Dict[str, Any]] = {}

    for model_key in models:
        model_cfg = MODELS[model_key]
        ports = model_cfg["ports"]
        model_name = model_cfg["name"]
        backbone = model_cfg["backbone"]
        memory_dir = _find_memory_dir(model_key)

        logger.info("=" * 60)
        logger.info("模型: %s (backbone=%s, ports=%s, memory=%s)", model_name, backbone, ports, memory_dir)
        logger.info("=" * 60)

        model_summaries: Dict[str, Any] = {}

        for bm in BENCHMARKS:
            bm_key = bm.get("name", "").lower().replace(" ", "_")
            if skip_set and (bm["family"] in skip_set or bm_key in skip_set):
                logger.info("跳过: %s", bm["name"])
                continue
            if only_set and bm["family"] not in only_set and bm_key not in only_set:
                continue
            bm_name = bm["name"]
            test_file = str(ROOT / bm["test_file"])
            output_dir = output_root / backbone / bm["family"]
            if bm.get("subset") and bm["subset"] != "full":
                output_dir = output_dir / bm["subset"]

            if not Path(test_file).exists():
                logger.warning("跳过 %s / %s: 测试文件不存在 %s", model_name, bm_name, test_file)
                continue

            logger.info("--- %s / %s ---", model_name, bm_name)
            result = await evaluate_single(
                config=config,
                ports=ports,
                model_name=model_name,
                test_file=test_file,
                mode=mode,
                threshold=threshold,
                top_k=top_k,
                concurrency=concurrency,
                retry_on_wrong=retry_on_wrong,
                output_dir=output_dir,
                memory_dir=str(memory_dir),
                timeout=timeout,
                retrieval_method=retrieval_method,
                resume=resume,
            )

            summary = result.get("summary", {})
            # BBH split: family is "bbh" for both, use family_subset to avoid overwrite
            key = f"{bm['family']}_{bm['subset']}" if bm.get("subset") and bm["subset"] != "full" else bm["family"]
            model_summaries[key] = summary
            perf = summary.get("performance", {})
            logger.info("  %s: %s/%s (%.4f)", bm_name, perf.get("correct", "?"), perf.get("scope_total", "?"), perf.get("accuracy", 0))

        all_summaries[model_key] = model_summaries

    # 生成合并 summary
    combined = _build_combined_summary(all_summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    combined_path = output_root / "combined_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    logger.info("合并 summary 已保存: %s", combined_path)

    # 打印合并表格
    _print_combined_table(combined)

    return combined


def _build_combined_summary(all_summaries: Dict[str, Any]) -> Dict[str, Any]:
    """构建合并 summary，包含 BBH 合并准确率。"""
    result: Dict[str, Any] = {"models": {}, "bbh_combined": {}}

    for model_key, summaries in all_summaries.items():
        model_entry: Dict[str, Any] = {}
        for bm_family, s in summaries.items():
            perf = s.get("performance", {})
            model_entry[bm_family] = {
                "accuracy": perf.get("accuracy"),
                "correct": perf.get("correct"),
                "scope_total": perf.get("scope_total"),
                "retrieval_used_ratio": s.get("retrieval", {}).get("retrieval_used_ratio"),
            }
        result["models"][model_key] = model_entry

        # BBH combined: merge Logic + Math (keys: bbh_logic, bbh_math)
        bbh_data = _merge_bbh(summaries)
        if bbh_data:
            result["bbh_combined"][model_key] = bbh_data

    return result


def _merge_bbh(summaries: Dict[str, Any]) -> Dict[str, Any] | None:
    """将 BBH Logic 和 BBH Math 合并为 BBH 整体指标。"""
    logic_sum = summaries.get("bbh_logic", {})
    math_sum = summaries.get("bbh_math", {})

    # BBH split format: two entries with family="bbh", subset="logic"/"math"
    logic_perf = logic_sum.get("performance", {}) if isinstance(logic_sum, dict) else {}
    math_perf = math_sum.get("performance", {}) if isinstance(math_sum, dict) else {}

    logic_correct = logic_perf.get("correct", 0)
    logic_total = logic_perf.get("scope_total", 0)
    math_correct = math_perf.get("correct", 0)
    math_total = math_perf.get("scope_total", 0)

    if logic_total + math_total == 0:
        return None

    combined_correct = logic_correct + math_correct
    combined_total = logic_total + math_total
    return {
        "accuracy": combined_correct / combined_total if combined_total else 0,
        "correct": combined_correct,
        "scope_total": combined_total,
        "logic": {"correct": logic_correct, "total": logic_total, "accuracy": logic_correct / logic_total if logic_total else 0},
        "math": {"correct": math_correct, "total": math_total, "accuracy": math_correct / math_total if math_total else 0},
    }


def _print_combined_table(combined: Dict[str, Any]) -> None:
    """打印合并结果表格。"""
    models_list = list(combined.get("models", {}).keys())
    if not models_list:
        return

    benchmark_families = list(combined["models"][models_list[0]].keys())
    # Move BBH variants to end, consolidate
    display_families = [f for f in benchmark_families if not f.startswith("bbh_")]
    bbh_variants = [f for f in benchmark_families if f.startswith("bbh_")]

    print("\n" + "=" * 90)
    print("Benchmark Results Summary")
    print("=" * 90)
    header = f"{'Benchmark':<20}" + "".join(f"{m:>20}" for m in models_list)
    print(header)
    print("-" * 90)

    for fam in display_families:
        row = f"{fam:<20}"
        for m in models_list:
            entry = combined["models"].get(m, {}).get(fam, {})
            acc = entry.get("accuracy")
            if acc is not None:
                correct = entry.get("correct", "?")
                total = entry.get("scope_total", "?")
                row += f"{acc:.4f} ({correct}/{total})".rjust(25)
            else:
                row += "N/A".rjust(25)
        print(row)

    for fam in bbh_variants:
        row = f"{fam:<20}"
        for m in models_list:
            entry = combined["models"].get(m, {}).get(fam, {})
            acc = entry.get("accuracy")
            if acc is not None:
                correct = entry.get("correct", "?")
                total = entry.get("scope_total", "?")
                row += f"{acc:.4f} ({correct}/{total})".rjust(25)
            else:
                row += "N/A".rjust(25)
        print(row)

    # BBH Combined row
    bbh_combined = combined.get("bbh_combined", {})
    if bbh_combined:
        row = f"{'BBH Combined':<20}"
        for m in models_list:
            entry = bbh_combined.get(m, {})
            acc = entry.get("accuracy")
            if acc is not None:
                correct = entry.get("correct", "?")
                total = entry.get("scope_total", "?")
                row += f"{acc:.4f} ({correct}/{total})".rjust(25)
            else:
                row += "N/A".rjust(25)
        print(row)

    print("=" * 90)


async def main():
    p = argparse.ArgumentParser(description="Benchmark 批量评估")
    p.add_argument("--models", nargs="+", default=["llama", "qwen"], help="要评估的模型列表")
    p.add_argument("--mode", default="guided", choices=["guided", "baseline", "direct", "base"])
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--retry-on-wrong", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--skip", nargs="*", default=[], help="跳过指定 benchmark (如 deepscaler)")
    p.add_argument("--only", nargs="*", default=[], help="只跑指定 benchmark")
    p.add_argument("--retrieval-method", default="dsd", choices=["dsd", "single"], help="检索方法")
    p.add_argument("--resume", action="store_true", help="增量评测: 只重测上次做错的题")
    p.add_argument("--output", default="output/benchmarks")
    args = p.parse_args()

    valid_models = [m for m in args.models if m in MODELS]
    if not valid_models:
        logger.error("无效模型: %s. 可用: %s", args.models, list(MODELS.keys()))
        sys.exit(1)

    await run_all(
        models=valid_models,
        mode=args.mode,
        threshold=args.threshold,
        top_k=args.top_k,
        concurrency=args.concurrency,
        retry_on_wrong=args.retry_on_wrong,
        timeout=args.timeout,
        output_root=Path(args.output),
        skip=args.skip,
        only=args.only,
        retrieval_method=args.retrieval_method,
        resume=args.resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
