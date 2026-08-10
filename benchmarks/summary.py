"""Benchmark 结果汇总 — 读取 eval_summary.json 生成合并报告。

用法:
  python benchmarks/summary.py --results-dir output/benchmarks
  python benchmarks/summary.py --results-dir output/benchmarks --format markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.config import BENCHMARKS

BBH_SPLIT = {"bbh_logic", "bbh_math"}


def _find_summary_files(results_dir: Path) -> Dict[str, Dict[str, Path]]:
    """扫描 results_dir 找到所有 model/benchmark/eval_summary.json。

    Returns: {model: {benchmark_family: Path(...)}}
    """
    result: Dict[str, Dict[str, Path]] = {}
    if not results_dir.exists():
        return result

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        result[model_name] = {}

        for bm_dir in sorted(model_dir.iterdir()):
            if not bm_dir.is_dir():
                continue
            summary_path = bm_dir / "eval_summary.json"
            if summary_path.exists():
                # infer family from directory structure
                family = bm_dir.name
                result[model_name][family] = summary_path

            # Check for subset dirs (e.g., bbh/logic/)
            for sub_dir in sorted(bm_dir.iterdir()):
                if sub_dir.is_dir():
                    sub_summary = sub_dir / "eval_summary.json"
                    if sub_summary.exists():
                        family = f"{bm_dir.name}_{sub_dir.name}"
                        result[model_name][family] = sub_summary

    return result


def _load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report(results_dir: Path) -> Dict[str, Any]:
    summaries = _find_summary_files(results_dir)

    if not summaries:
        print(f"[WARN] 未在 {results_dir} 找到任何 eval_summary.json")
        return {"models": {}, "bbh_combined": {}}

    report: Dict[str, Any] = {"models": {}, "bbh_combined": {}}

    for model_name, benchmarks in summaries.items():
        model_entry: Dict[str, Any] = {}
        bbh_parts: Dict[str, Dict[str, Any]] = {}

        for family, path in sorted(benchmarks.items()):
            s = _load_summary(path)
            perf = s.get("performance", {})
            entry = {
                "name": _family_display_name(family),
                "accuracy": perf.get("accuracy"),
                "correct": perf.get("correct"),
                "scope_total": perf.get("scope_total"),
                "mode": s.get("mode"),
                "retrieval_used_ratio": s.get("retrieval", {}).get("retrieval_used_ratio"),
            }
            model_entry[family] = entry

            if family in BBH_SPLIT:
                bbh_parts[family] = entry

        # Merge BBH
        if len(bbh_parts) >= 2:
            total_correct = sum(v.get("correct", 0) or 0 for v in bbh_parts.values())
            total_scope = sum(v.get("scope_total", 0) or 0 for v in bbh_parts.values())
            if total_scope > 0:
                report["bbh_combined"][model_name] = {
                    "accuracy": total_correct / total_scope,
                    "correct": total_correct,
                    "scope_total": total_scope,
                    "parts": {
                        k: {"correct": v.get("correct"), "total": v.get("scope_total")}
                        for k, v in sorted(bbh_parts.items())
                    },
                }

        report["models"][model_name] = model_entry

    return report


def _family_display_name(family: str) -> str:
    mapping = {
        "deepscaler": "DeepScaler",
        "math500": "MATH500",
        "gsm8k": "GSM8K",
        "bbh_logic": "BBH Logic",
        "bbh_math": "BBH Math",
        "svamp": "SVAMP",
        "general365": "General365",
    }
    return mapping.get(family, family)


def print_table(report: Dict[str, Any], fmt: str = "terminal") -> str | None:
    models = sorted(report.get("models", {}).keys())
    if not models:
        print("[WARN] 无模型数据")
        return None

    # Collect all benchmark families
    all_families: List[str] = []
    seen = set()
    for m in models:
        for fam in report["models"][m]:
            if fam not in seen:
                all_families.append(fam)
                seen.add(fam)

    # Move BBH parts to end
    display_families = [f for f in all_families if f not in BBH_SPLIT]
    bbh_families = [f for f in all_families if f in BBH_SPLIT]

    lines = []
    sep = "|" if fmt == "markdown" else " "

    if fmt == "markdown":
        header = "| Benchmark | " + " | ".join(models) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    else:
        header = f"{'Benchmark':<20}" + "".join(f"{m:>25}" for m in models)
        lines.append(header)
        lines.append("-" * (20 + 25 * len(models)))

    def _fmt_cell(entry: Dict[str, Any]) -> str:
        acc = entry.get("accuracy")
        if acc is None:
            return "N/A"
        correct = entry.get("correct", "?")
        total = entry.get("scope_total", "?")
        s = f"{acc:.4f} ({correct}/{total})"
        if fmt == "markdown":
            return s
        return s.rjust(25)

    for fam in display_families + bbh_families:
        name = _family_display_name(fam)
        cells = []
        for m in models:
            entry = report["models"][m].get(fam, {})
            cells.append(_fmt_cell(entry))
        if fmt == "markdown":
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        else:
            lines.append(f"{name:<20}" + "".join(cells))

    # BBH Combined row
    bbh_combined = report.get("bbh_combined", {})
    if bbh_combined and len(bbh_combined) > 1:
        if fmt == "markdown":
            combined_cells = []
            for m in models:
                entry = bbh_combined.get(m, {})
                acc = entry.get("accuracy")
                if acc is not None:
                    combined_cells.append(f"{acc:.4f} ({entry.get('correct','?')}/{entry.get('scope_total','?')})")
                else:
                    combined_cells.append("N/A")
            lines.append(f"| **BBH Combined** | " + " | ".join(combined_cells) + " |")
        else:
            combined_cells = []
            for m in models:
                entry = bbh_combined.get(m, {})
                acc = entry.get("accuracy")
                if acc is not None:
                    combined_cells.append(_fmt_cell({"accuracy": acc, "correct": entry.get("correct"), "scope_total": entry.get("scope_total")}))
                else:
                    combined_cells.append("N/A".rjust(25))
            lines.append(f"{'BBH Combined':<20}" + "".join(combined_cells))

    output = "\n".join(lines)
    print(output)
    return output


def main():
    p = argparse.ArgumentParser(description="Benchmark 结果汇总")
    p.add_argument("--results-dir", default="output/benchmarks", help="benchmark 结果根目录")
    p.add_argument("--format", default="terminal", choices=["terminal", "markdown", "json"], help="输出格式")
    p.add_argument("--output", help="保存到文件")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    report = build_report(results_dir)

    if args.format == "json":
        output = json.dumps(report, indent=2, ensure_ascii=False)
        print(output)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"[OK] 已保存: {args.output}")
        return

    table = print_table(report, fmt=args.format)

    if args.output and table:
        Path(args.output).write_text(table, encoding="utf-8")
        print(f"[OK] 已保存: {args.output}")


if __name__ == "__main__":
    main()
