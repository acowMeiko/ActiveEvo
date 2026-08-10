#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import save_json


def extract_gsm8k_answer(answer: str) -> str:
    text = str(answer or "")
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def prepare_gsm8k(src: Path) -> List[Dict]:
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "question": item["question"],
                "answer": extract_gsm8k_answer(item["answer"]),
                "difficulty_level": "GSM8K",
                "source_dataset": "gsm8k",
            })
    return rows


def prepare_math500(src: Path) -> List[Dict]:
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "question": item["problem"],
                "answer": item["answer"],
                "difficulty_level": f"MATH500-L{item.get('level', 'NA')}",
                "subject": item.get("subject", ""),
                "unique_id": item.get("unique_id", ""),
                "source_dataset": "math500",
            })
    return rows


def prepare_bbh(src_paths: List[Path]) -> List[Dict]:
    rows = []
    for src in src_paths:
        task_name = src.stem
        data = json.load(src.open("r", encoding="utf-8"))
        for ex in data.get("examples", []):
            rows.append({
                "question": ex["input"],
                "answer": ex["target"],
                "difficulty_level": f"BBH-{task_name}",
                "task_name": task_name,
                "source_dataset": "bbh",
            })
    return rows


def limit_rows(rows: List[Dict], limit: int) -> List[Dict]:
    if limit <= 0:
        return rows
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified Stage 3 benchmark inputs")
    parser.add_argument("--gsm8k-src", type=str, required=True)
    parser.add_argument("--math500-src", type=str, default=str(PROJECT_ROOT / "data/test/math_500_test.jsonl"))
    parser.add_argument("--bbh-dir", type=str, required=True)
    parser.add_argument("--bbh-tasks", nargs="*", default=None, help="Specific BBH task filenames without .json")
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "data/test/benchmarks"))
    parser.add_argument("--gsm8k-limit", type=int, default=0)
    parser.add_argument("--math500-limit", type=int, default=0)
    parser.add_argument("--bbh-limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_rows = limit_rows(prepare_gsm8k(Path(args.gsm8k_src)), args.gsm8k_limit)
    math500_rows = limit_rows(prepare_math500(Path(args.math500_src)), args.math500_limit)

    bbh_dir = Path(args.bbh_dir)
    if args.bbh_tasks:
        bbh_files = [bbh_dir / f"{name}.json" for name in args.bbh_tasks]
    else:
        bbh_files = sorted(p for p in bbh_dir.glob("*.json") if p.name != "dataset_infos.json")
    bbh_rows = limit_rows(prepare_bbh(bbh_files), args.bbh_limit)

    save_json(gsm8k_rows, str(out_dir / "gsm8k_stage3.json"), indent=2)
    save_json(math500_rows, str(out_dir / "math500_stage3.json"), indent=2)
    save_json(bbh_rows, str(out_dir / "bbh_stage3.json"), indent=2)

    print(json.dumps({
        "gsm8k": len(gsm8k_rows),
        "math500": len(math500_rows),
        "bbh": len(bbh_rows),
        "out_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
