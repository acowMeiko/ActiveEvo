#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.official_dataset_utils import evaluate_general365_prediction, evaluate_svamp_prediction


def _load_rows(path: Path) -> List[Dict]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["svamp", "general365"], required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    gold_rows = _load_rows(Path(args.gold))
    pred_rows = _load_rows(Path(args.predictions))

    pred_map = {}
    for row in pred_rows:
        key = str(row.get("id", "") or row.get("question", "")).strip()
        pred_map[key] = str(row.get("prediction", "") or row.get("model_answer", "") or row.get("answer", "")).strip()

    total = len(gold_rows)
    correct = 0
    details = []
    for gold in gold_rows:
        key = str(gold.get("id", "") or gold.get("question", "")).strip()
        prediction = pred_map.get(key, "")
        if args.dataset == "svamp":
            ok = evaluate_svamp_prediction(prediction, gold.get("answer", ""))
        else:
            ok = evaluate_general365_prediction(prediction, gold)
        correct += int(ok)
        details.append({"id": key, "is_correct": ok, "gold": gold.get("answer", ""), "prediction": prediction})

    summary = {
        "dataset": args.dataset,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
