#!/usr/bin/env python3
"""Fix existing eval_summary.json files: replace per-round stats with cumulative stats.

Usage: python scripts/fix_eval_cumulative.py --base-dir output/stage3_eval --rounds 4 --total 400
"""
import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_question_key(item: dict) -> str:
    return item.get("question", "") or item.get("problem", "")


def main():
    p = argparse.ArgumentParser(description="Fix eval_summary.json to use cumulative stats")
    p.add_argument("--base-dir", default="output/stage3_eval", help="Base output directory")
    p.add_argument("--prefix", default="llama_round", help="Round subdir prefix")
    p.add_argument("--rounds", type=int, default=4, help="Number of rounds")
    p.add_argument("--total", type=int, default=400, help="Total original questions")
    args = p.parse_args()

    base = Path(args.base_dir)
    cumulative_correct_texts = set()

    print("| Round | Cumulative Correct | Accuracy |")
    print("|-------|-------------------|----------|")

    for r in range(1, args.rounds + 1):
        rd = base / f"{args.prefix}{r}"
        results_file = rd / "eval_results.json"
        summary_file = rd / "eval_summary.json"

        if not results_file.exists():
            print(f"| Round {r} | (missing eval_results.json) |", file=sys.stderr)
            continue
        if not summary_file.exists():
            print(f"| Round {r} | (missing eval_summary.json) |", file=sys.stderr)
            continue

        results = load_json(results_file)
        # Collect correct questions from this round
        round_correct_texts = set()
        for entry in results:
            if entry.get("is_correct") or entry.get("status") == "carried_forward":
                round_correct_texts.add(get_question_key(entry))
        cumulative_correct_texts |= round_correct_texts

        cum_correct = len(cumulative_correct_texts)
        cum_accuracy = cum_correct / args.total

        # Update summary
        summary = load_json(summary_file)
        perf = summary.setdefault("performance", {})
        perf["cumulative_correct"] = cum_correct
        perf["cumulative_accuracy"] = cum_accuracy
        perf["per_round_correct"] = perf.get("correct", 0)
        perf["per_round_accuracy"] = perf.get("accuracy", 0.0)
        perf["correct"] = cum_correct
        perf["accuracy"] = cum_accuracy
        perf["total"] = args.total
        save_json(summary, summary_file)

        print(f"| Round {r} | {cum_correct} | {cum_accuracy:.4f} |")

    print(f"\nUpdated {args.rounds} eval_summary.json files under {base}")


if __name__ == "__main__":
    main()
