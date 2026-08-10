"""Build a reproducible, stratified DeepScaler evaluation subset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def normalize_level(value: object) -> int | None:
    text = str(value or "")
    for level in (1, 2, 3, 4):
        if str(level) in text:
            return level
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-level", type=int, default=50)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("DeepScaler input must be a JSON list")

    rng = random.Random(args.seed)
    selected = []
    for level in args.levels:
        pool = [
            (index, item) for index, item in enumerate(source)
            if isinstance(item, dict)
            and normalize_level(item.get("difficulty_level", item.get("level"))) == level
        ]
        if len(pool) < args.per_level:
            raise ValueError(f"Level {level} has only {len(pool)} items")
        for source_index, item in rng.sample(pool, args.per_level):
            copied = dict(item)
            copied["_sampling"] = {
                "source_index": source_index,
                "stratified_level": level,
                "seed": args.seed,
            }
            selected.append(copied)

    # Shuffle the final sequence so endpoint load is not confounded with level.
    rng.shuffle(selected)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(target),
        "seed": args.seed,
        "total": len(selected),
        "per_level": args.per_level,
        "levels": args.levels,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
