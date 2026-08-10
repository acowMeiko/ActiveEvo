#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import sys

from datasets import load_dataset
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.official_dataset_utils import (
    preprocess_general365_rows,
    preprocess_svamp_test_json,
    preprocess_svamp_train_csv,
)
DATA_DIR = ROOT / "data"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_svamp() -> None:
    train_root = DATA_DIR / "input" / "official" / "svamp"
    test_root = DATA_DIR / "test" / "official" / "svamp"
    for path in [
        train_root / "raw",
        train_root / "processed",
        train_root / "metadata",
        test_root / "raw",
        test_root / "processed",
        test_root / "metadata",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        import subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/arkilpatel/SVAMP", str(tmp / "SVAMP")],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        repo = tmp / "SVAMP"

        # raw files
        shutil.copy2(repo / "data" / "mawps-asdiv-a_svamp" / "train.csv", train_root / "raw" / "train.csv")
        shutil.copy2(repo / "data" / "mawps-asdiv-a_svamp" / "dev.csv", train_root / "raw" / "dev.csv")
        shutil.copy2(repo / "SVAMP.json", test_root / "raw" / "SVAMP.json")
        shutil.copy2(repo / "README.md", train_root / "metadata" / "README_upstream.md")
        shutil.copy2(repo / "LICENSE", train_root / "metadata" / "LICENSE")

        # processed
        train_rows = preprocess_svamp_train_csv(repo / "data" / "mawps-asdiv-a_svamp" / "train.csv")
        dev_rows = preprocess_svamp_train_csv(repo / "data" / "mawps-asdiv-a_svamp" / "dev.csv")
        test_rows = preprocess_svamp_test_json(repo / "SVAMP.json")

    _write_json(train_root / "processed" / "train.json", train_rows)
    _write_json(train_root / "processed" / "dev.json", dev_rows)
    _write_json(test_root / "processed" / "test.json", test_rows)
    _write_json(
        train_root / "metadata" / "dataset_info.json",
        {
            "dataset": "svamp",
            "official_source": "https://github.com/arkilpatel/SVAMP",
            "train_files": ["train.csv", "dev.csv"],
            "test_files": ["SVAMP.json"],
            "preprocessing": {
                "train": "Replace number0/number1 placeholders using Numbers, then concatenate Body + Ques into natural-language question.",
                "test": "Concatenate Body + Question and normalize numeric answer to plain string.",
            },
            "accuracy_extraction": {
                "type": "numeric",
                "rule": "Extract boxed/final/last numeric token and compare to gold with abs_tol=1e-6.",
            },
        },
    )
    _write_text(
        train_root / "metadata" / "PREPROCESSING_AND_EVAL.md",
        "# SVAMP\n\n"
        "Train preprocessing:\n"
        "- Official train/dev CSV uses `number0`, `number1`, ... placeholders.\n"
        "- We replace placeholders with tokens from the `Numbers` field.\n"
        "- Final processed `question` is `Body + Ques` after placeholder substitution.\n\n"
        "Test preprocessing:\n"
        "- Official challenge set is `SVAMP.json`.\n"
        "- Final processed `question` is `Body + Question`.\n"
        "- Gold answer is normalized to a plain numeric string.\n\n"
        "Accuracy extraction:\n"
        "- Dataset is numeric-only.\n"
        "- Extract answer from `\\boxed{}`, `Final Answer`, or the last numeric token.\n"
        "- Compare against gold with numeric tolerance `1e-6`.\n",
    )


def prepare_general365() -> None:
    train_root = DATA_DIR / "input" / "official" / "general365"
    test_root = DATA_DIR / "test" / "official" / "general365"
    for path in [
        train_root / "processed",
        train_root / "metadata",
        test_root / "raw",
        test_root / "processed",
        test_root / "metadata",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = snapshot_download(
            repo_id="meituan-longcat/General365_Public",
            repo_type="dataset",
            local_dir=tmpdir,
            allow_patterns=["*.parquet", "README.md", "*.json", ".huggingface.yaml"],
        )
        snapshot_dir = Path(snapshot_path)
        for path in snapshot_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(snapshot_dir)
                target = test_root / "raw" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    ds = load_dataset("meituan-longcat/General365_Public", split="test")
    processed_rows = preprocess_general365_rows(ds)
    _write_json(test_root / "processed" / "test.json", processed_rows)
    _write_json(train_root / "processed" / "train.json", [])
    _write_json(
        train_root / "metadata" / "dataset_info.json",
        {
            "dataset": "general365",
            "official_source": "https://huggingface.co/datasets/meituan-longcat/General365_Public",
            "available_splits": ["test"],
            "public_train_split_available": False,
            "test_size": len(processed_rows),
            "answer_types": sorted({row["answer_type"] for row in processed_rows}),
            "preprocessing": {
                "test": "Keep official question/answer/answer_type/float_round fields and normalize text spacing only.",
                "train": "No official public training split released by the dataset authors.",
            },
            "accuracy_extraction": {
                "number": "Prefer boxed/final answer; exact normalized LaTeX match first, then numeric compare with float_round if applicable.",
                "single_choice": "Extract final option label and compare exact letter.",
                "multiple_choice": "Extract option-label set and compare exact set.",
                "text": "Lowercased normalized exact match.",
            },
        },
    )
    _write_text(
        train_root / "metadata" / "NO_OFFICIAL_TRAIN_SPLIT.md",
        "# General365\n\n"
        "The official public release `meituan-longcat/General365_Public` currently exposes only a `test` split.\n"
        "No official training split is published in the public dataset card or dataset files.\n",
    )
    _write_text(
        test_root / "metadata" / "PREPROCESSING_AND_EVAL.md",
        "# General365\n\n"
        "Official public split:\n"
        "- `test` only, 720 examples.\n\n"
        "Preprocessing:\n"
        "- Keep official fields `id`, `question`, `answer`, `answer_type`, `float_round`.\n"
        "- Normalize whitespace only.\n\n"
        "Accuracy extraction by answer_type:\n"
        "- `number`: extract from `\\boxed{}` or final-answer line; prefer exact normalized match; if both sides parse as numbers, compare numerically, using `float_round` when provided.\n"
        "- `single_choice`: extract one final option label such as `A`, `B`, `C`, `D`.\n"
        "- `multiple_choice`: extract label set and compare exact set.\n"
        "- `text`: lowercase normalized exact match.\n",
    )


def main() -> None:
    prepare_svamp()
    prepare_general365()
    print("Prepared official datasets: SVAMP and General365.")


if __name__ == "__main__":
    main()
