#!/usr/bin/env python3
"""Rebuild Stage 1 diagnostic figures without model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_bundles(stage1_dir: Path) -> list[dict]:
    bundles: list[dict] = []
    for path in sorted((stage1_dir / "02_learning_packets").glob("*/level*_learning_packets.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            bundles.extend(item for item in payload if isinstance(item, dict))
    return bundles


def _values(bundle: dict) -> tuple[float, float | None, float | None, float | None]:
    signal = bundle.get("capability_signal", {})
    entropy = signal.get("entropy", {})
    return (
        float(signal.get("success_rate", 0.0)),
        entropy.get("h_token"), entropy.get("h_answer"), entropy.get("h_gap"),
    )


def generate_figures(stage1_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: install requirements.txt") from exc

    bundles = load_bundles(stage1_dir)
    if not bundles:
        raise ValueError(f"No Stage 1 bundles found below {stage1_dir}")
    output = stage1_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    colors = {"level1": "#4477AA", "level2": "#66CCEE", "level3": "#EE6677", "level4": "#AA3377"}
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for level in colors:
        rows = [item for item in bundles if item.get("level") == level]
        points = [_values(item) for item in rows]
        ax.scatter([p[0] for p in points if p[1] is not None], [p[1] for p in points if p[1] is not None],
                   s=14, alpha=.55, label=level, color=colors[level])
    for boundary in (.25, .5, .75):
        ax.axvline(boundary, color="gray", linewidth=.8, linestyle="--")
    ax.set(xlabel="Success rate (10 rollouts)", ylabel="Mean token entropy", title="Accuracy–entropy capability map")
    ax.legend(ncol=4, fontsize=8)
    path = output / "01_accuracy_entropy_map.png"; fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    data, labels = [], []
    for level in colors:
        rows = [item for item in bundles if item.get("level") == level]
        for key, label in (("correct_mean", "correct"), ("wrong_mean", "wrong"), ("h_token", "overall")):
            vals = [item["capability_signal"]["entropy"].get(key) for item in rows]
            vals = [float(value) for value in vals if isinstance(value, (int, float))]
            if vals:
                data.append(vals); labels.append(f"{level[-1]}-{label}")
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set(ylabel="Token entropy", title="Within-level trajectory entropy"); ax.tick_params(axis="x", rotation=45)
    path = output / "02_level_entropy_distributions.png"; fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    points = [_values(item) for item in bundles]
    points = [p for p in points if p[1] is not None and p[2] is not None]
    ax.scatter([p[1] for p in points], [p[2] for p in points], s=14, alpha=.55)
    ax.set(xlabel="Mean token entropy", ylabel="Final-answer entropy", title="Token vs answer uncertainty")
    path = output / "03_token_answer_entropy.png"; fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    manifests = stage1_dir / "selection_manifests"
    policies = ["random", "high_token_entropy", "low_token_entropy"]
    matrix = []
    for left in policies:
        left_ids = set(json.loads((manifests / f"budget_600_{left}.json").read_text())["question_ids"])
        row = []
        for right in policies:
            right_ids = set(json.loads((manifests / f"budget_600_{right}.json").read_text())["question_ids"])
            row.append(len(left_ids & right_ids) / max(1, len(left_ids | right_ids)))
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(5.4, 4.6)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3), ["Random", "High H", "Low H"]); ax.set_yticks(range(3), ["Random", "High H", "Low H"])
    for i, row in enumerate(matrix):
        for j, value in enumerate(row): ax.text(j, i, f"{value:.2f}", ha="center", va="center")
    ax.set_title("Selection overlap (Jaccard, budget=600)"); fig.colorbar(image, ax=ax)
    path = output / "04_selection_overlap.png"; fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, default=Path("output/stage1"))
    args = parser.parse_args()
    for path in generate_figures(args.stage1_dir):
        print(path)


if __name__ == "__main__":
    main()
