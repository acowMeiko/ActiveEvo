#!/usr/bin/env python3
"""Archive legacy artifacts and remove reproducible runtime clutter.

Run from anywhere. The default archive is kept outside the active code tree.
Use ``--dry-run`` to print actions without changing files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT.parents[2] / "archive" / f"code_cleanup_{date.today().isoformat()}"

DELETE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", "hf_cache", "logs"}
DELETE_FILE_SUFFIXES = {".log", ".out", ".bak", ".tmp", ".pyc", ".pyo"}
LATEX_SUFFIXES = {".aux", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"}

ARCHIVE_DIRS = {
    "memory-before": "legacy_state/memory-before",
    "output-before": "legacy_state/output-before",
    "output/archive": "legacy_state/output_archive",
    "paper/external": "research_history/paper_external",
    "refine-logs": "research_history/refine-logs",
    "review-stage": "research_history/review-stage",
}

EVIDENCE_SUMMARIES = {
    "paper/evidence/table4_evolution/summary.json",
    "paper/evidence/table4_evolution/qwen/summary.json",
    "paper/evidence/table4_evolution/llama/summary.json",
    "paper/evidence/table5_zone_migration/summary.json",
    "paper/evidence/table5_zone_migration/qwen/summary.json",
    "paper/evidence/table5_zone_migration/llama/summary.json",
}


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(path: Path):
    if path.is_file():
        yield path
    elif path.exists():
        yield from (item for item in sorted(path.rglob("*")) if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archive = args.archive.resolve()
    if PROJECT_ROOT == archive or PROJECT_ROOT in archive.parents:
        raise SystemExit("archive must be outside the active project tree")
    if archive.exists() and not args.dry_run:
        raise SystemExit(f"archive already exists; choose a new --archive path: {archive}")

    records: list[dict[str, object]] = []
    before_bytes = tree_size(PROJECT_ROOT)

    def record(action: str, source: Path, destination: Path | None, size: int, checksum: str = "", reason: str = ""):
        records.append({
            "action": action,
            "source": str(source),
            "destination": str(destination or ""),
            "bytes": size,
            "sha256": checksum,
            "reason": reason,
        })

    def delete_tree(path: Path, reason: str):
        if not path.exists():
            return
        size = tree_size(path)
        print(f"DELETE_DIR {path} ({size} bytes): {reason}")
        record("delete_dir", path, None, size, reason=reason)
        if not args.dry_run:
            shutil.rmtree(path)

    def delete_file(path: Path, reason: str):
        if not path.exists():
            return
        size = path.stat().st_size
        print(f"DELETE_FILE {path} ({size} bytes): {reason}")
        record("delete_file", path, None, size, reason=reason)
        if not args.dry_run:
            path.unlink()

    def archive_path(source: Path, destination: Path, reason: str):
        if not source.exists():
            return
        if destination.exists():
            raise SystemExit(f"archive destination already exists: {destination}")
        for item in file_inventory(source):
            relative = item.relative_to(source) if source.is_dir() else Path(item.name)
            target = destination / relative if source.is_dir() else destination
            record("archive", item, target, item.stat().st_size, sha256(item), reason)
        print(f"ARCHIVE {source} -> {destination} ({tree_size(source)} bytes): {reason}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

    # Remove reproducible caches and log trees first, including those nested in
    # legacy directories, so archives contain research state rather than noise.
    for path in sorted(PROJECT_ROOT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if ".git" in path.parts or not path.exists():
            continue
        if path.is_dir() and path.name in DELETE_DIR_NAMES:
            delete_tree(path, "regenerable cache or runtime log directory")

    for path in sorted(PROJECT_ROOT.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        suffix = ".synctex.gz" if path.name.endswith(".synctex.gz") else path.suffix
        if suffix in DELETE_FILE_SUFFIXES or suffix in LATEX_SUFFIXES:
            delete_file(path, "regenerable log, backup, temporary, or build artifact")

    for empty_name in ("checkpoints", "tmp_assoc_smoke", "tmp_test", "paper/build"):
        path = PROJECT_ROOT / empty_name
        if path.exists() and not any(path.rglob("*")):
            delete_tree(path, "empty runtime directory")

    for source_text, destination_text in ARCHIVE_DIRS.items():
        archive_path(PROJECT_ROOT / source_text, archive / destination_text, "legacy or research-history artifact")

    evidence_root = PROJECT_ROOT / "paper" / "evidence"
    for table_name in ("table4_evolution", "table5_zone_migration"):
        table_root = evidence_root / table_name
        if not table_root.exists():
            continue
        for item in sorted(table_root.rglob("*")):
            if not item.is_file():
                continue
            rel_project = item.relative_to(PROJECT_ROOT).as_posix()
            if rel_project in EVIDENCE_SUMMARIES:
                continue
            destination = archive / "paper_evidence_raw" / item.relative_to(evidence_root)
            archive_path(item, destination, "duplicate raw evidence; canonical source remains in output/memory")
        if not args.dry_run:
            for directory in sorted((p for p in table_root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                if not any(directory.iterdir()):
                    directory.rmdir()

    if args.dry_run:
        print(f"DRY_RUN actions={len(records)}")
        return 0

    archive.mkdir(parents=True, exist_ok=True)
    manifest_path = archive / "cleanup_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action", "source", "destination", "bytes", "sha256", "reason"])
        writer.writeheader()
        writer.writerows(records)

    after_bytes = tree_size(PROJECT_ROOT)
    summary = {
        "project_root": str(PROJECT_ROOT),
        "archive_root": str(archive),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "active_tree_reduction_bytes": before_bytes - after_bytes,
        "actions": len(records),
        "manifest": str(manifest_path),
    }
    (archive / "cleanup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
