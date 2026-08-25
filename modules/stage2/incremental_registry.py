"""Append-only Stage-2 Candidate Registry and round releases.

The Registry avoids repeated model calls across 40 -> 100 -> 600 -> all
expansions. A release freezes reviewed Principle versions; later repairs create
children and never overwrite their parents.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from modules.stage2.principle_test import (
    DEV8_BY_LEVEL,
    LEVELS,
    _json_hash,
    _load_bundles_by_level,
    _stable_key,
    audit_input_hash,
    candidates_from_generation,
    current_stage2_protocol,
    generation_input_hash,
    generate_raw_candidates,
    run_audit_ablation,
)
from utils.io_utils import atomic_save_json, load_json


ROUND_QUOTAS: dict[int, dict[str, int]] = {
    40: {"level1": 10, "level2": 10, "level3": 10, "level4": 10},
    100: {"level1": 25, "level2": 25, "level3": 25, "level4": 25},
    600: {"level1": 100, "level2": 200, "level3": 200, "level4": 100},
}


def _target_name(target: int | str, total: int | None = None) -> str:
    if str(target).lower() == "all":
        return f"target_all_{int(total or 0):04d}"
    return f"target_{int(target):04d}"


def _release_name(target_count: int) -> str:
    return f"round_{target_count:04d}"


def _ordered_pool(bundles: Sequence[Mapping[str, Any]], level: str, seed: int) -> list[dict]:
    pool = {str(item["question_id"]): dict(item) for item in bundles}
    required = [qid for qid in DEV8_BY_LEVEL[level] if qid in pool]
    remainder = sorted(
        (qid for qid in pool if qid not in required),
        key=lambda qid: _stable_key(qid, f"stage2_principle_test_{seed}"),
    )
    return [pool[qid] for qid in required + remainder]


def _resolve_quotas(
    target: int | str,
    available: Mapping[str, int],
) -> dict[str, int]:
    if str(target).lower() == "all":
        return {level: int(available[level]) for level in LEVELS}
    value = int(target)
    if value not in ROUND_QUOTAS:
        raise ValueError(
            f"Unsupported target={value}. Use 40, 100, 600, or all so Level quotas remain explicit."
        )
    quotas = dict(ROUND_QUOTAS[value])
    insufficient = {
        level: {"required": quotas[level], "available": int(available[level])}
        for level in LEVELS if int(available[level]) < quotas[level]
    }
    if insufficient:
        raise ValueError(f"Stage-1 eligible pool is too small: {insufficient}")
    return quotas


def _registry_generation_path(
    registry_root: Path, question_id: str, generation_protocol_hash: str
) -> Path:
    return registry_root / "questions" / question_id / generation_protocol_hash / "generation.json"


def _registry_audit_path(
    registry_root: Path,
    question_id: str,
    candidate_protocol_hash: str,
    audit_protocol: str,
    candidate_id: str,
) -> Path:
    return (
        registry_root / "questions" / question_id / candidate_protocol_hash /
        "audits" / audit_protocol / f"{candidate_id}.json"
    )


def prepare_incremental_target(
    config: Any,
    *,
    target: int | str,
    workspace_root: Path,
    registry_root: Path,
    backbone: str = "llama",
    seed: int = 42,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a nested target manifest and report reusable/missing questions."""
    by_level, input_hashes = _load_bundles_by_level(config, backbone)
    available = {level: len(by_level[level]) for level in LEVELS}
    quotas = _resolve_quotas(target, available)
    selected_by_level: dict[str, list[dict]] = {}
    for level in LEVELS:
        selected_by_level[level] = _ordered_pool(by_level[level], level, seed)[:quotas[level]]
    selected = [item for level in LEVELS for item in selected_by_level[level]]
    target_count = len(selected)
    target_name = _target_name(target, target_count)
    target_root = Path(workspace_root) / target_name
    manifest_path = target_root / "manifests" / "source40.json"
    protocol = dict(protocol or {})
    generation_hash = str(protocol.get("generation_protocol_hash", ""))
    candidate_hash = str(protocol.get("candidate_protocol_hash", ""))
    audit_protocol = str((protocol.get("auditor") or {}).get("protocol", ""))

    question_status = {}
    for bundle in selected:
        qid = str(bundle["question_id"])
        generation_path = _registry_generation_path(Path(registry_root), qid, generation_hash)
        generation_record = load_json(generation_path) if generation_path.exists() else {}
        generation_reusable = bool(
            generation_record.get("status") in {
                "complete", "no_reliable_principle", "terminal_failed"
            }
            and generation_record.get("generation_input_hash")
            == generation_input_hash(config, bundle)
        )
        generated_candidates = (
            candidates_from_generation(generation_record) if generation_reusable else []
        )
        audit_reusable = bool(
            generation_reusable
            and generated_candidates
            and candidate_hash
            and audit_protocol
            and all(
                (
                    (audit_path := _registry_audit_path(
                        Path(registry_root), qid, candidate_hash,
                        audit_protocol, candidate["candidate_id"],
                    )).exists()
                    and (audit_record := load_json(audit_path)).get("status") == "terminal"
                    and audit_record.get("audit_input_hash")
                    == audit_input_hash(config, candidate, bundle, audit_protocol)
                )
                for candidate in generated_candidates
            )
        )
        # A terminal generation with no candidate requires no audit call.
        if generation_reusable and generation_record.get("status") != "complete":
            audit_reusable = True
        question_status[qid] = {
            "generation": "reusable" if generation_reusable else "missing",
            "audit": "reusable" if audit_reusable else "missing",
            "bundle_hash": _json_hash(bundle),
        }

    manifest = {
        "schema_version": "incremental_target.1",
        "target": str(target),
        "target_name": target_name,
        "target_count": target_count,
        "level_quotas": quotas,
        "backbone": backbone,
        "seed": seed,
        "selection_salt": f"stage2_principle_test_{seed}",
        "source_question_ids_by_level": {
            level: [str(item["question_id"]) for item in selected_by_level[level]]
            for level in LEVELS
        },
        "source_question_ids": [str(item["question_id"]) for item in selected],
        "source_bundle_hashes": {
            str(item["question_id"]): _json_hash(item) for item in selected
        },
        "stage1_input_hashes": input_hashes,
        "protocol_hash": protocol.get("protocol_hash", ""),
        "generation_protocol_hash": generation_hash,
        "candidate_protocol_hash": candidate_hash,
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    if manifest_path.exists():
        previous = load_json(manifest_path)
        if previous.get("manifest_hash") != manifest["manifest_hash"]:
            raise RuntimeError(
                f"Frozen target manifest differs. Use a new workspace or explicit protocol migration: {manifest_path}"
            )
    atomic_save_json(manifest_path, manifest)
    atomic_save_json(target_root / "source_bundles" / "source40_bundles.json", selected)
    # The generator reads this compatibility path; targets after 40 do not use
    # test variants, so the list is intentionally empty.
    atomic_save_json(target_root / "source_bundles" / "reserve20_bundles.json", [])
    status_counts = {
        "generation_reusable": sum(v["generation"] == "reusable" for v in question_status.values()),
        "generation_missing": sum(v["generation"] == "missing" for v in question_status.values()),
        "audit_reusable": sum(v["audit"] == "reusable" for v in question_status.values()),
        "audit_missing": sum(v["audit"] == "missing" for v in question_status.values()),
    }
    status = {
        "target_root": str(target_root),
        "manifest_hash": manifest["manifest_hash"],
        "target_count": target_count,
        "level_quotas": quotas,
        **status_counts,
        "questions": question_status,
    }
    atomic_save_json(target_root / "registry_status.json", status)
    return status


async def run_incremental_candidate_round(
    config: Any,
    *,
    target: int | str,
    workspace_root: Path,
    registry_root: Path,
    state_root: Path,
    backbone: str = "llama",
    concurrency: int = 2,
) -> dict[str, Any]:
    """Generate/audit only Registry misses under the frozen Stage-2 protocol."""
    lock_path = Path(state_root) / "current_protocol.lock.json"
    if not lock_path.exists():
        raise RuntimeError("No frozen Stage-2 protocol. Complete and freeze the 40-question round first.")
    frozen = load_json(lock_path)
    audit_protocol = str((frozen.get("auditor") or {}).get("protocol", ""))
    current = current_stage2_protocol(
        config,
        audit_protocol=audit_protocol,
        intervention_arm=str(frozen.get("intervention_arm", "")),
    )
    if current.get("candidate_protocol_hash") != frozen.get("candidate_protocol_hash"):
        raise RuntimeError(
            "Current Stage-2 generation/audit code differs from the frozen protocol. "
            "Create an explicit protocol migration instead of mixing candidates."
        )
    before = prepare_incremental_target(
        config, target=target, workspace_root=workspace_root,
        registry_root=registry_root, backbone=backbone,
        seed=config.stage2.principle_test_seed, protocol=frozen,
    )
    target_root = Path(before["target_root"])
    atomic_save_json(target_root / "registry_status_before.json", before)
    generation = await generate_raw_candidates(
        config, test_root=target_root, concurrency=concurrency,
        registry_root=registry_root,
        generation_protocol_hash=frozen["generation_protocol_hash"],
    )
    audit = await run_audit_ablation(
        config, test_root=target_root, concurrency=concurrency,
        protocols=(audit_protocol,), registry_root=registry_root,
        candidate_protocol_hash=frozen["candidate_protocol_hash"],
    )
    after = prepare_incremental_target(
        config, target=target, workspace_root=workspace_root,
        registry_root=registry_root, backbone=backbone,
        seed=config.stage2.principle_test_seed, protocol=frozen,
    )
    atomic_save_json(target_root / "registry_status_after.json", after)
    result = {
        "target": str(target),
        "target_root": str(target_root),
        "before": {key: value for key, value in before.items() if key != "questions"},
        "generation": generation,
        "audit": audit,
        "after": {key: value for key, value in after.items() if key != "questions"},
        "quality_status": "awaiting_round_quality_check",
    }
    atomic_save_json(target_root / "incremental_round_summary.json", result)
    bank = load_json(target_root / "raw_candidates" / "candidate_bank.json")
    atomic_save_json(target_root / "quality_decision.template.json", {
        "approved": False,
        "target_count": int(after["target_count"]),
        "manifest_hash": after["manifest_hash"],
        "candidate_bank_hash": bank["candidate_bank_hash"],
        "protocol_hash": frozen["protocol_hash"],
        "reviewer": "",
        "reviewed_at": "",
        "rationale": "",
        "notes": [],
    })
    return result


def _load_effects(test_root: Path, arm: str) -> dict[str, dict[str, Any]]:
    path = test_root / "source_evaluation" / "interventions" / "candidate_effects.json"
    if not path.exists():
        return {}
    return {
        str(item["candidate_id"]): item
        for item in load_json(path)
        if isinstance(item, dict) and item.get("arm") == arm
    }


def _append_version_index(
    path: Path,
    *,
    version: Mapping[str, Any],
) -> None:
    payload = load_json(path) if path.exists() else {"versions": []}
    versions = payload.get("versions", []) if isinstance(payload, dict) else []
    version_id = str(version["version_id"])
    if not any(str(item.get("version_id")) == version_id for item in versions):
        versions.append(dict(version))
    atomic_save_json(path, {"versions": versions})


def _update_experiment_state(state_root: Path) -> None:
    releases = []
    for path in sorted((state_root / "releases").glob("round_*/release.json")):
        payload = load_json(path)
        releases.append(payload)
    lines = [
        "# Stage 2 Experiment State", "",
        "> 该文件由冻结命令更新。Release不可覆盖；Principle修复只能新增版本。", "",
        "## Frozen rounds", "",
        "| Round | Questions | Reused | New | Frozen principles | Protocol |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in releases:
        lines.append(
            f"| {item['release_id']} | {item['target_count']} | "
            f"{item['reused_question_count']} | {item['new_question_count']} | "
            f"{item['frozen_principle_count']} | `{item['protocol_hash'][:12]}` |"
        )
    lines.extend(["", "## Next action", ""])
    if releases:
        latest = releases[-1]
        next_target = 100 if latest["target_count"] < 100 else 600 if latest["target_count"] < 600 else "all"
        lines.append(
            f"Current release: `{latest['release_id']}`. Next target: `{next_target}`; "
            "reuse all hash-compatible frozen questions and generate only missing questions."
        )
    else:
        lines.append("No round has been frozen. Complete the 40-question quality calibration first.")
    (state_root / "STAGE2_EXPERIMENT_STATE.md").parent.mkdir(parents=True, exist_ok=True)
    (state_root / "STAGE2_EXPERIMENT_STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_stage2_round(
    config: Any,
    *,
    target_root: Path,
    registry_root: Path,
    state_root: Path,
    target_count: int,
    quality_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze a quality-approved round without overwriting any earlier release."""
    target_root = Path(target_root)
    state_root = Path(state_root)
    registry_root = Path(registry_root)
    calibration_report = target_root / "report" / "summary.json"
    if quality_decision_path is not None:
        quality = load_json(quality_decision_path)
        approved = quality.get("approved") is True
    elif calibration_report.exists():
        quality = load_json(calibration_report)
        approved = bool((quality.get("acceptance") or {}).get("ready_to_scale"))
    else:
        raise RuntimeError(
            "Missing round quality decision. Provide a JSON file with approved=true "
            "or complete the calibration report."
        )
    if not approved:
        raise RuntimeError("Round quality has not been approved; refusing to freeze Principles.")

    frozen_protocol_path = target_root / "source_evaluation" / "frozen_protocol.json"
    if frozen_protocol_path.exists():
        selected = load_json(frozen_protocol_path)
        audit_protocol = selected["audit_protocol"]
        intervention_arm = selected["intervention_arm"]
        protocol = current_stage2_protocol(
            config, audit_protocol=audit_protocol, intervention_arm=intervention_arm
        )
    else:
        lock_path = state_root / "current_protocol.lock.json"
        if not lock_path.exists():
            raise RuntimeError("Missing frozen protocol selection for this round")
        protocol = load_json(lock_path)
        audit_protocol = str(protocol["auditor"]["protocol"])
        intervention_arm = str(protocol.get("intervention_arm", ""))

    current_lock = state_root / "current_protocol.lock.json"
    if current_lock.exists():
        prior_lock = load_json(current_lock)
        if prior_lock.get("protocol_hash") != protocol["protocol_hash"]:
            raise RuntimeError(
                "A different protocol is already frozen. Create an explicit migration; "
                "no release or lock was changed."
            )

    manifest = load_json(target_root / "manifests" / "source40.json")
    if int(manifest.get("target_count", len(manifest.get("source_question_ids", [])))) != target_count:
        actual = len(manifest.get("source_question_ids", []))
        if actual != target_count:
            raise ValueError(f"Release target mismatch: requested={target_count} manifest={actual}")
    bank = load_json(target_root / "raw_candidates" / "candidate_bank.json")
    if quality_decision_path is not None:
        expected_quality_binding = {
            "target_count": target_count,
            "manifest_hash": manifest["manifest_hash"],
            "candidate_bank_hash": bank["candidate_bank_hash"],
            "protocol_hash": protocol["protocol_hash"],
        }
        mismatches = {
            key: {"expected": expected, "actual": quality.get(key)}
            for key, expected in expected_quality_binding.items()
            if quality.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Quality decision is not bound to this exact round: {mismatches}"
            )
        if not str(quality.get("reviewer", "")).strip() or not str(
            quality.get("rationale", "")
        ).strip():
            raise RuntimeError("Quality decision requires non-empty reviewer and rationale")
    effects = _load_effects(target_root, intervention_arm)
    audit_records = {}
    for path in sorted((target_root / "audit_variants" / audit_protocol).glob("candidate_*.json")):
        record = load_json(path)
        audit_records[str(record["candidate_id"])] = record

    # Validate all Registry writes before changing durable state.
    registry_writes: list[tuple[Path, Any]] = []
    for qid in manifest.get("source_question_ids", []):
        generation_path = target_root / "raw_candidates" / "questions" / f"{qid}.json"
        if generation_path.exists():
            registry_writes.append((
                _registry_generation_path(
                    registry_root, str(qid), protocol["generation_protocol_hash"]
                ),
                load_json(generation_path),
            ))
    for candidate_id, record in audit_records.items():
        qid = str(record["final_candidate"]["question_id"])
        registry_writes.append((
            _registry_audit_path(
                registry_root, qid, protocol["candidate_protocol_hash"],
                audit_protocol, candidate_id,
            ),
            record,
        ))
    for path, payload in registry_writes:
        if path.exists() and _json_hash(load_json(path)) != _json_hash(payload):
            raise RuntimeError(f"Registry artifact is immutable and differs: {path}")

    frozen_principles = []
    pending_versions: list[tuple[Path, dict[str, Any]]] = []
    status_counts: Counter[str] = Counter()
    for candidate in bank.get("candidates", []):
        candidate_id = str(candidate["candidate_id"])
        audit = audit_records.get(candidate_id)
        if not audit:
            status = "audit_missing"
            content = candidate["content"]
        else:
            content = audit["final_candidate"]["content"]
            decision = audit.get("final_decision")
            effect = effects.get(candidate_id, {})
            if decision != "pass":
                status = "rejected"
            elif effect.get("effect_label") == "harmful":
                status = "rejected_effect"
            elif effect.get("effect_label") == "helpful":
                status = "effect_validated"
            else:
                status = "audited"
        content_hash = _json_hash(content)
        versions_path = registry_root / "principles" / candidate_id / "versions.json"
        existing_versions = (
            load_json(versions_path).get("versions", []) if versions_path.exists() else []
        )
        version_id = (
            f"{candidate_id}_{protocol['candidate_protocol_hash'][:8]}_{content_hash[:8]}"
        )
        parent_version_id = ""
        if existing_versions and not any(
            str(item.get("version_id")) == version_id for item in existing_versions
        ):
            parent_version_id = str(existing_versions[-1].get("version_id", ""))
        item = {
            "candidate_id": candidate_id,
            "version_id": version_id,
            "parent_version_id": parent_version_id,
            "question_id": str(candidate["question_id"]),
            "level": candidate["level"],
            "status": status,
            "content": content,
            "content_hash": content_hash,
            "protocol_hash": protocol["protocol_hash"],
            "candidate_protocol_hash": protocol["candidate_protocol_hash"],
            "audit_protocol": audit_protocol,
            "effect": effects.get(candidate_id, {}),
        }
        frozen_principles.append(item)
        status_counts[status] += 1
        pending_versions.append((
            versions_path,
            {
                "version_id": version_id,
                "content_hash": content_hash,
                "status": status,
                "release_id": _release_name(target_count),
                "parent_version_id": parent_version_id,
            },
        ))

    principles_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in frozen_principles:
        principles_by_question[str(item["question_id"])].append(item)
    question_terminals = []
    for qid in map(str, manifest.get("source_question_ids", [])):
        generation_path = target_root / "raw_candidates" / "questions" / f"{qid}.json"
        generation = load_json(generation_path) if generation_path.exists() else {}
        principles = principles_by_question.get(qid, [])
        if not generation:
            terminal_status = "generation_missing"
        elif not principles:
            terminal_status = str(generation.get("status", "no_reliable_principle"))
        elif any(item["status"] == "effect_validated" for item in principles):
            terminal_status = "effect_validated"
        elif any(item["status"] == "audited" for item in principles):
            terminal_status = "audited"
        else:
            terminal_status = "rejected"
        question_terminals.append({
            "question_id": qid,
            "generation_status": generation.get("status", "missing"),
            "terminal_status": terminal_status,
            "candidate_ids": [item["candidate_id"] for item in principles],
            "principle_statuses": dict(Counter(item["status"] for item in principles)),
            "bundle_hash": manifest.get("source_bundle_hashes", {}).get(qid, ""),
        })

    release_id = _release_name(target_count)
    previous_releases = [
        path for path in sorted((state_root / "releases").glob("round_*/release.json"))
        if path.parent.name != release_id
    ]
    previous_questions: set[str] = set()
    parent_release = ""
    if previous_releases:
        previous = load_json(previous_releases[-1])
        previous_questions = set(previous.get("source_question_ids", []))
        parent_release = str(previous.get("release_id", ""))
    current_questions = set(map(str, manifest.get("source_question_ids", [])))
    if not previous_questions.issubset(current_questions):
        raise RuntimeError("New release is not a superset of the previous frozen release")

    release_dir = state_root / "releases" / release_id
    release = {
        "release_id": release_id,
        "parent_release": parent_release,
        "target_count": target_count,
        "source_question_ids": sorted(current_questions),
        "reused_question_count": len(previous_questions),
        "new_question_count": len(current_questions - previous_questions),
        "protocol_hash": protocol["protocol_hash"],
        "generation_protocol_hash": protocol["generation_protocol_hash"],
        "candidate_protocol_hash": protocol["candidate_protocol_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "candidate_bank_hash": bank["candidate_bank_hash"],
        "quality_decision_hash": _json_hash(quality),
        "frozen_principle_count": len(frozen_principles),
        "status_counts": dict(status_counts),
        "question_terminal_counts": dict(Counter(
            item["terminal_status"] for item in question_terminals
        )),
    }
    release["release_hash"] = _json_hash({
        "release": release,
        "principles": frozen_principles,
        "question_terminals": question_terminals,
    })
    release_path = release_dir / "release.json"
    if release_path.exists():
        previous = load_json(release_path)
        if previous.get("release_hash") != release["release_hash"]:
            raise RuntimeError(f"Release is immutable and already exists: {release_path}")
        return previous
    # Commit only after all immutable-input and protocol checks pass. The
    # release file is written last and acts as the commit marker.
    for path, payload in registry_writes:
        if not path.exists():
            atomic_save_json(path, payload)
    for versions_path, version in pending_versions:
        _append_version_index(versions_path, version=version)
    atomic_save_json(release_dir / "frozen_principles.json", frozen_principles)
    atomic_save_json(release_dir / "question_terminals.json", question_terminals)
    atomic_save_json(release_dir / "quality_decision.json", quality)
    atomic_save_json(state_root / "protocols" / f"{protocol['protocol_hash']}.json", protocol)
    if not current_lock.exists():
        atomic_save_json(current_lock, protocol)
    atomic_save_json(release_path, release)
    _update_experiment_state(state_root)
    return release


def incremental_registry_status(
    *, registry_root: Path, state_root: Path, workspace_root: Path
) -> dict[str, Any]:
    releases = [
        load_json(path)
        for path in sorted((Path(state_root) / "releases").glob("round_*/release.json"))
    ]
    question_dirs = list((Path(registry_root) / "questions").glob("*"))
    workspaces = [
        str(path) for path in sorted(Path(workspace_root).glob("target_*")) if path.is_dir()
    ]
    return {
        "registry_root": str(registry_root),
        "registered_question_count": len(question_dirs),
        "release_count": len(releases),
        "releases": releases,
        "current_protocol_exists": (Path(state_root) / "current_protocol.lock.json").exists(),
        "workspaces": workspaces,
    }
