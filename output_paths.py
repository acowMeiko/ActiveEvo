import re
from pathlib import Path


def _project_root_from_output(root: Path) -> Path:
    """Infer project root from Config.paths.output_dir."""
    root = Path(root)
    if root.name == "output":
        return root.parent
    return root.parent if root.is_absolute() else Path(".")


def memory_stage_root(root: Path) -> Path:
    return _project_root_from_output(root) / "memory"


def stage_output_dir(root: Path, stage: str, dataset: str, step: str) -> Path:
    project_root = _project_root_from_output(root)
    if stage == "stage1":
        return project_root / "learning_packet" / dataset / step
    if stage == "stage2":
        return project_root / "memory" / dataset / step
    return root / stage / dataset / step


def stage1_sampling_file(root: Path, dataset: str) -> Path:
    return stage_output_dir(root, "stage1", dataset, "01_sampling") / "sampling_results.json"


def stage1_mining_file(root: Path, dataset: str) -> Path:
    return stage_output_dir(root, "stage1", dataset, "02_mining") / "hard_negative_pairs.json"


def stage1_principle_file(root: Path, dataset: str) -> Path:
    return stage_output_dir(root, "stage1", dataset, "03_dpo") / "principle_dataset.json"


def stage1_principle_dpo_file(root: Path, dataset: str) -> Path:
    """旧流程产物文件名 (backward compat)."""
    return stage_output_dir(root, "stage1", dataset, "03_dpo") / "principle_dpo_dataset.json"


def stage1_dpo_file(root: Path, dataset: str) -> Path:
    """Deprecated: use stage1_principle_file instead."""
    return stage1_principle_file(root, dataset)


def stage1_diff_failed_file(root: Path, dataset: str) -> Path:
    return stage_output_dir(root, "stage1", dataset, "03_dpo") / "diff_failed.json"


def stage1_learning_packets_file(root: Path, dataset: str) -> Path:
    return stage_output_dir(root, "stage1", dataset, "04_learning_packets") / "learning_packets.json"


def stage1_learning_pairs_file(root: Path, dataset: str) -> Path:
    """新版 Stage 1 输出：仅含正负对 + 分区信息，不含 diff/principle。"""
    return stage_output_dir(root, "stage1", dataset, "03_learning_pairs") / "learning_pairs.json"


def stage2_memory_file(root: Path, dataset: str, backbone: str = "") -> Path:
    prefix = f"{backbone}_" if backbone else ""
    return stage_output_dir(root, "stage2", dataset, "01_memory_bank") / f"{prefix}semantic_memory_bank.json"


def stage2_summary_file(root: Path, dataset: str, backbone: str = "") -> Path:
    prefix = f"{backbone}_" if backbone else ""
    return stage_output_dir(root, "stage2", dataset, "01_memory_bank") / f"{prefix}stage2_summary.json"


def stage2_vector_dir(root: Path, dataset: str) -> Path:
    return stage_output_dir(root, "stage2", dataset, "02_vector_db")


def stage2_dsd_registry_dir(root: Path, dataset: str, backbone: str = "") -> Path:
    sub = backbone if backbone else None
    base = stage_output_dir(root, "stage2", dataset, "00_dsd_registry")
    return base / sub if sub else base


def stage2_dsd_registry_file(root: Path, dataset: str, backbone: str = "") -> Path:
    return stage2_dsd_registry_dir(root, dataset, backbone) / "question_dsd_registry.json"


def stage2_dsd_core_index_file(root: Path, dataset: str, backbone: str = "") -> Path:
    return stage2_dsd_registry_dir(root, dataset, backbone) / "core_skill.index"


def stage2_dsd_condition_index_file(root: Path, dataset: str, backbone: str = "") -> Path:
    return stage2_dsd_registry_dir(root, dataset, backbone) / "context_constraints.index"


def stage2_dsd_target_index_file(root: Path, dataset: str, backbone: str = "") -> Path:
    return stage2_dsd_registry_dir(root, dataset, backbone) / "task_objective.index"


def stage3_guided_results_file(root: Path, dataset: str, backbone: str = "") -> Path:
    prefix = f"{backbone}_" if backbone else ""
    return stage_output_dir(root, "stage3", dataset, "01_guided_eval") / f"{prefix}principle_guided_test_results.json"


def stage3_guided_summary_file(root: Path, dataset: str, backbone: str = "") -> Path:
    prefix = f"{backbone}_" if backbone else ""
    return stage_output_dir(root, "stage3", dataset, "01_guided_eval") / f"{prefix}principle_guided_test_summary.json"


def stage3_baseline_results_file(root: Path, dataset: str, backbone: str = "") -> Path:
    prefix = f"{backbone}_" if backbone else ""
    return stage_output_dir(root, "stage3", dataset, "02_baseline_eval") / f"{prefix}principle_baseline_test_results.json"


def stage3_baseline_summary_file(root: Path, dataset: str, backbone: str = "") -> Path:
    prefix = f"{backbone}_" if backbone else ""
    return stage_output_dir(root, "stage3", dataset, "02_baseline_eval") / f"{prefix}principle_baseline_test_summary.json"


def stage1_unified_sampling_file(root: Path, backbone: str = "") -> Path:
    """新 Stage 1: 全量采样结果 (6000题统一)."""
    prefix = f"{backbone}_" if backbone else ""
    return root / "stage1" / "01_sampling" / f"{prefix}unified_sampling_results.json"


def stage1_level_packets_dir(root: Path, backbone: str = "") -> Path:
    """新 Stage 1: 各 level learning packets 输出目录."""
    sub = backbone if backbone else "default"
    return root / "stage1" / "02_learning_packets" / sub


def stage1_level_packets_file(root: Path, level: int, backbone: str = "") -> Path:
    return stage1_level_packets_dir(root, backbone) / f"level{level}_learning_packets.json"


def stage3_progress_state_file(root: Path, scope_key: str = "default", backbone: str = "") -> Path:
    safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scope_key or "default")).strip("._") or "default"
    sub = backbone if backbone else "default"
    return root / "stage3" / "curriculum" / sub / safe_scope / "progress_state.json"
