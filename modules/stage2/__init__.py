"""Stage 2 Memory Factory.

Stage 2 turns Stage-1 trajectory evidence into auditable, versioned memories.
It deliberately contains no neighbor-task validation or Solver evaluation.
"""

from modules.stage2.factory import build_level_memory
from modules.stage2.generator_comparison import compare_generators, run_qwen_protocol_probe
from modules.stage2.pilot_runner import (
    compare_pilot_branches,
    evaluate_dev8_gate,
    pilot_status,
    prepare_pilot_manifest,
    run_generator_pilot,
)
from modules.stage2.source_validation import validate_source_principles
from modules.stage2.role_compare import run_kimi_synthesis_recovery, run_role_compare
from modules.stage2.protocol_calibration import (
    prepare_l23_protocol,
    run_l23_protocol_calibration,
)
from modules.stage2.incremental_registry import (
    freeze_stage2_round,
    incremental_registry_status,
    prepare_incremental_target,
    run_incremental_candidate_round,
)
from modules.stage2.principle_test import (
    build_principle_test_report,
    generate_raw_candidates,
    generate_variants,
    prepare_principle_test,
    principle_test_status,
    require_variant_approvals,
    run_audit_ablation,
    run_principle_test_smoke,
    run_provider_probe,
    run_robustness_effect_test,
    run_source_effect_test,
    run_transfer_effect_test,
)

__all__ = [
    "build_level_memory",
    "compare_generators",
    "run_qwen_protocol_probe",
    "prepare_pilot_manifest",
    "run_generator_pilot",
    "evaluate_dev8_gate",
    "pilot_status",
    "compare_pilot_branches",
    "validate_source_principles",
    "run_role_compare",
    "run_kimi_synthesis_recovery",
    "prepare_l23_protocol",
    "run_l23_protocol_calibration",
    "freeze_stage2_round",
    "incremental_registry_status",
    "prepare_incremental_target",
    "run_incremental_candidate_round",
    "prepare_principle_test",
    "run_provider_probe",
    "run_principle_test_smoke",
    "generate_variants",
    "require_variant_approvals",
    "generate_raw_candidates",
    "run_audit_ablation",
    "run_source_effect_test",
    "run_transfer_effect_test",
    "run_robustness_effect_test",
    "build_principle_test_report",
    "principle_test_status",
]
