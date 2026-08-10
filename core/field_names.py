# core/field_names.py
"""
语义字段名统一定义 — 领域无关的三维语义字段

新字段名 (领域无关):
  - core_skill          核心能力/方法
  - context_constraints  上下文约束与结构
  - task_objective       任务目标

旧字段名 (数学绑定, 向后兼容):
  - core_math_concept    → core_skill
  - conditions_structure → context_constraints
  - solution_target      → task_objective

所有代码统一通过 get_field() / get_field_with_fallback() 读取数据，
确保新旧字段名数据均可正常处理。
"""
from typing import Any, Dict, Optional

# ==================== 新字段名常量 ====================
CORE_FIELD = "core_skill"
CONDITION_FIELD = "context_constraints"
TARGET_FIELD = "task_objective"

# ==================== 旧字段名 (向后兼容) ====================
LEGACY_CORE_FIELD = "core_math_concept"
LEGACY_CONDITION_FIELD = "conditions_structure"
LEGACY_TARGET_FIELD = "solution_target"

# 新 → 旧 映射
FIELD_LEGACY_MAP = {
    CORE_FIELD: LEGACY_CORE_FIELD,
    CONDITION_FIELD: LEGACY_CONDITION_FIELD,
    TARGET_FIELD: LEGACY_TARGET_FIELD,
}

# 旧 → 新 映射
LEGACY_FIELD_MAP = {v: k for k, v in FIELD_LEGACY_MAP.items()}

# 三个字段的有序列表 (用于遍历)
SEMANTIC_FIELDS = [CORE_FIELD, CONDITION_FIELD, TARGET_FIELD]
LEGACY_SEMANTIC_FIELDS = [LEGACY_CORE_FIELD, LEGACY_CONDITION_FIELD, LEGACY_TARGET_FIELD]


def get_field(data: Dict[str, Any], field: str, default: str = "") -> str:
    """从字典中读取字段值，自动尝试新旧字段名。
    
    优先读取新字段名，如果为空则尝试旧字段名。
    """
    value = str(data.get(field, "") or "").strip()
    if value:
        return value
    
    # 尝试旧字段名
    legacy = FIELD_LEGACY_MAP.get(field)
    if legacy:
        value = str(data.get(legacy, "") or "").strip()
        if value:
            return value
    
    # 尝试新字段名 (如果传入的是旧字段名)
    new_field = LEGACY_FIELD_MAP.get(field)
    if new_field:
        value = str(data.get(new_field, "") or "").strip()
        if value:
            return value
    
    return default


def extract_semantic_triple(data: Dict[str, Any]) -> Dict[str, str]:
    """从字典中提取三维语义字段，自动兼容新旧字段名。
    
    Returns:
        dict with keys: core_skill, context_constraints, task_objective
    """
    return {
        CORE_FIELD: get_field(data, CORE_FIELD),
        CONDITION_FIELD: get_field(data, CONDITION_FIELD),
        TARGET_FIELD: get_field(data, TARGET_FIELD),
    }


def normalize_to_new_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """将字典中的旧字段名转换为新字段名（不删除旧字段，确保兼容）。"""
    result = dict(data)
    for new_field, legacy_field in FIELD_LEGACY_MAP.items():
        if new_field not in result or not str(result.get(new_field, "")).strip():
            legacy_value = str(result.get(legacy_field, "")).strip()
            if legacy_value:
                result[new_field] = legacy_value
    return result


def semantic_fields_complete(features: Dict[str, str]) -> bool:
    """检查三维语义字段是否都非空。"""
    return all(
        get_field(features, f)
        for f in SEMANTIC_FIELDS
    )
