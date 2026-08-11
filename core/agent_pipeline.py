"""
core/agent_pipeline.py — AgentPipeline 统一编排

三阶段 Agent 流水线:

  ┌─── Stage 1: 学习区挖掘 ────────────────────────────────────┐
  │  execute.sample_unstable()    # 多次推理，发现 learning-zone  │
  │  execute.mine_hard_negatives()# 挖掘困难负样本                │
  │  plan.extract_semantics()     # 三维语义特征提取              │
  │  plan.diff_analysis()         # 正确/错误路径差异分析          │
  │  plan.generate_principles()   # 可复用经验生成                │
  │  plan.crucible_test()         # 坩埚验证                     │
  └───────────────────────────────────────────────────────────────┘
                                  ↓
  ┌─── Stage 2: 记忆构建 ─────────────────────────────────────────┐
  │  memory.ingest()              # 经验入库 + 去重                │
  │  memory.build_indices()       # FAISS 三维向量索引             │
  │  memory.save() + snapshot()   # 持久化 + 快照                 │
  └───────────────────────────────────────────────────────────────┘
                                  ↓
  ┌─── Stage 3: 检索增强评估 ─────────────────────────────────────┐
  │  plan.extract_semantics()     # 提取测试题语义特征             │
  │  memory.retrieve()            # 三维门控检索匹配经验           │
  │  execute.answer_guided()      # 经验引导答题                  │
  │  execute.check_answer()       # 答案校验                     │
  └───────────────────────────────────────────────────────────────┘

使用方式:
  # 编程式
  pipeline = AgentPipeline(config, ports=[8000])
  await pipeline.run_full_curriculum(levels=[1,2,3,4])

  # CLI
  python -m core.agent_pipeline --levels 1 2 --ports 8000
"""
import asyncio
import json
import logging
import os
import re
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm

from core.config import Config
from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
from core.runtime_metrics import collect_runtime_metrics
from modules.plan_module import PlanModule
from modules.memory_module import MemoryModule
from modules.execute_module import (
    ExecuteModule,
    URLRouter,
    check_answer_match,
    extract_final_answer,
    filter_healthy_urls,
)
from modules.stage1_sampler import Stage1Sampler
from modules.stage1_miner import Stage1Miner
from modules.stage2_repair import RecursiveRepairProtocol
from modules.dsd_module import DSDRegistry
from output_paths import (
    stage2_memory_file,
    stage2_summary_file,
    stage1_learning_packets_file,
    stage1_mining_file,
    stage1_sampling_file,
    stage1_unified_sampling_file,
    stage1_level_packets_file,
    stage3_guided_results_file,
    stage3_guided_summary_file,
    stage3_progress_state_file,
)
from utils.io_utils import load_json, save_json
from utils.summary_utils import build_stage3_summary

logger = logging.getLogger(__name__)


STRICT_MEMORY_RECONCILE_THRESHOLD = 0.7  # lowered from 0.95: let LLM judge all semantically overlapping principles


def _build_stage3_retry_prompt(question: str, principle_text: str, previous_response: str) -> str:
    sections = [f"[Question]\n{question}"]
    if principle_text:
        sections.append(f"[Relevant Experiences]\n{principle_text}")
    sections.append(
        "[Retry Instruction]\n"
        "Your previous answer was likely incorrect.\n"
        "Re-solve the problem from scratch, re-check all constraints, and correct the final boxed answer if needed."
    )
    sections.append(f"[Previous Attempt]\n{previous_response}")
    return "\n\n".join(sections)


def _principle_text_for_memory(candidate: Dict[str, Any]) -> str:
    text = str(candidate.get("principle_text", "") or candidate.get("principle", "") or candidate.get("action_guidance", "") or "").strip()
    if text:
        return text
    parts = [str(candidate.get(key, "")).strip() for key in ("trigger_condition", "action_guidance", "scope")]
    return " ".join(part for part in parts if part)


def _normalize_principle_compare_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    return normalized.strip()


def _candidate_rank_key(candidate: Dict[str, Any]) -> tuple[float, float, int]:
    quality = candidate.get("quality_metrics", {}) if isinstance(candidate.get("quality_metrics", {}), dict) else {}
    rank_score = float(candidate.get("rank_score", quality.get("rank_score", 0.0)) or 0.0)
    neighbor_ratio = float(candidate.get("neighbor_pass_ratio", quality.get("overall_pass_ratio", 0.0)) or 0.0)
    principle_len = len(_principle_text_for_memory(candidate))
    return (rank_score, neighbor_ratio, principle_len)


def _fast_reconcile_principles(new_principle: str, existing_principle: str) -> Optional[Dict[str, str]]:
    new_norm = _normalize_principle_compare_text(new_principle)
    existing_norm = _normalize_principle_compare_text(existing_principle)
    if not new_norm or not existing_norm:
        return None
    if new_norm == existing_norm:
        return {
            "status": "heuristic",
            "decision": "merge_with_existing",
            "merged_principle": existing_principle if len(existing_principle) >= len(new_principle) else new_principle,
            "reason": "exact_text_match",
        }
    if new_norm in existing_norm:
        return {
            "status": "heuristic",
            "decision": "merge_with_existing",
            "merged_principle": existing_principle,
            "reason": "new_is_subspan_of_existing",
        }
    if existing_norm in new_norm:
        return {
            "status": "heuristic",
            "decision": "replace_existing",
            "merged_principle": new_principle,
            "reason": "new_subsumes_existing",
        }
    return None


def _normalize_level_tag(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(\d+)", text)
    return f"Level {match.group(1)}" if match else text


def _filter_test_data_by_level(test_data: List[Dict[str, Any]], eval_level: Optional[int]) -> List[Dict[str, Any]]:
    if eval_level is None:
        return test_data
    target = f"Level {int(eval_level)}"
    filtered = [item for item in test_data if _normalize_level_tag(item.get("difficulty_level") or item.get("level")) == target]
    return filtered


def _test_item_key(item: Dict[str, Any]) -> str:
    question = str(item.get("question") or item.get("problem") or "").strip()
    explicit_id = str(item.get("question_id") or item.get("id") or "").strip()
    base = explicit_id or question
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _load_stage3_progress_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"items": {}, "round_history": []}
    payload = load_json(str(state_path))
    if not isinstance(payload, dict):
        return {"items": {}, "round_history": []}
    payload.setdefault("items", {})
    payload.setdefault("round_history", [])
    return payload


def _stage3_test_scope_key(test_input: Optional[str]) -> str:
    if not test_input:
        return "default"
    path = Path(str(test_input))
    parent = path.parent.name if path.parent.name else "root"
    stem = path.stem or "dataset"
    return f"{parent}_{stem}"


def _bootstrap_stage3_progress_from_previous_round(
    progress_state: Dict[str, Any],
    *,
    root: Path,
    dataset: Optional[str],
    eval_level: Optional[int],
    test_data: List[Dict[str, Any]],
    backbone: str = "",
) -> Dict[str, Any]:
    state_items = progress_state.get("items", {})
    if state_items or eval_level is None or eval_level <= 1:
        return progress_state

    previous_dataset = None
    if dataset and re.search(r"dpo_level\d+", dataset):
        previous_dataset = re.sub(r"dpo_level(\d+)", lambda m: f"dpo_level{int(m.group(1)) - 1}", dataset)
    if not previous_dataset:
        previous_dataset = f"dpo_level{int(eval_level) - 1}"

    current_items_by_key = {_test_item_key(item): item for item in test_data}
    candidate_paths: List[Path] = []
    previous_results_path = stage3_guided_results_file(root, previous_dataset, backbone)
    if previous_results_path.exists():
        candidate_paths.append(previous_results_path)
    suite_root = root / "stage3_suite" / previous_dataset
    if suite_root.exists():
        candidate_paths.extend(sorted(suite_root.glob("*/principle_guided_test_results.json")))
    if not candidate_paths:
        return progress_state

    chosen_path: Optional[Path] = None
    previous_results: Optional[List[Dict[str, Any]]] = None
    best_overlap = 0
    for candidate_path in candidate_paths:
        try:
            candidate_results = load_json(str(candidate_path))
        except Exception:
            logger.warning("Cannot read bootstrap candidate results: %s", candidate_path)
            continue
        if not isinstance(candidate_results, list):
            continue
        overlap = 0
        for record in candidate_results:
            if not isinstance(record, dict):
                continue
            source_item = record.get("source_item")
            if not isinstance(source_item, dict) or not source_item:
                source_item = {
                    "question": record.get("question", ""),
                    "answer": record.get("expected_answer", ""),
                    "difficulty_level": record.get("difficulty_level", ""),
                }
            key = str(record.get("evaluation_key", "")).strip() or _test_item_key(source_item)
            if key in current_items_by_key:
                overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            chosen_path = candidate_path
            previous_results = candidate_results

    if not chosen_path or not previous_results or best_overlap <= 0:
        return progress_state

    boot_items: Dict[str, Any] = {}
    for record in previous_results:
        if not isinstance(record, dict):
            continue
        source_item = record.get("source_item")
        if not isinstance(source_item, dict) or not source_item:
            source_item = {
                "question": record.get("question", ""),
                "answer": record.get("expected_answer", ""),
                "difficulty_level": record.get("difficulty_level", ""),
            }
        key = str(record.get("evaluation_key", "")).strip() or _test_item_key(source_item)
        if key not in current_items_by_key:
            continue
        source_item = current_items_by_key[key]
        prior_correct = bool(record.get("ever_correct", record.get("is_correct", False)))
        boot_record = {
            **record,
            "evaluation_key": key,
            "source_item": source_item,
            "question": source_item.get("question") or source_item.get("problem") or record.get("question", ""),
            "expected_answer": source_item.get("answer", record.get("expected_answer", "")),
            "difficulty_level": _normalize_level_tag(source_item.get("difficulty_level") or source_item.get("level") or record.get("difficulty_level")),
            "carried_forward": prior_correct,
            "status": "carried_forward" if prior_correct else record.get("status", "bootstrap_unresolved"),
            "ever_correct": prior_correct,
            "first_correct_round": record.get("first_correct_round", (int(eval_level) - 1) if prior_correct else None),
        }
        boot_items[key] = {
            "evaluation_key": key,
            "question": boot_record["question"],
            "difficulty_level": boot_record["difficulty_level"],
            "source_item": source_item,
            "ever_correct": prior_correct,
            "first_correct_round": boot_record.get("first_correct_round"),
            "last_dataset": previous_dataset,
            "last_eval_level": int(eval_level) - 1,
            "record": boot_record,
        }

    if boot_items:
        progress_state["items"] = boot_items
        progress_state.setdefault("round_history", [])
        logger.info("Stage 3 bootstrap: restored %d previous-round results from %s (overlap=%d)", chosen_path, len(boot_items), best_overlap)
    return progress_state


def _build_progressive_stage3_input(
    test_data: List[Dict[str, Any]],
    eval_level: Optional[int],
    progress_state: Dict[str, Any],
) -> Dict[str, Any]:
    state_items = progress_state.get("items", {}) if isinstance(progress_state.get("items", {}), dict) else {}
    # New evaluation policy:
    # - Round 1: evaluate the full test set.
    # - Round N (N>1): evaluate only previously unresolved items.
    is_first_round = not bool(state_items)

    unresolved_records = []
    for key, payload in state_items.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("ever_correct"):
            continue
        source_item = payload.get("source_item", {})
        if isinstance(source_item, dict) and source_item:
            unresolved_records.append((key, source_item))

    eval_pool: List[Dict[str, Any]] = []
    eval_keys = set()
    if is_first_round:
        for item in test_data:
            key = _test_item_key(item)
            if key in eval_keys:
                continue
            eval_pool.append(item)
            eval_keys.add(key)
    else:
        for key, item in unresolved_records:
            if key not in eval_keys:
                eval_pool.append(item)
                eval_keys.add(key)

    carried_correct = []
    for key, payload in state_items.items():
        if not isinstance(payload, dict) or not payload.get("ever_correct"):
            continue
        record = dict(payload.get("record", {}) or {})
        if not record:
            continue
        record["status"] = "carried_forward"
        record["carried_forward"] = True
        record["evaluation_key"] = key
        carried_correct.append(record)

    return {
        "eval_pool": eval_pool,
        "carried_correct": carried_correct,
        "current_level_total": len(eval_pool),
        "unresolved_before_round": len(unresolved_records),
    }


def _merge_stage3_progress_state(
    progress_state: Dict[str, Any],
    evaluated_results: List[Dict[str, Any]],
    carried_correct: List[Dict[str, Any]],
    dataset: str,
    eval_level: Optional[int],
) -> Dict[str, Any]:
    state_items = progress_state.get("items", {}) if isinstance(progress_state.get("items", {}), dict) else {}

    for record in carried_correct:
        key = str(record.get("evaluation_key", "")).strip()
        if not key or key not in state_items:
            continue
        payload = state_items[key]
        payload["record"] = {**payload.get("record", {}), **record}
        payload["last_dataset"] = dataset
        payload["last_eval_level"] = eval_level

    for result in evaluated_results:
        key = str(result.get("evaluation_key", "")).strip()
        if not key:
            continue
        existing = state_items.get(key, {})
        previously_correct = bool(existing.get("ever_correct", False))
        now_correct = bool(result.get("is_correct", False))
        first_correct_round = existing.get("first_correct_round")
        if now_correct and first_correct_round is None:
            first_correct_round = eval_level

        state_items[key] = {
            "evaluation_key": key,
            "question": result.get("question", ""),
            "difficulty_level": result.get("difficulty_level", ""),
            "source_item": existing.get("source_item", result.get("source_item", {})),
            "ever_correct": previously_correct or now_correct,
            "first_correct_round": first_correct_round,
            "last_dataset": dataset,
            "last_eval_level": eval_level,
            "record": {
                **result,
                "ever_correct": previously_correct or now_correct,
                "first_correct_round": first_correct_round,
                "carried_forward": False,
            },
        }

    progress_state["items"] = state_items
    history = progress_state.get("round_history", [])
    if not isinstance(history, list):
        history = []
    history.append({"dataset": dataset, "eval_level": eval_level})
    progress_state["round_history"] = history
    return progress_state


def _build_stage3_complete_results(
    test_data: List[Dict[str, Any]],
    evaluated_results: List[Dict[str, Any]],
    carried_correct: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result_by_key: Dict[str, Dict[str, Any]] = {}
    for record in carried_correct:
        if isinstance(record, dict):
            key = str(record.get("evaluation_key", "")).strip()
            if key:
                result_by_key[key] = dict(record)
    for record in evaluated_results:
        if isinstance(record, dict):
            key = str(record.get("evaluation_key", "")).strip()
            if key:
                result_by_key[key] = dict(record)

    complete_results: List[Dict[str, Any]] = []
    for idx, item in enumerate(test_data):
        key = _test_item_key(item)
        record = dict(result_by_key.get(key, {}))
        if not record:
            record = {
                "status": "missing_from_progressive_eval",
                "is_correct": False,
                "retrieval_used": False,
                "retrieval_count": 0,
                "attempt_count": 0,
                "retry_used": False,
                "rescued_by_retry": False,
                "first_attempt_correct": False,
                "carried_forward": False,
            }
        record.setdefault("status", "success")
        record["idx"] = idx
        record["evaluation_key"] = key
        record["source_item"] = item
        record["question"] = item.get("question") or item.get("problem") or record.get("question", "")
        record["expected_answer"] = item.get("answer", record.get("expected_answer", ""))
        record["difficulty_level"] = _normalize_level_tag(item.get("difficulty_level") or item.get("level") or record.get("difficulty_level"))
        record.setdefault("carried_forward", False)
        complete_results.append(record)
    return complete_results


class AgentPipeline:
    """经验驱动的自进化推理 Agent。

    三大模块:
      - plan:    规划 (语义提取, diff, principle 生成)
      - memory:  记忆 (入库, 检索, 索引)
      - execute: 执行 (采样, 挖掘, 答题)
    """

    def __init__(
        self,
        config: Config,
        ports: Optional[Sequence[int]] = None,
        model_override: Optional[str] = None,
        memory_dir: str = "memory/v3",
        timeout: int = 600,
    ):
        self.config = config
        urls = URLRouter.build_urls(config.hyperparams.weak_model_url, ports) if ports else None
        if urls:
            urls = filter_healthy_urls(urls)
        self._urls = urls

        self.plan = PlanModule(
            config, urls=urls, model_override=model_override, timeout=timeout,
        )
        self.memory = MemoryModule(config, memory_dir=memory_dir)
        self.execute = ExecuteModule(
            config, ports=ports, urls=urls, model_override=model_override, timeout=timeout,
        )

    # ================================================================
    # Stage 1: 学习区挖掘 (新版)
    # ================================================================

    async def stage1_learn(
        self,
        dataset_path: Optional[str] = None,
        concurrency: int = 10,
        diff_concurrency: int = 10,
        k_trials: int = 10,
        temperature: float = 0.7,
        skip_sampling: bool = False,
        backbone: str = "",
        urls_override: Optional[List[str]] = None,
        model_override: Optional[str] = None,
        resume: bool = False,
        collect_all_samples: bool = False,
    ) -> Optional[Path]:
        """Stage 1: 全量采样 → 三区划分 → 等分四级 → mining + diff + 打包.

        Args:
            backbone: 模型标识,用于输出路径分离 (e.g. "qwen", "llama")
            urls_override: 覆盖默认 URL
            model_override: 覆盖默认 model
            resume: 启用 checkpoint 续跑, 中断后可恢复
            collect_all_samples: True = 收集所有 zone 样本 (No Learning Zone 消融)
        """
        tag = backbone or "default"
        with collect_runtime_metrics("stage1", {"backbone": tag}) as runtime:
            output_root = self.config.paths.output_dir
            _urls = urls_override or self.execute.urls
            _model = model_override or self.execute.model

            # ── 加载数据集 ──
            src = dataset_path or str(
                Path(self.config.paths.output_dir).parent / "data" / "deepscaler_merged_6000.json"
            )
            if not Path(src).exists():
                logger.error("Unified dataset not found: %s", src)
                return None
            tasks = load_json(src)
            if not isinstance(tasks, list) or not tasks:
                logger.error("Dataset is empty: %s", src)
                return None
            logger.info("Stage 1 [%s]: loaded %d tasks", tag, len(tasks))

            # ── Step 1.1: 采样 ──
            sampling_path = stage1_unified_sampling_file(output_root, backbone)
            sampling_ckpt = str(sampling_path).replace(".json", ".checkpoint.json")
            if skip_sampling and sampling_path.exists():
                logger.info("[Stage 1.1 %s skipped] using existing: %s", tag, sampling_path)
                sampling_data = load_json(str(sampling_path))
            else:
                # 已有完整文件直接复用（total>0 才认为有效）
                if not resume and sampling_path.exists():
                    existing = load_json(str(sampling_path))
                    if existing.get("summary", {}).get("total", 0) > 0:
                        logger.info("[Stage 1.1 %s] final file already exists (total=%d): %s",
                                     tag, existing["summary"]["total"], sampling_path)
                        sampling_data = existing
                    else:
                        logger.warning("[Stage 1.1 %s] empty/corrupt, deleting and re-sampling: %s",
                                       tag, sampling_path)
                        sampling_path.unlink(missing_ok=True)
                        # fall through to sampling below
                        sampling_data = None
                else:
                    sampling_data = None

                if sampling_data is None:
                    # 自动检测 checkpoint: 有则续跑，无则全新
                    ckpt_file = Path(sampling_ckpt)
                    if not resume and ckpt_file.exists():
                        logger.info("[Stage 1.1 %s] checkpoint detected, auto-resuming", tag)
                        resume = True

                    sampler = Stage1Sampler(
                        urls=_urls,
                        model=_model,
                        api_key=self.execute.api_key,
                        k_trials=k_trials,
                        temperature=temperature,
                        max_tokens=4096,
                        timeout=self.execute.timeout,
                    )
                    sampling_data = await sampler.sample_all(
                        tasks, concurrency=concurrency,
                        checkpoint_path=sampling_ckpt,
                        collect_all_samples=collect_all_samples,
                    )
                    Stage1Sampler.save(sampling_data, sampling_path)
                    # 跑完后删除 checkpoint
                    if ckpt_file.exists():
                        ckpt_file.unlink()

            summary = sampling_data.get("summary", {})
            level_data = sampling_data.get("level_data", {})

            # When collect_all_samples=True, regenerate level_data from full results
            # (needed when reusing sampling data originally created without this flag)
            if collect_all_samples:
                all_results = sampling_data.get("results", [])
                if all_results:
                    level_data = {"1": [], "2": [], "3": [], "4": []}
                    for r in all_results:
                        if r.get("zone") == "no_data":
                            continue
                        level = Stage1Sampler.level_from_result(r)
                        if level <= 0:
                            continue
                        r["assigned_level"] = level
                        level_data[str(level)].append(r)
                    logger.info(
                        "[%s] collect_all_samples: regenerated level_data from %d results → L1=%d L2=%d L3=%d L4=%d",
                        tag, len(all_results),
                        len(level_data["1"]), len(level_data["2"]),
                        len(level_data["3"]), len(level_data["4"]),
                    )

            if not level_data:
                logger.error("[%s] no samples to process (all zones empty)", tag)
                return None

            # ── Step 1.2: Mining + Diff + 打包 ──
            miner = Stage1Miner(
                urls=_urls,
                model=_model,
                api_key=self.execute.api_key,
                timeout=self.execute.timeout,
            )
            mining_result = await miner.run(level_data, concurrency=diff_concurrency)

            packets = mining_result.get("packets", {})
            stats = mining_result.get("stats", {})

            # 保存 —— 按 backbone 分目录
            packets_dir = stage1_level_packets_file(output_root, 1, backbone).parent
            Stage1Miner.save(packets, packets_dir)

            summary_path = packets_dir / "stage1_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                import json as _j
                _j.dump({
                    "backbone": tag, "model": _model,
                    "sampling": summary, "mining": stats,
                    "levels": {lv: len(packets.get(f"level{lv}", [])) for lv in ("1", "2", "3", "4")},
                }, f, ensure_ascii=False, indent=2)

            metrics_path = packets_dir / "stage1_runtime_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                import json as _jm
                _jm.dump(runtime.finish(), f, ensure_ascii=False, indent=2)

            logger.info(
                "Stage 1 [%s] done: total=%d | M=%d L=%d P=%d | pairs=%d (diff_ok=%d)",
                tag, summary.get("total", 0),
                summary.get("master_zone", 0), summary.get("learning_zone", 0),
                summary.get("panic_zone", 0),
                stats.get("total_pairs_found", 0), stats.get("diff_success", 0),
            )
            return packets_dir

    # ================================================================
    # Stage 1b: 后续轮次 Panic 区重采样
    # ================================================================

    async def stage1_panic_resample(
        self,
        backbone: str,
        k_trials: int = 10,
        concurrency: int = 64,
        diff_concurrency: int = 64,
    ) -> Dict[str, Any]:
        """Round 2+ Stage 1: 只对 panic 区样本重新 K-trial 采样。

        与首轮 Stage 1 不同:
        - 只处理 zone=="panic" 的样本 (不做全量)
        - 用当前 memory 引导答题 (已有经验积累)
        - SR > 0 → 按首轮相同阈值分 level, mine pairs + diff
        - 追加到对应 level 的 learning packet, 更新 unified_sampling_results
        """
        import json as _json

        output_root = self.config.paths.output_dir
        sampling_path = stage1_unified_sampling_file(output_root, backbone)
        if not sampling_path.exists():
            logger.warning("Sampling file not found: %s", sampling_path)
            return {"status": "no_sampling_file", "rescued": 0, "total_panic": 0}

        data = load_json(str(sampling_path))
        all_results: List[Dict[str, Any]] = data.get("results", [])
        panic_samples = [r for r in all_results if r.get("zone") == "panic"]
        if not panic_samples:
            logger.info("Stage 1b [%s]: no panic samples, skipping resample", backbone)
            return {"status": "no_panic", "rescued": 0, "total_panic": 0}

        logger.info(
            "Stage 1b [%s]: panic resample (panic=%d, K=%d, memory=%d)",
            backbone, len(panic_samples), k_trials, self.memory.principle_count,
        )

        urls = self.execute.urls
        model = self.execute.model
        api_key = self.execute.api_key
        timeout = self.execute.timeout
        semaphore = asyncio.Semaphore(concurrency)

        async def resample_one(sample: Dict[str, Any]) -> Dict[str, Any]:
            question = sample.get("question", "")
            expected = sample.get("expected_answer", "")
            async with semaphore:
                semantics = await self.plan.extract_semantics(question)

            principle_text = ""
            if self.memory.principle_count > 0:
                retrieved = self.memory.retrieve(semantics, top_k=3, threshold=0.5)
                passing = [r for r in retrieved if r.get("item", {}).get("principle")]
                if passing:
                    blocks = [f"[Experience {i}]\n- guidance: {r['item']['principle']}"
                              for i, r in enumerate(passing, 1)]
                    principle_text = "\n\n".join(blocks)

            correct_answers, wrong_answers, all_answers, trial_records = [], [], [], []
            sampler = Stage1Sampler(
                urls=urls, model=model, api_key=api_key,
                k_trials=k_trials, temperature=0.7,
                max_tokens=4096, timeout=timeout,
            )
            for trial in range(k_trials):
                entropy = {
                    "available": False,
                    "normalized_entropy": 1.0,
                    "trajectory_entropy": 1.0,
                    "token_count": 0,
                    "entropy_token_count": 0,
                    "avg_neg_logprob": None,
                    "method": "guided_text_only_default_max_entropy",
                }
                async with semaphore:
                    if principle_text:
                        response = await self.execute.answer_guided(question, principle_text)
                    else:
                        metadata = await sampler._single_inference_with_metadata(question)
                        response = metadata.get("text", "")
                        entropy = metadata.get("entropy") or entropy
                if not response:
                    response = "[EMPTY_RESPONSE]"
                is_correct = bool(response and self.execute.check_answer(response, expected))
                all_answers.append(response)
                trial_records.append({
                    "trial": trial + 1,
                    "response": response,
                    "is_correct": is_correct,
                    "extracted_answer": None,
                    "normalized_entropy": entropy.get("normalized_entropy", 1.0),
                    "trajectory_entropy": entropy.get("trajectory_entropy", entropy.get("normalized_entropy", 1.0)),
                    "entropy_available": bool(entropy.get("available")),
                    "entropy_method": entropy.get("method", "unknown"),
                    "token_count": entropy.get("token_count", 0),
                    "entropy_token_count": entropy.get("entropy_token_count", 0),
                    "avg_neg_logprob": entropy.get("avg_neg_logprob"),
                })
                if is_correct:
                    correct_answers.append(response)
                else:
                    wrong_answers.append(response)

            correct_count = len(correct_answers)
            result = {
                **sample,
                "all_answers": all_answers,
                "correct_trajectories": correct_answers,
                "wrong_trajectories": wrong_answers,
                "correct_count": correct_count,
                "wrong_count": len(wrong_answers),
                "trial_records": trial_records,
                "planned_trials": k_trials,
                "actual_trials": len(trial_records),
                "early_stopped": False,
            }
            return sampler.finalize_result(result)

        tasks = [resample_one(s) for s in panic_samples]
        rescued: List[Dict] = []
        still_panic: List[Dict] = []
        for coro in async_tqdm(
            asyncio.as_completed(tasks), total=len(tasks),
            desc=f"Stage 1b Panic Resample ({backbone})",
        ):
            r = await coro
            if r.get("zone") in {"learning", "master"}:
                rescued.append(r)
            else:
                still_panic.append(r)

        logger.info(
            "Stage 1b resample: %d/%d entered learning zone",
            len(rescued), len(panic_samples),
        )

        if not rescued:
            return {"status": "no_rescued", "rescued": 0, "total_panic": len(panic_samples)}

        # ── 更新 unified_sampling_results ──
        by_idx = {r["idx"]: r for r in rescued}
        for i, s in enumerate(all_results):
            if s.get("zone") == "panic" and s["idx"] in by_idx:
                updated = by_idx[s["idx"]]
                all_results[i] = updated

        backup = sampling_path.with_suffix(".json.bak")
        sampling_path.rename(backup)
        sampling_path.write_text(_json.dumps(
            {"results": all_results, "summary": data.get("summary", {})},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")

        # ── 按 level 分组, mine pairs + diff, 追加到 packet 文件 ──
        level_groups: Dict[int, List[Dict]] = {}
        for r in rescued:
            lv = r["assigned_level"]
            level_groups.setdefault(lv, []).append(r)

        miner = Stage1Miner(urls=urls, model=model, api_key=api_key, timeout=timeout)
        total_new_packets = 0
        for lv in sorted(level_groups):
            samples = level_groups[lv]
            mining_result = await miner.run(
                {str(lv): samples}, concurrency=diff_concurrency,
            )
            new_packets = mining_result.get("packets", {}).get(str(lv), [])

            packet_path = stage1_level_packets_file(output_root, lv, backbone)
            existing = []
            if packet_path.exists():
                existing = load_json(str(packet_path))
                if not isinstance(existing, list):
                    existing = []

            existing_ids = {str(p.get("question_id", p.get("idx", ""))) for p in existing}
            new_unique = [p for p in new_packets if str(p.get("idx", "")) not in existing_ids]
            existing.extend(new_unique)

            packet_path.parent.mkdir(parents=True, exist_ok=True)
            with open(packet_path, "w", encoding="utf-8") as f:
                _json.dump(existing, f, ensure_ascii=False, indent=2)
            total_new_packets += len(new_unique)
            logger.info("Stage 1b: level%d +%d packets (deduped)", lv, len(new_unique))

        result = {
            "status": "ok",
            "rescued": len(rescued),
            "total_panic": len(panic_samples),
            "new_packets": total_new_packets,
            "levels_updated": sorted(level_groups.keys()),
        }

        # Save panic resample summary
        summary_path = output_root / "stage1" / "02_learning_packets" / backbone / "panic_resample_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            _json.dump({
                "backbone": backbone,
                "k_trials": k_trials,
                "memory_principles": self.memory.principle_count,
                **result,
                "rescued_by_level": {str(lv): len(level_groups.get(lv, [])) for lv in sorted(level_groups)},
            }, f, ensure_ascii=False, indent=2)
        logger.info("Stage 1b summary saved: %s", summary_path)

        return result

    # ================================================================
    # Stage 2: 记忆构建
    # ================================================================

    async def stage2_memorize(
        self,
        dataset: str,
        round_id: int,
        validation_concurrency: int = 8,
        backbone: str = "",
        max_records: int = 0,
        run_tag: str = "",
        dsd_cache_dir: str = "",
        rebuild_dsd: bool = False,
        prepare_dsd_only: bool = False,
        prepare_candidates_only: bool = False,
        candidate_bank: str = "",
        neighbor_validation: bool = True,
        gate_only: bool = False,
        selection_file: str = "",
        artifact_dir: str = "",
        candidate_concurrency: int = 0,
    ) -> Dict[str, Any]:
        """Stage 2 v3: causally validate and version reusable memories."""
        from modules.stage2_v3 import run_stage2_v3

        return await run_stage2_v3(
            self,
            dataset=dataset,
            round_id=round_id,
            validation_concurrency=validation_concurrency,
            backbone=backbone,
            max_records=max_records,
            run_tag=run_tag,
            dsd_cache_dir=dsd_cache_dir,
            rebuild_dsd=rebuild_dsd,
            prepare_dsd_only=prepare_dsd_only,
            prepare_candidates_only=prepare_candidates_only,
            candidate_bank=candidate_bank,
            neighbor_validation=neighbor_validation,
            gate_only=gate_only,
            selection_file=selection_file,
            artifact_dir=artifact_dir,
            candidate_concurrency=candidate_concurrency,
        )

    async def stage2_memorize_legacy(
        self,
        dataset: str,
        round_id: int,
        validation_concurrency: int = 8,
        backbone: str = "",
    ) -> Dict[str, Any]:
        """Stage 2: DSD-guided principle selection and memory construction."""
        metrics_context = collect_runtime_metrics("stage2", {"dataset": dataset, "round_id": round_id})
        runtime = metrics_context.__enter__()
        self.memory.load()
        initial = self.memory.principle_count
        if initial > 0 and not self.memory.is_indexed:
            self.memory.build_indices()
        logger.info("Stage 2: memory build (dataset=%s, round=%d, backbone=%s, existing=%d)", dataset, round_id, backbone or "default", initial)

        # 优先使用新版路径 (output/stage1/02_learning_packets/{backbone}/level{N}_...), 回退旧路径
        _bk = backbone or os.environ.get("MEMUA_BACKBONE", "")
        _level_match = re.search(r"(\d+)", str(dataset))
        _level = int(_level_match.group(1)) if _level_match else round_id
        lp_path_new = stage1_level_packets_file(self.config.paths.output_dir, _level, _bk) if _bk else None
        lp_path_old = stage1_learning_packets_file(self.config.paths.output_dir, dataset)
        if lp_path_new and lp_path_new.exists():
            lp_path = lp_path_new
        else:
            lp_path = lp_path_old
        logger.info("Stage 2: reading learning packets: %s", lp_path)
        packets = load_json(str(lp_path)) if lp_path.exists() else []
        if not isinstance(packets, list) or not packets:
            logger.warning("Stage 1 output missing or empty, skipping Stage 2")
            summary = {"dataset": dataset, "round_id": round_id, "status": "empty_stage1"}
            summary["runtime_metrics"] = runtime.finish()
            metrics_context.__exit__(None, None, None)
            return summary

        registry = DSDRegistry(dataset=dataset, round_id=round_id, output_root=self.config.paths.output_dir, backbone=backbone)
        dsd_concurrency = max(1, int(os.environ.get("MEMUA_STAGE2_DSD_CONCURRENCY", "32")))
        logger.info("Stage 2: building DSD registry from %d learning packets", len(packets))
        registry_summary = await registry.build(packets, self.plan, concurrency=dsd_concurrency)
        registry.save()
        logger.info(
            "Stage 2: DSD registry ready with %d records; starting principle validation (dsd_concurrency=%d)",
            len(registry.records),
            dsd_concurrency,
        )

        stage2_units: List[Dict[str, Any]] = []
        stats = {"added": 0, "updated": 0, "duplicated": 0, "skipped": 0}
        stage2_neighbor_pass_ratio_threshold = float(
            os.environ.get("MEMUA_STAGE2_NEIGHBOR_PASS_RATIO_THRESHOLD", "0.6")
        )
        stage2_neighbor_top_k = max(1, int(os.environ.get("MEMUA_STAGE2_NEIGHBOR_TOP_K", "5")))
        max_principle_rounds = max(1, int(os.environ.get("MEMUA_STAGE2_MAX_GENERATION_ROUNDS", "3")))
        max_candidates_per_round = max(1, int(os.environ.get("MEMUA_STAGE2_MAX_CANDIDATES_PER_ROUND", "2")))
        candidate_total = 0
        source_pass_total = 0
        full_neighbor_pass_total = 0
        total_generation_rounds = 0
        attempted_stage2_units = 0

        def _principle_text(candidate: Dict[str, Any]) -> str:
            text = str(candidate.get("principle", "") or candidate.get("action_guidance", "") or "").strip()
            if text:
                return text
            parts = [str(candidate.get(key, "")).strip() for key in ("trigger_condition", "action_guidance", "scope")]
            return " ".join(part for part in parts if part)

        generation_semaphore = asyncio.Semaphore(max(1, int(os.environ.get("MEMUA_STAGE2_GENERATION_CONCURRENCY", str(validation_concurrency)))))
        reconcile_semaphore = asyncio.Semaphore(max(1, int(os.environ.get("MEMUA_STAGE2_RECONCILE_CONCURRENCY", "4"))))

        async def _run_reconcile_judge(
            question: str,
            new_principle: str,
            existing_question: str,
            existing_principle: str,
        ) -> Dict[str, Any]:
            async with reconcile_semaphore:
                return await self.plan.judge_principle_consistency(
                    question=question,
                    new_principle=new_principle,
                    existing_question=existing_question,
                    existing_principle=existing_principle,
                )

        stage2_repair_max_depth = max(1, int(os.environ.get("MEMUA_STAGE2_REPAIR_MAX_DEPTH", "3")))

        async def process_stage2_record(record: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, int]]:
            local_stats = {
                "candidate_total": 0,
                "source_pass_total": 0,
                "full_neighbor_pass_total": 0,
                "total_generation_rounds": 0,
                "repair_rounds": 0,
                "attempted_stage2_units": 0,
            }

            diff_ok = ((record.get("contrast") or {}).get("status") == "success")
            has_pair = all([
                str(record.get("question_text", "")).strip(),
                str(record.get("expected_answer", "")).strip(),
                str(record.get("chosen_trajectory", "")).strip(),
                str(record.get("hard_negative_trajectory", "")).strip(),
            ])
            source_dsd = record.get("source_dsd") or {}
            has_semantics = any(
                str(source_dsd.get(key, "")).strip()
                for key in (CORE_FIELD, CONDITION_FIELD, TARGET_FIELD)
            )
            stage2_candidate = bool(record.get("stage2_candidate", False)) or bool(
                (record.get("source_metadata") or {}).get("stage2_candidate", False)
            )
            if not stage2_candidate:
                stage2_candidate = diff_ok and has_pair

            if not stage2_candidate:
                return {"question_id": record["question_id"], "status": "skipped", "reason": "not_stage2_candidate"}, local_stats
            if not has_semantics:
                return {"question_id": record["question_id"], "status": "skipped", "reason": "missing_semantics"}, local_stats

            neighbors = registry.get_neighbors(record["question_id"], top_k=stage2_neighbor_top_k)
            if not neighbors:
                return {"question_id": record["question_id"], "status": "skipped", "reason": "no_neighbors"}, local_stats

            stage2_unit: Dict[str, Any] = {
                "question_id": record["question_id"],
                "source_question_text": record["question_text"],
                "question_text": record["question_text"],
                "expected_answer": record["expected_answer"],
                "source_dsd": record["source_dsd"],
                "neighbor_questions": neighbors,
                "status": "pending",
            }
            local_stats["attempted_stage2_units"] += 1

            # ── Step 1: 获取 diff (复用 Stage 1 或重新分析) ──
            stage1_diff = (record.get("contrast") or {}).get("differences", [])
            if isinstance(stage1_diff, list) and stage1_diff:
                diff_list = stage1_diff
            else:
                async with generation_semaphore:
                    diff_result = await self.plan.diff_analysis(
                        record["question_text"],
                        record.get("chosen_trajectory", ""),
                        record.get("hard_negative_trajectory", ""),
                        record["source_dsd"],
                        record.get("divergence_step_text", ""),
                        int(record.get("divergence_step_idx", 0) or 0),
                    )
                    diff_list = diff_result.get("differences", []) if diff_result.get("status") == "success" else []

            if not diff_list:
                stage2_unit["status"] = "no_diff"
                return stage2_unit, local_stats

            # ── Step 2: Self-CPE — 从 diff + 内省提取一条防御性原则 ──
            async with generation_semaphore:
                cpe_result = await self.plan.self_cpe_extract(
                    question=record["question_text"],
                    diff_list=diff_list,
                    source_dsd=record["source_dsd"],
                )

            if cpe_result.get("status") != "success":
                stage2_unit["status"] = "cpe_failed"
                return stage2_unit, local_stats

            local_stats["candidate_total"] += 1
            principle = cpe_result.get("principle", {})

            # ── Step 3: 邻域验证 + 递归修复协议 ──
            repair = RecursiveRepairProtocol(
                plan_module=self.plan,
                max_depth=stage2_repair_max_depth,
                pass_threshold=stage2_neighbor_pass_ratio_threshold,
                validation_concurrency=validation_concurrency,
            )
            repair_result = await repair.run(
                principle=principle,
                neighbors=neighbors,
                source_dsd=record["source_dsd"],
            )

            stage2_unit["repair_result"] = {
                "status": repair_result["status"],
                "repair_history": repair_result.get("repair_history", []),
            }
            local_stats["repair_rounds"] += len(repair_result.get("repair_history", []))

            if repair_result["status"] != "validated":
                local_stats["total_generation_rounds"] += 1 + local_stats["repair_rounds"]

                stage2_unit["status"] = "repair_discarded"
                return stage2_unit, local_stats

            # ── Step 4: 组装 validated_principle (兼容现有 memory 接口) ──
            final_validation = repair_result.get("final_validation", {})
            final_principle = repair_result["principle"]
            neighbor_results_raw = final_validation.get("results", [])
            neighbor_pass_count = final_validation.get("pass_count", 0)
            neighbor_total = final_validation.get("total", len(neighbors))
            neighbor_pass_ratio = final_validation.get("pass_ratio", 0.0)

            # 转为兼容格式
            neighbor_results = []
            for nr in neighbor_results_raw:
                neighbor_results.append({
                    "question_id": nr.get("neighbor_id", ""),
                    "question_text": nr.get("neighbor_question", ""),
                    "passed": nr.get("is_correct", False),
                    "response": nr.get("response", ""),
                    "attempt_count": nr.get("attempt_count", 1),
                    "rescued_by_retry": nr.get("rescued_by_retry", False),
                })

            principle_text = _principle_text_for_memory(final_principle)
            if not principle_text:
                parts = [
                    str(final_principle.get("trigger_condition", "")).strip(),
                    str(final_principle.get("action_guidance", "")).strip(),
                    str(final_principle.get("scope", "")).strip(),
                ]
                principle_text = " ".join(part for part in parts if part)

            validated_principle = {
                **final_principle,
                "content": dict(final_principle),
                "principle": principle_text,
                "principle_text": principle_text,
                "source_question_id": record["question_id"],
                "source_question_text": record["question_text"],
                "source_dsd": record["source_dsd"],
                "neighbor_pass_count": neighbor_pass_count,
                "neighbor_total_count": neighbor_total,
                "neighbor_pass_ratio": neighbor_pass_ratio,
                "neighbor_gate_pass": True,
                "neighbor_cases": neighbor_results,
                "validated_neighbors": neighbor_results,
                "qualified_for_memory": True,
                "rank_score": neighbor_pass_ratio,
                "status": "active",
                "round_id": round_id,
                "source_round": round_id,
                "source_dataset": dataset,
            }
            local_stats["full_neighbor_pass_total"] += 1
            local_stats["total_generation_rounds"] += 1 + local_stats["repair_rounds"]

            stage2_unit["validated_principle"] = validated_principle
            stage2_unit["status"] = "validated"

            return stage2_unit, local_stats

        validation_records = [
            record for record in registry.records
            if record.get("stage2_candidate", False) and ((record.get("contrast") or {}).get("status") == "success")
        ]
        logger.info(
            "Stage 2: validating %d candidate records (neighbor_top_k=%d, max_candidates_per_round=%d)",
            len(validation_records),
            stage2_neighbor_top_k,
            max_candidates_per_round,
        )
        validation_tasks = [process_stage2_record(record) for record in validation_records]
        done_count = 0
        total_count = len(validation_tasks)
        for coro in async_tqdm(asyncio.as_completed(validation_tasks), total=total_count, desc=f"Stage 2 validation ({dataset})"):
            stage2_unit, local_stats = await coro
            stage2_units.append(stage2_unit)
            candidate_total += local_stats["candidate_total"]
            source_pass_total += local_stats["source_pass_total"]
            full_neighbor_pass_total += local_stats["full_neighbor_pass_total"]
            total_generation_rounds += local_stats["total_generation_rounds"]
            attempted_stage2_units += local_stats["attempted_stage2_units"]
            done_count += 1
            if done_count % 10 == 0 or done_count == total_count:
                logger.info("Stage 2 validation progress: %d/%d (%s)", done_count, total_count, dataset)

        stage2_units.sort(key=lambda item: str(item.get("question_id", "")))

        stats = {"added": 0, "updated": 0, "duplicated": 0, "skipped": 0, "merged": 0, "replaced": 0}
        validated_principles = [unit["validated_principle"] for unit in stage2_units if unit.get("status") == "validated" and unit.get("validated_principle")]
        pending_memory_candidates = sorted(validated_principles, key=_candidate_rank_key, reverse=True)
        if validated_principles:
            logger.info("Stage 2: reconciling %d validated principles before memory ingest", len(validated_principles))
            if self.memory.principle_count > 0 and not self.memory.is_indexed:
                self.memory.build_indices()

            for principle in pending_memory_candidates:
                strict_matches = self.memory.retrieve_strict_semantic_matches(
                    principle.get("source_dsd", {}),
                    facet_threshold=STRICT_MEMORY_RECONCILE_THRESHOLD,
                    top_k=5,
                )

                if not strict_matches:
                    current = self.memory.ingest_validated_principles(
                        [principle], round_id, dataset, auto_dedup=False,
                    )
                    for key in ("added", "updated", "duplicated", "skipped"):
                        stats[key] += current.get(key, 0)
                    continue

                existing = strict_matches[0]["item"]
                existing_principle = str(existing.get("principle", "")).strip()
                new_principle = _principle_text_for_memory(principle)
                decision = _fast_reconcile_principles(new_principle, existing_principle)
                if decision is None:
                    decision = await _run_reconcile_judge(
                        question=str(principle.get("source_question_text", "")).strip(),
                        new_principle=new_principle,
                        existing_question=str(existing.get("question", "")).strip(),
                        existing_principle=existing_principle,
                    )
                principle["memory_reconciliation"] = {
                    "strict_matches": strict_matches,
                    "decision": decision,
                }

                action = decision.get("decision", "add_as_new")
                existing_id = str(existing.get("memory_id", "")).strip()
                merged_principle = str(decision.get("merged_principle", "")).strip() or new_principle

                if action == "replace_existing" and existing_id:
                    replacement_content = dict(principle.get("content", {}))
                    replacement_content["action_guidance"] = merged_principle
                    self.memory.memory.update(existing_id, {
                        "principle": merged_principle,
                        "principle_text": merged_principle,
                        "content": replacement_content,
                        **replacement_content,
                        "source_round": round_id,
                        "source_dataset": dataset,
                        "source_dsd": principle.get("source_dsd", {}),
                        "validated_neighbors": principle.get("validated_neighbors", []),
                        "quality_metrics": {
                            **(existing.get("quality_metrics", {}) or {}),
                            **(principle.get("quality_metrics", {}) or {}),
                            "memory_reconcile_action": "replace_existing",
                            "memory_reconcile_reason": decision.get("reason", ""),
                        },
                        "operation": "REPLACE",
                        "update_reason": decision.get("reason", ""),
                    })
                    stats["updated"] += 1
                    stats["replaced"] += 1
                elif action == "merge_with_existing" and existing_id:
                    merged_content = dict(principle.get("content", {}))
                    merged_content["action_guidance"] = merged_principle
                    self.memory.memory.update(existing_id, {
                        "principle": merged_principle,
                        "principle_text": merged_principle,
                        "content": merged_content,
                        **merged_content,
                        "validated_neighbors": principle.get("validated_neighbors", []),
                        "quality_metrics": {
                            **(existing.get("quality_metrics", {}) or {}),
                            **(principle.get("quality_metrics", {}) or {}),
                            "memory_reconcile_action": "merge_with_existing",
                            "memory_reconcile_reason": decision.get("reason", ""),
                        },
                        "operation": "MERGE",
                        "update_reason": decision.get("reason", ""),
                    })
                    stats["updated"] += 1
                    stats["merged"] += 1
                else:
                    current = self.memory.ingest_validated_principles(
                        [principle], round_id, dataset, auto_dedup=False,
                    )
                    for key in ("added", "updated", "duplicated", "skipped"):
                        stats[key] += current.get(key, 0)
        else:
            logger.warning("No validated principles passed the Stage 2 gate for %s", dataset)

        # 构建索引
        logger.info("Stage 2: saving final memory indices and snapshot for %s", dataset)
        self.memory.build_indices()
        self.memory.save_indices()

        # 保存 + 快照
        self.memory.save()
        self.memory.snapshot(round_id)

        final = self.memory.principle_count
        summary = {
            "dataset": dataset,
            "round_id": round_id,
            "total_stage2_units": len(stage2_units),
            "validation_records": len(validation_records),
            "neighbor_top_k": stage2_neighbor_top_k,
            "max_candidates_per_round": max_candidates_per_round,
            "total_candidates_generated": candidate_total,
            "source_pass_rate": (source_pass_total / candidate_total) if candidate_total else 0.0,
            "full_neighbor_pass_rate": (full_neighbor_pass_total / candidate_total) if candidate_total else 0.0,
            "average_generation_rounds": (total_generation_rounds / attempted_stage2_units) if attempted_stage2_units else 0.0,
            "pending_memory_candidates": len(pending_memory_candidates),
            "principles_inserted": stats.get("added", 0),
            "principles_updated": stats.get("updated", 0),
            "principles_merged": stats.get("merged", 0),
            "principles_replaced": stats.get("replaced", 0),
            "principles_rejected": stats.get("duplicated", 0) + stats.get("skipped", 0),
            "memory_before": initial,
            "memory_after": final,
            "registry_summary": registry_summary,
        }
        summary["runtime_metrics"] = runtime.finish()
        save_json(stage2_units, str(stage2_memory_file(self.config.paths.output_dir, dataset, backbone)))
        save_json(summary, str(stage2_summary_file(self.config.paths.output_dir, dataset, backbone)))
        metrics_context.__exit__(None, None, None)
        logger.info("Stage 2 done: %d → %d principles, round_%d snapshotted", initial, final, round_id)
        return summary

    # ================================================================
    # Stage 3: 检索增强评估
    # ================================================================

    async def stage3_evaluate(
        self,
        test_data: List[Dict[str, Any]],
        threshold: float = 0.55,
        top_k: int = 5,
        concurrency: int = 10,
        mode: str = "guided",
        retry_on_wrong: int = 2,
        eval_level: Optional[int] = None,
        dataset: Optional[str] = None,
        progress_state_path: Optional[Path] = None,
        test_scope_key: Optional[str] = None,
        backbone: str = "",
    ) -> Dict[str, Any]:
        """Stage 3: 使用记忆进行检索增强推理评估。

        流程:
          1. plan.extract_semantics()     — 生成测试题语义特征
          2. memory.retrieve()            — 三维门控检索匹配经验
          3. execute.answer_guided()      — 经验引导答题
          4. execute.check_answer()       — 答案校验

        Returns:
            {"results": [...], "summary": {...}}
        """
        metrics_context = collect_runtime_metrics(
            "stage3",
            {"dataset": dataset, "eval_level": eval_level, "test_scope_key": test_scope_key or "default"},
        )
        runtime = metrics_context.__enter__()
        if not self.memory.is_indexed:
            self.memory.load()
            if not self.memory.is_indexed:
                self.memory.build_indices()
                self.memory.save_indices()

        progress_path = (
            Path(progress_state_path)
            if progress_state_path
            else stage3_progress_state_file(self.config.paths.output_dir, test_scope_key or "default", backbone)
        )
        progress_state = _load_stage3_progress_state(progress_path)
        progress_state = _bootstrap_stage3_progress_from_previous_round(
            progress_state,
            root=self.config.paths.output_dir,
            dataset=dataset,
            eval_level=eval_level,
            test_data=test_data,
            backbone=backbone,
        )
        progressive_input = _build_progressive_stage3_input(test_data, eval_level, progress_state)
        filtered_test_data = progressive_input["eval_pool"]
        carried_correct = progressive_input["carried_correct"]
        semaphore = asyncio.Semaphore(concurrency)
        logger.info(
            "Stage 3: retrieval-augmented eval (mode=%s, %d items, threshold=%.2f, top_k=%d, concurrency=%d, backbone=%s)",
            mode, len(filtered_test_data), threshold, top_k, concurrency, backbone or "default",
        )

        async def solve_one(item: Dict, idx: int) -> Dict:
            question = item.get("question") or item.get("problem") or ""
            expected = item.get("answer", "")
            if not question or not expected:
                return {"idx": idx, "status": "skipped", "is_correct": False}

            async with semaphore:
                semantics = await self.plan.extract_semantics(question)

            use_principle = False
            principle_text = ""
            retrieval_count = 0
            retrieved_items: List[Dict[str, Any]] = []

            if mode == "guided":
                retrieved = self.memory.retrieve(semantics, top_k=top_k, threshold=threshold)
                passing = [r for r in retrieved if r.get("item", {}).get("principle")]
                if passing:
                    use_principle = True
                    retrieval_count = len(passing)
                    retrieved_items = passing
                    # 格式化多经验
                    blocks = []
                    for i, r in enumerate(passing, 1):
                        p = r["item"].get("principle", "")
                        score = r.get("score", 0.0)
                        structure_score = r.get("structure_score", score)
                        quality_score = r.get("quality_score", 0.5)
                        dims = r.get("dimension_similarities", {}) or {}
                        matched_facets = ", ".join(r.get("matched_facets", [])) or "none"
                        blocks.append(
                            "\n".join(
                                [
                                    f"[Experience {i}]",
                                    f"- association_score: {score:.2f}",
                                    f"- structure_alignment: {structure_score:.2f}",
                                    f"- experience_reliability: {quality_score:.2f}",
                                    (
                                        f"- facet_match: skill={dims.get(CORE_FIELD, 0.0):.2f}, "
                                        f"constraints={dims.get(CONDITION_FIELD, 0.0):.2f}, "
                                        f"objective={dims.get(TARGET_FIELD, 0.0):.2f}"
                                    ),
                                    f"- matched_facets: {matched_facets}",
                                    f"- guidance: {p}",
                                ]
                            )
                        )
                    principle_text = "\n\n".join(blocks)

            # 答题
            attempts = []
            total_attempts = min(3, 1 + max(0, retry_on_wrong))
            is_correct = False
            final_response = ""

            for attempt_idx in range(total_attempts):
                async with semaphore:
                    current_question = question
                    if attempt_idx > 0 and attempts:
                        current_question = _build_stage3_retry_prompt(question, principle_text, final_response)
                    if use_principle and principle_text:
                        response = await self.execute.answer_guided(current_question, principle_text)
                    elif mode == "baseline":
                        response = await self.execute.answer_baseline(current_question)
                    else:
                        response = await self.execute.answer_direct(current_question)

                current_correct = self.execute.check_answer(response, expected)
                attempts.append({
                    "attempt": attempt_idx + 1,
                    "is_correct": current_correct,
                    "response_preview": (response or "")[:300],
                })
                final_response = response
                if current_correct:
                    is_correct = True
                    break

            return {
                "idx": idx,
                "status": "success",
                "evaluation_key": _test_item_key(item),
                "source_item": item,
                "difficulty_level": _normalize_level_tag(item.get("difficulty_level") or item.get("level")),
                "question": question,
                "mode": mode,
                "retrieval_used": use_principle,
                "retrieval_count": retrieval_count,
                "retrieved_principles": retrieved_items,
                "model_answer": final_response,
                "expected_answer": expected,
                "is_correct": is_correct,
                "first_attempt_correct": attempts[0]["is_correct"] if attempts else False,
                "attempt_count": len(attempts),
                "retry_used": len(attempts) > 1,
                "rescued_by_retry": is_correct and any(not a["is_correct"] for a in attempts[:-1]),
                "attempts": attempts,
            }

        tasks = [solve_one(item, idx) for idx, item in enumerate(filtered_test_data)]
        results = []
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Stage 3 ({mode})"):
            results.append(await coro)

        results.sort(key=lambda x: x.get("idx", 0))
        progress_state = _merge_stage3_progress_state(
            progress_state=progress_state,
            evaluated_results=results,
            carried_correct=carried_correct,
            dataset=dataset or "",
            eval_level=eval_level,
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(progress_state, str(progress_path))

        complete_results = _build_stage3_complete_results(test_data, results, carried_correct)
        total_scope = len({_test_item_key(item) for item in test_data})
        cumulative_correct = sum(
            1
            for payload in progress_state.get("items", {}).values()
            if isinstance(payload, dict) and payload.get("ever_correct", False)
        )
        summary = build_stage3_summary(
            complete_results,
            scope_total=total_scope,
            correct=cumulative_correct,
            memory_experience_count=self.memory.principle_count,
            dataset=dataset,
            mode=mode,
            eval_level=eval_level,
            test_scope_key=test_scope_key or "default",
            runtime_metrics=runtime.finish(),
        )
        perf = summary["performance"]
        retrieval = summary["retrieval"]

        logger.info(
            "Stage 3 done: %d/%d correct (%.4f), current_round=%d, carried=%d, retrieval_used=%d",
            perf["correct"], perf["scope_total"], perf["accuracy"],
            len(results),
            len(carried_correct),
            retrieval["retrieved_samples"],
        )
        metrics_context.__exit__(None, None, None)
        return {"results": complete_results, "summary": summary}

    # ================================================================
    # Stage 3: 恐慌区救援
    # ================================================================

    async def stage3_remediate_panic(
        self,
        backbone: str,
        current_level: int,
        test_input: Path,
        k_trials: int = 5,
        concurrency: int = 64,
        mining_concurrency: int = 64,
        eval_concurrency: int = 64,
        eval_threshold: float = 0.55,
        eval_top_k: int = 5,
    ) -> Dict[str, Any]:
        """Stage 3: 评估当前 agent 能力 + 恐慌区救援。

        流程:
          1. 评估: 用当前 memory 在测试集上 guided answering → 测量 agent 能力
          2. 救援: 加载 panic 样本, K-trial guided answering, SR>0 → mine+dif
          3. 救活的追加到 level_{current_level+1}_learning_packets.json
        """
        import json as _json

        metrics_context = collect_runtime_metrics(
            "stage3", {"backbone": backbone, "current_level": current_level}
        )
        runtime = metrics_context.__enter__()

        target_level = min(current_level + 1, 4)
        output_root = self.config.paths.output_dir

        # 确保 memory 已加载
        if not self.memory.is_indexed:
            self.memory.load()
            if not self.memory.is_indexed:
                self.memory.build_indices()
                self.memory.save_indices()

        # ════════════════════════════════════════════════════════
        # Phase 1: 评估当前 agent 能力 (memory-guided, progressive)
        # ════════════════════════════════════════════════════════
        # 已做对的题不再重复测试, 直接 carried_forward
        test_data = load_json(str(test_input)) if test_input.exists() else []
        test_scope_key = _stage3_test_scope_key(str(test_input))
        progress_path = stage3_progress_state_file(output_root, test_scope_key, backbone)
        progress_state = _load_stage3_progress_state(progress_path)
        progress_state = _bootstrap_stage3_progress_from_previous_round(
            progress_state,
            root=output_root,
            dataset=f"dpo_level{current_level}",
            eval_level=current_level,
            test_data=test_data,
            backbone=backbone,
        )
        progressive_input = _build_progressive_stage3_input(test_data, current_level, progress_state)
        eval_pool = progressive_input["eval_pool"]
        carried_correct = progressive_input["carried_correct"]

        logger.info(
            "Stage 3.1: capability eval (level=%d, memory=%d principles, to_test=%d, carried=%d)",
            current_level, self.memory.principle_count, len(eval_pool), len(carried_correct),
        )

        eval_summary: Dict[str, Any] = {
            "scope_total": len(test_data) if isinstance(test_data, list) else 0,
            "tested": 0, "carried_correct": len(carried_correct),
            "correct": len(carried_correct), "accuracy": 0.0,
            "retrieval_used": 0, "memory_experience_count": self.memory.principle_count,
        }
        eval_results: List[Dict[str, Any]] = []

        if eval_pool:
            eval_sem = asyncio.Semaphore(eval_concurrency)

            async def eval_one(item: Dict) -> Dict:
                question = item.get("question") or item.get("problem") or ""
                expected = item.get("answer", "")
                if not question or not expected:
                    key = _test_item_key(item)
                    return {"evaluation_key": key, "is_correct": False, "retrieval_used": False,
                            "status": "skipped", "source_item": item,
                            "question": question, "expected_answer": expected}

                async with eval_sem:
                    semantics = await self.plan.extract_semantics(question)

                principle_text = ""
                retrieval_used = False
                if self.memory.principle_count > 0:
                    retrieved = self.memory.retrieve(semantics, top_k=eval_top_k, threshold=eval_threshold)
                    passing = [r for r in retrieved if r.get("item", {}).get("principle")]
                    if passing:
                        retrieval_used = True
                        blocks = []
                        for i, r in enumerate(passing, 1):
                            p = r["item"].get("principle", "")
                            blocks.append(f"[Experience {i}]\n- guidance: {p}")
                        principle_text = "\n\n".join(blocks)

                async with eval_sem:
                    if retrieval_used and principle_text:
                        response = await self.execute.answer_guided(question, principle_text)
                    else:
                        response = await self.execute.answer_direct(question)

                is_correct = self.execute.check_answer(response, expected)
                key = _test_item_key(item)
                return {
                    "evaluation_key": key, "source_item": item,
                    "question": question, "expected_answer": expected,
                    "is_correct": is_correct, "retrieval_used": retrieval_used,
                    "status": "success",
                }

            eval_tasks = [eval_one(item) for item in eval_pool]
            for coro in async_tqdm(
                asyncio.as_completed(eval_tasks), total=len(eval_tasks),
                desc=f"Stage 3.1 Eval (L{current_level})",
            ):
                eval_results.append(await coro)

            # 合并进展状态
            progress_state = _merge_stage3_progress_state(
                progress_state=progress_state,
                evaluated_results=eval_results,
                carried_correct=carried_correct,
                dataset=f"dpo_level{current_level}",
                eval_level=current_level,
            )
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(progress_state, str(progress_path))

            # 计算完整准确率 (含 carried_forward)
            scope_total = len(test_data) if isinstance(test_data, list) else 0
            cumulative_correct = sum(
                1 for payload in progress_state.get("items", {}).values()
                if isinstance(payload, dict) and payload.get("ever_correct", False)
            )
            tested_correct = sum(1 for r in eval_results if r.get("is_correct"))
            retrieval_used = sum(1 for r in eval_results if r.get("retrieval_used"))
            eval_summary = {
                "scope_total": max(scope_total, len(eval_pool) + len(carried_correct)),
                "tested": len(eval_pool),
                "carried_correct": len(carried_correct),
                "tested_correct": tested_correct,
                "correct": cumulative_correct,
                "accuracy": cumulative_correct / max(scope_total, 1) if scope_total > 0 else 0.0,
                "retrieval_used": retrieval_used,
                "memory_experience_count": self.memory.principle_count,
            }
            logger.info(
                "Stage 3.1 eval done: cumulative %d/%d correct (%.4f), this_round tested=%d correct=%d, retrieval_hit=%d/%d",
                cumulative_correct, scope_total, eval_summary["accuracy"],
                len(eval_pool), tested_correct, retrieval_used, len(eval_pool),
            )

            # Save standalone Phase 1 eval results
            eval_output_dir = output_root / "stage3" / "remediation"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            save_json(eval_results, str(eval_output_dir / f"{backbone}_L{current_level}_eval_results.json"))
            save_json(eval_summary, str(eval_output_dir / f"{backbone}_L{current_level}_eval_summary.json"))
        elif carried_correct:
            scope_total = len(test_data) if isinstance(test_data, list) else 0
            eval_summary = {
                "scope_total": scope_total,
                "tested": 0,
                "carried_correct": len(carried_correct),
                "tested_correct": 0,
                "correct": len(carried_correct),
                "accuracy": len(carried_correct) / max(scope_total, 1) if scope_total > 0 else 0.0,
                "retrieval_used": 0,
                "memory_experience_count": self.memory.principle_count,
            }
            logger.info(
                "Stage 3.1: all %d items carried forward, skipping test (accuracy=%.4f)",
                scope_total, eval_summary["accuracy"],
            )
            # Save summary even when all carried forward
            eval_output_dir = output_root / "stage3" / "remediation"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            save_json(eval_summary, str(eval_output_dir / f"{backbone}_L{current_level}_eval_summary.json"))
        else:
            logger.warning("Test set missing or empty, skipping eval: %s", test_input)

        # ════════════════════════════════════════════════════════
        # Phase 2: 恐慌区救援
        # ════════════════════════════════════════════════════════
        sampling_path = output_root / "stage1" / "01_sampling" / f"{backbone}_unified_sampling_results.json"
        if not sampling_path.exists():
            logger.warning("Sampling file not found: %s", sampling_path)
            metrics_context.__exit__(None, None, None)
            return {"status": "no_sampling_file", "evaluation": eval_summary,
                    "rescued": 0, "total_panic": 0}

        data = load_json(str(sampling_path))
        all_results: List[Dict[str, Any]] = data.get("results", [])
        panic_samples = [r for r in all_results if r.get("zone") == "panic"]
        if not panic_samples:
            logger.info("Stage 3.2: no panic samples to rescue")
            metrics_context.__exit__(None, None, None)
            rd = {"evaluation": eval_summary, "status": "no_panic_samples",
                  "rescued": 0, "total_panic": 0}
            return rd

        logger.info(
            "Stage 3.2: panic zone rescue (level=%d→%d, panic=%d, k=%d)",
            current_level, target_level, len(panic_samples), k_trials,
        )

        semaphore = asyncio.Semaphore(concurrency)
        urls = self.execute.urls or URLRouter.build_urls(
            self.config.hyperparams.weak_model_url, getattr(self.execute, 'ports', None)
        )
        router = URLRouter(urls)
        model = self.execute.model
        api_key = self.execute.api_key
        timeout = self.execute.timeout

        async def resample_one(sample: Dict[str, Any]) -> Dict[str, Any]:
            question = sample.get("question", "")
            expected = sample.get("expected_answer", "")
            async with semaphore:
                semantics = await self.plan.extract_semantics(question)

            principle_text = ""
            if self.memory.principle_count > 0:
                retrieved = self.memory.retrieve(semantics, top_k=3, threshold=0.5)
                passing = [r for r in retrieved if r.get("item", {}).get("principle")]
                if passing:
                    blocks = []
                    for i, r in enumerate(passing, 1):
                        p = r["item"].get("principle", "")
                        blocks.append(f"[Experience {i}]\n- guidance: {p}")
                    principle_text = "\n\n".join(blocks)

            correct_answers = []
            wrong_answers = []
            all_answers = []
            trial_records = []
            sampler = Stage1Sampler(
                urls=urls, model=model, api_key=api_key,
                k_trials=k_trials, temperature=0.7,
                max_tokens=4096, timeout=timeout,
            )
            for trial in range(k_trials):
                entropy = {
                    "available": False,
                    "normalized_entropy": 1.0,
                    "trajectory_entropy": 1.0,
                    "token_count": 0,
                    "entropy_token_count": 0,
                    "avg_neg_logprob": None,
                    "method": "guided_text_only_default_max_entropy",
                }
                async with semaphore:
                    if principle_text:
                        response = await self.execute.answer_guided(question, principle_text)
                    else:
                        metadata = await sampler._single_inference_with_metadata(question)
                        response = metadata.get("text", "")
                        entropy = metadata.get("entropy") or entropy
                if not response:
                    response = "[EMPTY_RESPONSE]"
                all_answers.append(response)
                is_correct = bool(self.execute.check_answer(response, expected))
                trial_records.append({
                    "trial": trial + 1,
                    "response": response,
                    "is_correct": is_correct,
                    "extracted_answer": None,
                    "normalized_entropy": entropy.get("normalized_entropy", 1.0),
                    "trajectory_entropy": entropy.get("trajectory_entropy", entropy.get("normalized_entropy", 1.0)),
                    "entropy_available": bool(entropy.get("available")),
                    "entropy_method": entropy.get("method", "unknown"),
                    "token_count": entropy.get("token_count", 0),
                    "entropy_token_count": entropy.get("entropy_token_count", 0),
                    "avg_neg_logprob": entropy.get("avg_neg_logprob"),
                })
                if is_correct:
                    correct_answers.append(response)
                else:
                    wrong_answers.append(response)

            correct_count = len(correct_answers)
            result = {
                **sample,
                "all_answers": all_answers,
                "correct_trajectories": correct_answers,
                "wrong_trajectories": wrong_answers,
                "correct_count": correct_count,
                "wrong_count": len(wrong_answers),
                "trial_records": trial_records,
                "planned_trials": k_trials,
                "actual_trials": len(trial_records),
                "early_stopped": False,
            }
            return sampler.finalize_result(result)

        rescue_tasks = [resample_one(s) for s in panic_samples]
        rescued_samples: List[Dict[str, Any]] = []
        still_panic: List[Dict[str, Any]] = []
        for coro in async_tqdm(
            asyncio.as_completed(rescue_tasks), total=len(rescue_tasks),
            desc=f"Stage 3.2 Rescue (L{current_level}→L{target_level})",
        ):
            r = await coro
            if r.get("zone") in {"learning", "master"}:
                rescued_samples.append(r)
            else:
                still_panic.append(r)

        logger.info(
            "Stage 3.2 rescue: %d/%d entered learning zone",
            len(rescued_samples), len(panic_samples),
        )

        # ── Phase 2b: 按 assigned_level 分组 mine + diff, 追加到对应 level packet ──
        new_unique_count = 0
        level_updated: Dict[int, int] = {}
        if rescued_samples:
            miner = Stage1Miner(
                urls=urls, model=model, api_key=api_key, timeout=timeout,
            )
            # 按 SR 分布到各 level, 而非全部塞进 target_level
            by_assigned: Dict[str, List[Dict]] = {}
            for r in rescued_samples:
                lv = max(1, min(4, int(r.get("assigned_level", target_level) or target_level)))
                by_assigned.setdefault(str(lv), []).append(r)

            mining_result = await miner.run(by_assigned, concurrency=mining_concurrency)
            for lv_str in sorted(by_assigned.keys()):
                lv_packets = mining_result.get("packets", {}).get(lv_str, [])
                if not lv_packets:
                    continue
                packet_path = stage1_level_packets_file(output_root, int(lv_str), backbone)
                existing_packets = []
                if packet_path.exists():
                    existing_packets = load_json(str(packet_path))
                    if not isinstance(existing_packets, list):
                        existing_packets = []

                existing_ids = {str(p.get("question_id", p.get("idx", ""))) for p in existing_packets}
                new_unique = [p for p in lv_packets if str(p.get("idx", "")) not in existing_ids]
                existing_packets.extend(new_unique)

                packet_path.parent.mkdir(parents=True, exist_ok=True)
                with open(packet_path, "w", encoding="utf-8") as f:
                    _json.dump(existing_packets, f, ensure_ascii=False, indent=2)
                new_unique_count += len(new_unique)
                level_updated[int(lv_str)] = len(new_unique)
                logger.info(
                    "Stage 3.2: level%s +%d packets (%d rescued samples, deduped)",
                    lv_str, len(new_unique), len(by_assigned[lv_str]),
                )

        # ── 保存 Stage 3 完整报告 (评估 + 救援) ──
        rescue_output_dir = output_root / "stage3" / "remediation"
        rescue_output_dir.mkdir(parents=True, exist_ok=True)
        rescue_path = rescue_output_dir / f"{backbone}_L{current_level}_rescue.json"
        remediation_stats = {
            "rescued_count": len(rescued_samples),
            "total_panic": len(panic_samples),
            "new_packets": new_unique_count,
            "levels_updated": level_updated,
            "rescued": [{"idx": r.get("idx"), "question": r.get("question", "")[:120], "sr": r["sr"], "assigned_level": r.get("assigned_level", 0)} for r in rescued_samples],
            "still_panic": [{"idx": r.get("idx"), "question": r.get("question", "")[:120]} for r in still_panic],
        }
        with open(rescue_path, "w", encoding="utf-8") as f:
            _json.dump({
                "backbone": backbone,
                "current_level": current_level,
                "target_level": target_level,
                "evaluation": eval_summary,
                "remediation": remediation_stats,
            }, f, ensure_ascii=False, indent=2)

        metrics_context.__exit__(None, None, None)

        status = "ok" if rescued_samples else "no_rescued"
        return {
            "status": status,
            "evaluation": eval_summary,
            "rescued": len(rescued_samples),
            "new_packets": new_unique_count,
            "levels_updated": level_updated,
            "total_panic": len(panic_samples),
            "target_level": target_level,
        }

    # ================================================================
    # 全流程: 多轮课程学习
    # ================================================================

    async def run_full_curriculum(
        self,
        levels: Sequence[int] = (1, 2, 3, 4),
        backbone: str = "",
        test_input: Optional[Path] = None,
        concurrency: int = 64,
        diff_concurrency: int = 64,
        stage2_concurrency: int = 64,
        eval_concurrency: int = 64,
        mining_concurrency: int = 64,
        skip_sampling: bool = False,
        skip_mining: bool = False,
        start_level: int = 1,
    ) -> Dict[str, Any]:
        """课程学习全流程 — 每个 level 就是一个 round.

        Round 1 (Level 1):
          Stage 1: 全局 K-trial 采样 → 三区划分 → 四级分 pack → mining + diff
          Stage 2: Level 1 记忆化 (DSD → Self-CPE → 验证)
          Stage 3: Panic 救援 (memory 引导答题, 救活→追加到高 level packet)

        Round 2-4 (Level 2-4):
          Stage 1: Panic 区重采样 (用累积 memory 引导, K=10, 救活→mining→追加)
          Stage 2: 当前 level 记忆化
          Stage 3: Panic 救援
        """
        all_summaries: Dict[str, Any] = {}

        active_levels = [l for l in levels if l >= start_level]
        if start_level > 1:
            logger.info("Skipping levels %s (start_level=%d)", [l for l in levels if l < start_level], start_level)
            # Ensure memory is loaded from disk (contains principles from skipped levels)
            if not self.memory.is_indexed:
                self.memory.load()
                if not self.memory.is_indexed:
                    self.memory.build_indices()
                    self.memory.save_indices()

        for level in active_levels:
            round_label = f"Round {level}/4"
            dataset = f"dpo_level{level}"
            logger.info("=" * 60)
            logger.info("%s: Level %d — %s", round_label, level, dataset)
            logger.info("=" * 60)

            all_summaries[f"round_{level}"] = {}

            # ═══ Stage 1 ═══
            if level == 1 and not skip_sampling:
                logger.info("Stage 1 (global): unified sampling + 3-zone + 4-level pack + mining + diff")
                packets_dir = await self.stage1_learn(
                    concurrency=concurrency,
                    diff_concurrency=diff_concurrency,
                    skip_sampling=False,
                    backbone=backbone,
                )
                all_summaries[f"round_{level}"]["stage1"] = str(packets_dir) if packets_dir else None
                if packets_dir is None:
                    logger.error("Stage 1 failed, cannot continue")
                    return all_summaries
            elif level > 1:
                logger.info("Stage 1 (panic resample): memory=%d principles", self.memory.principle_count)
                panic_result = await self.stage1_panic_resample(
                    backbone=backbone,
                    concurrency=concurrency,
                    diff_concurrency=diff_concurrency,
                )
                all_summaries[f"round_{level}"]["stage1"] = panic_result
                logger.info(
                    "Stage 1 done: %d/%d panic rescued -> +%d packets",
                    panic_result.get("rescued", 0),
                    panic_result.get("total_panic", 0),
                    panic_result.get("new_packets", 0),
                )
            else:
                all_summaries[f"round_{level}"]["stage1"] = "skipped"

            # ═══ Stage 2: 记忆化 ═══
            stage2_summary = await self.stage2_memorize(
                dataset=dataset,
                round_id=level,
                validation_concurrency=stage2_concurrency,
                backbone=backbone,
            )

            # ═══ Stage 3: Panic 救援 (Round 1-3) / 最终评估 (Round 4) ═══
            if level < 4:
                stage3_summary = await self.stage3_remediate_panic(
                    backbone=backbone,
                    current_level=level,
                    test_input=test_input or self.config.paths.test_dir / "deepscaler.json",
                    concurrency=stage2_concurrency,
                    eval_concurrency=eval_concurrency,
                    mining_concurrency=mining_concurrency,
                )
                _eval = stage3_summary.get("evaluation", {})
                rescued = stage3_summary.get("rescued", 0)
                total_panic = stage3_summary.get("total_panic", 0)
                new_pkts = stage3_summary.get("new_packets", 0)
                levels_updated = stage3_summary.get("levels_updated", {})
                lv_str = ", ".join(f"L{lv}:+{n}" for lv, n in sorted(levels_updated.items())) if levels_updated else "none"
                logger.info(
                    ">>> %s complete: "
                    "eval accuracy=%.4f (%d/%d) | "
                    "rescued %d/%d panic -> %s (%d packets) | "
                    "memory %d principles",
                    round_label,
                    _eval.get("accuracy", 0.0), _eval.get("correct", 0), _eval.get("scope_total", 0),
                    rescued, total_panic, lv_str, new_pkts,
                    self.memory.principle_count,
                )
            else:
                # Round 4: 最终 guided 评估
                logger.info("Stage 3 (final eval): guided evaluation with accumulated memory")
                test_path = test_input or self.config.paths.test_dir / "deepscaler.json"
                test_data = load_json(str(test_path))
                test_scope_key = _stage3_test_scope_key(str(test_path))
                eval_result = await self.stage3_evaluate(
                    test_data=test_data,
                    threshold=0.55,
                    top_k=5,
                    concurrency=eval_concurrency,
                    eval_level=level,
                    dataset=dataset,
                    progress_state_path=stage3_progress_state_file(self.config.paths.output_dir, test_scope_key, backbone),
                    test_scope_key=test_scope_key,
                    backbone=backbone,
                )
                stage3_summary = eval_result["summary"]

                # Save results and summary to disk
                out_root = self.config.paths.output_dir
                save_json(eval_result["results"], str(stage3_guided_results_file(out_root, dataset, backbone)))
                save_json(eval_result["summary"], str(stage3_guided_summary_file(out_root, dataset, backbone)))

                perf = stage3_summary.get("performance", {})
                logger.info(
                    ">>> %s complete: "
                    "final accuracy=%.4f (%d/%d) | "
                    "memory %d principles",
                    round_label,
                    perf.get("accuracy", 0.0), perf.get("correct", 0), perf.get("scope_total", 0),
                    self.memory.principle_count,
                )

            all_summaries[f"round_{level}"][dataset] = {
                "stage2": stage2_summary,
                "stage3": stage3_summary,
            }

        logger.info("=" * 60)
        logger.info("All %d rounds complete!", len(active_levels))
        logger.info("=" * 60)
        return all_summaries


# ── 辅助函数 ─────────────────────────────────────────────────


def _extract_key_lines(trajectory: str, max_lines: int = 50) -> str:
    """提取轨迹中含数学操作的关键行。"""
    _MATH_PATTERN = re.compile(
        r"[=<>≤≥∈∉∑∏∫\\]|\\frac|\\sqrt|\\boxed|\\cdot|\\times|Final Answer",
        re.IGNORECASE,
    )
    lines = trajectory.splitlines()
    key_lines = [line for line in lines if _MATH_PATTERN.search(line)]
    if len(key_lines) < 5:
        key_lines = lines
    return "\n".join(key_lines[:max_lines])


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

async def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MemMetaV3 Agent Pipeline — 统一编排",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全流程 (Stage 1 + Round 1-4)
  python -m core.agent_pipeline --levels 1 2 3 4 --ports 8000 8001

  # 从 Level 2 恢复 (跳过已完成的 Round 1)
  python -m core.agent_pipeline --levels 1 2 3 4 --start-level 2 --skip-sampling

  # 仅跑 Stage 3 (需要 memory/ 已有数据)
  python -m core.agent_pipeline --stage 3 --levels 1

  # 跳过采样和挖掘 (复用已有 Stage 1 中间结果)
  python -m core.agent_pipeline --levels 1 --skip-sampling --skip-mining
""",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="仅运行指定 stage")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="仅处理指定数据集 (与 --stage 配合)")
    parser.add_argument("--test-input", type=str, default=None,
                        help="测试集路径 (--stage 3 或全流程评估用, 默认 data/test/deepscaler.json)")
    parser.add_argument("--memory-dir", type=str, default="memory/v3")
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--ports", type=int, nargs="+", default=None)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--diff-concurrency", type=int, default=64)
    parser.add_argument("--eval-concurrency", type=int, default=64)
    parser.add_argument("--stage2-limit", type=int, default=0)
    parser.add_argument("--stage2-run-tag", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-mining", action="store_true")
    parser.add_argument("--start-level", type=int, default=1, help="从指定 level 开始 (1-4), 跳过已完成 level 的 Stage 2-3")
    parser.add_argument("--backbone", type=str, default="", help="模型标识 (qwen/llama), 用于定位新版 Stage 1 产物")
    args = parser.parse_args()

    config = Config.from_core_file()
    config.ensure_dirs()
    config.setup_logging()

    memory_dir = Path(args.memory_dir) / args.backbone if args.backbone else Path(args.memory_dir)
    pipeline = AgentPipeline(
        config=config,
        ports=args.ports,
        model_override=args.model,
        memory_dir=str(memory_dir),
    )

    if args.stage is not None:
        # 单 Stage 模式
        dataset = args.dataset or "dpo_level1"
        level = int(dataset.replace("dpo_level", "")) if "level" in dataset else 1
        if args.stage == 1:
            await pipeline.stage1_learn(
                dataset=dataset,
                concurrency=args.concurrency,
                diff_concurrency=args.diff_concurrency,
                skip_sampling=args.skip_sampling,
                backbone=args.backbone,
            )
        elif args.stage == 2:
            await pipeline.stage2_memorize(
                dataset=dataset,
                round_id=level,
                backbone=args.backbone,
                max_records=args.stage2_limit,
                run_tag=args.stage2_run_tag,
            )
        elif args.stage == 3:
            test_path = args.test_input or str(config.paths.test_dir / "deepscaler.json")
            test_data = load_json(test_path)
            test_scope_key = _stage3_test_scope_key(test_path)
            result = await pipeline.stage3_evaluate(
                test_data=test_data,
                threshold=args.threshold,
                top_k=args.top_k,
                concurrency=args.eval_concurrency,
                eval_level=level,
                dataset=dataset,
                progress_state_path=stage3_progress_state_file(config.paths.output_dir, test_scope_key, args.backbone),
                test_scope_key=test_scope_key,
                backbone=args.backbone,
            )
            from output_paths import stage3_guided_results_file, stage3_guided_summary_file
            out_root = config.paths.output_dir
            save_json(result["results"], str(stage3_guided_results_file(out_root, dataset, args.backbone)))
            save_json(result["summary"], str(stage3_guided_summary_file(out_root, dataset, args.backbone)))
    else:
        # 全流程
        test_input = Path(args.test_input) if args.test_input else None
        await pipeline.run_full_curriculum(
            levels=args.levels,
            backbone=args.backbone,
            test_input=test_input,
            concurrency=args.concurrency,
            diff_concurrency=args.diff_concurrency,
            stage2_concurrency=args.eval_concurrency,
            eval_concurrency=args.eval_concurrency,
            mining_concurrency=args.diff_concurrency,
            skip_sampling=args.skip_sampling,
            skip_mining=args.skip_mining,
            start_level=args.start_level,
        )


if __name__ == "__main__":
    asyncio.run(_cli_main())
