"""Weak-model runtime used by Stage 1 and legacy in-process Stage 3 helpers.

The public controller is the repository-root ``pipeline.py``. Stage 2 is not
implemented here: its independent Generator/Auditor/Store orchestration lives
in ``modules.stage2.factory`` so it cannot silently reuse the weak Solver.
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
from modules.stage1_miner import Stage1Miner, save_rankings_and_manifests, write_analysis_csv
from utils.output_paths import (
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
    if dataset and re.search(r"level\d+", dataset):
        previous_dataset = re.sub(r"level(\d+)", lambda m: f"level{int(m.group(1)) - 1}", dataset)
    if not previous_dataset:
        previous_dataset = f"level{int(eval_level) - 1}"

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
        k_trials: int = 10,
        experiment_seed: int = 42,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_logprobs: int = 5,
        entropy_histogram_bins: int = 20,
        max_tokens: int = 4096,
        task_batch_size: int = 1000,
        expected_dataset_sha256: str = "",
        expected_dataset_size: int = 6000,
        skip_sampling: bool = False,
        backbone: str = "",
        urls_override: Optional[List[str]] = None,
        model_override: Optional[str] = None,
        resume: bool = False,
        collect_all_samples: bool = False,
    ) -> Optional[Path]:
        """Stage 1: complete sampling -> accuracy levels -> entropy rankings.

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
                raise FileNotFoundError(f"Unified dataset not found: {src}")
            dataset_sha256 = hashlib.sha256(Path(src).read_bytes()).hexdigest()
            if expected_dataset_sha256 and dataset_sha256 != expected_dataset_sha256:
                raise ValueError(
                    f"Dataset SHA-256 mismatch: expected {expected_dataset_sha256}, got {dataset_sha256}"
                )
            tasks = load_json(src)
            if not isinstance(tasks, list) or not tasks:
                raise ValueError(f"Dataset is empty: {src}")
            if expected_dataset_size and len(tasks) != int(expected_dataset_size):
                raise ValueError(f"Expected {expected_dataset_size} tasks, found {len(tasks)}")
            missing = [index for index, item in enumerate(tasks) if not isinstance(item, dict) or any(
                not str(item.get(field, "")).strip() for field in ("question", "answer", "difficulty_level")
            )]
            if missing:
                logger.warning(
                    "Retaining %d invalid input record(s) outside Master/Learning/Panic: %s",
                    len(missing), missing[:10],
                )
            logger.info("Stage 1 [%s]: loaded %d tasks", tag, len(tasks))

            # ── Step 1.1: 采样 ──
            sampling_path = stage1_unified_sampling_file(output_root, backbone)
            sampling_ckpt = str(sampling_path).replace(".json", ".checkpoint.json")
            if skip_sampling:
                if not sampling_path.exists():
                    raise FileNotFoundError(
                        f"--skip-sampling was requested, but no Stage 1 sampling artifact exists: "
                        f"{sampling_path}"
                    )
                logger.info("[Stage 1.1 %s skipped] using existing: %s", tag, sampling_path)
                sampling_data = load_json(str(sampling_path))
            else:
                # 已有完整文件直接复用（total>0 才认为有效）
                if not resume and sampling_path.exists():
                    existing = load_json(str(sampling_path))
                    expected_config = {
                        "k_trials": k_trials,
                        "experiment_seed": experiment_seed,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_logprobs": top_logprobs,
                        "early_stop": False,
                    }
                    existing_config = existing.get("config", {}) if isinstance(existing, dict) else {}
                    compatible = (
                        existing.get("schema_version") == "stage1.1"
                        and all(existing_config.get(key) == value for key, value in expected_config.items())
                    )
                    compatible = compatible and existing.get("dataset_sha256") == dataset_sha256
                    if existing.get("summary", {}).get("total", 0) > 0 and compatible:
                        logger.info("[Stage 1.1 %s] final file already exists (total=%d): %s",
                                     tag, existing["summary"]["total"], sampling_path)
                        sampling_data = existing
                    elif existing.get("summary", {}).get("total", 0) > 0:
                        raise ValueError(
                            f"Existing Stage 1 output uses a different protocol/config: {sampling_path}. "
                            "Use another --backbone tag or archive the old output."
                        )
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
                        experiment_seed=experiment_seed,
                        temperature=temperature,
                        top_p=top_p,
                        top_logprobs=top_logprobs,
                        entropy_histogram_bins=entropy_histogram_bins,
                        max_tokens=max_tokens,
                        task_batch_size=task_batch_size,
                        timeout=self.execute.timeout,
                    )
                    sampling_data = await sampler.sample_all(
                        tasks, concurrency=concurrency,
                        checkpoint_path=sampling_ckpt,
                        collect_all_samples=collect_all_samples,
                    )
                    sampling_data["dataset_sha256"] = dataset_sha256
                    Stage1Sampler.save(sampling_data, sampling_path)
                    incomplete = int(sampling_data.get("summary", {}).get("incomplete", 0))
                    if incomplete == 0 and ckpt_file.exists():
                        ckpt_file.unlink()
                    if incomplete:
                        raise RuntimeError(
                            f"Stage 1 has {incomplete} incomplete question(s); checkpoint retained at {ckpt_file}. "
                            "Resume with the same configuration before building Learning Bundles."
                        )

            summary = sampling_data.get("summary", {})
            level_data = sampling_data.get("level_data", {})

            # Master/Learning/Panic are immutable capability zones.  The legacy
            # collect_all_samples flag may retain every result for analysis, but it
            # must never promote Master or Panic questions into Learning Levels.
            if collect_all_samples:
                logger.warning(
                    "[%s] collect_all_samples is retained only as a legacy analysis flag; "
                    "Master and Panic remain outside level1-level4 Learning Bundles.",
                    tag,
                )

            if not level_data:
                logger.error("[%s] no samples to process (all zones empty)", tag)
                return None

            # ── Step 1.2: deterministic evidence bundling and rankings ──
            miner = Stage1Miner(
                experiment_seed=experiment_seed,
                dataset_sha256=dataset_sha256,
            )
            mining_result = await miner.run(level_data)

            packets = mining_result.get("packets", {})
            stats = mining_result.get("stats", {})

            # 保存 —— 按 backbone 分目录
            packets_dir = stage1_level_packets_file(output_root, 1, backbone).parent
            Stage1Miner.save(packets, packets_dir)
            stage1_root = packets_dir.parent.parent
            save_rankings_and_manifests(mining_result.get("rankings", {}), stage1_root)
            write_analysis_csv(packets, stage1_root / "analysis" / "questions.csv")
            from scripts.stage1.analyze import generate_figures
            figure_paths = generate_figures(stage1_root)
            (stage1_root / "run_manifest.json").write_text(json.dumps({
                "schema_version": "stage1.1",
                "dataset_path": str(Path(src).resolve()),
                "dataset_sha256": dataset_sha256,
                "dataset_size": len(tasks),
                "invalid_input_indices": missing,
                "model": _model,
                "ports": _urls,
                "sampling_config": sampling_data.get("config", {}),
                "ranking_policies": ["random", "high_token_entropy", "low_token_entropy"],
                "figures": [str(path) for path in figure_paths],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

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
                "Stage 1 [%s] done: total=%d | M=%d L=%d P=%d | deterministic_pairs=%d",
                tag, summary.get("total", 0),
                summary.get("master_zone", 0), summary.get("learning_zone", 0),
                summary.get("panic_zone", 0),
                stats.get("total_pairs_found", 0),
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
            dataset=f"level{current_level}",
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
                dataset=f"level{current_level}",
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
