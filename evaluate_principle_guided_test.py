#!/usr/bin/env python
import argparse
import asyncio
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm.asyncio import tqdm as async_tqdm

from core.api_call import async_llm_call
from core.config import Config, apply_local_api_port
from core.runtime_metrics import collect_runtime_metrics
from generate_principle import extract_semantic_features
from retrieve_principles import OnlinePrincipleRetriever
from template.prompt_template import (
    PRINCIPLE_TO_RULES_PROMPT,
    WEAK_BASELINE_TEST_PROMPT,
    DIRECT_ANSWER_PROMPT,
    GUIDED_ANSWER_PROMPT,
)
from utils.io_utils import load_json, save_json
from utils.summary_utils import build_stage3_summary
from output_paths import stage2_memory_file, stage2_vector_dir, stage3_baseline_results_file, stage3_baseline_summary_file, stage3_guided_results_file, stage3_guided_summary_file
from utils.official_dataset_utils import evaluate_general365_prediction, evaluate_svamp_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_retry_prompt(question: str, principle_text: str, previous_response: str) -> str:
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


def soften_ground_rules(text: str) -> str:
    if not text:
        return ""

    softened = str(text)
    replacements = {
        "MUST ": "Consider ",
        "MUST\n": "Consider\n",
        "NEVER ": "Try not to ",
        "SHALL ": "You can ",
        "ONLY IF": "It may be most useful when",
    }
    for src, dst in replacements.items():
        softened = softened.replace(src, dst)

    softened = re.sub(r'(^|\n)-\s*MUST\b', r'\1- Consider', softened)
    softened = re.sub(r'(^|\n)\d+\.\s*MUST\b', lambda m: m.group(0).replace('MUST', 'Consider'), softened)
    return softened


def build_urls(base_url: str, ports: Optional[Sequence[int]] = None) -> List[str]:
    if not ports:
        return [base_url]
    return [re.sub(r':\d+(?=/|$)', f':{port}', base_url) for port in ports]


class DynamicRouter:
    """动态路由分配器，用于在函数内部动态获取空闲 GPU 端口。"""

    def __init__(self, urls: List[str]):
        self.urls = urls
        self.counter = 0

    def get_url(self) -> str:
        url = self.urls[self.counter % len(self.urls)]
        self.counter += 1
        return url


def sanitize_text(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = cleaned.replace("\x08", r"\b")
    cleaned = cleaned.replace("\x0c", r"\f")
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = cleaned.replace("\u2013", "-")
    cleaned = cleaned.replace("\u2014", "-")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    filtered = []
    for ch in cleaned:
        if ch in ("\n", "\t"):
            filtered.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        filtered.append(ch)
    return "".join(filtered).strip()


def _extract_boxed_value(text: str) -> Optional[str]:
    boxed_idx = text.find(r'\boxed{')
    if boxed_idx == -1:
        return None
    start_idx = boxed_idx + 7
    open_braces = 1
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            open_braces += 1
        elif text[i] == '}':
            open_braces -= 1
        if open_braces == 0:
            return text[start_idx:i].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """引入精准的括号平衡解析算法。"""
    text = sanitize_text(text)
    if not text:
        return None

    boxed = _extract_boxed_value(text)
    if boxed:
        return boxed

    patterns = [
        r'####\s*([^\n]+)',
        r'\$\s*([^\$]+)\$',
        r'Final Answer:\s*\$?([^\n]+)\$?',
        r'answer[:\s]+([^\n]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def normalize_answer_text(text: str) -> str:
    cleaned = sanitize_text(text).strip().lower()
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\displaystyle", "")
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\,", "").replace(" ", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(r"\(", "").replace(r"\)", "")
    cleaned = cleaned.replace(r"\[", "").replace(r"\]", "")
    cleaned = re.sub(r"^finalanswer:?", "", cleaned)
    cleaned = re.sub(r"^answer:?", "", cleaned)
    cleaned = cleaned.strip(".,;:!?")
    return cleaned.replace("\\", "")


def parse_bbh_options(question: str) -> Dict[str, str]:
    text = sanitize_text(question)
    if "Options:" not in text:
        return {}
    option_block = text.split("Options:", 1)[1]
    matches = re.findall(r'\(([A-Z])\)\s*(.+?)(?=\n\([A-Z]\)|$)', option_block, re.DOTALL)
    return {label: sanitize_text(content).strip() for label, content in matches}


def extract_answer_candidates(text: str) -> List[str]:
    text = sanitize_text(text)
    if not text:
        return []

    candidates: List[str] = []
    boxed = _extract_boxed_value(text)
    if boxed:
        candidates.append(boxed)

    patterns = [
        r'####\s*([^\n]+)',
        r'Final Answer:\s*(.+)',
        r'答案是?\s*([^\n，。]+)',
        r'answer[:\s]+([^\n]+)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            candidate = sanitize_text(match.group(1))
            if candidate:
                candidates.append(candidate)

    tail_lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates.extend(tail_lines[-3:])

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        key = normalize_answer_text(candidate)
        if key and key not in seen:
            seen.add(key)
            deduped.append(candidate.strip())
    return deduped


def extract_choice_label(text: str) -> Optional[str]:
    text = sanitize_text(text).strip()
    if not text:
        return None
    patterns = [
        r'^\(([A-Z])\)$',
        r'^\[?([A-Z])\]?$',
        r'(?:final answer|answer|option|choice)\s*[:：]?\s*\(?([A-Z])\)?\s*$',
        r'\(([A-Z])\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def canonicalize_expected_answer(question: Optional[str], expected: str) -> Dict[str, str]:
    raw = sanitize_text(expected)
    if "####" in raw:
        raw = raw.split("####")[-1].strip()

    options = parse_bbh_options(question or "")
    label = extract_choice_label(raw)
    if label and label in options:
        return {
            "kind": "choice",
            "label": label,
            "value": normalize_answer_text(raw),
            "option_text": normalize_answer_text(options[label]),
        }

    normalized = normalize_answer_text(raw)
    if normalized in {"yes", "no", "true", "false"}:
        return {"kind": "boolean", "value": normalized}

    return {"kind": "text", "value": normalized}


def check_answer_match(generated: str, expected: str, question: Optional[str] = None) -> bool:
    if not generated or not expected:
        return False

    expected_info = canonicalize_expected_answer(question, expected)
    expected_clean = expected_info.get("value", "")
    if not expected_clean:
        return False

    candidates = extract_answer_candidates(generated)
    if not candidates:
        generated_clean = normalize_answer_text(generated)
        if "\n" not in sanitize_text(generated) and len(generated_clean) <= max(len(expected_clean) + 8, 32):
            candidates = [generated_clean]
        else:
            return False

    if expected_info["kind"] == "choice":
        expected_label = expected_info.get("label", "")
        expected_option_text = expected_info.get("option_text", "")
        for candidate in candidates:
            cand_label = extract_choice_label(candidate)
            if cand_label and cand_label == expected_label:
                return True
            cand_clean = normalize_answer_text(candidate)
            if expected_option_text and cand_clean == expected_option_text:
                return True
        return False

    for candidate in candidates:
        cand_clean = normalize_answer_text(candidate)
        if cand_clean and cand_clean == expected_clean:
            return True
    return False


def check_answer_match_with_item(generated: str, item: Dict[str, Any]) -> bool:
    expected = item.get("answer", "")
    question = item.get("question") or item.get("problem") or ""
    source_dataset = str(item.get("source_dataset", "")).strip().lower()
    answer_type = str(item.get("answer_type", "")).strip().lower()

    if source_dataset == "svamp":
        return evaluate_svamp_prediction(generated, str(expected))

    if source_dataset == "general365" or answer_type in {"single_choice", "multiple_choice", "text"} or "float_round" in item:
        return evaluate_general365_prediction(generated, item)

    return check_answer_match(generated, expected, question=question)


def parse_level(value: str) -> str:
    text = (value or '').strip()
    if not text:
        return 'Unknown'
    match = re.search(r'(\d+)', text)
    return f"Level {match.group(1)}" if match else text


def format_principle_block(retrieved: Optional[Dict[str, Any]]) -> str:
    if not retrieved:
        return 'No reliable principle package was retrieved. Solve from first principles and do not mention principle execution.'
    matched = retrieved.get('matched_memory', {})
    principle = matched.get('golden_principle', '')
    principle_text = json.dumps(principle, ensure_ascii=False, indent=2) if isinstance(principle, dict) else str(principle)
    scores = retrieved.get('dimension_similarities', {})
    return (
        f"Similarity scores:\n"
        f"- core_skill: {scores.get('core_skill', scores.get('core_math_concept', 0.0)):.4f}\n"
        f"- context_constraints: {scores.get('context_constraints', scores.get('conditions_structure', 0.0)):.4f}\n"
        f"- task_objective: {scores.get('task_objective', scores.get('solution_target', 0.0)):.4f}\n"
        f"Matched memory question: {matched.get('question', '')}\n"
        f"Retrieved principle:\n{principle_text}"
    )


def format_multi_principle_block(retrievals: List[Dict[str, Any]]) -> str:
    """格式化多条通过门控的经验为引导式prompt块。"""
    if not retrievals:
        return ""
    blocks = []
    for i, r in enumerate(retrievals, 1):
        score = r.get('score', 0.0)
        structure_score = r.get('structure_score', score)
        quality_score = r.get('quality_score', 0.5)
        dims = r.get('dimension_similarities', {}) or {}
        matched_facets = ", ".join(r.get('matched_facets', [])) or "none"
        principle_text = extract_retrieved_principle_text(r)
        if not principle_text:
            continue
        blocks.append(
            "\n".join(
                [
                    f"[Experience {i}]",
                    f"- association_score: {score:.2f}",
                    f"- structure_alignment: {structure_score:.2f}",
                    f"- experience_reliability: {quality_score:.2f}",
                    (
                        f"- facet_match: skill={dims.get('core_skill', 0.0):.2f}, "
                        f"constraints={dims.get('context_constraints', 0.0):.2f}, "
                        f"objective={dims.get('task_objective', 0.0):.2f}"
                    ),
                    f"- matched_facets: {matched_facets}",
                    f"- guidance: {principle_text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def extract_retrieved_principle_text(retrieved: Optional[Dict[str, Any]]) -> str:
    if not retrieved:
        return ""
    matched = retrieved.get("matched_memory", {})
    principle = matched.get("golden_principle", "")
    if isinstance(principle, dict):
        experience = principle.get("experience_principle") or {}
        if isinstance(experience, dict):
            return str(experience.get("principle", "") or "").strip()
        return str(principle.get("principle", "") or "").strip()
    return str(principle or "").strip()


async def build_ground_rules(
    question: str,
    chosen_retrieval: Optional[Dict[str, Any]],
    model: str,
    router: DynamicRouter,
    api_key: str,
    timeout: int,
) -> str:
    if not chosen_retrieval:
        return ""
    if not extract_retrieved_principle_text(chosen_retrieval):
        return ""
    prompt = PRINCIPLE_TO_RULES_PROMPT.format(question=question, principle_block=format_principle_block(chosen_retrieval))
    response = await async_llm_call(
        prompt=prompt, model=model, url=router.get_url(), api_key=api_key, timeout=timeout, temperature=0.1, max_tokens=1024,
    )
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return soften_ground_rules(str(data.get('ground_rules', '') or '').strip())
    except Exception:
        pass
    return soften_ground_rules(response.strip())


async def solve_one(
    item: Dict[str, Any],
    idx: int,
    retriever: Optional[OnlinePrincipleRetriever],
    model: str,
    router: DynamicRouter,
    api_key: str,
    timeout: int,
    threshold: float,
    mode: str,
    retry_on_wrong: int,
    answer_max_tokens: int,
) -> Dict[str, Any]:
    question = item.get('question') or item.get('problem') or ''
    expected = item.get('answer', '')
    difficulty = parse_level(item.get('difficulty_level', 'Unknown'))

    if not question or not expected:
        return {'idx': idx, 'question': question, 'difficulty_level': difficulty, 'status': 'skipped', 'is_correct': False}

    semantic_result = await extract_semantic_features(
        question=question, model=model, url=router.get_url(), api_key=api_key, timeout=timeout, temperature=0.1,
    )
    if semantic_result.get('status') != 'success':
        return {'idx': idx, 'question': question, 'difficulty_level': difficulty, 'status': 'semantic_failed', 'is_correct': False}

    features = semantic_result.get('features', {})
    chosen_retrieval = None
    all_retrievals = []
    use_principle = False
    ground_rules = ""

    if mode == 'guided' and retriever is not None:
        retrieved = retriever.retrieve(features)
        passing = []
        for candidate in retrieved:
            principle_text = extract_retrieved_principle_text(candidate)
            if principle_text:
                passing.append(candidate)

        if passing:
            use_principle = True
            chosen_retrieval = passing[0]  # Best match for backward compat
            all_retrievals = passing

            # Format multi-principle block
            multi_block = format_multi_principle_block(passing)
            if multi_block:
                ground_rules = multi_block
            else:
                # Fallback to single-principle ground rules generation
                ground_rules = await build_ground_rules(
                    question=question, chosen_retrieval=chosen_retrieval, model=model,
                    router=router, api_key=api_key, timeout=timeout,
                )

        if use_principle and ground_rules:
            prompt = GUIDED_ANSWER_PROMPT.format(principle_text=ground_rules, question=question)
        else:
            use_principle = False
            chosen_retrieval = None
            all_retrievals = []
            ground_rules = ""
            prompt = DIRECT_ANSWER_PROMPT.format(question=question)
    else:
        prompt = WEAK_BASELINE_TEST_PROMPT.format(question=question)

    attempts: List[Dict[str, Any]] = []
    final_response = ""
    is_correct = False

    total_attempts = min(3, 1 + max(0, retry_on_wrong))
    for attempt_idx in range(total_attempts):
        current_prompt = prompt
        if attempt_idx > 0 and attempts:
            current_prompt = build_retry_prompt(question, ground_rules if use_principle else "", final_response)
        response = await async_llm_call(
            prompt=current_prompt, model=model, url=router.get_url(), api_key=api_key, timeout=timeout, temperature=0.0, max_tokens=answer_max_tokens,
        )
        current_correct = check_answer_match_with_item(response, item)
        attempts.append({
            'attempt': attempt_idx + 1,
            'is_correct': current_correct,
            'model_response': response,
        })
        final_response = response
        if current_correct:
            is_correct = True
            break

    return {
        'idx': idx,
        'status': 'success',
        'question': question,
        'difficulty_level': difficulty,
        'mode': mode,
        'semantic_features': features,
        'retrieval_used': use_principle,
        'retrieval_count': len(all_retrievals),
        'retrieved_principle': extract_retrieved_principle_text(chosen_retrieval),
        'similarity_scores': chosen_retrieval.get('dimension_similarities', {}) if chosen_retrieval else {},
        'model_answer': final_response,
        'expected_answer': expected,
        'is_correct': is_correct,
        'first_attempt_correct': attempts[0]['is_correct'] if attempts else False,
        'attempt_count': len(attempts),
        'retry_used': len(attempts) > 1,
        'rescued_by_retry': is_correct and any(not x.get('is_correct') for x in attempts[:-1]),
    }


class _MemoryRetrieverAdapter:
    """Adapter to use Memory class as a retriever in solve_one()."""

    def __init__(self, memory, top_k: int = 5, threshold: float = 0.55):
        self.memory = memory
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = self.memory.retrieve(features, top_k=self.top_k, threshold=self.threshold)
        # Convert Memory format to OnlinePrincipleRetriever format
        payload = []
        for r in results:
            item = r.get("item", {})
            principle_text = item.get("principle", "")
            payload.append({
                "matched_memory": {
                    "question": item.get("question", ""),
                    "golden_principle": principle_text,
                    "source_dataset": item.get("source_dataset", ""),
                },
                "dimension_similarities": r.get("dimension_similarities", {}),
                "score": r.get("score", 0.0),
                "structure_score": r.get("structure_score", r.get("score", 0.0)),
                "quality_score": r.get("quality_score", 0.5),
                "matched_facets": r.get("matched_facets", []),
                "memory_id": r.get("memory_id", ""),
                "principle_text": principle_text,
            })
        return payload


async def run() -> None:
    parser = argparse.ArgumentParser(description='Evaluate principle-guided retrieval on test set')
    parser.add_argument('--input', '-i', type=str, default=None)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--summary-output', type=str, default=None)
    parser.add_argument('--vector-db', type=str, default=None)
    parser.add_argument('--semantic-bank', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.55, help='三维必须同时超过的相似度阈值')
    parser.add_argument('--top-k', type=int, default=5, help='最大检索经验数')
    parser.add_argument('--memory-dir', type=str, default=None, help='Memory类目录(优先于vector-db)')
    parser.add_argument('--mode', choices=['baseline', 'guided'], default='guided')
    parser.add_argument('--concurrency', '-c', type=int, default=100, help='并发数')
    parser.add_argument('--model', '-m', type=str, default='MetaEVO')
    parser.add_argument('--url', type=str, default=None)
    parser.add_argument('--port', type=int, choices=[8000, 8001, 8002, 8003, 8008, 8009], default=None)
    parser.add_argument('--ports', type=int, nargs='+', default=None)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--embedding-model', type=str, default='BAAI/bge-m3')
    parser.add_argument('--prefer-faiss', action='store_true')
    parser.add_argument('--retry-on-wrong', type=int, default=2, help='If a response is judged wrong, retry this many additional attempts; total attempts are capped at 3.')
    parser.add_argument('--answer-max-tokens', type=int, default=4096, help='Max completion tokens for final answer generation attempts.')
    args = parser.parse_args()

    config = Config.from_core_file()
    config.ensure_dirs()
    config.setup_logging()

    input_path = args.input or str(config.paths.test_dir / 'test_filter.json')
    dataset_name = next((name for name in ["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"] if name in input_path), "dpo_level1")
    output_path = args.output or str(stage3_guided_results_file(config.paths.output_dir, dataset_name) if args.mode == 'guided' else stage3_baseline_results_file(config.paths.output_dir, dataset_name))
    summary_output = args.summary_output or str(stage3_guided_summary_file(config.paths.output_dir, dataset_name) if args.mode == 'guided' else stage3_baseline_summary_file(config.paths.output_dir, dataset_name))
    vector_db_path = args.vector_db or str(stage2_vector_dir(config.paths.output_dir, dataset_name))
    semantic_bank = args.semantic_bank or str(stage2_memory_file(config.paths.output_dir, dataset_name))

    llm_url = args.url or config.hyperparams.weak_model_url
    if args.port is not None and not args.ports:
        llm_url = apply_local_api_port(llm_url, args.port)

    router = DynamicRouter(build_urls(llm_url, args.ports))

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError('测试集必须是 JSON list')

    retriever = None
    memory_experience_count = 0
    if args.mode == 'guided':
        # Prefer Memory class if --memory-dir is given
        memory_dir = getattr(args, 'memory_dir', None)
        if memory_dir and Path(memory_dir).exists():
            from modules.memory import Memory
            mem = Memory(memory_dir)
            mem.load()
            core_embeddings = getattr(mem, "_core_embeddings", None)
            if core_embeddings is None or len(core_embeddings) == 0:
                mem.build_indices()
            memory_experience_count = len(getattr(mem, 'principles', []) or [])
            # Wrap Memory as a retriever-like object
            retriever = _MemoryRetrieverAdapter(mem, top_k=args.top_k, threshold=args.threshold)
        else:
            retriever = OnlinePrincipleRetriever(
                config=config, vector_db_path=vector_db_path, top_k=args.top_k,
                embedding_model_name=args.embedding_model, prefer_chroma=not args.prefer_faiss,
                semantic_memory_bank_path=semantic_bank,
                threshold=args.threshold,
            )

    metrics_context = collect_runtime_metrics(
        "stage3",
        {"input": input_path, "mode": args.mode, "dataset": dataset_name},
    )
    runtime = metrics_context.__enter__()

    semaphore = asyncio.Semaphore(args.concurrency)

    async def sem_task(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        async with semaphore:
            return await solve_one(
                item=item, idx=idx, retriever=retriever, model=args.model,
                router=router, api_key=config.hyperparams.weak_model_key,
                timeout=args.timeout, threshold=args.threshold, mode=args.mode,
                retry_on_wrong=args.retry_on_wrong, answer_max_tokens=args.answer_max_tokens,
            )

    tasks = [sem_task(item, idx) for idx, item in enumerate(data)]
    results: List[Dict[str, Any]] = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f'{args.mode.capitalize()} Test'):
        results.append(await coro)

    results.sort(key=lambda x: x.get('idx', 0))
    for item in results:
        item.pop('idx', None)

    valid_results = [r for r in results if r.get("status") == "success"]
    correct = sum(1 for r in valid_results if r.get("is_correct"))
    summary = build_stage3_summary(
        results,
        scope_total=len(results),
        correct=correct,
        memory_experience_count=memory_experience_count,
        dataset=dataset_name,
        mode=args.mode,
        runtime_metrics=runtime.finish(),
    )
    metrics_context.__exit__(None, None, None)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output_path)
    save_json(summary, summary_output)

    perf = summary["performance"]
    retrieval = summary["retrieval"]
    logger.info(f'=== {args.mode.capitalize()} Test Summary ({dataset_name}) ===')
    logger.info(f"Total accuracy: {perf['correct']}/{perf['scope_total']} = {perf['accuracy']:.4f}")
    logger.info(
        f"Retrieved experience: {retrieval['retrieved_samples']}/{perf['scope_total']} = "
        f"{retrieval['retrieval_used_ratio']:.4f}"
    )
    logger.info(f"Memory experiences: {retrieval['memory_experience_count']}")
    logger.info(
        f"Retrieved-experience accuracy: {retrieval['retrieved_correct']}/{retrieval['retrieved_samples']} = "
        f"{retrieval['retrieved_accuracy']:.4f}"
    )


if __name__ == '__main__':
    asyncio.run(run())
