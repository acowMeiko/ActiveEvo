#!/usr/bin/env python
"""
Stage 1 - Principle DPO generation.
Builds a DPO dataset from hard-negative pairs by extracting surface semantics,
comparing successful vs failed reasoning paths, generating candidate principles,
and using a guided answer crucible to select chosen/rejected principles.
"""
import sys
import json
import argparse
import asyncio
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm.asyncio import tqdm as async_tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.api_call import async_llm_call
from core.config import Config
from utils.output_paths import stage1_diff_failed_file, stage1_principle_file, stage1_mining_file
from template.prompt_template import (
    DIFF_PROMPT,
    GUIDED_ANSWER_PROMPT,
    SEMANTIC_SURFACE_EXTRACTION_PROMPT,
    STRUCTURED_PRINCIPLE_PROMPT,
)
from utils.io_utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _build_chat_completion_url(base_url: str, port: int) -> str:
    base = (base_url or '').strip() or 'http://localhost:8000/v1/chat/completions'
    return re.sub(r':\d+(?=/)', f':{port}', base)


def _extract_balanced_json(text: str) -> str:
    start = text.find('{')
    if start == -1:
        return ''

    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return ''


def _clean_json_string(text: str) -> str:
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _contains_cjk(text: str) -> bool:
    if not text:
        return False
    return re.search(r'[\u4e00-\u9fff]', text) is not None


_GENERIC_CORE_EXACT = {
    'structured reasoning',
    'problem solving',
    'mathematical reasoning',
    'general reasoning',
    'calculation',
    'reasoning',
    'math reasoning',
}

_GENERIC_CORE_PHRASES = (
    'structured reasoning and problem solving',
    'general mathematical reasoning',
)


def _semantics_too_generic(features: Dict[str, str]) -> bool:
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD
    core = str(features.get(CORE_FIELD, '')).strip().lower()
    constraints = str(features.get(CONDITION_FIELD, '')).strip().lower()
    objective = str(features.get(TARGET_FIELD, '')).strip().lower()
    if not core or not constraints or not objective:
        return True
    if core in _GENERIC_CORE_EXACT:
        return True
    if any(phrase in core for phrase in _GENERIC_CORE_PHRASES):
        return True
    if constraints in {'given conditions', 'problem conditions', 'some quantities are given'}:
        return True
    if objective in {'final answer', 'required quantity', 'unknown value'}:
        return True
    return False


def _semantic_fields_complete(features: Dict[str, str]) -> bool:
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
    return all(get_field(features, f) for f in [CORE_FIELD, CONDITION_FIELD, TARGET_FIELD])


def _heuristic_semantic_fallback(question: str) -> Dict[str, str]:
    from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD
    q = (question or '').strip()
    lower = q.lower()

    if any(token in lower for token in ['equally among', 'distributed equally', 'share equally', 'remainder']):
        concept = 'division with remainder'
        conditions = 'total quantity distributed equally among a fixed number of recipients'
        target = 'remaining quantity after equal distribution'
    elif 'probability' in lower or 'random' in lower:
        concept = 'probability calculation'
        conditions = 'discrete or geometric random selection under stated constraints'
        target = 'probability value'
    elif any(token in lower for token in ['triangle', 'tan ', 'sin ', 'cos ', 'hypotenuse']):
        concept = 'right triangle trigonometry'
        conditions = 'triangle with angle or side relations sufficient for trigonometric side matching'
        target = 'requested side length or angle value'
    elif any(token in lower for token in ['circle', 'polygon', 'coordinate', 'parallel', 'perpendicular']):
        concept = 'geometry relations'
        conditions = 'geometric figure with explicit angle, length, or incidence constraints'
        target = 'requested geometric measure'
    elif any(token in lower for token in ['ways', 'arrange', 'choose', 'permutation', 'combination', 'sequence', 'lineup', 'departure']):
        concept = 'combinatorial counting'
        conditions = 'finite selection or arrangement with explicit restrictions'
        target = 'number of valid configurations'
    elif any(token in lower for token in ['function', 'equation', 'polynomial', 'log', 'solve for']):
        concept = 'equation solving'
        conditions = 'algebraic relation with explicit variable constraints'
        target = 'value of the requested expression or variable'
    else:
        concept = 'arithmetic and algebraic manipulation'
        conditions = 'multiple given quantities linked by explicit arithmetic relations'
        target = 'requested numerical value'
    return {
        CORE_FIELD: concept,
        CONDITION_FIELD: conditions,
        TARGET_FIELD: target,
        # backward compat aliases
        'core_math_concept': concept,
        'conditions_structure': conditions,
        'solution_target': target,
    }


async def extract_semantic_features(
    question: str,
    model: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 600,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    if not question or not question.strip():
        return {"status": "invalid_input", "features": {}, "error": "empty_question"}

    config = Config.from_core_file()
    hp = config.hyperparams
    model = model or hp.weak_model_name
    url = url or hp.weak_model_url or 'http://localhost:8000/v1/chat/completions'
    api_key = api_key if api_key is not None else (hp.weak_model_key or 'EMPTY')

    generator = PrincipleGenerator(
        config=config,
        dataset_name="semantic_probe",
        urls=[url],
        timeout=timeout,
        model_override=model,
    )
    generator.api_key = api_key
    features = await generator.extract_surface_semantics(question.strip())
    return {
        "status": "success",
        "features": features,
        "raw": features,
    }


def _normalize_diff_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    differences = data.get('differences')
    if not isinstance(differences, list):
        return data

    normalized: List[Dict[str, str]] = []
    for item in differences:
        if not isinstance(item, dict):
            continue
        normalized.append({
            'Aspect': str(item.get('Aspect', item.get('aspect', ''))).strip(),
            'DivergenceStep': str(item.get('DivergenceStep', item.get('divergence_step', ''))).strip(),
            'CorrectThinking': str(item.get('CorrectThinking', item.get('HighQuality', item.get('correct_thinking', '')))).strip(),
            'IncorrectThinking': str(item.get('IncorrectThinking', item.get('LowQuality', item.get('incorrect_thinking', '')))).strip(),
            'WhyItMatters': str(item.get('WhyItMatters', item.get('Differences', item.get('why_it_matters', '')))).strip(),
        })

    if normalized:
        data['differences'] = normalized
    return data


def _normalize_principle_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    principles = data.get('principles')
    if not isinstance(principles, list):
        return data

    normalized: List[Dict[str, str]] = []
    for item in principles:
        if isinstance(item, str):
            item = {'scenario': '', 'principle': item}
        if not isinstance(item, dict):
            continue
        normalized.append({
            'scenario': str(item.get('scenario', item.get('Scenario', ''))).strip(),
            'principle': str(item.get('principle', item.get('Principle', ''))).strip(),
        })

    if normalized:
        data['principles'] = normalized
    return data


class PrincipleGenerator:
    def __init__(
        self,
        config: Config,
        dataset_name: str,
        urls: Optional[Sequence[str]] = None,
        timeout: int = 600,
        model_override: Optional[str] = None,
    ):
        self.config = config
        self.dataset_name = dataset_name
        self.timeout = timeout
        self.input_path = stage1_mining_file(config.paths.output_dir, dataset_name)

        hp = config.hyperparams
        self.model = model_override or hp.weak_model_name
        self.api_key = hp.weak_model_key or 'EMPTY'
        self.urls = list(urls) if urls else [hp.weak_model_url or 'http://localhost:8000/v1/chat/completions']
        self.req_counter = 0

    def _get_next_url(self) -> str:
        url = self.urls[self.req_counter % len(self.urls)]
        self.req_counter += 1
        return url

    async def call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        return await async_llm_call(
            prompt=prompt,
            model=self.model,
            url=self._get_next_url(),
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )

    def _json_candidates(self, text: str) -> List[str]:
        stripped = (text or '').strip()
        if not stripped:
            return []

        candidates = [stripped]
        if '```json' in stripped:
            try:
                candidates.append(stripped.split('```json', 1)[1].split('```', 1)[0].strip())
            except Exception:
                pass
        elif '```' in stripped:
            try:
                candidates.append(stripped.split('```', 1)[1].split('```', 1)[0].strip())
            except Exception:
                pass

        balanced = _extract_balanced_json(stripped)
        if balanced:
            candidates.append(balanced)

        start = stripped.find('{')
        end = stripped.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start:end + 1])

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    def extract_json(self, text: str, task: str = 'generic') -> Optional[Dict[str, Any]]:
        for candidate in self._json_candidates(text):
            for variant in [candidate, _clean_json_string(candidate)]:
                try:
                    parsed = json.loads(variant)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                if task == 'diff':
                    return _normalize_diff_fields(parsed)
                if task == 'principle':
                    return _normalize_principle_fields(parsed)
                return parsed
        return None

    async def _repair_json_with_model(self, raw_text: str, task: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None

        if task == 'diff':
            schema_hint = '{"differences":[{"Aspect":"...","DivergenceStep":"...","CorrectThinking":"...","IncorrectThinking":"...","WhyItMatters":"..."}]}'
        else:
            schema_hint = '{"principles":[{"scenario":"...","principle":"..."}]}'

        prompt = (
            'You are a JSON repair tool. Convert the malformed output below into one valid JSON object. '
            'Output raw JSON only, no markdown, no explanation. Preserve content whenever possible.\n\n'
            f'Task: {task}\n'
            f'Schema example: {schema_hint}\n\n'
            f'Malformed output:\n{raw_text}'
        )
        try:
            repaired = await self.call_llm(prompt, temperature=0.0, max_tokens=2048)
        except Exception:
            return None
        return self.extract_json(repaired, task=task)

    async def _repair_semantics_with_model(self, question: str, raw_text: str) -> Optional[Dict[str, str]]:
        from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD
        prompt = (
            'You are repairing a semantic-extraction result for a problem.\n'
            'Return valid JSON only with exactly these keys: '
            '"core_skill", "context_constraints", "task_objective".\n'
            'All values must be concise English.\n'
            'Match the Mem retrieval style: concrete, discriminative, and not overly generic.\n'
            'Bad core_skill examples: "structured reasoning", "problem solving", "calculation".\n'
            'Good core_skill examples: "division with remainder", "permutations with restrictions", "right triangle trigonometry".\n'
            'If the malformed output is incomplete, infer the missing field from the question.\n'
            'Do not output Chinese. Do not output markdown.\n\n'
            f'Question:\n{question}\n\n'
            f'Malformed or incomplete extraction:\n{raw_text}'
        )
        try:
            repaired = await self.call_llm(prompt, temperature=0.0, max_tokens=1024)
        except Exception:
            return None
        parsed = self.extract_json(repaired)
        if not parsed:
            return None
        features = {
            CORE_FIELD: str(parsed.get('core_skill', parsed.get('core_math_concept', parsed.get('Core Math Concept', '')))).strip(),
            CONDITION_FIELD: str(parsed.get('context_constraints', parsed.get('conditions_structure', parsed.get('Conditions Structure', '')))).strip(),
            TARGET_FIELD: str(parsed.get('task_objective', parsed.get('solution_target', parsed.get('Solution Target', '')))).strip(),
        }
        # backward compat aliases
        features['core_math_concept'] = features[CORE_FIELD]
        features['conditions_structure'] = features[CONDITION_FIELD]
        features['solution_target'] = features[TARGET_FIELD]
        if any(_contains_cjk(value) for value in features.values()):
            return None
        return features if _semantic_fields_complete(features) and not _semantics_too_generic(features) else None

    async def extract_surface_semantics(self, question: str) -> Dict[str, str]:
        from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD
        base_prompt = SEMANTIC_SURFACE_EXTRACTION_PROMPT.format(question=question)
        retry_prompt = (
            base_prompt
            + "\n\nReminder:\n"
            + "- Output valid JSON only.\n"
            + "- Every value must be written in English only.\n"
            + "- Do not output any Chinese characters.\n"
        )

        last_response = ''
        for prompt in [base_prompt, retry_prompt]:
            response = await self.call_llm(prompt, temperature=0.0, max_tokens=1024)
            last_response = response or ''
            parsed = self.extract_json(response)
            if not parsed:
                continue

            features = {
                CORE_FIELD: str(parsed.get('core_skill', parsed.get('core_math_concept', parsed.get('Core Math Concept', '')))).strip(),
                CONDITION_FIELD: str(parsed.get('context_constraints', parsed.get('conditions_structure', parsed.get('Conditions Structure', '')))).strip(),
                TARGET_FIELD: str(parsed.get('task_objective', parsed.get('solution_target', parsed.get('Solution Target', '')))).strip(),
            }
            # backward compat aliases
            features['core_math_concept'] = features[CORE_FIELD]
            features['conditions_structure'] = features[CONDITION_FIELD]
            features['solution_target'] = features[TARGET_FIELD]
            if any(_contains_cjk(value) for value in [features[CORE_FIELD], features[CONDITION_FIELD], features[TARGET_FIELD]]):
                logger.warning("Semantic extraction returned non-English fields; retrying with stricter English-only reminder.")
                continue
            if _semantic_fields_complete(features) and not _semantics_too_generic(features):
                features['_semantics_source'] = 'model'
                return features

        repaired = await self._repair_semantics_with_model(question, last_response)
        if repaired:
            repaired['_semantics_source'] = 'model'
            return repaired

        logger.warning('Semantic extraction remained incomplete; using heuristic fallback semantics.')
        fallback = _heuristic_semantic_fallback(question)
        fallback['_semantics_source'] = 'fallback'
        return fallback

    async def diff_analysis(self, question: str, chosen_path: str, rejected_path: str, surface_semantics: Dict[str, str]) -> Dict[str, Any]:
        from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
        prompt = DIFF_PROMPT.format(
            core_skill=get_field(surface_semantics, CORE_FIELD),
            context_constraints=get_field(surface_semantics, CONDITION_FIELD),
            task_objective=get_field(surface_semantics, TARGET_FIELD),
            question=question,
            chosen_path=chosen_path,
            rejected_path=rejected_path,
        )

        attempts = [
            {'temperature': 0.2, 'max_tokens': 4096, 'label': 'initial'},
            {'temperature': 0.0, 'max_tokens': 3072, 'label': 'retry_strict'},
        ]
        last_response = ''
        parse_stage = 'no_response'

        for attempt in attempts:
            try:
                response = await self.call_llm(prompt, temperature=attempt['temperature'], max_tokens=attempt['max_tokens'])
            except Exception as exc:
                return {'status': f'error: {exc}', 'differences': [], 'parse_stage': attempt['label'], 'raw': last_response, 'raw_preview': last_response[:500]}

            last_response = response or ''
            parsed = self.extract_json(last_response, task='diff')
            if parsed and isinstance(parsed.get('differences'), list) and parsed.get('differences'):
                return {'status': 'success', 'differences': parsed['differences'], 'parse_stage': attempt['label']}

            repaired = await self._repair_json_with_model(last_response, task='diff')
            if repaired and isinstance(repaired.get('differences'), list) and repaired.get('differences'):
                return {'status': 'success', 'differences': repaired['differences'], 'parse_stage': f"{attempt['label']}_repair"}

            parse_stage = f"{attempt['label']}_repair_failed"

        return {'status': 'parse_failed', 'differences': [], 'parse_stage': parse_stage, 'raw': last_response, 'raw_preview': last_response[:500]}

    async def principle_generation(self, question: str, diff_list: List[Dict[str, Any]], surface_semantics: Dict[str, str]) -> Dict[str, Any]:
        from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
        prompt = STRUCTURED_PRINCIPLE_PROMPT.format(
            question=question,
            core_skill=get_field(surface_semantics, CORE_FIELD),
            context_constraints=get_field(surface_semantics, CONDITION_FIELD),
            task_objective=get_field(surface_semantics, TARGET_FIELD),
            diff_list=json.dumps({'differences': diff_list}, ensure_ascii=False),
        )

        response = await self.call_llm(prompt, temperature=0.3, max_tokens=2048)
        parsed = self.extract_json(response, task='principle')
        if not parsed or not isinstance(parsed.get('principles'), list):
            repaired = await self._repair_json_with_model(response, task='principle')
            parsed = repaired if repaired and isinstance(repaired.get('principles'), list) else None

        if not parsed or not parsed.get('principles'):
            return {'status': 'parse_failed', 'principles': [], 'raw': response, 'raw_preview': response[:500]}

        return {'status': 'success', 'principles': parsed['principles']}

    def extract_final_answer(self, text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'Final Answer:\s*\\boxed\{([^}]+)\}',
            r'Final Answer:\s*([^\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip().strip('$')
        return None

    def check_answer_match(self, generated: str, expected: str) -> bool:
        generated_answer = (self.extract_final_answer(generated) or generated or '').strip().lower()
        expected_answer = (expected or '').strip().lower()
        if not generated_answer or not expected_answer:
            return False
        return expected_answer in generated_answer or generated_answer in expected_answer

    def _extract_principle_text(self, principle: Any) -> str:
        if isinstance(principle, str):
            return principle.strip()
        if isinstance(principle, dict):
            return str(principle.get('principle', '')).strip()
        return ''

    async def crucible_test(self, question: str, principle: Dict[str, str], expected_answer: str) -> Dict[str, Any]:
        principle_text = self._extract_principle_text(principle)
        if not principle_text:
            return {'principle': principle, 'response': '', 'is_correct': False, 'error': 'empty_principle'}

        prompt = GUIDED_ANSWER_PROMPT.format(principle_text=principle_text, question=question)
        try:
            response = await self.call_llm(prompt, temperature=0.0, max_tokens=4096)
        except Exception as exc:
            return {'principle': principle, 'response': '', 'is_correct': False, 'error': str(exc)}

        return {
            'principle': principle,
            'response': response[:1200],
            'is_correct': self.check_answer_match(response, expected_answer),
        }

    def _build_instruction(self) -> str:
        return (
            'Analyze the following math problem, along with a Successful Path and a Failed Path.\n'
            'First, identify the "Thinking Gaps" between the two paths. Then, perform a "Meta-Abstraction" '
            'to derive ONE highly condensed, actionable reasoning principle.\n'
            'The chosen principle should help solve the problem correctly, while the rejected principle should be weaker.'
        )

    def _build_input(self, question: str, chosen_path: str, rejected_path: str) -> str:
        return (
            f'[Question]:\n{question}\n\n'
            f'[Successful Path]:\n{chosen_path}\n\n'
            f'[Failed Path]:\n{rejected_path}'
        )

    def build_principle_record(
        self,
        question: str,
        chosen_path: str,
        rejected_path: str,
        chosen_principle: str,
        rejected_principle: str,
        diff_list: List[Dict[str, Any]],
        surface_semantics: Dict[str, str],
    ) -> Dict[str, Any]:
        from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
        return {
            'instruction': self._build_instruction(),
            'input': self._build_input(question, chosen_path, rejected_path),
            'core_skill': get_field(surface_semantics, CORE_FIELD),
            'context_constraints': get_field(surface_semantics, CONDITION_FIELD),
            'task_objective': get_field(surface_semantics, TARGET_FIELD),
            # backward compat aliases
            'core_math_concept': get_field(surface_semantics, CORE_FIELD),
            'conditions_structure': get_field(surface_semantics, CONDITION_FIELD),
            'solution_target': get_field(surface_semantics, TARGET_FIELD),
            'chosen': chosen_principle,
            'rejected': rejected_principle,
            'diff': json.dumps({'differences': diff_list}, ensure_ascii=False),
        }

    async def process_single(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = str(item.get('question', '')).strip()
        expected_answer = str(item.get('expected_answer', '')).strip()
        chosen_path = str(item.get('chosen_path', '')).strip()
        rejected_path = str(item.get('rejected_path', '')).strip()
        if not all([question, expected_answer, chosen_path, rejected_path]):
            return None

        surface_semantics = await self.extract_surface_semantics(question)
        diff_result = await self.diff_analysis(question, chosen_path, rejected_path, surface_semantics)
        if diff_result.get('status') != 'success':
            return {
                'idx': item.get('idx', 0),
                'status': 'diff_failed',
                'diff_status': diff_result.get('status', ''),
                'parse_stage': diff_result.get('parse_stage', ''),
                'question': question,
                'expected_answer': expected_answer,
                'chosen_path': chosen_path,
                'rejected_path': rejected_path,
                'surface_semantics': surface_semantics,
                'raw_preview': diff_result.get('raw_preview', ''),
                'raw': diff_result.get('raw', ''),
            }

        diff_list = diff_result.get('differences', [])
        principle_result = await self.principle_generation(question, diff_list, surface_semantics)
        if principle_result.get('status') != 'success':
            return {
                'idx': item.get('idx', 0),
                'status': 'principle_failed',
                'diff_list': diff_list,
                'raw_preview': principle_result.get('raw_preview', ''),
                'raw': principle_result.get('raw', ''),
            }

        principles = principle_result.get('principles', [])
        crucible_results = await asyncio.gather(
            *(self.crucible_test(question, p, expected_answer) for p in principles),
            return_exceptions=True,
        )

        passed: List[str] = []
        failed: List[str] = []
        for idx, result in enumerate(crucible_results):
            principle_text = self._extract_principle_text(principles[idx]) if idx < len(principles) else ''
            if isinstance(result, Exception):
                failed.append(principle_text)
                continue
            if result.get('is_correct'):
                passed.append(self._extract_principle_text(result.get('principle')))
            else:
                failed.append(self._extract_principle_text(result.get('principle')))

        passed = [p for p in passed if p]
        failed = [p for p in failed if p and p not in passed]
        if not passed or not failed:
            return {'idx': item.get('idx', 0), 'status': 'no_valid_principles', 'diff_list': diff_list}

        principle_record = self.build_principle_record(
            question=question,
            chosen_path=chosen_path,
            rejected_path=rejected_path,
            chosen_principle=passed[0],
            rejected_principle=failed[0],
            diff_list=diff_list,
            surface_semantics=surface_semantics,
        )
        return {'idx': item.get('idx', 0), 'status': 'success', 'principle_record': principle_record}

    async def run(self, concurrency: int = 20, checkpoint_interval: int = 10) -> Dict[str, Any]:
        if not self.input_path.exists():
            logger.error(f'Input file not found: {self.input_path}')
            return {'error': f'Input file not found: {self.input_path}'}

        divergent_results = load_json(str(self.input_path))
        if not isinstance(divergent_results, list):
            return {'error': 'invalid_data_format'}

        semaphore = asyncio.Semaphore(concurrency)
        output_path = stage1_principle_file(self.config.paths.output_dir, self.dataset_name)
        diff_failed_path = stage1_diff_failed_file(self.config.paths.output_dir, self.dataset_name)
        diff_summary_path = diff_failed_path.with_name('diff_failure_summary.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async def sem_task(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.process_single(record)

        processed: List[Dict[str, Any]] = []
        checkpoint_count = 0
        tasks = [sem_task(item) for item in divergent_results]

        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Generating Principles'):
            result = await coro
            if result:
                processed.append(result)
            checkpoint_count += 1
            if checkpoint_count % checkpoint_interval == 0:
                self._save_intermediate(processed, output_path, diff_failed_path, diff_summary_path)

        self._save_intermediate(processed, output_path, diff_failed_path, diff_summary_path)
        stats = {
            'total_divergent': len(divergent_results),
            'success': sum(1 for r in processed if r.get('status') == 'success'),
            'diff_failed': sum(1 for r in processed if r.get('status') == 'diff_failed'),
            'principle_failed': sum(1 for r in processed if r.get('status') == 'principle_failed'),
            'no_valid_principles': sum(1 for r in processed if r.get('status') == 'no_valid_principles'),
        }
        logger.info(f'Principle dataset saved to: {output_path}')
        logger.info(f'Diff failures saved to: {diff_failed_path}')
        logger.info(f'Diff failure summary saved to: {diff_summary_path}')
        return stats

    def _collect_diff_failed_items(self, processed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                'idx': r.get('idx', 0),
                'diff_status': r.get('diff_status', ''),
                'parse_stage': r.get('parse_stage', ''),
                'question': r.get('question', ''),
                'expected_answer': r.get('expected_answer', ''),
                'chosen_path': r.get('chosen_path', ''),
                'rejected_path': r.get('rejected_path', ''),
                'surface_semantics': r.get('surface_semantics', {}),
                'raw_preview': r.get('raw_preview', ''),
                'raw': r.get('raw', ''),
            }
            for r in processed if r.get('status') == 'diff_failed'
        ]

    def _build_diff_failure_summary(
        self,
        diff_failed_items: List[Dict[str, Any]],
        processed: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        status_counter = Counter(item.get('diff_status') or 'unknown' for item in diff_failed_items)
        parse_stage_counter = Counter(item.get('parse_stage') or 'unknown' for item in diff_failed_items)
        by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for item in diff_failed_items:
            stage = item.get('parse_stage') or 'unknown'
            if len(by_stage[stage]) < 3:
                by_stage[stage].append({
                    'idx': item.get('idx', 0),
                    'diff_status': item.get('diff_status', ''),
                    'question_preview': str(item.get('question', ''))[:200],
                    'raw_preview': str(item.get('raw_preview', ''))[:300],
                })

        summary = {
            'dataset': self.dataset_name,
            'totals': {
                'processed': len(processed),
                'success': sum(1 for r in processed if r.get('status') == 'success'),
                'diff_failed': len(diff_failed_items),
                'principle_failed': sum(1 for r in processed if r.get('status') == 'principle_failed'),
                'no_valid_principles': sum(1 for r in processed if r.get('status') == 'no_valid_principles'),
            },
            'diff_status_counts': dict(status_counter),
            'parse_stage_counts': dict(parse_stage_counter),
            'top_failure_examples': dict(by_stage),
        }
        return summary

    def _log_diff_failure_summary(self, summary: Dict[str, Any]) -> None:
        totals = summary.get('totals', {})
        diff_failed = totals.get('diff_failed', 0)
        processed = totals.get('processed', 0)
        logger.info(
            'Diff failure summary | dataset=%s | diff_failed=%s/%s | diff_status=%s | parse_stage=%s',
            self.dataset_name,
            diff_failed,
            processed,
            summary.get('diff_status_counts', {}),
            summary.get('parse_stage_counts', {}),
        )
        for stage, examples in summary.get('top_failure_examples', {}).items():
            if not examples:
                continue
            first = examples[0]
            logger.info(
                'Example failure | stage=%s | idx=%s | status=%s | question=%s',
                stage,
                first.get('idx', 0),
                first.get('diff_status', ''),
                first.get('question_preview', ''),
            )

    def _save_intermediate(
        self,
        processed: List[Dict[str, Any]],
        output_path: Path,
        diff_failed_path: Path,
        diff_summary_path: Path,
    ) -> None:
        principle_items = [r['principle_record'] for r in processed if r.get('status') == 'success' and r.get('principle_record')]
        diff_failed_items = self._collect_diff_failed_items(processed)
        diff_summary = self._build_diff_failure_summary(diff_failed_items, processed)
        save_json(principle_items, str(output_path))
        save_json(diff_failed_items, str(diff_failed_path))
        save_json(diff_summary, str(diff_summary_path))
        self._log_diff_failure_summary(diff_summary)

    async def run_from_file(self, input_path: str, output_path: Optional[str] = None, concurrency: int = 20) -> Dict[str, Any]:
        original_input = self.input_path
        self.input_path = Path(input_path)
        try:
            stats = await self.run(concurrency=concurrency)
            if output_path and 'error' not in stats:
                generated_path = stage1_principle_file(self.config.paths.output_dir, self.dataset_name)
                save_json(load_json(str(generated_path)), output_path)
            return stats
        finally:
            self.input_path = original_input


# Backward compatibility alias
PrincipleDPGenerator = PrincipleGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate principle dataset from hard-negative pairs.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', default='')
    parser.add_argument('--concurrency', type=int, default=20)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--ports', nargs='*', type=int, default=[])
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    config = Config.from_core_file()
    config.ensure_dirs()

    urls: Optional[List[str]] = None
    if args.ports:
        base_url = config.hyperparams.weak_model_url
        urls = [_build_chat_completion_url(base_url, port) for port in args.ports]

    generator = PrincipleGenerator(
        config=config,
        dataset_name=args.dataset,
        urls=urls,
        timeout=args.timeout,
        model_override=args.model or None,
    )
    stats = await generator.run(concurrency=args.concurrency)
    logger.info(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if 'error' not in stats else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(_async_main(args))


if __name__ == '__main__':
    raise SystemExit(main())
