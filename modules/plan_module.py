"""
modules/plan_module.py — PlanModule 类

Agent 的「规划」模块，负责：
  - 语义特征提取 (三维: core_skill / context_constraints / task_objective)
  - Diff 分析 (对比正确/错误推理路径)
  - Principle 生成 (从 diff 中抽象出可复用经验)
  - Crucible 坩埚验证 (用生成的 principle 引导答题，检验是否有效)
"""
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from core.api_call import async_llm_call
from core.config import Config
from core.field_names import CORE_FIELD, CONDITION_FIELD, TARGET_FIELD, get_field
from modules.execute_module import URLRouter, filter_healthy_urls
from modules.principle_schema import validate_condition_card, validate_principle_content
from template.prompt_template import (
    DIFF_PROMPT,
    GUIDED_ANSWER_PROMPT,
    HOLISTIC_DESCRIPTION_PROMPT,
    MEMORY_ACTION_BINDING_PROMPT,
    MEMORY_APPLICABILITY_PROMPT,
    REPAIR_DIAGNOSIS_PROMPT,
    REPAIR_PATCH_PROMPT,
    SEMANTIC_SURFACE_EXTRACTION_PROMPT,
    STRUCTURED_PRINCIPLE_PROMPT,
)

logger = logging.getLogger(__name__)


# ── JSON helpers (移植自 generate_principle.py) ───────────────────────


def _extract_balanced_json(text: str) -> str:
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [index for index in (object_start, array_start) if index >= 0]
    if not starts:
        return ""
    start = min(starts)
    stack: List[str] = []
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or ch != stack[-1]:
                return ""
            stack.pop()
            if not stack:
                return text[start : idx + 1]
    return ""


def _clean_json_string(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\\(?![\"\\\/bfnrtu])", r"\\\\", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


_GENERIC_CORE_EXACT = {
    "structured reasoning",
    "problem solving",
    "mathematical reasoning",
    "general reasoning",
    "calculation",
    "reasoning",
    "math reasoning",
}

_GENERIC_CORE_PHRASES = (
    "structured reasoning and problem solving",
    "general mathematical reasoning",
)


def _semantics_too_generic(features: Dict[str, str]) -> bool:
    core = str(features.get(CORE_FIELD, "")).strip().lower()
    constraints = str(features.get(CONDITION_FIELD, "")).strip().lower()
    objective = str(features.get(TARGET_FIELD, "")).strip().lower()
    if not core or not constraints or not objective:
        return True
    if core in _GENERIC_CORE_EXACT:
        return True
    if any(phrase in core for phrase in _GENERIC_CORE_PHRASES):
        return True
    if constraints in {"given conditions", "problem conditions", "some quantities are given"}:
        return True
    if objective in {"final answer", "required quantity", "unknown value"}:
        return True
    return False


def _normalize_diff_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    differences = data.get("differences")
    if not isinstance(differences, list):
        return data
    normalized = []
    for item in differences:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "Aspect": str(item.get("Aspect", item.get("aspect", ""))).strip(),
                "DivergenceStep": str(item.get("DivergenceStep", item.get("divergence_step", ""))).strip(),
                "CorrectThinking": str(
                    item.get("CorrectThinking", item.get("HighQuality", item.get("correct_thinking", "")))
                ).strip(),
                "WrongThinking": str(
                    item.get("WrongThinking", item.get("IncorrectThinking", item.get("LowQuality", item.get("incorrect_thinking", ""))))
                ).strip(),
                "CausalError": str(item.get("CausalError", item.get("causal_error", ""))).strip(),
                "WhyItMatters": str(
                    item.get("WhyItMatters", item.get("Differences", item.get("why_it_matters", "")))
                ).strip(),
            }
        )
    if normalized:
        data["differences"] = normalized
    return data


def _normalize_principle_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    principles = data.get("principles")
    if not isinstance(principles, list):
        return data
    normalized = []
    for item in principles:
        if isinstance(item, str):
            item = {"action_guidance": item}
        if not isinstance(item, dict):
            continue
        principle_text = str(item.get("principle", item.get("Principle", ""))).strip()
        action_guidance = str(item.get("action_guidance", item.get("ActionGuidance", ""))).strip()
        if not action_guidance and principle_text:
            action_guidance = principle_text
        evidence = item.get("required_evidence", item.get("RequiredEvidence", []))
        if isinstance(evidence, str):
            evidence = [evidence.strip()] if evidence.strip() else []
        elif isinstance(evidence, list):
            evidence = [str(value).strip() for value in evidence if isinstance(value, str) and value.strip()]
        failure_signals = item.get("failure_signals", item.get("FailureSignals", []))
        if isinstance(failure_signals, str):
            failure_signals = [failure_signals.strip()] if failure_signals.strip() else []
        elif isinstance(failure_signals, list):
            failure_signals = [
                str(value).strip() for value in failure_signals
                if isinstance(value, str) and value.strip()
            ]
        normalized.append(
            {
                "introspection": str(item.get("introspection", item.get("Introspection", ""))).strip(),
                "trigger_condition": str(item.get("trigger_condition", item.get("TriggerCondition", ""))).strip(),
                "required_evidence": evidence,
                "exclusion_condition": str(item.get("exclusion_condition", item.get("ExclusionCondition", ""))).strip(),
                "action_guidance": action_guidance,
                "verification_step": str(item.get("verification_step", item.get("VerificationStep", ""))).strip(),
                "generalization_level": str(item.get("generalization_level", item.get("GeneralizationLevel", "family"))).strip(),
                "scope": str(item.get("scope", item.get("Scope", ""))).strip(),
                "failure_mode": str(item.get("failure_mode", item.get("FailureMode", ""))).strip(),
                "failure_signals": failure_signals,
            }
        )
    if normalized:
        data["principles"] = normalized
    return data


def _heuristic_semantic_fallback(question: str) -> Dict[str, str]:
    lower = (question or "").lower()
    if any(token in lower for token in ["equally among", "distributed equally", "share equally", "remainder"]):
        concept = "division with remainder"
        conditions = "total quantity distributed equally among a fixed number of recipients"
        target = "remaining quantity after equal distribution"
    elif "probability" in lower or "random" in lower:
        concept = "probability calculation"
        conditions = "discrete or geometric random selection under stated constraints"
        target = "probability value"
    elif any(t in lower for t in ["triangle", "tan ", "sin ", "cos ", "hypotenuse"]):
        concept = "right triangle trigonometry"
        conditions = "triangle with angle or side relations sufficient for trigonometric side matching"
        target = "requested side length or angle value"
    elif any(t in lower for t in ["circle", "polygon", "coordinate", "parallel", "perpendicular"]):
        concept = "geometry relations"
        conditions = "geometric figure with explicit angle, length, or incidence constraints"
        target = "requested geometric measure"
    elif any(t in lower for t in ["ways", "arrange", "choose", "permutation", "combination", "sequence", "lineup", "departure"]):
        concept = "combinatorial counting"
        conditions = "finite selection or arrangement with explicit restrictions"
        target = "number of valid configurations"
    elif any(t in lower for t in ["function", "equation", "polynomial", "log", "solve for"]):
        concept = "equation solving"
        conditions = "algebraic relation with explicit variable constraints"
        target = "value of the requested expression or variable"
    else:
        concept = "arithmetic and algebraic manipulation"
        conditions = "multiple given quantities linked by explicit arithmetic relations"
        target = "requested numerical value"
    return {
        CORE_FIELD: concept,
        CONDITION_FIELD: conditions,
        TARGET_FIELD: target,
        "core_math_concept": concept,
        "conditions_structure": conditions,
        "solution_target": target,
    }


# ── PlanModule 类 ─────────────────────────────────────────────────────


class PlanModule:
    """Agent 的规划模块。

    使用方式:
        plan = PlanModule(config, urls=[...])
        semantics = await plan.extract_semantics(question)
        diff = await plan.diff_analysis(question, chosen, rejected, semantics)
        principles = await plan.generate_principles(question, diff, semantics)
        result = await plan.crucible_test(question, principle, expected)
    """

    def __init__(
        self,
        config: Config,
        urls: Optional[Sequence[str]] = None,
        model_override: Optional[str] = None,
        timeout: int = 600,
        seed: Optional[int] = None,
    ):
        hp = config.hyperparams
        self.model = model_override or hp.weak_model_name
        self.api_key = hp.weak_model_key or "EMPTY"
        initial_urls = list(urls) if urls else [hp.weak_model_url]
        self.urls = filter_healthy_urls(initial_urls)
        self.timeout = timeout
        self.seed = seed
        self.router = URLRouter(self.urls)

    # ── URL 轮询 ──

    def _next_url(self) -> str:
        return self.router.next()

    # ── LLM 调用 ──

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> str:
        attempts = max(1, self.router.attempt_budget())
        last_response = ""
        for _ in range(attempts):
            url = self._next_url()
            request_seed = self.seed if seed is None else seed
            response = await async_llm_call(
                prompt=prompt,
                model=self.model,
                url=url,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                retries=1,
                **({"seed": request_seed} if request_seed is not None else {}),
            )
            if response:
                self.router.report_success(url)
                return response
            # Empty response conflates timeout, server error, and parse-empty.
            # Continue on the next endpoint without quarantining a GPU; only
            # explicit connection/health failures may change endpoint health.
            logger.warning("Plan/Gate request returned empty; trying a different endpoint without quarantine: %s", url)
            last_response = response
        return last_response

    # ── JSON 解析 ──

    def _json_candidates(self, text: str) -> List[str]:
        stripped = (text or "").strip()
        if not stripped:
            return []
        candidates = [stripped]
        if "```json" in stripped:
            try:
                candidates.append(stripped.split("```json", 1)[1].split("```", 1)[0].strip())
            except Exception:
                pass
        elif "```" in stripped:
            try:
                candidates.append(stripped.split("```", 1)[1].split("```", 1)[0].strip())
            except Exception:
                pass
        balanced = _extract_balanced_json(stripped)
        if balanced:
            candidates.append(balanced)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1])
        seen = set()
        deduped = []
        for c in candidates:
            if c and c not in seen:
                deduped.append(c)
                seen.add(c)
        return deduped

    def _extract_json(self, text: str, task: str = "generic") -> Optional[Dict[str, Any]]:
        for candidate in self._json_candidates(text):
            for variant in [candidate, _clean_json_string(candidate)]:
                try:
                    parsed = json.loads(variant)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                if task == "diff":
                    return _normalize_diff_fields(parsed)
                if task == "principle":
                    return _normalize_principle_fields(parsed)
                return parsed
        return None

    def _extract_gate_json(self, text: str) -> tuple[Optional[Dict[str, Any]], str]:
        """Normalize the two common Gate response shapes without losing auditability.

        The documented protocol is ``{"judgments": [...]}``, while some local
        models emit the semantically equivalent top-level ``[...]``. Both are
        executable after normalization; ``parse_stage`` records which shape was
        actually returned so protocol compliance remains measurable.
        """
        for candidate in self._json_candidates(text):
            for variant in (candidate, _clean_json_string(candidate)):
                try:
                    parsed = json.loads(variant)
                except Exception:
                    continue
                if isinstance(parsed, dict) and isinstance(parsed.get("judgments"), list):
                    return parsed, "wrapped_object"
                if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                    return {"judgments": parsed}, "direct_list_normalized"
                if isinstance(parsed, dict) and parsed.get("memory_id"):
                    return {"judgments": [parsed]}, "single_judgment_normalized"
        return None, "parse_failed"

    def _extract_stage2_principle_json(self, text: str) -> tuple[Optional[Dict[str, Any]], str]:
        """Parse the canonical Stage-2 Principle envelope without coercing fields.

        The canonical output is ``{"principle": {...}}``. A direct content
        object or a one-element ``principles`` array is structurally
        equivalent and can be normalized losslessly; field types remain
        subject to the strict schema validator.
        """
        required = {
            "introspection", "trigger_condition", "required_evidence",
            "failure_signals", "exclusion_condition", "action_guidance", "verification_step",
            "generalization_level", "scope", "failure_mode",
        }
        for candidate in self._json_candidates(text):
            for variant in (candidate, _clean_json_string(candidate)):
                try:
                    parsed = json.loads(variant)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                if isinstance(parsed.get("principle"), dict):
                    return {"principle": parsed["principle"]}, "wrapped_object"
                principles = parsed.get("principles")
                if isinstance(principles, list) and len(principles) == 1 and isinstance(principles[0], dict):
                    return {"principle": principles[0]}, "single_list_normalized"
                if required.issubset(parsed.keys()):
                    return {"principle": parsed}, "direct_object_normalized"
        return None, "parse_failed"

    async def _repair_json(self, raw: str, task: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        if task == "diff":
            schema = '{"differences":[{"Aspect":"...","DivergenceStep":"...","CorrectThinking":"...","IncorrectThinking":"...","WhyItMatters":"..."}]}'
        elif task == "principle":
            schema = (
                '{"principles":[{"introspection":"...","trigger_condition":"...",'
                '"required_evidence":["..."],"exclusion_condition":"...",'
                '"failure_signals":["..."],'
                '"action_guidance":"...","verification_step":"...",'
                '"generalization_level":"family","scope":"...","failure_mode":"..."}]}'
            )
        elif task == "stage2_principle":
            schema = (
                '{"principle":{"introspection":"...","trigger_condition":"...",'
                '"required_evidence":["..."],"exclusion_condition":"...",'
                '"failure_signals":["..."],'
                '"action_guidance":"...","verification_step":"...",'
                '"generalization_level":"family","scope":"...","failure_mode":"..."}}'
            )
        else:
            schema = '{"result":"..."}'
        prompt = (
            "You are a JSON repair tool. Convert the malformed output into one valid JSON object. "
            "Output raw JSON only, no markdown, no explanation.\n\n"
            f"Task: {task}\nSchema example: {schema}\n\nMalformed output:\n{raw}"
        )
        try:
            repaired = await self._call_llm(prompt, temperature=0.0, max_tokens=2048)
        except Exception:
            return None
        if task == "stage2_principle":
            parsed, _ = self._extract_stage2_principle_json(repaired)
            return parsed
        return self._extract_json(repaired, task=task)

    async def _repair_semantics(self, question: str, raw: str) -> Optional[Dict[str, str]]:
        prompt = (
            "You are repairing a semantic-extraction result for a problem.\n"
            "Return valid JSON only with exactly these keys: "
            '"core_skill", "context_constraints", "task_objective".\n'
            "All values must be concise English.\n"
            "Match the Mem retrieval style: concrete, discriminative, and not overly generic.\n"
            'Bad core_skill examples: "structured reasoning", "problem solving", "calculation".\n'
            'Good core_skill examples: "division with remainder", "permutations with restrictions", "right triangle trigonometry".\n'
            "If the malformed output is incomplete, infer the missing field from the question.\n"
            "Do not output Chinese. Do not output markdown.\n\n"
            f"Question:\n{question}\n\nMalformed or incomplete extraction:\n{raw}"
        )
        try:
            repaired = await self._call_llm(prompt, temperature=0.0, max_tokens=1024)
        except Exception:
            return None
        parsed = self._extract_json(repaired)
        if not parsed:
            return None
        features = {
            CORE_FIELD: str(parsed.get("core_skill", parsed.get("core_math_concept", ""))).strip(),
            CONDITION_FIELD: str(parsed.get("context_constraints", parsed.get("conditions_structure", ""))).strip(),
            TARGET_FIELD: str(parsed.get("task_objective", parsed.get("solution_target", ""))).strip(),
        }
        features["core_math_concept"] = features[CORE_FIELD]
        features["conditions_structure"] = features[CONDITION_FIELD]
        features["solution_target"] = features[TARGET_FIELD]
        if any(_contains_cjk(v) for v in [features[CORE_FIELD], features[CONDITION_FIELD], features[TARGET_FIELD]]):
            return None
        from core.field_names import semantic_fields_complete
        return features if semantic_fields_complete(features) and not _semantics_too_generic(features) else None

    # ================================================================
    # 公共接口
    # ================================================================

    async def extract_semantics(self, question: str) -> Dict[str, str]:
        """提取问题的三维语义特征。

        Returns:
            dict with keys: core_skill, context_constraints, task_objective
            以及旧字段名兼容别名。
        """
        base_prompt = SEMANTIC_SURFACE_EXTRACTION_PROMPT.format(question=question)
        retry_prompt = (
            base_prompt
            + "\n\nReminder:\n"
            + "- Output valid JSON only.\n"
            + "- Every value must be written in English only.\n"
            + "- Do not output any Chinese characters.\n"
        )
        last_response = ""
        for prompt in [base_prompt, retry_prompt]:
            response = await self._call_llm(prompt, temperature=0.0, max_tokens=1024)
            last_response = response or ""
            parsed = self._extract_json(response)
            if not parsed:
                continue
            features = {
                CORE_FIELD: str(parsed.get("core_skill", parsed.get("core_math_concept", ""))).strip(),
                CONDITION_FIELD: str(parsed.get("context_constraints", parsed.get("conditions_structure", ""))).strip(),
                TARGET_FIELD: str(parsed.get("task_objective", parsed.get("solution_target", ""))).strip(),
            }
            features["core_math_concept"] = features[CORE_FIELD]
            features["conditions_structure"] = features[CONDITION_FIELD]
            features["solution_target"] = features[TARGET_FIELD]
            if any(_contains_cjk(v) for v in [features[CORE_FIELD], features[CONDITION_FIELD], features[TARGET_FIELD]]):
                continue
            from core.field_names import semantic_fields_complete
            if semantic_fields_complete(features) and not _semantics_too_generic(features):
                features["_semantics_source"] = "model"
                return features

        repaired = await self._repair_semantics(question, last_response)
        if repaired:
            repaired["_semantics_source"] = "model"
            return repaired

        logger.warning("Semantic extraction failed; using heuristic fallback.")
        fallback = _heuristic_semantic_fallback(question)
        fallback["_semantics_source"] = "fallback"
        return fallback

    async def extract_holistic_description(self, question: str) -> str:
        """Generate a single holistic retrieval key via LLM (single-vector ablation).

        Instead of decomposing into 3 facets, the LLM produces one concise English
        sentence capturing the skill + constraints + objective.
        """
        prompt = HOLISTIC_DESCRIPTION_PROMPT.format(question=question)
        response = await self._call_llm(prompt, temperature=0.0, max_tokens=512)
        if response:
            parsed = self._extract_json(response)
            if parsed and isinstance(parsed, dict):
                desc = str(parsed.get("description", "")).strip()
                if desc and not _contains_cjk(desc):
                    return desc
        # Fallback: use the question text directly (NOT DSD fields — this is an ablation)
        logger.warning("Holistic description generation failed; using raw question as fallback.")
        return question.strip()

    async def diff_analysis(
        self,
        question: str,
        chosen_path: str,
        rejected_path: str,
        surface_semantics: Dict[str, str],
        divergence_step_text: str = "",
        divergence_step_idx: int = 0,
    ) -> Dict[str, Any]:
        """对比正确/错误推理路径，返回差异分析。

        Returns:
            {"status": "success"|"parse_failed", "differences": [...], ...}
        """
        # 构造可选的分叉提示块
        if divergence_step_text:
            divergence_hint = (
                f"[Divergence Hint]\n"
                f"The two paths begin to formally diverge at reasoning Step {divergence_step_idx + 1}. "
                f"The correct path at this step reads:\n\"{divergence_step_text}\"\n"
                f"Focus your DivergenceStep analysis on this critical juncture.\n"
            )
        else:
            divergence_hint = ""

        prompt = DIFF_PROMPT.format(
            core_skill=get_field(surface_semantics, CORE_FIELD),
            context_constraints=get_field(surface_semantics, CONDITION_FIELD),
            task_objective=get_field(surface_semantics, TARGET_FIELD),
            divergence_hint=divergence_hint,
            question=question,
            chosen_path=chosen_path,
            rejected_path=rejected_path,
        )
        attempts = [
            {"temperature": 0.2, "max_tokens": 4096, "label": "initial"},
            {"temperature": 0.0, "max_tokens": 3072, "label": "retry_strict"},
        ]
        last_response = ""
        parse_stage = "no_response"

        for attempt in attempts:
            try:
                response = await self._call_llm(prompt, temperature=attempt["temperature"], max_tokens=attempt["max_tokens"])
            except Exception as exc:
                return {"status": f"error: {exc}", "differences": [], "parse_stage": attempt["label"]}
            last_response = response or ""
            parsed = self._extract_json(last_response, task="diff")
            if parsed and isinstance(parsed.get("differences"), list) and parsed["differences"]:
                return {"status": "success", "differences": parsed["differences"], "parse_stage": attempt["label"]}
            repaired = await self._repair_json(last_response, task="diff")
            if repaired and isinstance(repaired.get("differences"), list) and repaired["differences"]:
                return {"status": "success", "differences": repaired["differences"], "parse_stage": f"{attempt['label']}_repair"}
            parse_stage = f"{attempt['label']}_repair_failed"

        return {"status": "parse_failed", "differences": [], "parse_stage": parse_stage, "raw_preview": last_response[:500]}

    async def generate_principles(
        self,
        question: str,
        diff_list: List[Dict[str, Any]],
        surface_semantics: Dict[str, str],
        validation_feedback: str = "",
    ) -> Dict[str, Any]:
        """从差异分析中生成经验原则。

        Returns:
            {"status": "success"|"parse_failed", "principles": [...]}
        """
        prompt = STRUCTURED_PRINCIPLE_PROMPT.format(
            question=question,
            core_skill=get_field(surface_semantics, CORE_FIELD),
            context_constraints=get_field(surface_semantics, CONDITION_FIELD),
            task_objective=get_field(surface_semantics, TARGET_FIELD),
            diff_list=json.dumps({"differences": diff_list}, ensure_ascii=False),
            feedback_block=validation_feedback.strip() or "No prior validation feedback.",
        )
        response = await self._call_llm(prompt, temperature=0.3, max_tokens=2048)
        parsed = self._extract_json(response, task="principle")
        if not parsed or not isinstance(parsed.get("principles"), list):
            repaired = await self._repair_json(response, task="principle")
            parsed = repaired if repaired and isinstance(repaired.get("principles"), list) else None
        if not parsed or not parsed.get("principles"):
            return {"status": "parse_failed", "principles": [], "raw_preview": (response or "")[:500]}
        normalized_principles = []
        schema_errors = []
        for index, principle in enumerate(parsed["principles"]):
            normalized, errors = validate_principle_content(principle, require_extended=True)
            if errors:
                schema_errors.append({"index": index, "errors": errors})
            else:
                normalized_principles.append(normalized)
        if not normalized_principles:
            return {
                "status": "schema_failed",
                "principles": [],
                "schema_errors": schema_errors,
                "raw_preview": (response or "")[:500],
            }
        return {
            "status": "success",
            "principles": normalized_principles,
            "schema_errors": schema_errors,
        }

    async def crucible_test(
        self,
        question: str,
        principle: Dict[str, str],
        expected_answer: str,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """坩埚验证：用 principle 引导答题，检查是否正确。"""
        if isinstance(principle, dict):
            principle_text = str(principle.get("principle", "")).strip()
            if not principle_text:
                parts = [
                    str(principle.get("trigger_condition", "")).strip(),
                    str(principle.get("action_guidance", "")).strip(),
                    str(principle.get("scope", "")).strip(),
                ]
                principle_text = " ".join(part for part in parts if part)
        else:
            principle_text = str(principle).strip()
        if not principle_text:
            return {"principle": principle, "response": "", "is_correct": False, "error": "empty_principle"}
        max_attempts = max(1, int(max_attempts or 1))
        prompt = GUIDED_ANSWER_PROMPT.format(principle_text=principle_text, question=question)
        attempts: List[Dict[str, Any]] = []
        response = ""
        error_message = ""
        is_correct = False
        for attempt_idx in range(max_attempts):
            current_prompt = prompt
            if attempts:
                previous_response = attempts[-1].get("response", "")
                current_prompt = (
                    f"{prompt}\n\n"
                    "[Retry Instruction]\n"
                    "Your previous answer was likely incorrect.\n"
                    "Re-check the constraints, recompute carefully, and correct the final boxed answer if needed.\n"
                    "Use the same relevant experience as soft guidance, but do not repeat the same mistake.\n\n"
                    f"[Previous Attempt]\n{previous_response}\n"
                )
            try:
                response = await self._call_llm(current_prompt, temperature=0.0, max_tokens=4096)
            except Exception as exc:
                error_message = str(exc)
                attempts.append({"attempt": attempt_idx + 1, "response": "", "is_correct": False, "error": error_message})
                break
            is_correct = _check_answer_match(response, expected_answer)
            attempts.append({
                "attempt": attempt_idx + 1,
                "response": (response or "")[:1200],
                "is_correct": is_correct,
            })
            if is_correct:
                break
        return {
            "principle": principle,
            "response": (response or "")[:1200],
            "is_correct": is_correct,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "first_attempt_correct": attempts[0]["is_correct"] if attempts else False,
            "rescued_by_retry": is_correct and any(not item.get("is_correct") for item in attempts[:-1]),
            "passed": is_correct,
            "error": error_message,
        }

    async def judge_principle_consistency(
        self,
        *,
        question: str,
        new_principle: str,
        existing_question: str,
        existing_principle: str,
    ) -> Dict[str, Any]:
        """Judge whether a new validated principle should replace, merge, or be added.

        Returns:
            {
                "status": "success"|"parse_failed",
                "decision": "replace_existing"|"merge_with_existing"|"add_as_new",
                "merged_principle": "...",
                "reason": "..."
            }
        """
        prompt = (
            "You are reconciling two principles for semantically near-identical math reasoning questions.\n"
            "Decide whether the new principle should replace the old one, be merged with it, or be stored as a new separate principle.\n"
            "Output valid JSON only with keys: decision, merged_principle, reason.\n"
            'Allowed decisions: "replace_existing", "merge_with_existing", "add_as_new".\n'
            "Use replace_existing when the new principle is strictly better and subsumes the old one.\n"
            "Use merge_with_existing when both contain complementary useful guidance.\n"
            "Use add_as_new when they are not actually consistent enough in actionable reasoning content.\n\n"
            f"[New Source Question]\n{question}\n\n"
            f"[Existing Source Question]\n{existing_question}\n\n"
            f"[New Principle]\n{new_principle}\n\n"
            f"[Existing Principle]\n{existing_principle}\n"
        )
        response = await self._call_llm(prompt, temperature=0.0, max_tokens=1024)
        parsed = self._extract_json(response)
        if not parsed:
            parsed = await self._repair_json(response, task="generic")
        if not parsed:
            return {"status": "parse_failed", "decision": "add_as_new", "merged_principle": new_principle, "reason": "parse_failed"}

        decision = str(parsed.get("decision", "add_as_new")).strip()
        if decision not in {"replace_existing", "merge_with_existing", "add_as_new"}:
            decision = "add_as_new"
        merged_principle = str(parsed.get("merged_principle", "")).strip() or new_principle
        reason = str(parsed.get("reason", "")).strip()
        return {
            "status": "success",
            "decision": decision,
            "merged_principle": merged_principle,
            "reason": reason,
        }

    async def judge_memory_applicability(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        confidence_threshold: float = 0.70,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Compare source/current structure before a memory reaches the solver.

        This is a separate routing call, so source text and routing metadata
        cannot anchor the downstream solver. Parse failure is fail-closed.
        """
        routing_payload = []
        for candidate in candidates:
            content = candidate.get("content", {})
            retrieval = candidate.get("retrieval", {}) if isinstance(candidate.get("retrieval"), dict) else {}
            source_item = candidate.get("source_item")
            if not isinstance(source_item, dict):
                source_item = retrieval.get("item", {}) if isinstance(retrieval.get("item"), dict) else {}
            source_dsd = source_item.get("source_dsd", {}) if isinstance(source_item.get("source_dsd"), dict) else {}
            routing_payload.append({
                "memory_id": candidate.get("memory_id", ""),
                "source_question": source_item.get("question", source_item.get("source_question_text", "")),
                "source_structure": {
                    "core_skill": source_dsd.get("core_skill", source_item.get("core_skill", "")),
                    "context_constraints": source_dsd.get(
                        "context_constraints", source_item.get("context_constraints", "")
                    ),
                    "task_objective": source_dsd.get("task_objective", source_item.get("task_objective", "")),
                },
                "trigger_condition": content.get("trigger_condition", ""),
                "required_evidence": content.get("required_evidence", content.get("trigger_condition", "")),
                "exclusion_condition": content.get("exclusion_condition", ""),
                "scope": content.get("scope", ""),
                "failure_mode": content.get("failure_mode", ""),
                "failure_signals": content.get("failure_signals", []),
                "action_guidance": content.get("action_guidance", ""),
                "verification_step": content.get("verification_step", ""),
                "generalization_level": content.get("generalization_level", "family"),
            })
        prompt = MEMORY_APPLICABILITY_PROMPT.format(
            question=question,
            candidates_json=json.dumps(routing_payload, ensure_ascii=False, indent=2),
        )
        response = await self._call_llm(prompt, temperature=0.0, max_tokens=max_tokens)
        parsed, parse_stage = self._extract_gate_json(response)
        raw_judgments = parsed.get("judgments", []) if isinstance(parsed, dict) else []
        expected_ids = [str(candidate.get("memory_id", "")) for candidate in candidates]
        returned_ids = [
            str(item.get("memory_id", ""))
            for item in raw_judgments
            if isinstance(item, dict)
        ]
        protocol_errors = []
        try:
            strict_value = json.loads((response or "").strip())
        except Exception:
            strict_value = None
        strict_wire_format = isinstance(strict_value, dict) and isinstance(
            strict_value.get("judgments"), list
        )
        # A complete object inside a Markdown fence (optionally followed by an
        # explanation) is losslessly normalizable.  This fixes presentation
        # noise only; direct lists, missing IDs, wrong types, and incomplete
        # judgments remain fail-closed below.
        format_normalized = False
        if not strict_wire_format and parse_stage == "wrapped_object" and isinstance(parsed, dict):
            parse_stage = "wrapped_object_normalized"
            format_normalized = True
        elif not strict_wire_format:
            protocol_errors.append("top_level_must_be_raw_object_with_judgments_array")
        if len(returned_ids) != len(set(returned_ids)):
            protocol_errors.append("duplicate_memory_id")
        missing_ids = sorted(set(expected_ids) - set(returned_ids))
        extra_ids = sorted(set(returned_ids) - set(expected_ids))
        if missing_ids:
            protocol_errors.append("missing_candidate_ids:" + ",".join(missing_ids))
        if extra_ids:
            protocol_errors.append("unexpected_candidate_ids:" + ",".join(extra_ids))
        required_boolean_fields = (
            "trigger_match", "required_evidence_match", "structural_match",
            "scope_match", "contraindicated",
        )
        for item in raw_judgments:
            if not isinstance(item, dict):
                protocol_errors.append("judgment_must_be_object")
                continue
            memory_id = str(item.get("memory_id", "")) or "<missing>"
            for field in required_boolean_fields:
                if not isinstance(item.get(field), bool):
                    protocol_errors.append(f"{memory_id}:{field}_must_be_boolean")
            confidence_value = item.get("confidence")
            if (
                isinstance(confidence_value, bool)
                or not isinstance(confidence_value, (int, float))
                or not 0.0 <= float(confidence_value) <= 1.0
            ):
                protocol_errors.append(f"{memory_id}:confidence_must_be_number_0_to_1")
            for field in ("violated_conditions", "source_specific_elements"):
                if not isinstance(item.get(field), list):
                    protocol_errors.append(f"{memory_id}:{field}_must_be_array")
            if not isinstance(item.get("reason"), str) or not str(item.get("reason", "")).strip():
                protocol_errors.append(f"{memory_id}:reason_must_be_nonempty_string")
        by_id = {
            str(item.get("memory_id", "")): item
            for item in raw_judgments
            if isinstance(item, dict) and item.get("memory_id")
        }
        judgments = []
        selected_ids = []
        for candidate in candidates:
            memory_id = str(candidate.get("memory_id", ""))
            raw = by_id.get(memory_id, {})
            try:
                confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            trigger_match = raw.get("trigger_match") is True
            required_evidence_match = raw.get("required_evidence_match") is True
            structural_match = raw.get("structural_match") is True
            scope_match = raw.get("scope_match") is True
            contraindicated = raw.get("contraindicated") is True
            accepted = (
                trigger_match and required_evidence_match and structural_match
                and scope_match and not contraindicated
                and confidence >= confidence_threshold
            )
            violated_conditions = raw.get("violated_conditions", [])
            if not isinstance(violated_conditions, list):
                violated_conditions = [str(violated_conditions)] if violated_conditions else []
            source_specific_elements = raw.get("source_specific_elements", [])
            if not isinstance(source_specific_elements, list):
                source_specific_elements = [str(source_specific_elements)] if source_specific_elements else []
            judgment = {
                "memory_id": memory_id,
                "trigger_match": trigger_match,
                "required_evidence_match": required_evidence_match,
                "structural_match": structural_match,
                "scope_match": scope_match,
                "contraindicated": contraindicated,
                "violated_conditions": [str(value) for value in violated_conditions],
                "source_specific_elements": [str(value) for value in source_specific_elements],
                "confidence": confidence,
                "accepted": accepted,
                "reason": str(raw.get("reason", "missing_or_invalid_judgment")).strip(),
            }
            judgments.append(judgment)
            if accepted:
                selected_ids.append(memory_id)
        return {
            "status": "success" if parsed else "parse_failed",
            "parse_stage": parse_stage,
            "format_parsed": bool(parsed),
            "format_normalized": format_normalized,
            "strict_wire_format": strict_wire_format,
            "protocol_compliant": bool(parsed) and not protocol_errors,
            "protocol_errors": protocol_errors,
            "confidence_threshold": confidence_threshold,
            "judgments": judgments,
            "selected_ids": selected_ids,
            "raw_response": response,
            "raw_preview": (response or "")[:500],
        }

    async def adapt_memory_for_question(
        self,
        question: str,
        candidate: Dict[str, Any],
        gate_judgment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Bind an accepted action to the current problem without solving it."""
        if not gate_judgment.get("accepted"):
            return {
                "status": "reject",
                "condition_card": {},
                "schema_errors": [],
                "reason": "structural_gate_rejected",
                "raw_response": "",
            }
        content = candidate.get("content", {}) if isinstance(candidate.get("content"), dict) else {}
        retrieval = candidate.get("retrieval", {}) if isinstance(candidate.get("retrieval"), dict) else {}
        source_item = candidate.get("source_item")
        if not isinstance(source_item, dict):
            source_item = retrieval.get("item", {}) if isinstance(retrieval.get("item"), dict) else {}
        source_dsd = source_item.get("source_dsd", {}) if isinstance(source_item.get("source_dsd"), dict) else {}
        payload = {
            "source_question": source_item.get("question", source_item.get("source_question_text", "")),
            "source_structure": {
                "core_skill": source_dsd.get("core_skill", source_item.get("core_skill", "")),
                "context_constraints": source_dsd.get(
                    "context_constraints", source_item.get("context_constraints", "")
                ),
                "task_objective": source_dsd.get("task_objective", source_item.get("task_objective", "")),
            },
            "principle": {
                "introspection": content.get("introspection", ""),
                "trigger_condition": content.get("trigger_condition", ""),
                "required_evidence": content.get("required_evidence", []),
                "exclusion_condition": content.get("exclusion_condition", ""),
                "failure_mode": content.get("failure_mode", ""),
                "failure_signals": content.get("failure_signals", []),
                "action_guidance": content.get("action_guidance", ""),
                "verification_step": content.get("verification_step", ""),
                "scope": content.get("scope", ""),
                "generalization_level": content.get("generalization_level", "family"),
            },
            "gate_judgment": gate_judgment,
        }
        prompt = MEMORY_ACTION_BINDING_PROMPT.format(
            question=question,
            candidate_json=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        response = await self._call_llm(prompt, temperature=0.0, max_tokens=2048)
        parsed = self._extract_json(response, task="generic") or {}
        format_normalizations = []
        # Local models often express the requested 3-6 advisory sentences and
        # 2-4 checks as JSON string arrays.  Joining a pure list[str] is
        # lossless presentation normalization; mixed/empty lists still fail
        # the strict condition-card validator.
        for field in ("suggested_action", "verification"):
            raw_value = parsed.get(field)
            if (
                isinstance(raw_value, list) and raw_value
                and all(isinstance(item, str) and item.strip() for item in raw_value)
            ):
                parsed[field] = " ".join(" ".join(item.split()) for item in raw_value)
                format_normalizations.append(f"{field}:list_str_to_string")
        card, schema_errors = validate_condition_card(parsed)
        status = card.get("status", "reject")
        if schema_errors:
            status = "reject"
            card["status"] = "reject"
        return {
            "status": status,
            "condition_card": card if status == "use" else {},
            "schema_errors": schema_errors,
            "format_normalizations": format_normalizations,
            "source_principle": content,
            "reason": str(parsed.get("reason", "parse_failed_or_rejected")).strip(),
            "raw_response": response,
        }

    async def repair_diagnose(
        self,
        principle: Dict[str, str],
        failed_validations: List[Dict[str, Any]],
        source_context: str = "",
    ) -> Dict[str, Any]:
        """诊断原则失败原因: vague / over_general / misdirected / other.

        Returns:
            {"status": "success"|"parse_failed",
             "diagnosis": "vague"|"over_general"|"misdirected"|"other",
             "reason": "...", "suggested_fix_direction": "..."}
        """
        failed_cases_parts = []
        for i, fv in enumerate(failed_validations, 1):
            q = fv.get("question_text", fv.get("question", ""))
            a = fv.get("expected_answer", "")
            r = fv.get("response", "")[:300]
            failed_cases_parts.append(
                f"[Case {i}]\nQuestion: {q}\nExpected: {a}\nModel Response: {r}\n"
            )
        failed_cases_text = "\n".join(failed_cases_parts) if failed_cases_parts else "(no cases provided)"

        prompt = REPAIR_DIAGNOSIS_PROMPT.format(
            introspection=principle.get("introspection", ""),
            trigger_condition=principle.get("trigger_condition", ""),
            required_evidence=principle.get("required_evidence", ""),
            exclusion_condition=principle.get("exclusion_condition", ""),
            action_guidance=principle.get("action_guidance", ""),
            verification_step=principle.get("verification_step", ""),
            generalization_level=principle.get("generalization_level", "family"),
            scope=principle.get("scope", ""),
            failure_mode=principle.get("failure_mode", ""),
            failed_cases=failed_cases_text,
            source_context=source_context or "(source context not available)",
        )
        response = await self._call_llm(prompt, temperature=0.0, max_tokens=1024)
        parsed = self._extract_json(response, task="generic")
        if not parsed:
            parsed = await self._repair_json(response, task="generic")
        if not parsed:
            return {
                "status": "parse_failed",
                "diagnosis": "other",
                "reason": "parse_failed",
                "suggested_fix_direction": "",
            }
        return {
            "status": "success",
            "diagnosis": str(parsed.get("diagnosis", "other")).strip(),
            "reason": str(parsed.get("reason", "")).strip(),
            "suggested_fix_direction": str(parsed.get("suggested_fix_direction", "")).strip(),
        }

    async def repair_patch(
        self,
        principle: Dict[str, str],
        diagnosis_result: Dict[str, Any],
        failed_example: Dict[str, Any],
        source_dsd: Dict[str, str],
    ) -> Dict[str, Any]:
        """根据诊断结果修补原则。

        Returns:
            {"status": "success"|"parse_failed",
             "principle": {"introspection", "trigger_condition", "action_guidance",
                           "scope", "failure_mode", "repair_status"}}
        """
        fe_question = failed_example.get("question_text", failed_example.get("question", ""))
        fe_expected = failed_example.get("expected_answer", "")
        fe_response = failed_example.get("response", "")[:300]
        failed_example_text = (
            f"Question: {fe_question}\n"
            f"Expected Answer: {fe_expected}\n"
            f"Model Response (with this principle): {fe_response}"
        )

        prompt = REPAIR_PATCH_PROMPT.format(
            introspection=principle.get("introspection", ""),
            trigger_condition=principle.get("trigger_condition", ""),
            required_evidence=principle.get("required_evidence", ""),
            exclusion_condition=principle.get("exclusion_condition", ""),
            action_guidance=principle.get("action_guidance", ""),
            verification_step=principle.get("verification_step", ""),
            generalization_level=principle.get("generalization_level", "family"),
            scope=principle.get("scope", ""),
            failure_mode=principle.get("failure_mode", ""),
            diagnosis=diagnosis_result.get("diagnosis", "other"),
            reason=diagnosis_result.get("reason", ""),
            suggested_fix_direction=diagnosis_result.get("suggested_fix_direction", ""),
            failed_example=failed_example_text,
            core_skill=get_field(source_dsd, CORE_FIELD),
            context_constraints=get_field(source_dsd, CONDITION_FIELD),
            task_objective=get_field(source_dsd, TARGET_FIELD),
        )
        response = await self._call_llm(prompt, temperature=0.2, max_tokens=1536)
        parsed = self._extract_json(response, task="generic")
        if not parsed:
            parsed = await self._repair_json(response, task="generic")
        principle = parsed.get("principle", {}) if parsed else {}
        repair_status = str(principle.get("repair_status", "repaired")).strip().lower()
        if repair_status not in {"repaired", "unfixable"}:
            repair_status = "repaired"
        normalized, errors = validate_principle_content(principle, require_extended=True)
        if errors:
            return {
                "status": "schema_failed",
                "schema_errors": errors,
                "raw_preview": (response or "")[:500],
            }
        normalized["repair_status"] = repair_status
        return {"status": "success", "principle": normalized}


# ── 答案匹配 (轻量级，仅供 crucible_test 内部使用) ─────────────────


def _extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"Final Answer:\s*\\boxed\{([^}]+)\}",
        r"Final Answer:\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL)
        if m:
            return m.group(1).strip().strip("$")
    return None


def _check_answer_match(generated: str, expected: str) -> bool:
    gen = (_extract_final_answer(generated) or generated or "").strip().lower()
    exp = (expected or "").strip().lower()
    if not gen or not exp:
        return False
    return exp in gen or gen in exp
