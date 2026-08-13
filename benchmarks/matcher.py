"""Strict answer matching for the benchmark evaluation path.

The matcher intentionally separates *answer extraction* from *equivalence
checking*.  Long model responses are never graded by looking for their last
numeric token: a number in an unfinished derivation is not a final answer.
"""
from __future__ import annotations

import math
import re
import signal
import threading
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)
_MAX_SYMBOLIC_INPUT_CHARS = 256
_MAX_SYMBOLIC_OPERATORS = 64
_MAX_SYMBOLIC_IDENTIFIERS = 8
_MAX_NUMERIC_EXPONENT = 100
_SYMPY_TIMEOUT_SECONDS = 0.75
_FINAL_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$"
)


def _balanced_braced_values(text: str, marker: str) -> List[str]:
    """Return all complete braced values following ``marker``."""
    values: List[str] = []
    pos = 0
    while True:
        idx = text.find(marker, pos)
        if idx < 0:
            break
        start = idx + len(marker)
        depth = 1
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[start:end].strip())
                    pos = end + 1
                    break
        else:
            pos = start
    return values


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the last *complete* ``\\boxed{...}`` expression."""
    if not text:
        return None
    values: List[tuple[int, str]] = []
    for marker in (r"\boxed{", "\x08oxed{", "boxed{"):
        pos = 0
        while True:
            idx = text.find(marker, pos)
            if idx < 0:
                break
            extracted = _balanced_braced_values(text[idx:], marker)
            if extracted:
                values.append((idx, extracted[0]))
            pos = idx + len(marker)
    return max(values, key=lambda pair: pair[0])[1] if values else None


def extract_final_answer(text: str, *, allow_short_direct: bool = True) -> Optional[str]:
    """Extract an explicit final answer without mining intermediate work.

    Priority is the last complete box, then the last ``Final Answer:`` line.
    A short one-line response is accepted for datasets that return bare labels.
    """
    if not text:
        return None
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    matches = list(_FINAL_LINE_RE.finditer(text))
    if matches:
        value = matches[-1].group(1).strip().strip("$").strip()
        return value or None
    stripped = text.strip()
    if allow_short_direct and "\n" not in stripped and len(stripped) <= 128:
        return stripped
    return None


def _extract_last_number(text: str) -> Optional[float]:
    numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _extract_choice_labels(text: str) -> List[str]:
    return [label.upper() for label in re.findall(r"\b([A-Ja-j])\b", text)]


def _clean_text(text: str) -> str:
    return (text or "").strip()


def _strip_wrappers(text: str) -> str:
    value = (text or "").strip()
    boxed = extract_last_boxed(value)
    if boxed is not None:
        value = boxed
    value = value.strip().strip("$").strip()
    value = re.sub(r"^\\?\(|\\?\)$", "", value)
    return value.strip().rstrip(".,;:")


def _normalized_text(text: str) -> str:
    value = _strip_wrappers(text).lower()
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\displaystyle", "").replace(r"\,", "")
    value = value.replace("−", "-").replace(" ", "")
    return value


def _replace_latex_command(expr: str, command: str, replacement) -> str:
    """Replace repeated one-argument LaTeX commands with balanced braces."""
    marker = "\\" + command + "{"
    while marker in expr:
        start = expr.rfind(marker)
        content_start = start + len(marker)
        depth = 1
        end = content_start
        while end < len(expr) and depth:
            if expr[end] == "{":
                depth += 1
            elif expr[end] == "}":
                depth -= 1
            end += 1
        if depth:
            break
        content = expr[content_start:end - 1]
        expr = expr[:start] + replacement(content) + expr[end:]
    return expr


def _latex_to_sympy_text(text: str) -> str:
    expr = _normalized_text(text)
    expr = expr.replace(r"\pi", "pi").replace("π", "pi")
    expr = expr.replace(r"\cdot", "*").replace(r"\times", "*")
    expr = expr.replace(r"\div", "/").replace(r"\%", "/100")
    expr = expr.replace("°", "").replace(r"^\circ", "")
    expr = _replace_latex_command(expr, "sqrt", lambda x: f"sqrt({x})")
    # Fractions require two balanced arguments; process the innermost first.
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_pattern.search(expr):
        expr = frac_pattern.sub(r"((\1)/(\2))", expr)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("^", "**")
    return expr


def _symbolic_input_safe(text: str) -> bool:
    """Reject model prose and expressions capable of pathological expansion.

    Exact normalized answers are accepted before this check.  SymPy is only a
    fallback for small mathematical expressions, never a general text parser.
    """
    raw = _strip_wrappers(text)
    if not raw or len(raw) > _MAX_SYMBOLIC_INPUT_CHARS:
        return False
    expr = _latex_to_sympy_text(raw)
    if len(expr) > _MAX_SYMBOLIC_INPUT_CHARS:
        return False
    if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\\{}=\s]*", raw):
        return False
    if sum(raw.count(operator) for operator in "+-*/^=") > _MAX_SYMBOLIC_OPERATORS:
        return False
    if raw.count("{") != raw.count("}") or raw.count("(") != raw.count(")"):
        return False
    identifiers = re.findall(r"[A-Za-z]+", expr)
    allowed_words = {"pi", "sqrt", "oo", "e"}
    if any(identifier.lower() not in allowed_words and len(identifier) != 1 for identifier in identifiers):
        return False
    if len(set(identifier.lower() for identifier in identifiers)) > _MAX_SYMBOLIC_IDENTIFIERS:
        return False
    for exponent in re.findall(r"(?:\*\*|\^)\s*\(?\s*([-+]?\d+)", expr):
        try:
            if abs(int(exponent)) > _MAX_NUMERIC_EXPONENT:
                return False
        except ValueError:
            return False
    # Deeply nested powers are a common source of explosive symbolic growth.
    if expr.count("**") > 4:
        return False
    return True


class _SymbolicTimeout(TimeoutError):
    pass


@contextmanager
def _symbolic_timeout(seconds: float):
    """Wall-clock guard for SymPy on the main thread (Linux deployment path)."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum, _frame):
        raise _SymbolicTimeout("symbolic equivalence timed out")

    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, max(0.01, float(seconds)))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _parse_symbolic(text: str) -> Optional[sp.Expr]:
    try:
        return parse_expr(
            _latex_to_sympy_text(text),
            local_dict={"pi": sp.pi, "sqrt": sp.sqrt, "oo": sp.oo},
            transformations=_TRANSFORMS,
            evaluate=True,
        )
    except _SymbolicTimeout:
        raise
    except Exception:
        return None


def _simple_number(text: str) -> Optional[Fraction]:
    """Parse integers, decimals and plain/LaTeX rational numbers without SymPy."""
    value = _normalized_text(text)
    latex_fraction = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", value)
    plain_fraction = re.fullmatch(r"([-+]?\d+)/([-+]?\d+)", value)
    try:
        if latex_fraction:
            denominator = int(latex_fraction.group(2))
            return Fraction(int(latex_fraction.group(1)), denominator) if denominator else None
        if plain_fraction:
            denominator = int(plain_fraction.group(2))
            return Fraction(int(plain_fraction.group(1)), denominator) if denominator else None
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value):
            return Fraction(Decimal(value))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None
    return None


def answers_equivalent_detailed(candidate: str, expected: str) -> Dict[str, Any]:
    """Compare extracted answers and expose the safe matching decision path."""
    decision: Dict[str, Any] = {
        "correct": False,
        "status": "not_matched",
        "method": "none",
        "reason": "",
    }
    if not candidate or not expected:
        decision["reason"] = "missing_candidate_or_expected"
        return decision
    cand_norm = _normalized_text(candidate)
    exp_norm = _normalized_text(expected)
    if cand_norm == exp_norm:
        return {"correct": True, "status": "matched", "method": "exact", "reason": "normalized_exact"}

    # Preserve exact semantics for answer labels and booleans.
    if re.fullmatch(r"\(?[a-j]\)?", cand_norm) and re.fullmatch(r"\(?[a-j]\)?", exp_norm):
        correct = cand_norm.strip("()") == exp_norm.strip("()")
        return {"correct": correct, "status": "matched" if correct else "not_matched",
                "method": "choice_label", "reason": ""}
    if cand_norm in {"true", "false"} or exp_norm in {"true", "false"}:
        correct = cand_norm == exp_norm
        return {"correct": correct, "status": "matched" if correct else "not_matched",
                "method": "boolean", "reason": ""}

    candidate_number = _simple_number(candidate)
    expected_number = _simple_number(expected)
    if candidate_number is not None and expected_number is not None:
        correct = candidate_number == expected_number
        return {"correct": correct, "status": "matched" if correct else "not_matched",
                "method": "numeric_fraction", "reason": ""}

    unsafe = []
    if not _symbolic_input_safe(candidate):
        unsafe.append("candidate")
    if not _symbolic_input_safe(expected):
        unsafe.append("expected")
    if unsafe:
        return {"correct": False, "status": "symbolic_skipped", "method": "none",
                "reason": "unsafe_symbolic_" + "_and_".join(unsafe)}

    try:
        with _symbolic_timeout(_SYMPY_TIMEOUT_SECONDS):
            cand_expr = _parse_symbolic(candidate)
            exp_expr = _parse_symbolic(expected)
            if cand_expr is None or exp_expr is None:
                return {"correct": False, "status": "not_matched", "method": "symbolic",
                        "reason": "symbolic_parse_failed"}
            if int(sp.count_ops(cand_expr)) > _MAX_SYMBOLIC_OPERATORS:
                return {"correct": False, "status": "symbolic_skipped", "method": "symbolic",
                        "reason": "candidate_operation_limit"}
            if int(sp.count_ops(exp_expr)) > _MAX_SYMBOLIC_OPERATORS:
                return {"correct": False, "status": "symbolic_skipped", "method": "symbolic",
                        "reason": "expected_operation_limit"}
            difference = sp.simplify(cand_expr - exp_expr)
            if difference == 0:
                return {"correct": True, "status": "matched", "method": "symbolic", "reason": ""}
            if not difference.free_symbols:
                correct = abs(float(sp.N(difference, 15))) <= 1e-9
                return {"correct": correct, "status": "matched" if correct else "not_matched",
                        "method": "symbolic_numeric", "reason": ""}
    except _SymbolicTimeout:
        return {"correct": False, "status": "symbolic_timeout", "method": "symbolic",
                "reason": f"exceeded_{_SYMPY_TIMEOUT_SECONDS}_seconds"}
    except (TypeError, ValueError, ArithmeticError, MemoryError, RecursionError) as exc:
        return {"correct": False, "status": "symbolic_skipped", "method": "symbolic",
                "reason": type(exc).__name__}
    return decision


def answers_equivalent(candidate: str, expected: str) -> bool:
    """Compatibility boolean wrapper around the auditable matcher."""
    return bool(answers_equivalent_detailed(candidate, expected)["correct"])


def evaluate_svamp(generated: str, gold_answer: str) -> bool:
    candidate = extract_final_answer(generated)
    return bool(candidate) and answers_equivalent(candidate, str(gold_answer))


def evaluate_general365(generated: str, item: Dict[str, Any]) -> bool:
    answer_type = _clean_text(item.get("answer_type", "")).lower()
    expected_raw = _clean_text(item.get("answer", ""))
    candidate = extract_final_answer(generated)
    if candidate is None:
        return False

    if answer_type == "single_choice":
        expected_labels = _extract_choice_labels(expected_raw)
        candidate_labels = _extract_choice_labels(candidate)
        return bool(expected_labels) and candidate_labels[:1] == expected_labels[:1]
    if answer_type == "multiple_choice":
        expected_labels = sorted(set(_extract_choice_labels(expected_raw)))
        candidate_labels = sorted(set(_extract_choice_labels(candidate)))
        return bool(expected_labels) and candidate_labels == expected_labels
    if answer_type == "text":
        return _normalized_text(candidate) == _normalized_text(expected_raw)

    if "float_round" in item:
        candidate_num = _extract_last_number(candidate)
        expected_num = _extract_last_number(expected_raw)
        if candidate_num is None or expected_num is None:
            return False
        return round(candidate_num) == round(expected_num)
    return answers_equivalent(candidate, expected_raw)


def check_answer(generated: str, expected: str) -> bool:
    """Strict generic matcher used by DeepScaler/MATH/GSM8K/BBH."""
    candidate = extract_final_answer(generated)
    expected_value = extract_final_answer(expected) or _strip_wrappers(expected)
    return bool(candidate and expected_value) and answers_equivalent(candidate, expected_value)


def check_answer_detailed(generated: str, expected: str) -> Dict[str, Any]:
    """Strict match with extraction and a persisted audit reason."""
    candidate = extract_final_answer(generated)
    expected_value = extract_final_answer(expected) or _strip_wrappers(expected)
    if not candidate:
        return {"correct": False, "status": "not_matched", "method": "extraction",
                "reason": "missing_explicit_final_answer", "candidate": None}
    decision = answers_equivalent_detailed(candidate, expected_value)
    return {**decision, "candidate": candidate}


def check_answer_for_item(generated: str, item: Dict[str, Any]) -> bool:
    source = str(item.get("source_dataset", "")).strip().lower()
    answer_type = str(item.get("answer_type", "")).strip().lower()
    expected = item.get("answer", "")
    if source == "svamp":
        return evaluate_svamp(generated, str(expected))
    if source == "general365" or answer_type in {"single_choice", "multiple_choice", "text"} or "float_round" in item:
        return evaluate_general365(generated, item)
    return check_answer(generated, str(expected))
