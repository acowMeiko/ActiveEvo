from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _clean_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def _extract_boxed_value(text: str) -> Optional[str]:
    text = _clean_text(text)
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return None
    idx = start + len(marker)
    depth = 1
    chars: List[str] = []
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
        idx += 1
    return None


def _extract_final_answer_line(text: str) -> Optional[str]:
    text = _clean_text(text)
    patterns = [
        r"###\s*The final answer is:\s*(.+)",
        r"Final Answer:\s*(.+)",
        r"answer[:：]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _normalize_basic_answer(text: Any) -> str:
    value = _clean_text(text).lower()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\displaystyle", "")
    value = value.replace("\\,", "")
    value = value.replace("$", "")
    value = value.replace(" ", "")
    value = re.sub(r"^#+thefinalansweris:?", "", value)
    value = re.sub(r"^finalanswer:?", "", value)
    value = re.sub(r"^answer:?", "", value)
    return value.strip(".,;:!?")


def _extract_last_number(text: str) -> Optional[float]:
    text = _clean_text(text).replace(",", "")
    candidates = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
    if not candidates:
        return None
    token = candidates[-1]
    if "/" in token and token.count("/") == 1:
        left, right = token.split("/")
        try:
            return float(left) / float(right)
        except Exception:
            return None
    try:
        return float(token)
    except Exception:
        return None


def _extract_choice_labels(text: str) -> List[str]:
    text = _clean_text(text)
    found = re.findall(r"\b([A-Z])\b", text.upper())
    deduped: List[str] = []
    for label in found:
        if label not in deduped:
            deduped.append(label)
    return deduped


def _replace_number_tokens(text: str, numbers: List[str]) -> str:
    result = str(text)
    for idx, number in enumerate(numbers):
        result = re.sub(rf"\bnumber{idx}\b", number, result)
    return _clean_text(result)


def preprocess_svamp_train_csv(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            numbers = [token.strip() for token in str(row.get("Numbers", "")).split() if token.strip()]
            body = _replace_number_tokens(row.get("Body", ""), numbers)
            ques = _replace_number_tokens(row.get("Ques", ""), numbers)
            question = _replace_number_tokens(row.get("Question", ""), numbers)
            natural_question = _clean_text(f"{body} {ques}".strip()) or question
            rows.append({
                "id": f"svamp-train-{idx}",
                "question": natural_question,
                "answer": _clean_text(row.get("Answer", "")),
                "equation": _clean_text(row.get("Equation", "")),
                "numbers": numbers,
                "answer_type": "number",
                "source_dataset": "svamp",
                "official_split": csv_path.stem,
                "raw_question_with_placeholders": _clean_text(row.get("Question", "")),
                "raw_body_with_placeholders": _clean_text(row.get("Body", "")),
                "raw_ques_with_placeholders": _clean_text(row.get("Ques", "")),
            })
    return rows


def preprocess_svamp_test_json(json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for item in data:
        question = _clean_text(f"{item.get('Body', '')} {item.get('Question', '')}")
        answer = item.get("Answer", "")
        try:
            answer = str(int(float(answer))) if float(answer).is_integer() else str(float(answer))
        except Exception:
            answer = _clean_text(answer)
        rows.append({
            "id": _clean_text(item.get("ID", "")),
            "question": question,
            "answer": answer,
            "equation": _clean_text(item.get("Equation", "")),
            "type": _clean_text(item.get("Type", "")),
            "answer_type": "number",
            "source_dataset": "svamp",
            "official_split": "test",
        })
    return rows


def preprocess_general365_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for row in rows:
        processed.append({
            "id": _clean_text(row.get("id", "")),
            "question": _clean_text(row.get("question", "")),
            "answer": _clean_text(row.get("answer", "")),
            "answer_type": _clean_text(row.get("answer_type", "")),
            "float_round": _clean_text(row.get("float_round", "0")) or "0",
            "source_dataset": "general365",
            "official_split": "test",
        })
    return processed


def evaluate_svamp_prediction(prediction: str, gold_answer: str) -> bool:
    expected = _extract_last_number(str(gold_answer))
    if expected is None:
        return False
    boxed = _extract_boxed_value(prediction or "")
    final = _extract_final_answer_line(prediction or "")
    candidate = _extract_last_number(boxed or final or prediction or "")
    if candidate is None:
        return False
    return math.isclose(candidate, expected, rel_tol=1e-9, abs_tol=1e-6)


def evaluate_general365_prediction(prediction: str, gold_row: Dict[str, Any]) -> bool:
    answer_type = _clean_text(gold_row.get("answer_type", "")).lower()
    expected_raw = _clean_text(gold_row.get("answer", ""))
    boxed = _extract_boxed_value(prediction or "")
    final = _extract_final_answer_line(prediction or "")
    candidate_text = boxed or final or _clean_text(prediction or "")

    if answer_type == "single_choice":
        expected_labels = _extract_choice_labels(expected_raw)
        candidate_labels = _extract_choice_labels(candidate_text)
        return bool(expected_labels) and candidate_labels[:1] == expected_labels[:1]

    if answer_type == "multiple_choice":
        expected_labels = sorted(set(_extract_choice_labels(expected_raw)))
        candidate_labels = sorted(set(_extract_choice_labels(candidate_text)))
        return bool(expected_labels) and candidate_labels == expected_labels

    if answer_type == "text":
        return _normalize_basic_answer(candidate_text) == _normalize_basic_answer(expected_raw)

    expected_norm = _normalize_basic_answer(_extract_boxed_value(expected_raw) or expected_raw)
    candidate_norm = _normalize_basic_answer(candidate_text)
    if candidate_norm == expected_norm and expected_norm:
        return True

    float_round = str(gold_row.get("float_round", "0") or "0").strip()
    expected_num = _extract_last_number(expected_raw)
    candidate_num = _extract_last_number(candidate_text)
    if expected_num is not None and candidate_num is not None:
        if float_round.isdigit() and int(float_round) > 0:
            decimals = int(float_round)
            return round(candidate_num, decimals) == round(expected_num, decimals)
        return math.isclose(candidate_num, expected_num, rel_tol=1e-9, abs_tol=1e-6)
    return False

