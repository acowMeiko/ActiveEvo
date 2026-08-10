"""Re-check zone classifications independently of the original answer matcher."""
import json
import re
import sys
sys.path.insert(0, '.')

from utils.io_utils import load_json


def normalize_answer(text):
    cleaned = (text or "").strip().lower()
    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")
    cleaned = cleaned.replace("\\displaystyle", "")
    cleaned = cleaned.replace("\\dfrac", "\\frac")
    cleaned = cleaned.replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("$", "")
    cleaned = re.sub(r"\\\(|\\\)", "", cleaned)
    cleaned = re.sub(r"\\\[|\\\]", "", cleaned)
    cleaned = re.sub(r"^finalanswer:?", "", cleaned)
    cleaned = cleaned.strip(".,;:!?")
    cleaned = cleaned.replace("\\", "")
    return cleaned


def extract_boxed(text):
    if not text:
        return None
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + 7
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                raw = text[start:i].strip()
                raw = raw.strip("$")
                return raw.strip()
    return None


def check_answer_correct(ans_text, expected_norm):
    """More lenient check: extract boxed, normalize, compare."""
    boxed = extract_boxed(ans_text)
    if boxed:
        boxed_norm = normalize_answer(boxed)
        if boxed_norm == expected_norm:
            return True
        # Check cross-contain (for cases with extra text)
        if expected_norm and boxed_norm:
            if expected_norm in boxed_norm or boxed_norm in expected_norm:
                return True
        return False
    # Fallback: full text match
    ans_norm = normalize_answer(ans_text)
    if expected_norm and ans_norm:
        if expected_norm in ans_norm or ans_norm in expected_norm:
            return True
    return False


for lv in ["dpo_level1", "dpo_level2", "dpo_level3", "dpo_level4"]:
    data = load_json(f"learning_packet/{lv}/01_sampling/sampling_results.json")
    results = data["results"]

    recheck = {"early_stop": 0, "all_correct": 0, "all_wrong": 0, "no_data": 0}
    mismatches = []

    for r in results:
        expected = r.get("expected_answer", "")
        expected_norm = normalize_answer(expected)
        all_answers = r.get("all_answers", [])

        correct_count = 0
        wrong_count = 0

        for ans in all_answers:
            if check_answer_correct(ans, expected_norm):
                correct_count += 1
            else:
                wrong_count += 1

        has_correct = correct_count > 0
        has_wrong = wrong_count > 0
        orig_status = r.get("status", "unknown")

        if has_correct and has_wrong:
            recheck["early_stop"] += 1
        elif has_correct and not has_wrong:
            recheck["all_correct"] += 1
        elif not has_correct and has_wrong:
            recheck["all_wrong"] += 1
        else:
            recheck["no_data"] += 1

        # Detect mismatches
        our_status = (
            "early_stop" if (has_correct and has_wrong)
            else "all_correct" if (has_correct and not has_wrong)
            else "all_wrong" if (not has_correct and has_wrong)
            else "no_data"
        )
        if orig_status == "early_stop" and our_status != "early_stop":
            mismatches.append((r, our_status, correct_count, wrong_count))
        elif orig_status == "all_correct" and our_status != "all_correct":
            mismatches.append((r, our_status, correct_count, wrong_count))
        elif orig_status == "all_wrong" and our_status != "all_wrong":
            mismatches.append((r, our_status, correct_count, wrong_count))
        elif orig_status == "max_trials_exceeded" and our_status != "all_wrong":
            mismatches.append((r, our_status, correct_count, wrong_count))

    orig = data["summary"]
    print(f"=== {lv} ===")
    print(f"  Original summary:    early={orig['early_stop']}, master={orig['all_correct']}, "
          f"panic_raw={orig['all_wrong']}, max_trials={orig.get('max_trials_exceeded', 0)}")
    print(f"  Original all_wrong+max_trials: {orig['all_wrong'] + orig.get('max_trials_exceeded', 0)}")
    print(f"  Re-checked:          learning={recheck['early_stop']}, master={recheck['all_correct']}, "
          f"panic={recheck['all_wrong']}, no_data={recheck['no_data']}")
    print(f"  Classification mismatches: {len(mismatches)}")

    if mismatches:
        # Show first few mismatches with details
        for m in mismatches[:5]:
            r, our, cc, wc = m
            ans_previews = []
            for a in r.get("all_answers", [])[:2]:
                boxed = extract_boxed(a)
                fa = a.split("Final Answer:")[-1].strip()[:80] if "Final Answer:" in a else a[:80]
                ans_previews.append(f"boxed={boxed}, tail={fa}")
            print(f"    idx={r['idx']}: orig={r.get('status')} -> our={our} "
                  f"(c={cc},w={wc}, orig_c={r.get('correct_count')}, orig_w={r.get('wrong_count')})")
            print(f"      expected={r.get('expected_answer','')[:60]}")
            for ap in ans_previews[:2]:
                print(f"      answer: {ap[:120]}")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches) - 5} more")
    print()
