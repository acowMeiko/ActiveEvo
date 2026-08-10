#!/usr/bin/env python
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_principle_guided_test import check_answer_match


def main() -> None:
    cases = [
        {
            "name": "gsm8k_hash_answer",
            "question": "How much is 9 times 2?",
            "expected": "#### 18",
            "generated": "We compute 9*2=18.\nFinal Answer: 18",
            "should_pass": True,
        },
        {
            "name": "math_boxed_fraction",
            "question": "Convert (0,3) to polar coordinates.",
            "expected": "\\left( 3, \\frac{\\pi}{2} \\right)",
            "generated": "Therefore, the polar coordinates are \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.",
            "should_pass": True,
        },
        {
            "name": "bbh_choice_label",
            "question": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938",
            "expected": "(B)",
            "generated": "Let's think carefully.\nFinal Answer: (B)",
            "should_pass": True,
        },
        {
            "name": "bbh_choice_text",
            "question": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938",
            "expected": "(B)",
            "generated": "Tomorrow is 12/25/1937.\nAnswer: 12/25/1937",
            "should_pass": True,
        },
        {
            "name": "bbh_boolean_yes",
            "question": "Is the statement supported?",
            "expected": "yes",
            "generated": "Final Answer: yes",
            "should_pass": True,
        },
        {
            "name": "bbh_boolean_false",
            "question": "not ( True ) and ( True ) is",
            "expected": "False",
            "generated": "Answer: False",
            "should_pass": True,
        },
        {
            "name": "bbh_wrong_choice",
            "question": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938",
            "expected": "(B)",
            "generated": "Final Answer: (C)",
            "should_pass": False,
        },
    ]

    results = []
    for case in cases:
        matched = check_answer_match(case["generated"], case["expected"], question=case["question"])
        results.append({
            "name": case["name"],
            "matched": matched,
            "should_pass": case["should_pass"],
            "ok": matched == case["should_pass"],
        })

    out_path = PROJECT_ROOT / "data/test/benchmarks/answer_matcher_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["ok"])
    print(json.dumps({
        "passed": passed,
        "total": len(results),
        "report": str(out_path),
    }, ensure_ascii=False, indent=2))

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
