from benchmarks.matcher import check_answer, extract_final_answer


def test_uses_last_boxed_answer():
    text = r"Intermediate: \boxed{7}. Final Answer: \boxed{9}"
    assert extract_final_answer(text) == "9"
    assert check_answer(text, "9")
    assert not check_answer(text, "7")


def test_symbolic_false_positive_regressions():
    assert not check_answer(r"Final Answer: \boxed{\frac{8\sqrt{3}}{3}}", r"\sqrt{3}")
    assert not check_answer(r"Final Answer: \boxed{17+12\sqrt{2}}", r"3+2\sqrt{2}")


def test_symbolic_equivalence():
    assert check_answer(r"Final Answer: \boxed{\frac{\sqrt{12}}{2}}", r"\sqrt{3}")
    assert check_answer(r"Final Answer: \boxed{0.5}", r"\frac{1}{2}")


def test_long_unfinished_reasoning_is_not_an_answer():
    text = "We need to solve the problem. The requested expression is 2\\sqrt{3}.\nLet us continue by..."
    assert not check_answer(text, r"2\sqrt{3}")


def test_short_direct_answer_remains_supported():
    assert check_answer("53", "53")
    assert check_answer("B", "(B)")
