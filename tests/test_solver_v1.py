import asyncio
from types import MethodType

from modules.execute_module import (
    ExecuteModule,
    SOLVER_V1_FINALIZER_MAX_TOKENS,
    SOLVER_V1_PROMPT_HASHES,
    SOLVER_V1_REASONER_MAX_TOKENS,
    _finalizer_format_valid,
    _reasoner_format_valid,
    _repetition_diagnostics,
)
from template.prompt_template import (
    SOLVER_V1_FINALIZER_PROMPT,
    SOLVER_V1_GUIDED_REASONER_PROMPT,
    SOLVER_V1_REASONER_PROMPT,
)


def test_frozen_solver_v1_limits_and_hashes():
    assert SOLVER_V1_REASONER_MAX_TOKENS == 8192
    assert SOLVER_V1_FINALIZER_MAX_TOKENS == 128
    assert set(SOLVER_V1_PROMPT_HASHES) == {"reasoner_base", "reasoner_guided", "finalizer"}
    assert all(len(value) == 64 for value in SOLVER_V1_PROMPT_HASHES.values())


def test_reasoner_prompts_have_same_contract():
    base = SOLVER_V1_REASONER_PROMPT.format(question="What is 2+2?")
    guided = SOLVER_V1_GUIDED_REASONER_PROMPT.format(
        question="What is 2+2?",
        principle_text=(
            "[Conditional Experience Card 1]\n"
            "When to consider: Addition is requested.\n"
            "Evidence found:\n  - Need: An addition expression. | Current fact: The problem asks 2+2.\n"
            "Do not use if: No arithmetic operation is present.\n"
            "Failure to avoid: Avoid changing the operation.\n"
            "Suggested action: Add the terms directly.\n"
            "Verification: Subtract one addend from the result."
        ),
    )
    for prompt in (base, guided):
        assert "Method:" in prompt
        assert "Reasoning:" in prompt
        assert "Check:" in prompt
        assert "a separate finalizer" in prompt
    assert "retrieval_score" not in guided
    assert "No external experience is provided" in base
    assert "Experience Decision:" in guided
    assert "use | ignore" in guided
    assert "Conditional Experience Card" in guided
    assert "conditional suggestion, not as evidence" in base
    assert "not evidence" in guided
    assert "memory IDs" in guided


def test_finalizer_contract_and_format_check():
    prompt = SOLVER_V1_FINALIZER_PROMPT.format(
        question="What is 2+2?", reasoner_output="Reasoning: 2+2=4"
    )
    assert "Output exactly one line" in prompt
    assert _finalizer_format_valid(r"Final Answer: \boxed{4}")
    assert not _finalizer_format_valid("Reasoning first\nFinal Answer: \\boxed{4}")


def test_reasoner_format_and_loop_diagnostics():
    valid = "Method:\nUse addition.\nReasoning:\n2+2=4.\nCheck:\n4-2=2."
    assert _reasoner_format_valid(valid)
    loop = " ".join(["repeat this same algebraic manipulation forever"] * 20)
    diagnostics = _repetition_diagnostics(loop)
    assert diagnostics["likely_loop"]
    assert diagnostics["hard_loop"]
    assert diagnostics["consecutive_repeat_count"] >= 3


def test_reused_math_terms_do_not_trigger_hard_loop():
    text = (
        "Method:\nUse the quadratic formula and then verify both roots.\n"
        "Reasoning:\nThe discriminant is b^2-4ac. Substitute a, b, and c, "
        "simplify the square root, and compute the two candidate roots.\n"
        "Check:\nSubstitute each candidate into the original quadratic equation."
    )
    diagnostics = _repetition_diagnostics(text)
    assert not diagnostics["hard_loop"]


def test_repetition_penalty_is_reasoner_only():
    executor = ExecuteModule.__new__(ExecuteModule)
    executor.seed = None
    calls = []

    async def fake_call(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "text": "Method:\nSolve.\nReasoning:\n2+2=4.\nCheck:\n4-2=2.",
                "usage": {"completion_tokens": 12},
                "raw": {"choices": [{"finish_reason": "stop"}]},
            }, "http://test"
        return {
            "text": r"Final Answer: \boxed{4}",
            "usage": {"completion_tokens": 6},
            "raw": {"choices": [{"finish_reason": "stop"}]},
        }, "http://test"

    executor._call_with_router_metadata = MethodType(fake_call, executor)
    result = asyncio.run(executor.solve_v1(
        "What is 2+2?", reasoner_repetition_penalty=1.05, reasoner_temperature=0.2
    ))
    assert calls[0]["repetition_penalty"] == 1.05
    assert calls[0]["temperature"] == 0.2
    assert calls[1]["repetition_penalty"] == 1.0
    assert calls[1]["temperature"] == 0.0
    assert result["reasoner"]["repetition_penalty"] == 1.05
    assert result["reasoner"]["temperature"] == 0.2
    assert result["finalizer"]["repetition_penalty"] == 1.0
