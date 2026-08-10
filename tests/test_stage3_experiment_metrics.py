from tools.evaluate_stage3_gate_adaptation import condition_metrics, paired_effect


def _condition(correct: bool, injected: bool):
    return {
        "correct": correct,
        "memory_injected": injected,
        "solver": {
            "final_answer": "1" if correct else "0",
            "reasoner": {
                "hit_token_limit": False,
                "repetition": {"hard_loop": False, "likely_loop": False},
                "format_valid": True,
                "completion_tokens": 10,
            },
            "finalizer": {"format_valid": True},
        },
    }


def test_raw_and_adapted_metrics_exclude_noneligible_base_only_rows():
    rows = [
        {
            "experiment_eligible": False,
            "conditions": {"base": _condition(False, False)},
        },
        {
            "experiment_eligible": True,
            "conditions": {
                "base": _condition(False, False),
                "raw": _condition(False, True),
                "adapted": _condition(True, True),
            },
        },
    ]
    assert condition_metrics(rows, "base")["count"] == 2
    assert condition_metrics(rows, "raw")["count"] == 1
    assert condition_metrics(rows, "adapted")["correct"] == 1
    effect = paired_effect(rows, "raw", "adapted")
    assert effect["eligible_count"] == 1
    assert effect["rescues"] == 1
