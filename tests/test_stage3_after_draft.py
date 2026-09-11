import json
from pathlib import Path

from scripts.stage3.evaluate import after_draft_decision, load_baseline_cache


def test_after_draft_decision_parses_both_labels():
    assert after_draft_decision("Principle Processing:\ncorrected_draft - fixed") == "corrected_draft"
    assert after_draft_decision("Principle Processing: verified_no_change - valid") == "verified_no_change"
    assert after_draft_decision("Method:\nNo marker") == "missing"


def test_load_baseline_cache_uses_idx_and_seed(tmp_path: Path):
    output = tmp_path / "baseline"
    output.mkdir()
    (output / "eval_results.json").write_text(
        json.dumps(
            [
                {
                    "idx": 7,
                    "rollouts": [
                        {
                            "seed": 42,
                            "status": "complete",
                            "correct": True,
                            "response": {"reasoner": {"output": "Method:\nA draft."}},
                        },
                        {
                            "seed": 43,
                            "status": "complete",
                            "correct": False,
                            "response": {"reasoner": {"output": ""}},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    cache = load_baseline_cache(str(output))
    assert sorted(cache) == [(7, 42)]
    assert cache[(7, 42)]["correct"] is True
