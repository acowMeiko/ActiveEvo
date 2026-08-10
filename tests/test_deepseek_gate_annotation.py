import json

from tools.deepseek_gate_annotation import (
    evaluate,
    extract_pairs,
    parse_annotation,
    public_task,
    stratified_sample,
)


def _gate_records(item_count=30, candidates_per_item=5):
    records = []
    for item_idx in range(item_count):
        candidates = []
        judgments = []
        for rank in range(1, candidates_per_item + 1):
            memory_id = f"m_{item_idx}_{rank}"
            candidates.append({
                "memory_id": memory_id,
                "rerank_rank": rank,
                "rerank_score": 1.0 / rank,
                "retrieval_score": 0.5,
                "source_question": f"source question {memory_id}",
                "source_structure": {"core_skill": "algebra"},
                "principle": {
                    "trigger_condition": "A linear equation is present.",
                    "required_evidence": ["The unknown occurs linearly."],
                    "action_guidance": "Isolate the unknown.",
                },
            })
            judgments.append({
                "memory_id": memory_id,
                "accepted": rank == 1 and item_idx % 2 == 0,
            })
        records.append({
            "idx": item_idx,
            "question": f"current question {item_idx}",
            "retrieval_trace": {
                "gate_candidates": candidates,
                "gate_result": {
                    "parse_stage": "wrapped_object",
                    "judgments": judgments,
                },
            },
        })
    return records


def _annotation(label):
    applicable = label == "applicable"
    uncertain = label == "uncertain"
    return {
        "label": label,
        "trigger_match": applicable if not uncertain else False,
        "required_evidence_match": applicable if not uncertain else False,
        "structural_match": applicable if not uncertain else False,
        "scope_match": applicable if not uncertain else False,
        "contraindicated": False if applicable else True,
        "confidence": 0.9,
        "matched_evidence": [],
        "violated_conditions": [] if applicable else ["mismatch"],
        "source_specific_risk": [],
        "reason": "The structural prerequisites were checked independently.",
    }


def test_parse_annotation_enforces_consistent_schema():
    parsed, errors = parse_annotation(json.dumps(_annotation("applicable")))
    assert not errors
    assert parsed["label"] == "applicable"

    conflicting = _annotation("applicable")
    conflicting["structural_match"] = False
    _, errors = parse_annotation(json.dumps(conflicting))
    assert "applicable_label_conflicts_with_fields" in errors


def test_prepare_sampling_is_100_and_api_view_is_blind():
    pairs = extract_pairs(_gate_records())
    sampled = stratified_sample(pairs, sample_size=100, seed=42)
    assert len(pairs) == 150
    assert len(sampled) == 100
    assert any(pair["hidden_reference"]["llama_accepted"] for pair in sampled)

    visible = public_task(sampled[0])
    assert set(visible) == {
        "pair_id", "current_question", "source_question", "source_structure", "principle"
    }
    assert "hidden_reference" not in visible
    assert "rerank_score" not in json.dumps(visible)
    assert "llama_accepted" not in json.dumps(visible)

    smoke = stratified_sample(pairs, sample_size=10, seed=42, accepted_cap=5)
    assert len(smoke) == 10
    assert sum(pair["hidden_reference"]["llama_accepted"] for pair in smoke) == 5


def test_evaluate_computes_confusion_matrix_and_writes_review_files(tmp_path):
    tasks = extract_pairs(_gate_records(item_count=1, candidates_per_item=3))
    labels = tmp_path / "labels.jsonl"
    rows = [
        {"pair_id": tasks[0]["pair_id"], "status": "success", "annotation": _annotation("applicable")},
        {"pair_id": tasks[1]["pair_id"], "status": "success", "annotation": _annotation("applicable")},
        {"pair_id": tasks[2]["pair_id"], "status": "success", "annotation": _annotation("reject")},
    ]
    labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    metrics = evaluate(tasks, labels, tmp_path)
    assert metrics["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 1}
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert (tmp_path / "annotation_adjudication.csv").exists()
    assert (tmp_path / "gate_evaluation_report.md").exists()
