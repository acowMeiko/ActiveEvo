from modules.stage2.generator_comparison import build_balanced_manifest


def test_balanced_manifest_is_stable_and_excludes_dev_questions() -> None:
    packets = {
        f"level{level}": [{"idx": str(level * 100 + index)} for index in range(30)]
        for level in range(1, 5)
    }
    first = build_balanced_manifest(
        packets, per_level=25, seed=42, excluded_question_ids={"100", "200", "300", "400"}
    )
    second = build_balanced_manifest(
        packets, per_level=25, seed=42, excluded_question_ids={"100", "200", "300", "400"}
    )
    assert first == second
    assert first["actual_count"] == 100
    assert all(len(first["question_ids_by_level"][f"level{level}"]) == 25 for level in range(1, 5))
    assert not set(first["question_ids"]).intersection({"100", "200", "300", "400"})
