from types import MethodType

from modules.reranker_module import QwenApplicabilityReranker


def test_rank_is_score_only_and_rerank_keeps_threshold_behavior():
    reranker = QwenApplicabilityReranker.__new__(QwenApplicabilityReranker)

    def fake_score(self, question, candidates):
        return [0.2, 0.9, 0.6]

    reranker.score = MethodType(fake_score, reranker)
    candidates = [
        {"memory_id": "low", "content": {}},
        {"memory_id": "high", "content": {}},
        {"memory_id": "middle", "content": {}},
    ]
    ranked = reranker.rank("question", candidates, top_k=2)
    assert [candidate["memory_id"] for candidate in ranked] == ["high", "middle"]
    selected, all_ranked = reranker.rerank(
        "question", candidates, threshold=0.7, max_selected=1
    )
    assert [candidate["memory_id"] for candidate in selected] == ["high"]
    assert [candidate["memory_id"] for candidate in all_ranked] == ["high", "middle", "low"]
