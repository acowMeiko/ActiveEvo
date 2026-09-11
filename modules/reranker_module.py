"""Instruction-aware applicability reranking for recalled memories."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from modules.principle_schema import build_memory_views

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION = (
    "Given a complete math problem and a candidate past experience, decide whether "
    "the experience's trigger, mathematical objects, constraints, objective, and "
    "recommended operation all apply to this problem. Reject topic-only similarity, "
    "mismatched variables or geometric objects, and advice that could change the "
    "problem's meaning. Return yes only when the experience is directly executable."
)


class QwenApplicabilityReranker:
    """Qwen3-Reranker yes/no probability wrapper loaded once per evaluation."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        max_length: int = 2048,
        batch_size: int = 4,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
        ).to(self.device).eval()
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n'
        )
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        logger.info("Applicability reranker loaded from %s on %s", model_path, self.device)

    @staticmethod
    def _document(candidate: Dict[str, Any]) -> str:
        content = candidate["content"]
        routing = build_memory_views(content)["routing_view"]
        return "\n".join(
            [
                f"Trigger: {routing['trigger_condition']}",
                f"Required evidence: {routing['required_evidence']}",
                f"Reject when: {routing['exclusion_condition']}",
                f"Scope: {routing['scope']}",
                f"Failure to avoid: {routing['failure_mode']}",
                f"Action: {routing['action_guidance']}",
                f"Verification: {routing['verification_step']}",
                f"Generalization level: {routing['generalization_level']}",
            ]
        )

    def _format(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    @torch.inference_mode()
    def score(self, question: str, candidates: Sequence[Dict[str, Any]]) -> List[float]:
        texts = [self._format(question, self._document(candidate)) for candidate in candidates]
        scores: List[float] = []
        body_limit = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                texts[start:start + self.batch_size],
                padding=False,
                truncation=True,
                max_length=body_limit,
                return_attention_mask=False,
            )
            encoded["input_ids"] = [
                self.prefix_tokens + ids + self.suffix_tokens for ids in encoded["input_ids"]
            ]
            inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits[:, -1, :]
            binary = torch.stack(
                [logits[:, self.false_token_id], logits[:, self.true_token_id]], dim=1
            )
            scores.extend(torch.softmax(binary.float(), dim=1)[:, 1].cpu().tolist())
        return scores

    def rerank(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        threshold: float,
        max_selected: int = 1,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ranked = self.rank(question, candidates)
        selected = [item for item in ranked if item["rerank_score"] >= threshold][:max_selected]
        return selected, ranked

    def rank(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Score and sort candidates without treating relevance as reliability.

        This is used before the structural Application Gate: a high-recall
        retriever supplies candidates, the reranker narrows the batch, and the
        Gate makes the actual accept/reject decision.
        """
        probabilities = self.score(question, candidates)
        ranked: List[Dict[str, Any]] = []
        for candidate, probability in zip(candidates, probabilities):
            record = dict(candidate)
            record["rerank_score"] = float(probability)
            ranked.append(record)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_k] if top_k is not None else ranked


class BGECrossEncoderReranker:
    """Lightweight BGE cross-encoder for structured Memory relevance ranking."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        max_length: int = 1024,
        batch_size: int = 16,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
        ).to(self.device).eval()
        logger.info("BGE cross-encoder loaded from %s on %s", model_path, self.device)

    @staticmethod
    def query_text(question: str, semantics: Dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Problem: {question}",
                f"Core skill: {semantics.get('core_skill', semantics.get('core_math_concept', ''))}",
                f"Constraints: {semantics.get('context_constraints', semantics.get('conditions_structure', ''))}",
                f"Objective: {semantics.get('task_objective', semantics.get('solution_target', ''))}",
            ]
        )

    @staticmethod
    def document_text(candidate: Dict[str, Any]) -> str:
        content = candidate.get("content", {})
        retrieval = candidate.get("retrieval", {})
        item = retrieval.get("item", {}) if isinstance(retrieval, dict) else {}
        source = {
            "core_skill": item.get("core_skill", item.get("core_math_concept", "")),
            "context_constraints": item.get("context_constraints", item.get("conditions_structure", "")),
            "task_objective": item.get("task_objective", item.get("solution_target", "")),
        }
        evidence = content.get("required_evidence", [])
        signals = content.get("failure_signals", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if isinstance(signals, str):
            signals = [signals]
        return "\n".join(
            [
                f"Source core skill: {source['core_skill']}",
                f"Source constraints: {source['context_constraints']}",
                f"Source objective: {source['task_objective']}",
                f"Trigger: {content.get('trigger_condition', '')}",
                f"Required evidence: {'; '.join(map(str, evidence))}",
                f"Reject when: {content.get('exclusion_condition', '')}",
                f"Scope: {content.get('scope', '')}",
                f"Failure mode: {content.get('failure_mode', '')}",
                f"Failure signals: {'; '.join(map(str, signals))}",
                f"Action: {content.get('action_guidance', '')}",
                f"Verification: {content.get('verification_step', '')}",
            ]
        )

    @torch.inference_mode()
    def score(
        self,
        question: str,
        semantics: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
    ) -> List[float]:
        query = self.query_text(question, semantics)
        documents = [self.document_text(candidate) for candidate in candidates]
        scores: List[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits.float().reshape(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores

    def rank(
        self,
        question: str,
        semantics: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        probabilities = self.score(question, semantics, candidates)
        ranked = []
        for candidate, probability in zip(candidates, probabilities):
            row = dict(candidate)
            row["rerank_score"] = float(probability)
            row["pre_rerank_score"] = float(row.get("score", 0.0))
            row["score"] = float(probability)
            row["retrieval_score"] = float(probability)
            row["retrieval_method"] = "bge_cross_encoder"
            ranked.append(row)
        ranked.sort(key=lambda row: (row["rerank_score"], row["pre_rerank_score"]), reverse=True)
        return ranked[: max(0, int(top_k))]
