"""Automatic metric evaluators for QA, summarization, and classification."""

from __future__ import annotations

import re
from collections import Counter

from llm_judge_eval.evaluators.base import BaseEvaluator
from llm_judge_eval.schemas import EvalMethod, EvalRecord, ScoreDetail, TaskType


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return float(_normalize(prediction) == _normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0
    pred_ngrams = _ngrams(pred_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    if not ref_ngrams:
        return 0.0
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return overlap / sum(ref_ngrams.values())


def classification_accuracy(prediction: str, reference: str) -> float:
    return exact_match(prediction, reference)


def classification_f1(prediction: str, reference: str) -> float:
    # For single-label classification, F1 collapses to accuracy when labels match.
    return classification_accuracy(prediction, reference)


class AutomaticEvaluator(BaseEvaluator):
    """Compute task-specific automatic metrics."""

    TASK_METRICS: dict[TaskType, list[str]] = {
        TaskType.QA: ["exact_match", "token_f1"],
        TaskType.SUMMARIZATION: ["rouge_1", "rouge_2", "token_f1"],
        TaskType.CLASSIFICATION: ["accuracy", "f1"],
    }

    def evaluate(self, records: list[EvalRecord]) -> list[ScoreDetail]:
        scores: list[ScoreDetail] = []
        for record in records:
            if record.reference is None:
                continue
            metrics = self._score_record(record)
            for metric, value in metrics.items():
                scores.append(
                    ScoreDetail(
                        record_id=record.id,
                        method=EvalMethod.AUTOMATIC,
                        metric=metric,
                        score=float(value),
                    )
                )
        return scores

    def _score_record(self, record: EvalRecord) -> dict[str, float]:
        pred, ref = record.prediction, record.reference or ""
        if record.task == TaskType.QA:
            return {
                "exact_match": exact_match(pred, ref),
                "token_f1": token_f1(pred, ref),
            }
        if record.task == TaskType.SUMMARIZATION:
            return {
                "rouge_1": rouge_n(pred, ref, n=1),
                "rouge_2": rouge_n(pred, ref, n=2),
                "token_f1": token_f1(pred, ref),
            }
        if record.task == TaskType.CLASSIFICATION:
            return {
                "accuracy": classification_accuracy(pred, ref),
                "f1": classification_f1(pred, ref),
            }
        raise ValueError(f"Unsupported task: {record.task}")
