"""Base evaluator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_judge_eval.schemas import EvalRecord, ScoreDetail


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, records: list[EvalRecord]) -> list[ScoreDetail]:
        raise NotImplementedError
