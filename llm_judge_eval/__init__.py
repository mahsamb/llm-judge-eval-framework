"""LLM Judge Evaluation Framework — compare human, LLM-judge, and automatic metrics."""

from llm_judge_eval.runner import EvaluationRunner
from llm_judge_eval.schemas import EvalRecord, EvalResult, TaskType

__version__ = "0.1.0"
__all__ = ["EvaluationRunner", "EvalRecord", "EvalResult", "TaskType"]
