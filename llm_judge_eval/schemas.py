"""Data models for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    QA = "qa"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


class EvalMethod(str, Enum):
    AUTOMATIC = "automatic"
    HUMAN = "human"
    LLM_JUDGE = "llm_judge"


@dataclass
class EvalRecord:
    """One model prediction to evaluate."""

    id: str
    task: TaskType
    input_text: str
    prediction: str
    reference: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreDetail:
    """A single metric score for one record."""

    record_id: str
    method: EvalMethod
    metric: str
    score: float
    rationale: str | None = None
    raw_response: str | None = None


@dataclass
class EvalResult:
    """Aggregated output from a full evaluation run."""

    scores: list[ScoreDetail]
    summary: dict[str, Any]
    correlations: dict[str, float] = field(default_factory=dict)
