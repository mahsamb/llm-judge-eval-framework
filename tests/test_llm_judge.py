"""Tests for LLM-as-judge evaluator."""

import json

import pytest

from llm_judge_eval.config import LLMJudgeConfig
from llm_judge_eval.evaluators.llm_judge import LLMJudgeEvaluator
from llm_judge_eval.schemas import EvalRecord, TaskType


def test_mock_judge_exact_match():
    evaluator = LLMJudgeEvaluator(LLMJudgeConfig(provider="mock"))
    record = EvalRecord(
        id="1",
        task=TaskType.QA,
        input_text="Capital of France?",
        prediction="Paris",
        reference="Paris",
    )
    scores = evaluator.evaluate([record])
    assert len(scores) == 1
    assert scores[0].score == 5.0
    assert "exactly" in (scores[0].rationale or "").lower()


def test_mock_judge_wrong_answer():
    evaluator = LLMJudgeEvaluator(LLMJudgeConfig(provider="mock"))
    record = EvalRecord(
        id="2",
        task=TaskType.QA,
        input_text="When did WWII end?",
        prediction="1944",
        reference="1945",
    )
    scores = evaluator.evaluate([record])
    assert scores[0].score == 2.0


def test_parse_response_json():
    parsed = LLMJudgeEvaluator._parse_response('{"score": 4, "rationale": "good"}')
    assert parsed["score"] == 4


def test_parse_response_with_extra_text():
    parsed = LLMJudgeEvaluator._parse_response('Here is my answer: {"score": 3, "rationale": "ok"}')
    assert parsed["score"] == 3


def test_openai_requires_api_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    evaluator = LLMJudgeEvaluator(LLMJudgeConfig(provider="openai", api_key=None))
    with pytest.raises(ValueError, match="API key"):
        evaluator._openai_judge("test prompt")
