"""Tests for automatic metrics."""

from llm_judge_eval.evaluators.automatic import (
    exact_match,
    rouge_n,
    token_f1,
)
from llm_judge_eval.evaluators.automatic import AutomaticEvaluator
from llm_judge_eval.schemas import EvalRecord, TaskType


def test_exact_match_true():
    assert exact_match("Paris", "paris") == 1.0


def test_exact_match_false():
    assert exact_match("1944", "1945") == 0.0


def test_token_f1_partial_overlap():
    score = token_f1("100 degrees Celsius", "100°C at sea level")
    assert score == 0.0


def test_token_f1_identical():
    assert token_f1("hello world", "hello world") == 1.0


def test_rouge_n_perfect():
    assert rouge_n("the cat sat", "the cat sat", n=1) == 1.0


def test_automatic_evaluator_qa():
    records = [
        EvalRecord(
            id="1",
            task=TaskType.QA,
            input_text="Q?",
            prediction="Paris",
            reference="Paris",
        )
    ]
    scores = AutomaticEvaluator().evaluate(records)
    metrics = {s.metric: s.score for s in scores}
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
