"""Tests for human evaluator."""

from pathlib import Path

from llm_judge_eval.evaluators.human import HumanEvaluator
from llm_judge_eval.schemas import EvalRecord, TaskType

SAMPLE = Path(__file__).resolve().parent.parent / "llm_judge_eval" / "sample_data"


def test_human_evaluator_loads_scores():
    records = [
        EvalRecord(
            id="qa1",
            task=TaskType.QA,
            input_text="Q?",
            prediction="Paris",
            reference="Paris",
        )
    ]
    scores = HumanEvaluator(SAMPLE / "qa_human_scores.csv").evaluate(records)
    overall = [s for s in scores if s.metric == "overall"]
    assert len(overall) == 1
    assert overall[0].score == 5.0
    assert overall[0].rationale is not None
