"""Integration tests for the evaluation pipeline."""

from pathlib import Path

from llm_judge_eval.config import EvaluationConfig, LLMJudgeConfig
from llm_judge_eval.runner import EvaluationRunner

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "llm_judge_eval" / "sample_data"


def test_runner_qa_end_to_end(tmp_path):
    config = EvaluationConfig(
        task="qa",
        output_dir=tmp_path,
        run_automatic=True,
        run_human=True,
        run_llm_judge=True,
        human_scores_path=SAMPLE / "qa_human_scores.csv",
        llm_judge=LLMJudgeConfig(provider="mock"),
    )
    result = EvaluationRunner(config).run(SAMPLE / "qa_predictions.csv", run_name="qa_test")

    assert result.summary["n_records"] == 5
    assert result.summary["n_scores"] > 0
    assert (tmp_path / "qa_test_scores.csv").exists()
    assert (tmp_path / "qa_test_report.txt").exists()


def test_runner_skip_human(tmp_path):
    config = EvaluationConfig(
        task="classification",
        output_dir=tmp_path,
        run_automatic=True,
        run_human=False,
        run_llm_judge=True,
        llm_judge=LLMJudgeConfig(provider="mock"),
    )
    result = EvaluationRunner(config).run(
        SAMPLE / "classification_predictions.csv",
        run_name="cls_test",
    )
    methods = result.summary["methods"]
    assert "human" not in methods
    assert "automatic" in methods
    assert "llm_judge" in methods
