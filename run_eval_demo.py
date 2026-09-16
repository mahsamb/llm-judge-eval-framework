#!/usr/bin/env python3
"""Demo: compare human eval, LLM-as-judge, and automatic metrics."""

from pathlib import Path

from llm_judge_eval.config import EvaluationConfig, LLMJudgeConfig
from llm_judge_eval.runner import EvaluationRunner

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "llm_judge_eval" / "sample_data"


def run_task(task: str, dataset: str, human_scores: str) -> None:
    config = EvaluationConfig(
        task=task,
        output_dir=ROOT / "eval_outputs" / task,
        run_automatic=True,
        run_human=True,
        run_llm_judge=True,
        human_scores_path=SAMPLE / human_scores,
        llm_judge=LLMJudgeConfig(provider="mock"),  # use "openai" + API key for real judge
    )
    result = EvaluationRunner(config).run(SAMPLE / dataset, run_name=task)
    print(f"\n=== {task.upper()} ===")
    print(f"Records: {result.summary['n_records']}")
    for name, value in sorted(result.correlations.items()):
        if value == value:  # skip NaN
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    run_task("qa", "qa_predictions.csv", "qa_human_scores.csv")
    run_task("summarization", "summarization_predictions.csv", "summarization_human_scores.csv")
    run_task("classification", "classification_predictions.csv", "classification_human_scores.csv")
    print("\nReports written to eval_outputs/")
