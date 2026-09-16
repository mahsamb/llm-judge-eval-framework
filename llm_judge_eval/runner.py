"""Main evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from llm_judge_eval.config import EvaluationConfig
from llm_judge_eval.correlation.agreement import compute_pairwise_correlations
from llm_judge_eval.data_loader import load_records
from llm_judge_eval.evaluators import AutomaticEvaluator, HumanEvaluator, LLMJudgeEvaluator
from llm_judge_eval.reporting.reports import export_report, scores_to_dataframe
from llm_judge_eval.schemas import EvalMethod, EvalRecord, EvalResult, TaskType


class EvaluationRunner:
    """Run human, LLM-judge, and automatic evaluation and compare them."""

    DEFAULT_PRIMARY_METRICS = {
        TaskType.QA: "exact_match",
        TaskType.SUMMARIZATION: "rouge_1",
        TaskType.CLASSIFICATION: "accuracy",
    }

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.task = TaskType(config.task)

    def run(
        self,
        dataset_path: Path | str,
        run_name: str = "eval_run",
    ) -> EvalResult:
        records = load_records(dataset_path, self.task)
        all_scores = []

        if self.config.run_automatic:
            all_scores.extend(AutomaticEvaluator().evaluate(records))

        if self.config.run_human:
            if not self.config.human_scores_path:
                raise ValueError("human_scores_path required when run_human=True")
            all_scores.extend(HumanEvaluator(self.config.human_scores_path).evaluate(records))

        if self.config.run_llm_judge:
            all_scores.extend(LLMJudgeEvaluator(self.config.llm_judge).evaluate(records))

        scores_df = scores_to_dataframe(all_scores)
        metric_map = self._primary_metric_map()
        methods = sorted(scores_df["method"].unique())
        correlations = compute_pairwise_correlations(scores_df, methods, metric_map)

        summary = {
            "task": self.task.value,
            "n_records": len(records),
            "n_scores": len(all_scores),
            "methods": methods,
        }

        export_report(
            scores_df=scores_df,
            summary=summary,
            correlations=correlations,
            output_dir=self.config.output_dir,
            run_name=run_name,
        )

        return EvalResult(scores=all_scores, summary=summary, correlations=correlations)

    def _primary_metric_map(self) -> dict[str, str]:
        auto_metric = (
            self.config.primary_automatic_metric
            or self.DEFAULT_PRIMARY_METRICS[self.task]
        )
        return {
            EvalMethod.AUTOMATIC.value: auto_metric,
            EvalMethod.HUMAN.value: self.config.primary_human_metric,
            EvalMethod.LLM_JUDGE.value: self.config.primary_llm_metric,
        }
