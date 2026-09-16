"""Command-line interface for evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_judge_eval.config import EvaluationConfig, LLMJudgeConfig
from llm_judge_eval.runner import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Judge Evaluation Framework")
    parser.add_argument("--task", required=True, choices=["qa", "summarization", "classification"])
    parser.add_argument("--dataset", required=True, type=Path, help="CSV/JSON with predictions")
    parser.add_argument("--human-scores", type=Path, help="CSV with human annotations")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--run-name", default="eval_run")
    parser.add_argument("--skip-automatic", action="store_true")
    parser.add_argument("--skip-human", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--llm-provider", default="mock", choices=["mock", "openai"])
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional; prefer OPENAI_API_KEY environment variable",
    )
    args = parser.parse_args()

    if not args.skip_human and args.human_scores is None:
        parser.error("--human-scores is required unless --skip-human is set")

    config = EvaluationConfig(
        task=args.task,
        output_dir=args.output_dir,
        run_automatic=not args.skip_automatic,
        run_human=not args.skip_human,
        run_llm_judge=not args.skip_llm_judge,
        human_scores_path=args.human_scores,
        llm_judge=LLMJudgeConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key=args.openai_api_key,
        ),
    )

    result = EvaluationRunner(config).run(args.dataset, run_name=args.run_name)
    print(f"Evaluated {result.summary['n_records']} records.")
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
