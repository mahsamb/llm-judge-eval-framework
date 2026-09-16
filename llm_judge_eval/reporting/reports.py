"""Generate pandas-based evaluation reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_judge_eval.schemas import ScoreDetail


def scores_to_dataframe(scores: list[ScoreDetail]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": s.record_id,
                "method": s.method.value,
                "metric": s.metric,
                "score": s.score,
                "rationale": s.rationale,
            }
            for s in scores
        ]
    )


def build_summary_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std per method and metric."""
    summary = (
        scores_df.groupby(["method", "metric"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "score_mean", "std": "score_std", "count": "n"})
    )
    summary["score_std"] = summary["score_std"].fillna(0.0)
    return summary


def build_record_level_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Wide format: one row per record, columns per method/metric."""
    wide = scores_df.pivot_table(
        index="record_id",
        columns=["method", "metric"],
        values="score",
        aggfunc="first",
    )
    wide.columns = [f"{method}__{metric}" for method, metric in wide.columns]
    return wide.reset_index()


def export_report(
    scores_df: pd.DataFrame,
    summary: dict,
    correlations: dict[str, float],
    output_dir: Path,
    run_name: str = "eval_run",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    scores_path = output_dir / f"{run_name}_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    paths["scores"] = scores_path

    summary_df = build_summary_table(scores_df)
    summary_path = output_dir / f"{run_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    record_path = output_dir / f"{run_name}_by_record.csv"
    build_record_level_table(scores_df).to_csv(record_path, index=False)
    paths["by_record"] = record_path

    if correlations:
        corr_df = pd.DataFrame(
            [{"comparison": k, "value": v} for k, v in sorted(correlations.items())]
        )
        corr_path = output_dir / f"{run_name}_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        paths["correlations"] = corr_path

    report_path = output_dir / f"{run_name}_report.txt"
    lines = [
        "LLM Judge Evaluation Report",
        "=" * 40,
        "",
        "Run summary:",
    ]
    for key, value in summary.items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Aggregate scores:"])
    for _, row in summary_df.iterrows():
        lines.append(
            f"  [{row['method']}/{row['metric']}] mean={row['score_mean']:.4f} "
            f"std={row['score_std']:.4f} n={int(row['n'])}"
        )
    if correlations:
        lines.extend(["", "Method correlations:"])
        for name, value in sorted(correlations.items()):
            if pd.notna(value):
                lines.append(f"  {name}: {value:.4f}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["report"] = report_path

    return paths
