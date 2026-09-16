"""Load and align human evaluation scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_judge_eval.evaluators.base import BaseEvaluator
from llm_judge_eval.schemas import EvalMethod, EvalRecord, ScoreDetail


class HumanEvaluator(BaseEvaluator):
    """Load human scores from CSV and align to evaluation records."""

    REQUIRED_COLUMNS = {"id", "overall"}

    def __init__(self, scores_path: Path | str):
        self.scores_path = Path(scores_path)
        self.df = pd.read_csv(self.scores_path)
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Human scores CSV missing columns: {sorted(missing)}")

    def evaluate(self, records: list[EvalRecord]) -> list[ScoreDetail]:
        record_ids = {r.id for r in records}
        scores: list[ScoreDetail] = []
        metric_cols = [c for c in self.df.columns if c not in {"id", "rationale"}]

        for _, row in self.df.iterrows():
            record_id = str(row["id"])
            if record_id not in record_ids:
                continue
            for metric in metric_cols:
                value = row[metric]
                if pd.isna(value):
                    continue
                scores.append(
                    ScoreDetail(
                        record_id=record_id,
                        method=EvalMethod.HUMAN,
                        metric=str(metric),
                        score=float(value),
                        rationale=str(row["rationale"]) if "rationale" in row and pd.notna(row.get("rationale")) else None,
                    )
                )
        return scores
