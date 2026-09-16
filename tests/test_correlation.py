"""Tests for correlation and agreement metrics."""

import pandas as pd

from llm_judge_eval.correlation.agreement import _cohens_kappa, compute_pairwise_correlations


def test_cohens_kappa_perfect_agreement():
    x = pd.Series([5, 4, 3, 2, 1])
    y = pd.Series([5, 4, 3, 2, 1])
    assert _cohens_kappa(x, y) == 1.0


def test_cohens_kappa_constant_scores():
    x = pd.Series([5, 5, 5])
    y = pd.Series([5, 5, 5])
    assert _cohens_kappa(x, y) == 1.0


def test_pairwise_correlations():
    scores_df = pd.DataFrame(
        [
            {"record_id": "a", "method": "human", "metric": "overall", "score": 5.0},
            {"record_id": "a", "method": "llm_judge", "metric": "overall", "score": 5.0},
            {"record_id": "b", "method": "human", "metric": "overall", "score": 2.0},
            {"record_id": "b", "method": "llm_judge", "metric": "overall", "score": 2.0},
            {"record_id": "c", "method": "human", "metric": "overall", "score": 4.0},
            {"record_id": "c", "method": "llm_judge", "metric": "overall", "score": 3.0},
        ]
    )
    metric_map = {"human": "overall", "llm_judge": "overall"}
    results = compute_pairwise_correlations(scores_df, ["human", "llm_judge"], metric_map)
    assert "human_vs_llm_judge_pearson" in results
    assert results["human_vs_llm_judge_pearson"] > 0.9
