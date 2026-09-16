"""Statistical agreement between evaluation methods."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    value = x.corr(y, method=method)
    return float(value) if pd.notna(value) else float("nan")


def _cohens_kappa(x: pd.Series, y: pd.Series, bins: int = 5) -> float:
    """Cohen's kappa on binned scores (useful when scales are ordinal)."""
    if len(x) < 2:
        return float("nan")
    if x.nunique() < 2 and y.nunique() < 2:
        return 1.0 if (x == y).all() else 0.0

    combined = pd.concat([x, y])
    if combined.nunique() < 2:
        return 1.0 if (x == y).all() else 0.0

    edges = [combined.min()] + [
        combined.min() + (combined.max() - combined.min()) * i / bins for i in range(1, bins)
    ] + [combined.max() + 1e-9]
    edges = sorted(set(edges))

    x_cat = pd.cut(x, bins=edges, labels=False, include_lowest=True)
    y_cat = pd.cut(y, bins=edges, labels=False, include_lowest=True)

    labels = sorted(set(x_cat.dropna()) | set(y_cat.dropna()))
    n = len(x)
    if n == 0:
        return float("nan")

    matrix = {(a, b): 0 for a in labels for b in labels}
    for a, b in zip(x_cat, y_cat):
        if pd.isna(a) or pd.isna(b):
            continue
        matrix[(int(a), int(b))] += 1

    po = sum(matrix[(l, l)] for l in labels) / n
    row_marginals = {l: sum(matrix[(l, b)] for b in labels) / n for l in labels}
    col_marginals = {l: sum(matrix[(a, l)] for a in labels) / n for l in labels}
    pe = sum(row_marginals[l] * col_marginals[l] for l in labels)

    if math.isclose(1 - pe, 0.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return float((po - pe) / (1 - pe))


def compute_pairwise_correlations(
    scores_df: pd.DataFrame,
    methods: Iterable[str],
    metric_map: dict[str, str],
) -> dict[str, float]:
    """
    Compare aligned primary metrics across evaluation methods.

    Returns keys like 'human_vs_llm_judge_pearson'.
    """
    aligned = scores_df.pivot_table(
        index="record_id",
        columns=["method", "metric"],
        values="score",
        aggfunc="first",
    )

    results: dict[str, float] = {}
    method_list = list(methods)

    for i, method_a in enumerate(method_list):
        metric_a = metric_map.get(method_a)
        if not metric_a:
            continue
        col_a = (method_a, metric_a)
        if col_a not in aligned.columns:
            continue

        for method_b in method_list[i + 1 :]:
            metric_b = metric_map.get(method_b)
            if not metric_b:
                continue
            col_b = (method_b, metric_b)
            if col_b not in aligned.columns:
                continue

            pair = aligned[[col_a, col_b]].dropna()
            if pair.empty:
                continue

            x = pair[col_a]
            y = pair[col_b]
            prefix = f"{method_a}_vs_{method_b}"
            results[f"{prefix}_pearson"] = _safe_corr(x, y, "pearson")
            results[f"{prefix}_spearman"] = _safe_corr(x, y, "spearman")

            # Kappa only for ordinal scales (human / llm), not binary automatic metrics
            if method_a in {"human", "llm_judge"} and method_b in {"human", "llm_judge"}:
                results[f"{prefix}_kappa"] = _cohens_kappa(x, y)

    return results
