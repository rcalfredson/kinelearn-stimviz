from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def ci95_bounds(
    mean: pd.Series | np.ndarray,
    sem: pd.Series | np.ndarray,
    n: pd.Series | np.ndarray,
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """Return two-sided 95% Student's t-confidence bounds."""
    critical_value = t.ppf(0.975, np.asarray(n) - 1)
    half_width = critical_value * sem
    return mean - half_width, mean + half_width


def add_summary_statistics(
    df: pd.DataFrame,
    value_col: str = "value",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate means and confidence intervals for aligned traces."""
    if group_cols is None:
        group_cols = []

    grouped = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
    lo, hi = ci95_bounds(grouped["mean"], grouped["sem"], grouped["n"])
    grouped["ci_low"] = lo
    grouped["ci_high"] = hi
    return grouped
