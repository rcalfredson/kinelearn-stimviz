from __future__ import annotations

import pandas as pd

from .stats import add_summary_statistics


def summarize_by_event(aligned_windows: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate rows within each event while retaining the time axis."""
    group_cols = ["entity_id", "event_id", "event_type", "event_time", "behavior", "relative_time"]
    extra_cols = [
        col
        for col in aligned_windows.columns
        if col not in group_cols and col != "value"
    ]
    group_cols = group_cols + extra_cols
    return (
        aligned_windows.groupby(group_cols, dropna=False)["value"]
        .mean()
        .reset_index()
        .sort_values(["behavior", "entity_id", "event_id", "relative_time"])
        .reset_index(drop=True)
    )


def summarize_by_subject(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Average across events per subject, behavior, and time bin."""
    base_cols = ["entity_id", "behavior", "relative_time"]
    extra_cols = [
        col
        for col in event_summary.columns
        if col not in {"event_id", "event_type", "event_time", "value"} and col not in base_cols
    ]
    group_cols = base_cols + extra_cols
    return (
        event_summary.groupby(group_cols, dropna=False)["value"]
        .mean()
        .reset_index()
        .sort_values(["behavior", "entity_id", "relative_time"])
        .reset_index(drop=True)
    )


def summarize_by_group(
    subject_summary: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Average across subjects with pointwise 95% Student's t-confidence intervals."""
    if group_cols is None:
        group_cols = []
    summary_cols = group_cols + ["behavior", "relative_time"]
    return add_summary_statistics(
        subject_summary,
        value_col="value",
        group_cols=summary_cols,
    ).sort_values(summary_cols).reset_index(drop=True)
