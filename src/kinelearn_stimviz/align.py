from __future__ import annotations

import numpy as np
import pandas as pd


def select_events(
    events_df: pd.DataFrame,
    *,
    subset: str = "all",
    count: int | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Select all, first N, or last N events within each event group."""
    if subset not in {"all", "first", "last"}:
        raise ValueError(f"Unsupported event subset: {subset}")
    if count is not None and count <= 0:
        raise ValueError("Event count must be a positive integer.")

    if subset == "all" or count is None:
        return events_df.copy()

    if group_cols is None:
        group_cols = ["entity_id", "event_type"]

    sorted_events = events_df.sort_values(group_cols + ["event_time", "event_id"]).copy()
    selected = []
    for _, group in sorted_events.groupby(group_cols, sort=False, dropna=False):
        if subset == "first":
            selected.append(group.head(count))
        else:
            selected.append(group.tail(count))

    return pd.concat(selected, ignore_index=True)


def infer_time_axis(
    behavior_df: pd.DataFrame,
    *,
    pre: float,
    post: float,
    bin_size: float | None = None,
) -> np.ndarray:
    """Infer an evenly spaced relative-time axis for aligned windows."""
    if bin_size is None:
        diffs = (
            behavior_df.sort_values(["entity_id", "behavior", "time"])
            .groupby(["entity_id", "behavior"])["time"]
            .diff()
            .dropna()
        )
        if diffs.empty:
            raise ValueError("Could not infer a bin size from the behavior table; provide --bin-size.")
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.empty:
            raise ValueError("Behavior time column has no positive spacing; provide --bin-size.")
        bin_size = float(positive_diffs.median())

    steps = int(np.floor((pre + post) / bin_size + 1e-9)) + 1
    return np.round(np.linspace(-pre, -pre + bin_size * (steps - 1), steps), 9)


def align_event_times(
    trace: pd.DataFrame,
    *,
    event_time: float,
    time_axis: np.ndarray,
    tolerance: float | None = None,
) -> pd.DataFrame:
    """Align one trace segment to a single event using nearest-bin matching."""
    if tolerance is None:
        if len(time_axis) < 2:
            raise ValueError("Need at least two time bins or an explicit tolerance.")
        tolerance = float(np.diff(time_axis).mean() / 2 + 1e-9)

    target = pd.DataFrame({"relative_time": time_axis})
    segment = trace.copy()
    segment["relative_time"] = segment["time"] - event_time
    segment = segment.sort_values("relative_time")[["relative_time", "value"]]

    aligned = pd.merge_asof(
        target,
        segment,
        on="relative_time",
        direction="nearest",
        tolerance=tolerance,
    )
    return aligned


def build_aligned_windows(
    behavior_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    pre: float,
    post: float,
    time_axis: np.ndarray | None = None,
    bin_size: float | None = None,
    metadata_df: pd.DataFrame | None = None,
    behaviors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build stimulus-aligned behavior windows in long format.

    Returns one row per event, behavior, and relative-time bin with columns such as
    `entity_id`, `event_id`, `event_type`, `behavior`, `relative_time`, and `value`.
    """
    if time_axis is None:
        time_axis = infer_time_axis(behavior_df, pre=pre, post=post, bin_size=bin_size)

    if behaviors is not None:
        behavior_df = behavior_df[behavior_df["behavior"].isin(behaviors)].copy()

    event_records: list[pd.DataFrame] = []
    for event in events_df.itertuples(index=False):
        event_start = float(event.event_time) - pre
        event_end = float(event.event_time) + post
        entity_behavior = behavior_df[
            (behavior_df["entity_id"] == event.entity_id)
            & (behavior_df["time"] >= event_start)
            & (behavior_df["time"] <= event_end)
        ]

        for behavior_name, trace in entity_behavior.groupby("behavior", sort=False):
            aligned = align_event_times(
                trace,
                event_time=float(event.event_time),
                time_axis=time_axis,
            )
            aligned["event_id"] = event.event_id
            aligned["entity_id"] = event.entity_id
            aligned["event_type"] = event.event_type
            aligned["event_time"] = float(event.event_time)
            aligned["behavior"] = behavior_name
            event_records.append(aligned)

    if not event_records:
        raise ValueError("No aligned windows were created. Check entity ids, event times, and window size.")

    windows = pd.concat(event_records, ignore_index=True)
    ordered_columns = [
        "entity_id",
        "event_id",
        "event_type",
        "event_time",
        "behavior",
        "relative_time",
        "value",
    ]
    windows = windows[ordered_columns]

    if metadata_df is not None:
        windows = windows.merge(metadata_df, on="entity_id", how="left")

    return windows.sort_values(["behavior", "entity_id", "event_id", "relative_time"]).reset_index(drop=True)
