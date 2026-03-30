from __future__ import annotations

import numpy as np
import pandas as pd


def infer_frame_rate(
    behavior_df: pd.DataFrame,
    *,
    fps: float | None = None,
    frame_col: str = "frame_index",
) -> float:
    """Infer fps from frame/time columns, or return the provided fps."""
    if fps is not None:
        return fps
    if frame_col not in behavior_df.columns:
        raise ValueError("Frame alignment requires a frame_index column or an explicit fps.")

    diffs = (
        behavior_df.sort_values(["entity_id", "behavior", frame_col])
        .groupby(["entity_id", "behavior"])[["time", frame_col]]
        .diff()
        .dropna()
    )
    diffs = diffs[(diffs[frame_col] > 0) & (diffs["time"] > 0)]
    if diffs.empty:
        raise ValueError("Could not infer fps from behavior table; provide --fps.")
    seconds_per_frame = (diffs["time"] / diffs[frame_col]).median()
    return float(1.0 / seconds_per_frame)


def _compute_tolerance(time_axis: np.ndarray, tolerance: float | None = None) -> float:
    if tolerance is not None:
        return tolerance
    if len(time_axis) < 2:
        raise ValueError("Need at least two time bins or an explicit tolerance.")
    return float(np.diff(time_axis).mean() / 2 + 1e-9)


def _align_trace_values(
    times: np.ndarray,
    values: np.ndarray,
    *,
    event_time: float,
    time_axis: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Align one numeric trace to an event using nearest-time matching."""
    target_times = event_time + time_axis
    aligned = np.full(len(target_times), np.nan, dtype=float)

    right_idx = np.searchsorted(times, target_times, side="left")
    left_idx = right_idx - 1

    best_idx = np.full(len(target_times), -1, dtype=int)
    best_dist = np.full(len(target_times), np.inf, dtype=float)

    valid_left = left_idx >= 0
    if np.any(valid_left):
        left_dist = np.abs(times[left_idx[valid_left]] - target_times[valid_left])
        best_idx[valid_left] = left_idx[valid_left]
        best_dist[valid_left] = left_dist

    valid_right = right_idx < len(times)
    if np.any(valid_right):
        right_dist = np.abs(times[right_idx[valid_right]] - target_times[valid_right])
        update_mask = right_dist < best_dist[valid_right]
        valid_positions = np.flatnonzero(valid_right)
        update_positions = valid_positions[update_mask]
        best_idx[update_positions] = right_idx[update_positions]
        best_dist[update_positions] = right_dist[update_positions]

    matched = best_idx >= 0
    within_tolerance = matched & (best_dist <= tolerance)
    aligned[within_tolerance] = values[best_idx[within_tolerance]]
    return aligned


def _prepare_behavior_lookup(behavior_df: pd.DataFrame) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Pre-split behavior traces once so events do not rescan the full table."""
    lookup: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    sorted_df = behavior_df.sort_values(["entity_id", "behavior", "time"])
    for (entity_id, behavior_name), group in sorted_df.groupby(["entity_id", "behavior"], sort=False):
        entity_key = str(entity_id)
        behavior_key = str(behavior_name)
        if entity_key not in lookup:
            lookup[entity_key] = {}
        lookup[entity_key][behavior_key] = (
            group["time"].to_numpy(dtype=float),
            group["value"].to_numpy(dtype=float),
        )
    return lookup


def _prepare_frame_lookup(
    behavior_df: pd.DataFrame,
    *,
    frame_col: str,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Pre-split behavior traces by frame index for direct frame-offset slicing."""
    lookup: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    sorted_df = behavior_df.sort_values(["entity_id", "behavior", frame_col])
    for (entity_id, behavior_name), group in sorted_df.groupby(["entity_id", "behavior"], sort=False):
        entity_key = str(entity_id)
        behavior_key = str(behavior_name)
        if entity_key not in lookup:
            lookup[entity_key] = {}
        lookup[entity_key][behavior_key] = (
            group[frame_col].to_numpy(dtype=int),
            group["value"].to_numpy(dtype=float),
        )
    return lookup


def _infer_frame_time_axis(
    *,
    pre: float,
    post: float,
    fps: float,
) -> np.ndarray:
    """Build the legacy-style time axis used by direct frame slicing."""
    n_frames = int(np.ceil((pre + post) * fps))
    return np.arange(-pre, post, 1.0 / fps)[:n_frames]


def _align_trace_frames(
    frames: np.ndarray,
    values: np.ndarray,
    *,
    start_frame: int,
    n_frames: int,
) -> np.ndarray:
    """Align one trace to an event using direct frame-offset slicing."""
    target_frames = start_frame + np.arange(n_frames, dtype=int)
    aligned = np.full(n_frames, np.nan, dtype=float)
    idx = np.searchsorted(frames, target_frames, side="left")
    valid = (idx < len(frames)) & (frames[idx] == target_frames)
    aligned[valid] = values[idx[valid]]
    return aligned


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
    tolerance = _compute_tolerance(time_axis, tolerance)

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
    alignment_mode: str = "nearest",
    fps: float | None = None,
    frame_col: str = "frame_index",
) -> pd.DataFrame:
    """
    Build stimulus-aligned behavior windows in long format.

    Returns one row per event, behavior, and relative-time bin with columns such as
    `entity_id`, `event_id`, `event_type`, `behavior`, `relative_time`, and `value`.
    """
    if alignment_mode not in {"nearest", "frame"}:
        raise ValueError(f"Unsupported alignment mode: {alignment_mode}")

    if behaviors is not None:
        behavior_df = behavior_df[behavior_df["behavior"].isin(behaviors)].copy()
        behavior_order = list(dict.fromkeys(behaviors))
    else:
        behavior_order = list(behavior_df["behavior"].drop_duplicates())

    if time_axis is None:
        if alignment_mode == "frame":
            fps = infer_frame_rate(behavior_df, fps=fps, frame_col=frame_col)
            time_axis = _infer_frame_time_axis(pre=pre, post=post, fps=fps)
        else:
            time_axis = infer_time_axis(behavior_df, pre=pre, post=post, bin_size=bin_size)

    if alignment_mode == "frame":
        fps = infer_frame_rate(behavior_df, fps=fps, frame_col=frame_col)
        if frame_col not in behavior_df.columns:
            raise ValueError("Frame alignment requires a behavior table with a frame_index column.")
        behavior_lookup = _prepare_frame_lookup(behavior_df, frame_col=frame_col)
        n_frames = len(time_axis)
    else:
        tolerance = _compute_tolerance(time_axis)
        behavior_lookup = _prepare_behavior_lookup(behavior_df)

    event_records: list[pd.DataFrame] = []
    for event in events_df.sort_values(["entity_id", "event_time", "event_id"]).itertuples(index=False):
        entity_id = str(event.entity_id)
        entity_behavior = behavior_lookup.get(entity_id)
        if not entity_behavior:
            continue

        event_time = float(event.event_time)
        event_start = event_time - pre
        event_end = event_time + post

        for behavior_name in behavior_order:
            trace = entity_behavior.get(behavior_name)
            if trace is None:
                continue

            if alignment_mode == "frame":
                frames, values = trace
                start_frame = int(np.floor(event_start * fps))
                stop_frame = start_frame + n_frames
                left = np.searchsorted(frames, start_frame, side="left")
                right = np.searchsorted(frames, stop_frame, side="left")
                if left >= right:
                    continue
                aligned_values = _align_trace_frames(
                    frames[left:right],
                    values[left:right],
                    start_frame=start_frame,
                    n_frames=n_frames,
                )
            else:
                times, values = trace
                left = np.searchsorted(times, event_start, side="left")
                right = np.searchsorted(times, event_end, side="right")
                if left >= right:
                    continue
                aligned_values = _align_trace_values(
                    times[left:right],
                    values[left:right],
                    event_time=event_time,
                    time_axis=time_axis,
                    tolerance=tolerance,
                )
            aligned = pd.DataFrame(
                {
                    "entity_id": entity_id,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "event_time": event_time,
                    "behavior": behavior_name,
                    "relative_time": time_axis,
                    "value": aligned_values,
                }
            )
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
