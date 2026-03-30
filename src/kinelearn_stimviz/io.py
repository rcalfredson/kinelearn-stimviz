from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


DEFAULT_EVENT_COLUMNS = {
    "entity": "entity_id",
    "time": "event_time",
    "type": "event_type",
}

DEFAULT_BEHAVIOR_COLUMNS = {
    "entity": "entity_id",
    "time": "time",
    "behavior": "behavior",
    "value": "value",
}


def read_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet table based on file extension."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_stimulus_events(
    path: str | Path,
    *,
    entity_col: str = "entity_id",
    time_col: str = "event_time",
    event_type_col: str = "event_type",
    event_id_col: str = "event_id",
) -> pd.DataFrame:
    """Load and validate a stimulus-event table."""
    events = read_table(path).copy()
    required = [entity_col, time_col]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(f"Stimulus event table is missing columns: {missing}")

    if event_type_col not in events.columns:
        events[event_type_col] = "stimulus"
    if event_id_col not in events.columns:
        events[event_id_col] = np.arange(len(events), dtype=int)

    events = events.rename(
        columns={
            entity_col: "entity_id",
            time_col: "event_time",
            event_type_col: "event_type",
            event_id_col: "event_id",
        }
    )
    events["event_time"] = pd.to_numeric(events["event_time"], errors="raise")
    return events.sort_values(["entity_id", "event_time", "event_id"]).reset_index(drop=True)


def load_metadata(
    path: str | Path | None,
    *,
    entity_col: str = "entity_id",
) -> pd.DataFrame | None:
    if path is None:
        return None

    metadata = read_table(path).copy()
    if entity_col not in metadata.columns:
        raise ValueError(f"Metadata table is missing the entity column: {entity_col}")
    return metadata.rename(columns={entity_col: "entity_id"})


def _normalize_wide_behavior_table(
    df: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    frame_col: str | None = None,
    behavior_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    reserved = {entity_col, time_col}
    if frame_col and frame_col in df.columns:
        reserved.add(frame_col)
    if behavior_cols is None:
        behavior_cols = [col for col in df.columns if col not in reserved]
    behavior_cols = list(behavior_cols)
    if not behavior_cols:
        raise ValueError("No behavior columns were provided or inferred for wide-format input.")

    id_vars = [entity_col, time_col]
    if frame_col and frame_col in df.columns:
        id_vars.append(frame_col)
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=behavior_cols,
        var_name="behavior",
        value_name="value",
    )
    rename_map = {entity_col: "entity_id", time_col: "time"}
    if frame_col and frame_col in df.columns:
        rename_map[frame_col] = "frame_index"
    return long_df.rename(columns=rename_map)


def _normalize_long_behavior_table(
    df: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    behavior_col: str,
    value_col: str,
    frame_col: str | None = None,
) -> pd.DataFrame:
    required = [entity_col, time_col, behavior_col, value_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Behavior table is missing columns: {missing}")
    normalized = df.rename(
        columns={
            entity_col: "entity_id",
            time_col: "time",
            behavior_col: "behavior",
            value_col: "value",
        }
    )
    keep_cols = ["entity_id", "time", "behavior", "value"]
    if frame_col and frame_col in normalized.columns:
        normalized = normalized.rename(columns={frame_col: "frame_index"})
        keep_cols.insert(2, "frame_index")
    return normalized[keep_cols]


def adapt_kinelearn_predictions(
    df: pd.DataFrame,
    *,
    entity_col: str = "__stem__",
    time_col: str | None = None,
    frame_col: str | None = "__frame__",
    fps: float | None = None,
    behavior_col: str = "behavior",
    probability_col: str | None = "predicted_probability",
    label_col: str | None = None,
    probability_prefix: str = "prob_",
    label_prefix: str = "pred_",
) -> pd.DataFrame:
    """
    Adapt KineLearn-style frame-level predictions into the package's long schema.

    Supports either:
    - long format with one row per frame and behavior
    - wide KineLearn output with one row per frame and columns like `prob_<behavior>`
    """
    if entity_col not in df.columns:
        raise ValueError(f"KineLearn prediction table is missing entity column: {entity_col}")

    frame_values = (
        pd.to_numeric(df[frame_col], errors="raise")
        if frame_col and frame_col in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    if time_col and time_col in df.columns:
        time_values = pd.to_numeric(df[time_col], errors="raise")
    elif frame_col and frame_col in df.columns and fps:
        time_values = frame_values / fps
    else:
        raise ValueError("Provide either a timestamp column or a frame index column together with fps.")

    if behavior_col in df.columns:
        if probability_col and probability_col in df.columns:
            values = pd.to_numeric(df[probability_col], errors="raise")
        elif label_col and label_col in df.columns:
            values = pd.to_numeric(df[label_col], errors="raise")
        else:
            raise ValueError("Provide either a probability column or a label column present in the table.")

        return pd.DataFrame(
            {
                "entity_id": df[entity_col].astype(str),
                "frame_index": frame_values,
                "time": time_values,
                "behavior": df[behavior_col].astype(str),
                "value": values,
            }
        )

    value_cols = [col for col in df.columns if col.startswith(probability_prefix)]
    value_prefix = probability_prefix
    if not value_cols:
        value_cols = [col for col in df.columns if col.startswith(label_prefix)]
        value_prefix = label_prefix
    if not value_cols:
        raise ValueError(
            "KineLearn prediction table must either include a behavior column or "
            f"wide columns starting with {probability_prefix!r} or {label_prefix!r}."
        )

    id_vars = [entity_col]
    if frame_col and frame_col in df.columns:
        id_vars.append(frame_col)
    if time_col and time_col in df.columns:
        id_vars.append(time_col)

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="behavior",
        value_name="value",
    )
    long_df["behavior"] = long_df["behavior"].str.replace(
        f"^{value_prefix}",
        "",
        regex=True,
    )
    long_df = long_df.rename(columns={entity_col: "entity_id"})
    if frame_col and frame_col in long_df.columns:
        long_df = long_df.rename(columns={frame_col: "frame_index"})
    else:
        long_df["frame_index"] = np.nan
    if time_col and time_col in long_df.columns:
        long_df = long_df.rename(columns={time_col: "time"})
    else:
        long_df["time"] = pd.to_numeric(long_df["frame_index"], errors="raise") / fps

    keep_cols = ["entity_id", "frame_index", "time", "behavior", "value"]
    return long_df[keep_cols]


def load_behavior_table(
    path: str | Path,
    *,
    input_format: str = "long",
    entity_col: str = "entity_id",
    time_col: str = "time",
    behavior_col: str = "behavior",
    value_col: str = "value",
    behavior_cols: Iterable[str] | None = None,
    frame_col: str | None = "frame_index",
    fps: float | None = None,
    probability_col: str | None = "predicted_probability",
    label_col: str | None = None,
    probability_prefix: str = "prob_",
    label_prefix: str = "pred_",
) -> pd.DataFrame:
    """Load a behavior time-series table and normalize it to long format."""
    df = read_table(path).copy()

    if input_format == "long":
        normalized = _normalize_long_behavior_table(
            df,
            entity_col=entity_col,
            time_col=time_col,
            behavior_col=behavior_col,
            value_col=value_col,
            frame_col=frame_col,
        )
    elif input_format == "wide":
        normalized = _normalize_wide_behavior_table(
            df,
            entity_col=entity_col,
            time_col=time_col,
            frame_col=frame_col,
            behavior_cols=behavior_cols,
        )
    elif input_format == "kinelearn_predictions":
        normalized = adapt_kinelearn_predictions(
            df,
            entity_col=entity_col,
            time_col=time_col if time_col in df.columns else None,
            frame_col=frame_col,
            fps=fps,
            behavior_col=behavior_col,
            probability_col=probability_col,
            label_col=label_col,
            probability_prefix=probability_prefix,
            label_prefix=label_prefix,
        )
    else:
        raise ValueError(f"Unsupported behavior input format: {input_format}")

    normalized["entity_id"] = normalized["entity_id"].astype(str)
    normalized["behavior"] = normalized["behavior"].astype(str)
    normalized["time"] = pd.to_numeric(normalized["time"], errors="raise")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="raise")
    if "frame_index" in normalized.columns:
        normalized["frame_index"] = pd.to_numeric(normalized["frame_index"], errors="coerce")
        return normalized.sort_values(["entity_id", "behavior", "frame_index", "time"]).reset_index(drop=True)
    return normalized.sort_values(["entity_id", "behavior", "time"]).reset_index(drop=True)
