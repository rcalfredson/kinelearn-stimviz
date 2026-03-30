from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_TS_RE = re.compile(r"(\d{8}_\d{6})")


@dataclass
class LegacyPairMetadata:
    entity_id: str
    prediction_path: str
    led_path: str
    chunk_path: str
    video_start_time: str
    frame_rate: float
    on_time_ms: int
    off_time_ms: int
    period_s: float
    n_events: int
    n_frames: int


def load_legacy_manifest(path: str | Path) -> list[tuple[Path, Path]]:
    """Load the old JSON list of [prediction_csv, led_csv] pairs."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pairs: list[tuple[Path, Path]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Legacy manifest item {idx} is not a [prediction_path, led_path] pair.")
        pairs.append((Path(item[0]), Path(item[1])))
    return pairs


def parse_legacy_led_file(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Parse the old LED CSV structure used by the legacy scripts."""
    path = Path(path)
    raw = pd.read_csv(path, header=None)
    meta = raw.iloc[0]
    on_ms = int(str(meta[3]).split(":")[-1].strip().replace("ms", ""))
    off_ms = int(str(meta[4]).split(":")[-1].strip().replace("ms", ""))
    events = (
        pd.read_csv(path, skiprows=1)["Event Time"]
        .dropna()
        .map(datetime.fromisoformat)
        .to_list()
    )
    if not events:
        raise ValueError(f"No LED events found in {path}")

    event_df = pd.DataFrame({"event_datetime": events})
    info = {
        "on_time_ms": on_ms,
        "off_time_ms": off_ms,
        "period_s": (on_ms + off_ms) / 1000.0,
    }
    return event_df, info


def find_chunk_file(
    prediction_path: str | Path,
    *,
    led_path: str | Path,
    timestamp_re: re.Pattern[str] = DEFAULT_TS_RE,
    chunk_pattern: str = "chunk_data_log_{ts}.csv",
) -> Path:
    """Locate the legacy chunk log needed to anchor LED time to video time."""
    prediction_path = Path(prediction_path)
    led_path = Path(led_path)
    match = timestamp_re.search(prediction_path.name)
    if not match:
        raise ValueError(f"Could not extract timestamp token from prediction filename: {prediction_path.name}")
    ts_stub = match.group(1)
    chunk_path = led_path.parent / chunk_pattern.format(ts=ts_stub)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Expected chunk log next to LED file: {chunk_path}")
    return chunk_path


def load_video_start_time(chunk_path: str | Path) -> datetime:
    chunk_df = pd.read_csv(chunk_path)
    if "Datetime" not in chunk_df.columns:
        raise ValueError(f"Chunk log is missing Datetime column: {chunk_path}")
    return datetime.fromisoformat(str(chunk_df.loc[0, "Datetime"]))


def infer_entity_id(prediction_path: str | Path, *, mode: str = "prediction_stem") -> str:
    prediction_path = Path(prediction_path)
    if mode == "prediction_stem":
        return prediction_path.stem
    raise ValueError(f"Unsupported entity id mode: {mode}")


def convert_prediction_table(
    prediction_path: str | Path,
    *,
    entity_id: str,
    frame_rate: float,
    behavior_prefix: str = "Pred_",
) -> tuple[pd.DataFrame, list[str], int]:
    """Convert a wide prediction CSV into the package's long behavior table."""
    prediction_df = pd.read_csv(prediction_path).fillna(0)
    behavior_cols = [col for col in prediction_df.columns if col.startswith(behavior_prefix)]
    if not behavior_cols:
        raise ValueError(f"No behavior columns starting with {behavior_prefix!r} found in {prediction_path}")

    prediction_df = prediction_df.reset_index(drop=True)
    prediction_df["frame_index"] = prediction_df.index.astype(int)
    prediction_df["time"] = prediction_df["frame_index"] / frame_rate
    long_df = prediction_df.melt(
        id_vars=["frame_index", "time"],
        value_vars=behavior_cols,
        var_name="behavior",
        value_name="value",
    )
    long_df["behavior"] = long_df["behavior"].str.replace(f"^{re.escape(behavior_prefix)}", "", regex=True)
    long_df["entity_id"] = entity_id
    long_df = long_df[["entity_id", "frame_index", "time", "behavior", "value"]]
    behavior_names = [re.sub(f"^{re.escape(behavior_prefix)}", "", col) for col in behavior_cols]
    return long_df, behavior_names, len(prediction_df)


def convert_event_table(
    led_path: str | Path,
    *,
    entity_id: str,
    prediction_path: str | Path,
    frame_rate: float,
    chunk_pattern: str,
    timestamp_re: re.Pattern[str],
    event_type: str = "stimulus",
) -> tuple[pd.DataFrame, LegacyPairMetadata]:
    event_df, stim_info = parse_legacy_led_file(led_path)
    chunk_path = find_chunk_file(
        prediction_path,
        led_path=led_path,
        timestamp_re=timestamp_re,
        chunk_pattern=chunk_pattern,
    )
    video_start_time = load_video_start_time(chunk_path)
    event_df["event_time"] = (
        event_df["event_datetime"] - video_start_time
    ).dt.total_seconds()
    event_df["entity_id"] = entity_id
    event_df["event_type"] = event_type
    event_df = event_df[["entity_id", "event_time", "event_type", "event_datetime"]]

    metadata = LegacyPairMetadata(
        entity_id=entity_id,
        prediction_path=str(prediction_path),
        led_path=str(led_path),
        chunk_path=str(chunk_path),
        video_start_time=video_start_time.isoformat(),
        frame_rate=frame_rate,
        on_time_ms=stim_info["on_time_ms"],
        off_time_ms=stim_info["off_time_ms"],
        period_s=stim_info["period_s"],
        n_events=len(event_df),
        n_frames=0,
    )
    return event_df, metadata


def convert_legacy_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    frame_rate: float = 60.0,
    behavior_prefix: str = "Pred_",
    chunk_pattern: str = "chunk_data_log_{ts}.csv",
    timestamp_regex: str = r"(\d{8}_\d{6})",
    entity_id_mode: str = "prediction_stem",
    event_type: str = "stimulus",
) -> dict[str, Path]:
    """Convert an old JSON manifest into normalized tables for the new CLI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_re = re.compile(timestamp_regex)
    pairs = load_legacy_manifest(manifest_path)

    behavior_tables: list[pd.DataFrame] = []
    event_tables: list[pd.DataFrame] = []
    metadata_rows: list[LegacyPairMetadata] = []
    next_event_id = 0

    for prediction_path, led_path in pairs:
        entity_id = infer_entity_id(prediction_path, mode=entity_id_mode)
        behavior_df, _, n_frames = convert_prediction_table(
            prediction_path,
            entity_id=entity_id,
            frame_rate=frame_rate,
            behavior_prefix=behavior_prefix,
        )
        event_df, metadata = convert_event_table(
            led_path,
            entity_id=entity_id,
            prediction_path=prediction_path,
            frame_rate=frame_rate,
            chunk_pattern=chunk_pattern,
            timestamp_re=timestamp_re,
            event_type=event_type,
        )
        event_df = event_df.copy()
        event_df["event_id"] = range(next_event_id, next_event_id + len(event_df))
        next_event_id += len(event_df)

        metadata.n_frames = n_frames
        behavior_tables.append(behavior_df)
        event_tables.append(event_df[["entity_id", "event_id", "event_time", "event_type", "event_datetime"]])
        metadata_rows.append(metadata)

    behavior_out = output_dir / "behavior_long.csv"
    events_out = output_dir / "stimulus_events.csv"
    metadata_out = output_dir / "metadata.csv"

    pd.concat(behavior_tables, ignore_index=True).to_csv(behavior_out, index=False)
    pd.concat(event_tables, ignore_index=True).to_csv(events_out, index=False)
    pd.DataFrame([row.__dict__ for row in metadata_rows]).to_csv(metadata_out, index=False)

    return {
        "behavior": behavior_out,
        "events": events_out,
        "metadata": metadata_out,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinelearn-stimviz-convert-legacy",
        description="Convert old JSON pair manifests into tabular inputs for kinelearn-stimviz.",
    )
    parser.add_argument("manifest", help="Path to the legacy JSON manifest of [prediction_csv, led_csv] pairs.")
    parser.add_argument("--output-dir", required=True, help="Directory where normalized output tables will be written.")
    parser.add_argument("--fps", type=float, default=60.0, help="Frame rate used to convert frame index to time.")
    parser.add_argument("--behavior-prefix", default="Pred_", help="Prefix used by behavior columns in legacy prediction CSVs.")
    parser.add_argument(
        "--chunk-pattern",
        default="chunk_data_log_{ts}.csv",
        help="Legacy chunk log filename pattern, where {ts} is extracted from prediction filenames.",
    )
    parser.add_argument(
        "--timestamp-regex",
        default=r"(\d{8}_\d{6})",
        help="Regex used to extract the timestamp token from prediction filenames.",
    )
    parser.add_argument(
        "--entity-id-mode",
        default="prediction_stem",
        choices=["prediction_stem"],
        help="How to derive entity ids for the normalized output tables.",
    )
    parser.add_argument("--event-type", default="stimulus", help="Event type label to assign in the output event table.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    outputs = convert_legacy_manifest(
        args.manifest,
        output_dir=args.output_dir,
        frame_rate=args.fps,
        behavior_prefix=args.behavior_prefix,
        chunk_pattern=args.chunk_pattern,
        timestamp_regex=args.timestamp_regex,
        entity_id_mode=args.entity_id_mode,
        event_type=args.event_type,
    )
    print(f"Wrote behavior table: {outputs['behavior']}")
    print(f"Wrote stimulus event table: {outputs['events']}")
    print(f"Wrote metadata table: {outputs['metadata']}")


if __name__ == "__main__":
    main()
