from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from .io import read_table, read_yaml
from .legacy_convert import (
    DEFAULT_TS_RE,
    find_chunk_file,
    load_legacy_manifest,
    load_video_start_time,
    parse_legacy_led_file,
    write_output_table,
)


def load_entity_map(path: str | Path | None) -> dict[str, str]:
    """Load an optional entity-id remapping table."""
    if path is None:
        return {}
    mapping_df = read_table(path)
    required = {"source_entity_id", "target_entity_id"}
    missing = required.difference(mapping_df.columns)
    if missing:
        raise ValueError(f"Entity map is missing columns: {sorted(missing)}")
    return dict(
        zip(
            mapping_df["source_entity_id"].astype(str),
            mapping_df["target_entity_id"].astype(str),
        )
    )


def _normalize_manifest_records(payload: object) -> pd.DataFrame:
    if isinstance(payload, list):
        records = payload
    else:
        raise ValueError("YAML event manifest must be a list of records.")
    manifest_df = pd.DataFrame(records)
    if manifest_df.empty:
        raise ValueError("Event manifest is empty.")
    return manifest_df


def load_event_manifest(
    path: str | Path,
    *,
    entity_col: str = "entity_id",
    led_col: str = "led_path",
    chunk_col: str = "chunk_path",
    video_start_col: str = "video_start_time",
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """Load a generic event-source manifest from CSV, Parquet, or YAML."""
    path = Path(path)
    if path.suffix.lower() in {".yml", ".yaml"}:
        manifest_df = _normalize_manifest_records(read_yaml(path))
    else:
        manifest_df = read_table(path)

    if entity_col not in manifest_df.columns:
        raise ValueError(f"Event manifest is missing entity column: {entity_col}")
    if led_col not in manifest_df.columns:
        raise ValueError(f"Event manifest is missing LED path column: {led_col}")
    if chunk_col not in manifest_df.columns and video_start_col not in manifest_df.columns:
        raise ValueError(
            f"Event manifest must include either {chunk_col!r} or {video_start_col!r}."
        )

    rename_map = {
        entity_col: "entity_id",
        led_col: "led_path",
    }
    if chunk_col in manifest_df.columns:
        rename_map[chunk_col] = "chunk_path"
    if video_start_col in manifest_df.columns:
        rename_map[video_start_col] = "video_start_time"
    if event_type_col in manifest_df.columns:
        rename_map[event_type_col] = "event_type"
    manifest_df = manifest_df.rename(columns=rename_map).copy()
    if "event_type" not in manifest_df.columns:
        manifest_df["event_type"] = "stimulus"
    return manifest_df


def build_events_from_manifest(
    manifest_df: pd.DataFrame,
    *,
    default_event_type: str = "stimulus",
    entity_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build normalized stimulus events from a generic manifest table."""
    if entity_map is None:
        entity_map = {}

    event_tables: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    next_event_id = 0

    for row in manifest_df.itertuples(index=False):
        raw_entity_id = str(row.entity_id)
        entity_id = entity_map.get(raw_entity_id, raw_entity_id)
        led_path = Path(row.led_path)
        event_df, stim_info = parse_legacy_led_file(led_path)

        if hasattr(row, "chunk_path") and pd.notna(row.chunk_path):
            video_start_time = load_video_start_time(Path(row.chunk_path))
            chunk_path = str(Path(row.chunk_path))
        elif hasattr(row, "video_start_time") and pd.notna(row.video_start_time):
            video_start_time = pd.Timestamp(row.video_start_time).to_pydatetime()
            chunk_path = None
        else:
            raise ValueError(f"Row for {raw_entity_id} lacks both chunk_path and video_start_time.")

        event_type = getattr(row, "event_type", default_event_type) or default_event_type
        event_df["event_time"] = (event_df["event_datetime"] - video_start_time).dt.total_seconds()
        event_df["entity_id"] = entity_id
        event_df["event_type"] = event_type
        event_df["event_id"] = range(next_event_id, next_event_id + len(event_df))
        next_event_id += len(event_df)
        event_tables.append(
            event_df[["entity_id", "event_id", "event_time", "event_type", "event_datetime"]]
        )
        source_rows.append(
            {
                "entity_id": entity_id,
                "source_entity_id": raw_entity_id,
                "led_path": str(led_path),
                "chunk_path": chunk_path,
                "video_start_time": video_start_time.isoformat(),
                "event_type": event_type,
                "on_time_ms": stim_info["on_time_ms"],
                "off_time_ms": stim_info["off_time_ms"],
                "period_s": stim_info["period_s"],
                "n_events": len(event_df),
            }
        )

    return (
        pd.concat(event_tables, ignore_index=True),
        pd.DataFrame(source_rows),
    )


def build_events_from_legacy_pairs(
    manifest_path: str | Path,
    *,
    default_event_type: str = "stimulus",
    timestamp_regex: str = r"(\d{8}_\d{6})",
    chunk_pattern: str = "chunk_data_log_{ts}.csv",
    entity_id_mode: str = "prediction_stem",
    entity_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build normalized stimulus events directly from old JSON pair manifests."""
    if entity_map is None:
        entity_map = {}
    timestamp_re = DEFAULT_TS_RE if timestamp_regex == DEFAULT_TS_RE.pattern else re.compile(timestamp_regex)

    event_tables: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    next_event_id = 0

    for prediction_path, led_path in load_legacy_manifest(manifest_path):
        if entity_id_mode != "prediction_stem":
            raise ValueError(f"Unsupported entity id mode: {entity_id_mode}")
        raw_entity_id = prediction_path.stem
        entity_id = entity_map.get(raw_entity_id, raw_entity_id)
        event_df, stim_info = parse_legacy_led_file(led_path)
        chunk_path = find_chunk_file(
            prediction_path,
            led_path=led_path,
            timestamp_re=timestamp_re,
            chunk_pattern=chunk_pattern,
        )
        video_start_time = load_video_start_time(chunk_path)
        event_df["event_time"] = (event_df["event_datetime"] - video_start_time).dt.total_seconds()
        event_df["entity_id"] = entity_id
        event_df["event_type"] = default_event_type
        event_df["event_id"] = range(next_event_id, next_event_id + len(event_df))
        next_event_id += len(event_df)
        event_tables.append(
            event_df[["entity_id", "event_id", "event_time", "event_type", "event_datetime"]]
        )
        source_rows.append(
            {
                "entity_id": entity_id,
                "source_entity_id": raw_entity_id,
                "prediction_path": str(prediction_path),
                "led_path": str(led_path),
                "chunk_path": str(chunk_path),
                "video_start_time": video_start_time.isoformat(),
                "event_type": default_event_type,
                "on_time_ms": stim_info["on_time_ms"],
                "off_time_ms": stim_info["off_time_ms"],
                "period_s": stim_info["period_s"],
                "n_events": len(event_df),
            }
        )

    return (
        pd.concat(event_tables, ignore_index=True),
        pd.DataFrame(source_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinelearn-stimviz-build-events",
        description="Build stimulus event tables from LED logs and manifest files.",
    )
    parser.add_argument("manifest", help="Event-source manifest. Supports a generic table/YAML or legacy JSON pair list.")
    parser.add_argument("--output-dir", required=True, help="Directory where normalized event tables will be written.")
    parser.add_argument(
        "--source-format",
        default="table",
        choices=["table", "legacy-pairs-json"],
        help="Interpret the manifest as a generic table/YAML or the old JSON pair format.",
    )
    parser.add_argument("--event-type", default="stimulus", help="Default event type label to assign.")
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "parquet", "both"],
        help="Whether to write normalized tables as CSV, Parquet, or both.",
    )
    parser.add_argument("--entity-map", help="Optional CSV/Parquet table with source_entity_id and target_entity_id.")

    parser.add_argument("--entity-col", default="entity_id", help="Entity column for generic manifest tables.")
    parser.add_argument("--led-col", default="led_path", help="LED-log path column for generic manifest tables.")
    parser.add_argument("--chunk-col", default="chunk_path", help="Chunk-log path column for generic manifest tables.")
    parser.add_argument("--video-start-col", default="video_start_time", help="Video-start timestamp column for generic manifest tables.")
    parser.add_argument("--event-type-col", default="event_type", help="Optional event-type column for generic manifest tables.")

    parser.add_argument(
        "--chunk-pattern",
        default="chunk_data_log_{ts}.csv",
        help="Legacy chunk log filename pattern for legacy-pairs-json mode.",
    )
    parser.add_argument(
        "--timestamp-regex",
        default=r"(\d{8}_\d{6})",
        help="Regex used to extract the timestamp token from legacy prediction filenames.",
    )
    parser.add_argument(
        "--entity-id-mode",
        default="prediction_stem",
        choices=["prediction_stem"],
        help="How to derive entity ids in legacy-pairs-json mode before optional remapping.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_map = load_entity_map(args.entity_map)

    if args.source_format == "legacy-pairs-json":
        events_table, sources_table = build_events_from_legacy_pairs(
            args.manifest,
            default_event_type=args.event_type,
            timestamp_regex=args.timestamp_regex,
            chunk_pattern=args.chunk_pattern,
            entity_id_mode=args.entity_id_mode,
            entity_map=entity_map,
        )
    else:
        manifest_df = load_event_manifest(
            args.manifest,
            entity_col=args.entity_col,
            led_col=args.led_col,
            chunk_col=args.chunk_col,
            video_start_col=args.video_start_col,
            event_type_col=args.event_type_col,
        )
        events_table, sources_table = build_events_from_manifest(
            manifest_df,
            default_event_type=args.event_type,
            entity_map=entity_map,
        )

    outputs = {
        "events": write_output_table(events_table, output_dir / "stimulus_events", args.output_format),
        "sources": write_output_table(sources_table, output_dir / "event_sources", args.output_format),
    }
    for path in outputs["events"]:
        print(f"Wrote stimulus event table: {path}")
    for path in outputs["sources"]:
        print(f"Wrote event source table: {path}")


if __name__ == "__main__":
    main()
