from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .legacy_convert import SUPPORTED_OUTPUT_FORMATS, write_output_table


TABLE_STEMS = ("behavior_long", "stimulus_events", "metadata")
INPUT_FORMATS = {"auto", "csv", "parquet"}


def find_table_path(input_dir: str | Path, stem: str, input_format: str = "auto") -> Path:
    """Find one converted table in an input directory."""
    input_dir = Path(input_dir)
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"Unsupported input format: {input_format}")

    suffixes = [".parquet", ".csv"] if input_format == "auto" else [f".{input_format}"]
    matches = [input_dir / f"{stem}{suffix}" for suffix in suffixes if (input_dir / f"{stem}{suffix}").exists()]
    if not matches:
        expected = ", ".join(f"{stem}{suffix}" for suffix in suffixes)
        raise FileNotFoundError(f"Could not find {expected} in {input_dir}")
    return matches[0]


def read_converted_table(input_dir: str | Path, stem: str, input_format: str = "auto") -> pd.DataFrame:
    """Read a converted table from CSV or Parquet."""
    path = find_table_path(input_dir, stem, input_format=input_format)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def validate_converted_tables(tables: dict[str, pd.DataFrame], *, input_dir: str | Path) -> None:
    required_columns = {
        "behavior_long": {"entity_id", "frame_index", "time", "behavior", "value"},
        "stimulus_events": {"entity_id", "event_time"},
        "metadata": {"entity_id"},
    }
    for stem, required in required_columns.items():
        missing = required.difference(tables[stem].columns)
        if missing:
            raise ValueError(f"{input_dir}/{stem} is missing columns: {sorted(missing)}")

    behavior_entities = set(tables["behavior_long"]["entity_id"].astype(str))
    event_entities = set(tables["stimulus_events"]["entity_id"].astype(str))
    metadata_entities = set(tables["metadata"]["entity_id"].astype(str))
    missing_behavior = event_entities.difference(behavior_entities)
    missing_metadata = event_entities.difference(metadata_entities)
    if missing_behavior:
        raise ValueError(
            f"{input_dir} has event entities missing from behavior_long: {sorted(missing_behavior)[:5]}"
        )
    if missing_metadata:
        raise ValueError(
            f"{input_dir} has event entities missing from metadata: {sorted(missing_metadata)[:5]}"
        )


def merge_converted_dirs(
    input_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    input_format: str = "auto",
    output_format: str = "parquet",
    regenerate_event_ids: bool = True,
) -> dict[str, list[Path]]:
    """Merge converted kinelearn-stimviz table directories into one dataset."""
    if len(input_dirs) < 2:
        raise ValueError("Provide at least two converted input directories to merge.")
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")

    grouped_tables: dict[str, list[pd.DataFrame]] = {stem: [] for stem in TABLE_STEMS}
    for input_dir in input_dirs:
        tables = {
            stem: read_converted_table(input_dir, stem, input_format=input_format)
            for stem in TABLE_STEMS
        }
        validate_converted_tables(tables, input_dir=input_dir)
        for stem, table in tables.items():
            grouped_tables[stem].append(table)

    merged = {
        stem: pd.concat(tables, ignore_index=True)
        for stem, tables in grouped_tables.items()
    }
    if regenerate_event_ids:
        events = merged["stimulus_events"].sort_values(["entity_id", "event_time"]).reset_index(drop=True)
        events["event_id"] = range(len(events))
        merged["stimulus_events"] = events

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        stem: write_output_table(table, output_dir / stem, output_format)
        for stem, table in merged.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinelearn-stimviz-merge",
        description="Merge converted kinelearn-stimviz table directories into one plot-ready dataset.",
    )
    parser.add_argument(
        "--input",
        dest="input_dirs",
        action="append",
        required=True,
        help="Converted input directory containing behavior_long, stimulus_events, and metadata tables. Repeat for each cohort.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where merged tables will be written.")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=sorted(INPUT_FORMATS),
        help="Input table format. Auto prefers Parquet when both CSV and Parquet are present.",
    )
    parser.add_argument(
        "--output-format",
        default="parquet",
        choices=sorted(SUPPORTED_OUTPUT_FORMATS),
        help="Whether to write merged tables as CSV, Parquet, or both.",
    )
    parser.add_argument(
        "--keep-event-ids",
        action="store_true",
        help="Preserve input event_id values instead of regenerating a unique merged sequence.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = merge_converted_dirs(
        args.input_dirs,
        output_dir=args.output_dir,
        input_format=args.input_format,
        output_format=args.output_format,
        regenerate_event_ids=not args.keep_event_ids,
    )
    for stem, paths in outputs.items():
        for path in paths:
            print(f"Wrote merged {stem} table: {path}")


if __name__ == "__main__":
    main()
