from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import summarize_by_event, summarize_by_group, summarize_by_subject
from .align import build_aligned_windows, select_events
from .io import load_behavior_table, load_metadata, load_stimulus_events
from .plotting import plot_psth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinelearn-stimviz",
        description="Generate PSTH-style plots from generic stimulus-event and behavior tables.",
    )
    parser.add_argument("--events", required=True, help="Stimulus event table in CSV or Parquet format.")
    parser.add_argument("--behavior", required=True, help="Behavior time-series table in CSV or Parquet format.")
    parser.add_argument("--metadata", help="Optional metadata/group table in CSV or Parquet format.")
    parser.add_argument("--output", default="plots/psth.png", help="Output plot path.")

    parser.add_argument("--behavior-format", default="long", choices=["long", "wide", "kinelearn_predictions"])
    parser.add_argument("--entity-col", default="entity_id", help="Entity identifier column shared across tables.")
    parser.add_argument("--time-col", default="time", help="Behavior timestamp column.")
    parser.add_argument("--behavior-col", default="behavior", help="Behavior-name column for long-format input.")
    parser.add_argument("--value-col", default="value", help="Behavior value column for long-format input.")
    parser.add_argument("--behavior-cols", nargs="*", help="Explicit behavior columns for wide-format input.")

    parser.add_argument("--event-time-col", default="event_time", help="Stimulus event timestamp column.")
    parser.add_argument("--event-type-col", default="event_type", help="Stimulus event type column.")
    parser.add_argument("--event-id-col", default="event_id", help="Optional unique event id column.")
    parser.add_argument(
        "--event-subset",
        default="all",
        choices=["all", "first", "last"],
        help="Select all events or the first/last N events within each entity and event type.",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        help="Number of events to keep when using --event-subset first or last.",
    )

    parser.add_argument("--frame-col", default="frame_index", help="Frame index column when frame-based alignment or prediction adaptation is used.")
    parser.add_argument("--fps", type=float, help="Frames per second for prediction adaptation or frame-based alignment.")
    parser.add_argument("--probability-col", default="predicted_probability", help="Prediction score column for KineLearn tables.")
    parser.add_argument("--label-col", help="Fallback label column if probability is unavailable.")

    parser.add_argument("--pre", type=float, default=1.0, help="Seconds before each event.")
    parser.add_argument("--post", type=float, default=2.0, help="Seconds after each event.")
    parser.add_argument("--bin-size", type=float, help="Aligned window bin size in seconds.")
    parser.add_argument(
        "--alignment-mode",
        default="nearest",
        choices=["nearest", "frame"],
        help="Use nearest-time matching or direct frame-offset slicing for alignment.",
    )
    parser.add_argument("--behaviors", nargs="*", help="Subset of behaviors to include.")
    parser.add_argument("--group-col", default="group", help="Metadata column used for cohort comparisons.")
    parser.add_argument("--title", help="Optional plot title.")
    parser.add_argument("--ylabel", default="Proportion / score", help="Y-axis label.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    events = load_stimulus_events(
        args.events,
        entity_col=args.entity_col,
        time_col=args.event_time_col,
        event_type_col=args.event_type_col,
        event_id_col=args.event_id_col,
    )
    behavior = load_behavior_table(
        args.behavior,
        input_format=args.behavior_format,
        entity_col=args.entity_col,
        time_col=args.time_col,
        behavior_col=args.behavior_col,
        value_col=args.value_col,
        behavior_cols=args.behavior_cols,
        frame_col=args.frame_col,
        fps=args.fps,
        probability_col=args.probability_col,
        label_col=args.label_col,
    )
    metadata = load_metadata(args.metadata, entity_col=args.entity_col)
    events = select_events(
        events,
        subset=args.event_subset,
        count=args.event_count,
    )

    aligned = build_aligned_windows(
        behavior,
        events,
        pre=args.pre,
        post=args.post,
        bin_size=args.bin_size,
        metadata_df=metadata,
        behaviors=args.behaviors,
        alignment_mode=args.alignment_mode,
        fps=args.fps,
        frame_col=args.frame_col,
    )
    event_summary = summarize_by_event(aligned)
    subject_summary = summarize_by_subject(event_summary)

    if metadata is not None and args.group_col in subject_summary.columns:
        group_summary = summarize_by_group(subject_summary, group_cols=[args.group_col])
        group_col = args.group_col
    else:
        group_summary = summarize_by_group(subject_summary, group_cols=[])
        group_col = None

    output = plot_psth(
        group_summary,
        output_path=args.output,
        behavior_order=args.behaviors,
        group_col=group_col,
        title=args.title,
        ylabel=args.ylabel,
    )
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
