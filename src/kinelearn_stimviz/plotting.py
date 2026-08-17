from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    sns = None


_REFERENCE_FONT_SIZE = 10.0
_EXTRA_Y_HEADROOM_PER_SCALE = 0.2


def _tight_layout(fig, *, has_title: bool) -> None:
    if has_title:
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()


def _resize_figure_for_font(
    fig,
    axes,
    font_size: float | None,
    *,
    has_title: bool,
) -> None:
    """Grow only the non-plot area needed by text at the requested size."""
    if font_size is None:
        _tight_layout(fig, has_title=has_title)
        return

    scale = font_size / _REFERENCE_FONT_SIZE
    flat_axes = list(axes.flat)
    text_objects = fig.findobj(match=Text)
    requested_sizes = [text.get_fontsize() for text in text_objects]

    # Measure the data rectangles with the same content at the 10-point
    # reference size, then restore the requested sizes and add only enough
    # canvas to recover those rectangle dimensions.
    for text, requested_size in zip(text_objects, requested_sizes, strict=True):
        text.set_fontsize(requested_size / scale)
    _tight_layout(fig, has_title=has_title)
    width, height = fig.get_size_inches()
    reference_axes_width = max(ax.get_position().width for ax in flat_axes) * width
    reference_axes_height = sum(ax.get_position().height for ax in flat_axes) * height

    for text, requested_size in zip(text_objects, requested_sizes, strict=True):
        text.set_fontsize(requested_size)
    _tight_layout(fig, has_title=has_title)
    requested_axes_width = max(ax.get_position().width for ax in flat_axes) * width
    requested_axes_height = sum(ax.get_position().height for ax in flat_axes) * height

    resized_width = width + reference_axes_width - requested_axes_width
    resized_height = height + reference_axes_height - requested_axes_height
    fig.set_size_inches(resized_width, resized_height, forward=True)
    _tight_layout(fig, has_title=has_title)


def _add_font_scaled_y_headroom(ax, font_size: float | None) -> None:
    """Add empty data space above an axes for larger-font plot elements."""
    if font_size is None or font_size <= _REFERENCE_FONT_SIZE:
        return

    lower, upper = ax.get_ylim()
    scale = font_size / _REFERENCE_FONT_SIZE
    extra = (upper - lower) * _EXTRA_Y_HEADROOM_PER_SCALE * (scale - 1.0)
    ax.set_ylim(lower, upper + extra)


def _anchor_y_ticks_at_zero(ax) -> None:
    """Start nonnegative data at zero and always label zero when in-range."""
    data_lower = ax.dataLim.ymin
    data_upper = ax.dataLim.ymax
    data_span = data_upper - data_lower
    zero_tolerance = max(abs(data_span), 1.0) * 1e-12
    if np.isfinite(data_lower) and data_lower >= -zero_tolerance:
        ax.set_ylim(bottom=0)

    lower, upper = ax.get_ylim()
    if not (lower <= 0 <= upper):
        return

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], min_n_ticks=2)
    )


def plot_psth(
    group_summary: pd.DataFrame,
    *,
    output_path: str | Path,
    behavior_order: list[str] | None = None,
    group_col: str | None = None,
    group_order: list[str] | None = None,
    behavior_labels: dict[str, str] | None = None,
    group_labels: dict[str, str] | None = None,
    group_colors: dict[str, str] | None = None,
    show_gridlines: bool = False,
    show_subplot_titles: bool = True,
    legend_title: str | None = None,
    show_legend_title: bool = True,
    show_group_ns: bool = False,
    title: str | None = None,
    ylabel: str = "Value",
    xlabel: str = "Time (s)",
    font_size: float | None = None,
    figsize: tuple[float, float] | None = None,
    event_band: tuple[float, float] = (0.0, 0.05),
    annotation: str | None = None,
    annotation_x: float = 0.98,
    annotation_y: float = 0.95,
    annotation_box_alpha: float = 0.65,
    annotation_box_pad: float = 0.25,
) -> Path:
    """Plot one PSTH-style panel per behavior with confidence intervals.

    ``font_size`` anchors text that uses the middle font role. Titles are
    scaled to Matplotlib's ``large`` ratio and secondary text to its ``small``
    ratio. The canvas grows only by the additional room needed for text beyond
    a 10-point reference, leaving the data rectangles approximately unchanged.
    """
    if font_size is not None and (not math.isfinite(font_size) or font_size <= 0):
        raise ValueError("font_size must be a positive, finite number.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="white")
        base_palette = sns.color_palette("deep")
    else:
        plt.style.use("ggplot")
        base_palette = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    medium_font = {"fontsize": font_size} if font_size is not None else {}
    small_font_size = font_size * (5 / 6) if font_size is not None else None
    large_font = {"fontsize": font_size * 1.2} if font_size is not None else {}

    if behavior_order is None:
        behavior_order = list(group_summary["behavior"].drop_duplicates())
    if behavior_labels is None:
        behavior_labels = {}
    if group_labels is None:
        group_labels = {}
    if group_colors is None:
        group_colors = {}

    n_behaviors = len(behavior_order)
    if figsize is None:
        figsize = (8, max(3.0, 3.2 * n_behaviors))
    fig, axes = plt.subplots(
        n_behaviors,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    if group_col and group_col in group_summary.columns:
        group_values = list(group_summary[group_col].dropna().drop_duplicates())
        if group_order:
            ordered_values = [value for value in group_order if value in group_values]
            remaining_values = [value for value in group_values if value not in ordered_values]
            group_values = ordered_values + remaining_values
    else:
        group_values = [None]
    if legend_title is None:
        legend_title = group_col

    for idx, behavior in enumerate(behavior_order):
        ax = axes[idx, 0]
        behavior_df = group_summary[group_summary["behavior"] == behavior]
        ax.axvspan(event_band[0], event_band[1], color="#838383", alpha=1.0, lw=0)

        for group_idx, group_value in enumerate(group_values):
            color = base_palette[group_idx % len(base_palette)]
            if group_value is None:
                plot_df = behavior_df
                label = "all"
            else:
                color = group_colors.get(str(group_value), color)
                plot_df = behavior_df[behavior_df[group_col] == group_value]
                label = group_labels.get(str(group_value), str(group_value))
            if plot_df.empty:
                continue
            if show_group_ns and "n" in plot_df.columns:
                n_subjects = int(plot_df["n"].max())
                label = f"{label} (n = {n_subjects})"

            ax.plot(plot_df["relative_time"], plot_df["mean"], label=label, color=color, lw=2)
            ax.fill_between(
                plot_df["relative_time"],
                plot_df["ci_low"],
                plot_df["ci_high"],
                color=color,
                alpha=0.2,
            )

        if show_subplot_titles:
            ax.set_title(behavior_labels.get(str(behavior), str(behavior)), **large_font)
        ax.set_ylabel(ylabel, **medium_font)
        if font_size is not None:
            ax.tick_params(axis="both", labelsize=small_font_size)
        ax.grid(show_gridlines)
        ax.axvline(0.0, color="black", lw=1, alpha=0.2)
        if annotation:
            ax.text(
                annotation_x,
                annotation_y,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=small_font_size if font_size is not None else None,
                bbox={
                    "boxstyle": f"round,pad={annotation_box_pad}",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": annotation_box_alpha,
                },
            )
        if idx == 0 and len(group_values) > 1:
            legend_kwargs = {}
            if font_size is not None:
                legend_kwargs = {
                    "fontsize": small_font_size,
                    "title_fontsize": font_size,
                }
            ax.legend(
                title=legend_title if show_legend_title else None,
                loc="upper left",
                **legend_kwargs,
            )
        _add_font_scaled_y_headroom(ax, font_size)
        _anchor_y_ticks_at_zero(ax)

    axes[-1, 0].set_xlabel(xlabel, **medium_font)
    if title:
        fig.suptitle(title, **large_font)
    _resize_figure_for_font(fig, axes, font_size, has_title=bool(title))

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
