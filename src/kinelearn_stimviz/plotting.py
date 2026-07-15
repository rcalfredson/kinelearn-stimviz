from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    sns = None


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
    figsize: tuple[float, float] | None = None,
    event_band: tuple[float, float] = (0.0, 0.05),
    annotation: str | None = None,
    annotation_x: float = 0.98,
    annotation_y: float = 0.95,
    annotation_box_alpha: float = 0.65,
    annotation_box_pad: float = 0.25,
) -> Path:
    """Plot one PSTH-style panel per behavior with confidence intervals."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="white")
        base_palette = sns.color_palette("deep")
    else:
        plt.style.use("ggplot")
        base_palette = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

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
            ax.set_title(behavior_labels.get(str(behavior), str(behavior)))
        ax.set_ylabel(ylabel)
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
                bbox={
                    "boxstyle": f"round,pad={annotation_box_pad}",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": annotation_box_alpha,
                },
            )
        if idx == 0 and len(group_values) > 1:
            ax.legend(title=legend_title if show_legend_title else None)

    axes[-1, 0].set_xlabel(xlabel)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
