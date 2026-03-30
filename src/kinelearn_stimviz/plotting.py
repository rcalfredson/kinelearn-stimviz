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
    title: str | None = None,
    ylabel: str = "Value",
    event_band: tuple[float, float] = (0.0, 0.05),
) -> Path:
    """Plot one PSTH-style panel per behavior with confidence intervals."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="whitegrid")
        base_palette = sns.color_palette("deep")
    else:
        plt.style.use("ggplot")
        base_palette = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    if behavior_order is None:
        behavior_order = list(group_summary["behavior"].drop_duplicates())

    n_behaviors = len(behavior_order)
    fig, axes = plt.subplots(
        n_behaviors,
        1,
        figsize=(8, max(3.0, 3.2 * n_behaviors)),
        sharex=True,
        squeeze=False,
    )

    if group_col and group_col in group_summary.columns:
        group_values = list(group_summary[group_col].dropna().drop_duplicates())
        palette = base_palette[: max(len(group_values), 1)]
    else:
        group_values = [None]
        palette = [base_palette[0]]

    for idx, behavior in enumerate(behavior_order):
        ax = axes[idx, 0]
        behavior_df = group_summary[group_summary["behavior"] == behavior]
        ax.axvspan(event_band[0], event_band[1], color="#d9d9d9", alpha=0.35, lw=0)

        for color, group_value in zip(palette, group_values):
            if group_value is None:
                plot_df = behavior_df
                label = "all"
            else:
                plot_df = behavior_df[behavior_df[group_col] == group_value]
                label = str(group_value)
            if plot_df.empty:
                continue

            ax.plot(plot_df["relative_time"], plot_df["mean"], label=label, color=color, lw=2)
            ax.fill_between(
                plot_df["relative_time"],
                plot_df["ci_low"],
                plot_df["ci_high"],
                color=color,
                alpha=0.2,
            )

        ax.set_title(behavior)
        ax.set_ylabel(ylabel)
        ax.axvline(0.0, color="black", lw=1, alpha=0.7)
        if idx == 0 and len(group_values) > 1:
            ax.legend(title=group_col)

    axes[-1, 0].set_xlabel("Time relative to event (s)")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
