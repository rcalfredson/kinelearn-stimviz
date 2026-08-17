from __future__ import annotations

import pandas as pd
import pytest

from kinelearn_stimviz.cli import build_parser
from kinelearn_stimviz.plotting import plot_psth


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "behavior": ["walking", "walking", "walking", "walking"],
            "group": ["control", "control", "treated", "treated"],
            "relative_time": [-1.0, 1.0, -1.0, 1.0],
            "mean": [0.1, 0.2, 0.2, 0.3],
            "ci_low": [0.05, 0.15, 0.15, 0.25],
            "ci_high": [0.15, 0.25, 0.25, 0.35],
        }
    )


def test_cli_accepts_font_size() -> None:
    args = build_parser().parse_args(
        ["--events", "events.csv", "--behavior", "behavior.csv", "--font-size", "11.5"]
    )

    assert args.font_size == 11.5


def test_font_size_anchors_medium_text_and_scales_titles(tmp_path, monkeypatch) -> None:
    closed_figures = []
    monkeypatch.setattr("kinelearn_stimviz.plotting.plt.close", closed_figures.append)

    plot_psth(
        _summary(),
        output_path=tmp_path / "plot.png",
        group_col="group",
        font_size=10,
        title="Overall title",
        annotation="Panel note",
    )

    fig = closed_figures[0]
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert ax.title.get_fontsize() == pytest.approx(12)
    assert fig._suptitle.get_fontsize() == pytest.approx(12)
    assert ax.xaxis.label.get_fontsize() == pytest.approx(10)
    assert ax.yaxis.label.get_fontsize() == pytest.approx(10)
    assert all(label.get_fontsize() == pytest.approx(10 * 5 / 6) for label in ax.get_xticklabels())
    assert ax.texts[0].get_fontsize() == pytest.approx(10 * 5 / 6)
    assert all(text.get_fontsize() == pytest.approx(10 * 5 / 6) for text in legend.get_texts())
    assert legend.get_title().get_fontsize() == pytest.approx(10)
    assert legend._loc == 2  # Matplotlib's upper-left location code.


def test_larger_font_expands_margins_without_scaling_plot_area(tmp_path, monkeypatch) -> None:
    closed_figures = []
    monkeypatch.setattr("kinelearn_stimviz.plotting.plt.close", closed_figures.append)

    for font_size in (10, 20):
        plot_psth(
            _summary(),
            output_path=tmp_path / f"plot-{font_size}.png",
            group_col="group",
            font_size=font_size,
            figsize=(5, 4),
            annotation="Stimulus",
        )

    small_fig, large_fig = closed_figures
    small_size = small_fig.get_size_inches()
    large_size = large_fig.get_size_inches()
    small_ax = small_fig.axes[0].get_window_extent().transformed(
        small_fig.dpi_scale_trans.inverted()
    )
    large_ax = large_fig.axes[0].get_window_extent().transformed(
        large_fig.dpi_scale_trans.inverted()
    )

    assert large_size[0] > small_size[0]
    assert large_size[1] > small_size[1]
    assert large_size[0] < small_size[0] * 1.5
    assert large_size[1] < small_size[1] * 1.5
    assert large_ax.width == pytest.approx(small_ax.width, rel=0.03)
    assert large_ax.height == pytest.approx(small_ax.height, rel=0.03)

    small_ylim = small_fig.axes[0].get_ylim()
    large_ylim = large_fig.axes[0].get_ylim()
    assert large_ylim[0] == pytest.approx(small_ylim[0])
    assert large_ylim[1] > small_ylim[1]


def test_nonnegative_y_axis_uses_zero_as_a_labeled_endpoint(tmp_path, monkeypatch) -> None:
    closed_figures = []
    monkeypatch.setattr("kinelearn_stimviz.plotting.plt.close", closed_figures.append)

    plot_psth(_summary(), output_path=tmp_path / "plot.png")

    ax = closed_figures[0].axes[0]
    assert ax.get_ylim()[0] == pytest.approx(0)
    assert any(tick == pytest.approx(0) for tick in ax.get_yticks())


def test_negative_y_data_retains_its_lower_range(tmp_path, monkeypatch) -> None:
    closed_figures = []
    monkeypatch.setattr("kinelearn_stimviz.plotting.plt.close", closed_figures.append)
    summary = _summary()
    summary.loc[0, "ci_low"] = -0.1

    plot_psth(summary, output_path=tmp_path / "plot.png")

    ax = closed_figures[0].axes[0]
    assert ax.get_ylim()[0] < 0
    assert any(tick == pytest.approx(0) for tick in ax.get_yticks())


@pytest.mark.parametrize("font_size", [0, -1, float("inf"), float("nan")])
def test_font_size_must_be_positive_and_finite(tmp_path, font_size) -> None:
    with pytest.raises(ValueError, match="positive, finite"):
        plot_psth(_summary(), output_path=tmp_path / "plot.png", font_size=font_size)
