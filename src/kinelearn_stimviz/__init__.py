"""Small tools for stimulus-aligned behavioral analysis and plotting."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "adapt_kinelearn_predictions",
    "build_aligned_windows",
    "infer_time_axis",
    "load_behavior_table",
    "load_metadata",
    "load_stimulus_events",
    "plot_psth",
    "select_events",
    "summarize_by_event",
    "summarize_by_group",
    "summarize_by_subject",
]


def __getattr__(name: str) -> Any:
    if name in {"adapt_kinelearn_predictions", "load_behavior_table", "load_metadata", "load_stimulus_events"}:
        module = import_module(".io", __name__)
        return getattr(module, name)
    if name in {"build_aligned_windows", "infer_time_axis", "select_events"}:
        module = import_module(".align", __name__)
        return getattr(module, name)
    if name in {"summarize_by_event", "summarize_by_group", "summarize_by_subject"}:
        module = import_module(".aggregate", __name__)
        return getattr(module, name)
    if name == "plot_psth":
        module = import_module(".plotting", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
