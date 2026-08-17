# kinelearn-stimviz

`kinelearn-stimviz` is a small downstream analysis package for stimulus-aligned behavior visualization. Its main job is to take exported behavior predictions plus stimulus-event tables and turn them into peri-stimulus summaries such as PSTH-style plots.

This package is intentionally narrower and more assay-specific than [KineLearn](https://www.github.com/rcalfredson/kinelearn) itself. KineLearn handles the core pose-to-behavior modeling workflow: feature extraction, splitting, training, evaluation, inference, and lightweight native visualizations such as timeline plots. `kinelearn-stimviz` sits one step downstream and focuses on event-aligned analyses that depend on external experimental timing, cohort structure, and assay-specific interpretation.

In practice, the workflow looks like this:

1. Use KineLearn to generate frame-level predictions or probabilities.
2. Prepare a stimulus-event table on the same time base as those predictions.
3. Load the exported tables into `kinelearn-stimviz`.
4. Align behavior traces to stimulus times.
5. Aggregate per event, per subject, and per group.
6. Plot PSTH-like summaries with confidence intervals.

## Relationship To KineLearn

`kinelearn-stimviz` is designed to interoperate with [KineLearn](https://www.github.com/rcalfredson/kinelearn) outputs, but it does not import KineLearn internals. The boundary is file-based:

- KineLearn writes exported tables and manifests.
- `kinelearn-stimviz` reads those exported tables together with stimulus-event annotations.

That separation keeps the plotting layer lightweight and reusable while avoiding tight coupling to model-training code.

Typical KineLearn inputs to this package include:

- `frame_predictions.parquet`
- `per_behavior_metrics.csv`
- `train_manifest.yml`

The most direct integration point is frame-level prediction output. For example, if a KineLearn prediction table contains:

- `__stem__`
- `__frame__`
- one or more `prob_<behavior>` columns
- optional `pred_<behavior>` columns

then `kinelearn-stimviz` can adapt that into its normalized behavior schema with:

- `entity_id <- __stem__`
- `frame_index <- __frame__`
- `time <- __frame__ / fps`
- `behavior <- <behavior name extracted from prob_<behavior>>`
- `value <- prob_<behavior>`

KineLearn itself already covers quick qualitative inspection through timeline plotting. `kinelearn-stimviz` is meant for the next step: specialized, event-aligned summaries.

When stimulus times are stored separately, for example in LED log files, `kinelearn-stimviz` also includes a helper for building normalized `stimulus_events` tables before plotting.

## Core Data Contract

The package works from a small normalized schema.

### 1. Stimulus event table

Required columns:

- `entity_id`: recording / video / subject identifier shared with the behavior table
- `event_time`: event time in seconds on the same clock as the behavior table

Optional columns:

- `event_id`: unique event identifier; generated automatically if omitted
- `event_type`: event label such as `laser`, `tone`, or `stimulus`
- any additional metadata columns

Example:

```csv
entity_id,event_id,event_time,event_type
fly_a,0,1.5,laser
fly_a,1,3.5,laser
fly_b,2,1.4,laser
```

### 2. Behavior time-series table

The package supports three input styles.

Long format:

- `entity_id`
- `time`
- `behavior`
- `value`

Wide format:

- `entity_id`
- `time`
- one or more behavior columns such as `walking`, `grooming`, `freezing`

KineLearn-style frame predictions:

- either wide KineLearn output with `__stem__`, `__frame__`, and `prob_<behavior>` columns
- or long per-frame/per-behavior tables with an entity id, a behavior column, and a probability column

Internally, behavior data are normalized to:

- `entity_id`
- `time`
- `behavior`
- `value`

### 3. Optional metadata / group table

Required:

- `entity_id`

Typical optional columns:

- `group`
- `cohort`
- `genotype`
- `session`

This table is merged after alignment so cohort-level summaries can be plotted cleanly.

## What The Package Does

The current package is intentionally modest in scope:

- stimulus-aligned windows around one event table at a time
- one or more behaviors per run
- binary or proportion-like traces
- per-event, per-subject, and per-group aggregation
- configurable all / first N / last N event selection
- PSTH-style plots with pointwise 95% Student's t-confidence intervals

The implementation is organized under:

```text
src/kinelearn_stimviz/
  io.py
  align.py
  aggregate.py
  stats.py
  plotting.py
  cli.py
  legacy_convert.py
```

## Typical Workflow

Install locally:

```bash
pip install -e .
```

### Plotting from normalized tables

If your stimulus times still live in LED log files, build a normalized event table first. The generic event-builder helper accepts either:

- a manifest table with explicit `entity_id`, `led_path`, and either `chunk_path` or `video_start_time`
- or the older JSON pair format used by the legacy workflow

Generic manifest example:

```bash
kinelearn-stimviz-build-events \
  event_manifest.csv \
  --output-dir converted/events \
  --source-format table \
  --output-format parquet
```

Legacy-pairs example with optional remapping from older source ids to KineLearn stems:

```bash
kinelearn-stimviz-build-events \
  video_lists/retinal_fed_29Aug.json \
  --output-dir converted/retinal_fed_events \
  --source-format legacy-pairs-json \
  --entity-map entity_map.csv \
  --event-type led \
  --output-format parquet
```

This writes:

- `stimulus_events.csv` / `.parquet`
- `event_sources.csv` / `.parquet`

The `event_sources` table is a provenance record describing which LED logs and timing anchors were used to build the final event table.

Run the example:

```bash
kinelearn-stimviz \
  --events examples/data/stimulus_events.csv \
  --behavior examples/data/behavior_long.csv \
  --metadata examples/data/metadata.csv \
  --output examples/output/example_psth.png \
  --pre 1.0 \
  --post 2.0 \
  --group-col group \
  --title "Synthetic stimulus-aligned behaviors"
```

Use only the first 10 or last 10 events per entity and event type:

```bash
kinelearn-stimviz \
  --events stimulus_events.csv \
  --behavior behavior.csv \
  --event-subset first \
  --event-count 10 \
  --output first10_psth.png

kinelearn-stimviz \
  --events stimulus_events.csv \
  --behavior behavior.csv \
  --event-subset last \
  --event-count 10 \
  --output last10_psth.png
```

If you want alignment to follow direct frame-offset slicing more closely, use frame mode with tables that include `frame_index`:

```bash
kinelearn-stimviz \
  --events stimulus_events.parquet \
  --behavior behavior_long.parquet \
  --metadata metadata.parquet \
  --alignment-mode frame \
  --fps 60 \
  --pre 1.0 \
  --post 3.0 \
  --behaviors back_leg_together genitalia_extension \
  --output last10_frame_aligned.png
```

### Choosing an alignment mode

`kinelearn-stimviz` supports two alignment modes:

- `nearest`: align by matching each target time bin to the nearest observed sample in time
- `frame`: align by converting the window to frame offsets and slicing exact frame positions

As a rule of thumb:

- use `frame` when your behavior data are genuinely frame-based, include `frame_index`, and you want the analysis to stay as close as possible to direct frame slicing in older workflows
- use `nearest` when your data are more naturally time-based, may be irregularly sampled, or come from mixed exported tables where absolute time is the more natural alignment unit

In practice:

- `frame` tends to preserve fine frame-level peaks and valleys more faithfully
- `nearest` is the more general and portable default for heterogeneous tabular inputs

For historical reproducibility, especially when comparing against older frame-sliced PSTH plots, `frame` is often the better fit. For newer table-based workflows built around exported timestamps, `nearest` is usually the safer default.

You can also point the CLI directly at a KineLearn-style frame prediction table:

```bash
kinelearn-stimviz \
  --events stimulus_events.csv \
  --behavior frame_predictions.parquet \
  --behavior-format kinelearn_predictions \
  --entity-col __stem__ \
  --frame-col __frame__ \
  --fps 60 \
  --alignment-mode frame \
  --behaviors genitalia_extension \
  --output psth.png
```

## Why The Legacy Converter Exists

The main package is designed around exported tables, not old project-specific manifests. The legacy converter exists only as a migration bridge for older datasets that were organized as JSON lists of `[prediction_csv, led_csv]` pairs.

Use it when you want to bring older datasets into the new table-based workflow:

```bash
kinelearn-stimviz-convert-legacy \
  video_lists/retinal_fed_29Aug.json \
  --output-dir converted/retinal_fed_29Aug \
  --fps 60 \
  --event-type led \
  --group retinal_fed \
  --output-format parquet
```

This writes normalized tables such as:

- `behavior_long.parquet`
- `stimulus_events.parquet`
- `metadata.parquet`

You can also use `--output-format csv` or `--output-format both`. Parquet output requires a parquet engine such as `pyarrow`.

The converter is intentionally isolated from the main plotting path. It uses the older prediction CSV plus LED log pairing and the legacy `chunk_data_log_<timestamp>.csv` lookup only to migrate older experiments into the same schema that newer KineLearn-derived workflows can use directly.

Use the default Pred_ columns for reproducing original proportion PSTHs.
Use --behavior-prefix Prob_ for probability-valued PSTHs, which will not match the original plot scale/profile exactly.

For legacy cohort comparisons, run the converter once per JSON manifest with a different `--group` label, merge the converted directories, then plot with `--group-col group`:

```bash
kinelearn-stimviz-convert-legacy \
  video_lists/no_retinal_29Aug.json \
  --output-dir converted/no_retinal_29Aug \
  --fps 60 \
  --event-type led \
  --group no_retinal \
  --output-format parquet

kinelearn-stimviz-merge \
  --input converted/retinal_fed_29Aug \
  --input converted/no_retinal_29Aug \
  --output-dir converted/legacy_combined_29Aug \
  --output-format parquet
```

By default, the converter reads binary behavior columns with the `Pred_` prefix; pass `--behavior-prefix Prob_` if you want probability traces instead.

For a standalone, paper-style panel, you can override display labels without changing the internal behavior or group names:

```bash
kinelearn-stimviz \
  --events converted/legacy_combined_29Aug/stimulus_events.parquet \
  --behavior converted/legacy_combined_29Aug/behavior_long.parquet \
  --metadata converted/legacy_combined_29Aug/metadata.parquet \
  --output plots/hind_leg_groom_last10.png \
  --event-subset last \
  --event-count 10 \
  --alignment-mode frame \
  --fps 60 \
  --pre 1 \
  --post 3 \
  --group-col group \
  --behaviors back_leg_together \
  --behavior-label "back_leg_together=Hind legs grooming" \
  --group-label "retinal_fed=Retinal-fed" \
  --group-label "no_retinal=Non-retinal-fed" \
  --group-order retinal_fed no_retinal \
  --group-color "retinal_fed=tab:orange" \
  --group-color "no_retinal=tab:blue" \
  --show-gridlines \
  --hide-subplot-titles \
  --hide-legend-title \
  --show-group-ns \
  --fig-width 5.2 \
  --fig-height 3.8 \
  --font-size 11 \
  --ylabel "Proportion" \
  --xlabel "Time (s)" \
  --title "Hind legs grooming" \
  --annotation "0.2 Hz stimulation" \
  --annotation-box-alpha 0.65
```

`--font-size` sets the middle-sized plot text (axis labels and legend titles)
in points. Subplot and overall titles are larger at 1.2 times that size; tick
labels, legend entries, and in-plot annotations are smaller at 5/6 of that
size. If the option is omitted, Matplotlib's current theme defaults are used
unchanged.
When a font size is supplied, the default or explicitly requested dimensions
describe the 10-point reference layout. The canvas grows only by the extra
margin space required by larger text; the data plotting area does not scale
with the font. Larger sizes also extend the upper y-limit to create empty space
inside the plot for the legend and annotation area without changing the lower
y-limit.

For nonnegative data, the y-axis explicitly starts at zero and zero is retained
as a labeled major tick. A tiny numerical tolerance prevents floating-point
noise around zero from defeating this behavior. Data with genuinely negative
values keep their negative lower range while still labeling zero.

## Design Notes

This package preserves the useful analysis structure from older one-off plotting scripts while generalizing away lab-specific assumptions:

- no hard-coded behavior names
- no hard-coded filename patterns in the main analysis path
- no fixed first-10 / last-10 workflow branches
- no dependence on KineLearn source imports
- no mixing of plotting with embedded statistical testing logic

The result is a small, reusable layer for stimulus-aligned analysis that can sit downstream of KineLearn or any other tool that can export compatible tables.
