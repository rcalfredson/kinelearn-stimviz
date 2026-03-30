# kinelearn-stimviz

`kinelearn-stimviz` is a small standalone Python package for stimulus-aligned behavioral analysis and PSTH-style plotting from exported tables. It is designed to interoperate with KineLearn outputs, but it does not import KineLearn code or rely on KineLearn internals.

The package focuses on a simple pipeline:

1. Load a stimulus event table.
2. Load a behavior time-series table.
3. Align behavior traces to stimulus times.
4. Aggregate aligned windows per event, per subject, and per group.
5. Plot PSTH-like summaries with confidence intervals.

## Why this repo exists

This package grew out of an earlier set of one-off analysis scripts that already contained a useful conceptual pipeline:

- Parse stimulus timing and behavior traces.
- Build per-event aligned windows.
- Average across pulses, then subjects.
- Plot means with uncertainty.

That core idea is preserved here. What changed is the packaging and the data contract:

- No hard-coded filename pattern assumptions.
- No hard-coded behavior names.
- No special treatment of only the first or last pulses.
- No dependence on lab-specific LED log layouts.
- No coupling to KineLearn source code.

## Minimal input contract

The package normalizes inputs into a small set of standard columns.

### 1. Stimulus event table

Required columns:

- `entity_id`: recording / video / subject identifier shared with the behavior table
- `event_time`: event time in seconds on the same clock as the behavior table

Optional columns:

- `event_id`: unique event identifier; generated automatically if omitted
- `event_type`: event label such as `laser`, `tone`, or `stimulus`
- any extra metadata columns

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

- entity id column such as `video_id` or `stem`
- `behavior`
- either a timestamp column or `frame_index` plus `fps`
- either `predicted_probability` or a binary `predicted_label`

Normalized internal schema:

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

This table is merged after alignment so group summaries can be computed cleanly.

## KineLearn interoperability

This repo is intended to consume exported KineLearn artifacts, not KineLearn internals.

Relevant KineLearn outputs may include:

- `train_manifest.yml`
- `per_behavior_metrics.csv`
- `frame_predictions.parquet`

The most direct integration point is `frame_predictions.parquet`. If it contains columns like:

- `video_id` or `stem`
- `frame_index` or a timestamp
- `behavior`
- `predicted_probability`
- optional `predicted_label`
- optional `true_label`

then `kinelearn-stimviz` can adapt that table into the normalized behavior schema with:

- `entity_id <- video_id`
- `time <- frame_index / fps` or an existing timestamp column
- `behavior <- behavior`
- `value <- predicted_probability` or `predicted_label`

External stimulus-event tables can then be joined by `entity_id` and aligned by time.

## Repository structure

```text
src/kinelearn_stimviz/
  __init__.py
  io.py
  align.py
  aggregate.py
  stats.py
  plotting.py
  cli.py
examples/
  data/
    behavior_long.csv
    stimulus_events.csv
    metadata.csv
```

## MVP scope

The current MVP intentionally stays small:

- binary or proportion-like traces
- one event table at a time
- one or more behaviors
- optional cohort comparison through a metadata table
- configurable all / first N / last N event selection
- PSTH-like mean traces with 95% confidence intervals

## Example workflow

Install locally:

```bash
pip install -e .
```

Convert an old JSON pair manifest into normalized tables:

```bash
kinelearn-stimviz-convert-legacy \
  old/retinal_fed_29Aug.json \
  --output-dir converted/retinal_fed_29Aug \
  --fps 60 \
  --event-type laser
```

This writes:

- `converted/retinal_fed_29Aug/behavior_long.csv`
- `converted/retinal_fed_29Aug/stimulus_events.csv`
- `converted/retinal_fed_29Aug/metadata.csv`

The converter is intentionally legacy-specific. It uses the older prediction CSV plus LED log pairing and the legacy `chunk_data_log_<timestamp>.csv` lookup only for migration into the new table-based workflow.

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

You can also point the CLI at a KineLearn-style frame prediction table:

```bash
kinelearn-stimviz \
  --events stimulus_events.csv \
  --behavior frame_predictions.parquet \
  --behavior-format kinelearn_predictions \
  --entity-col video_id \
  --behavior-col behavior \
  --frame-col frame_index \
  --fps 30 \
  --probability-col predicted_probability \
  --output psth.png
```

## Design lineage

- The overall parse -> align -> aggregate -> plot pipeline.
- The idea of building aligned windows around stimulus onsets.
- The fly or subject level averaging step before cohort summaries.
- PSTH-style plotting with uncertainty bands.

## What was generalized or dropped

Generalized:

- Stimulus events are now generic tabular inputs.
- Behaviors are arbitrary columns or long-format labels.
- KineLearn frame predictions can be adapted from exported tables.
- Grouping is handled through metadata tables instead of script-specific branches.
- first-N and last-N event selection are supported as configurable options rather than fixed workflow branches

Dropped or replaced:

- `chunk_data_log_<timestamp>.csv` filename assumptions.
- timestamp parsing from filenames.
- fixed behavior lists such as `back_leg_together` and `genitalia_extension`.
- rigid `first 10` and `last 10` pulse handling baked into the analysis flow.
- lab-specific LED CSV parsing logic.
- mixing plotting with hypothesis tests in a single plotting step.
