# Paper 3 Oracle Harm Headroom Checkpoint

This note defines the Gate 1 surface for the semantic-gap program.

## Goal

Measure whether query-conditioned geometry and combined structural features add ranking value **inside a semantic shortlist** on semantic-memory benchmarks.

## Implemented Surface

- runner:
  - [run_paper3_harm_oracle_probe.sh](/Users/pranav/Documents/RT/scripts/run_paper3_harm_oracle_probe.sh)
- notebook:
  - [rt_paper3_harm_oracle_runner.ipynb](/Users/pranav/Documents/RT/notebooks/rt_paper3_harm_oracle_runner.ipynb)
- core module:
  - [harm_oracle_study.py](/Users/pranav/Documents/RT/paper3_codec/harm_oracle_study.py)

## Outputs

Each run emits:

- `candidate_rows.csv`
- `summary.json`
- `report.md`

The canonical publication path for promoted runs is:

- `artifacts/paper3/<oracle-study-name>`

## Gate 1

Promote geometry refinement only if, within the semantic shortlist, query-geometry or combined structural features improve oracle ranking over semantic-only by at least one of:

- `Δ Kendall tau >= +0.03`
- `Δ top-5 oracle recall >= +5 percentage points`

This gate must be met on at least one of:

- `MSC valid`
- `LongMemEval-S cleaned`

If not, geometry refinement remains diagnostic only on semantic-memory benchmarks.

## Status

Implemented. No canonical promoted oracle artifact is recorded in this note yet.
