# Paper 3 Semantic-First Query-Geometry Checkpoint

This note defines the Gate 2 surface for the semantic-first hybrid codec.

## Goal

Test whether semantic shortlisting plus query-conditioned geometry v2 refinement can beat the best semantic incumbent on at least one public semantic-memory benchmark.

## Implemented Surface

- runner:
  - [run_paper3_semantic_query_geom_probe.sh](/Users/pranav/Documents/RT/scripts/run_paper3_semantic_query_geom_probe.sh)
- notebook:
  - [rt_paper3_semantic_query_geom_runner.ipynb](/Users/pranav/Documents/RT/notebooks/rt_paper3_semantic_query_geom_runner.ipynb)
- core implementation:
  - [run_paper3.py](/Users/pranav/Documents/RT/paper3_codec/run_paper3.py)
  - [query_geometry.py](/Users/pranav/Documents/RT/paper3_codec/query_geometry.py)
  - [policies.py](/Users/pranav/Documents/RT/paper3_codec/policies.py)

## New Public Policies

- `query_conditioned_geometry_v2`
- `query_conditioned_geometry_keep_compress_drop_v2`
- `semantic_query_conditioned_geometry_keep_compress_drop`
- `semantic_query_conditioned_geometry_keep_compress_drop_no_query`
- `semantic_query_conditioned_geometry_keep_compress_drop_no_support`

## Gate 2

Promote the hybrid only if, on `qwen25_15b`, it beats the benchmark-specific best semantic incumbent on at least one of:

- `MSC valid`
- `LongMemEval-S cleaned`

at budget `0.20` or `0.35`, under both:

- row-level significance
- conversation-level significance

The hybrid must also avoid a significant regression on the hard stress set at `0.50`.

## Status

Implemented. No canonical promoted hybrid artifact is recorded in this note yet.
