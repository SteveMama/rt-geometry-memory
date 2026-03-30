# Paper 3 Query-Conditioned Geometry Smoke Checkpoint

This note records the first practical implementation of query-conditioned geometry for Paper 3.

The goal was not to prove the full theorem story. The goal was to test a geometrically cleaner empirical version:

> express the query signal in the same tangent-space coordinates as the conversation-motion vectors, then score turns by query-projected local geometry.

## What was implemented

The new implementation adds two policy families:

- `query_conditioned_geometry`
- `query_conditioned_geometry_keep_compress_drop`

Code paths:

- query-conditioned geometry features:
  - [`paper3_codec/query_geometry.py`](/Users/pranav/Documents/RT/paper3_codec/query_geometry.py)
- Paper 3 runner integration:
  - [`paper3_codec/run_paper3.py`](/Users/pranav/Documents/RT/paper3_codec/run_paper3.py)
- bounded runner:
  - [`scripts/run_paper3_query_conditioned_geometry_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_query_conditioned_geometry_probe.sh)
- Colab notebook:
  - [`notebooks/rt_paper3_query_conditioned_geometry_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_query_conditioned_geometry_runner.ipynb)

## Geometric construction

For each prior turn index `t`, we compute a query direction in the same tangent space as the local conversation motion:

1. normalize the conversation states onto the sphere
2. define the current query state as the target-turn state
3. compute a tangent query vector at turn `t` using the sphere log map
4. project local geometric motion onto that tangent query direction

The implemented features are:

- query-projected local curvature
- query-projected local subspace energy

These are then combined with a small ambient-geometry term to produce:

- `query_conditioned_geometry`

This is intentionally a practical empirical feature, not a finalized theorem object.

## Smoke-run setup

These runs were bounded engineering smoke tests.

- model: `qwen25_05b`
- budgets: `0.20`, `0.35`
- conversations per benchmark: `2`
- target-turn stride: `6`
- max target turns per conversation: `6`
- segment span: `3`

Benchmarks:

- `msc_valid`
- `locomo10`

## MSC smoke result

Study:

- [`paper3_query_geom_smoke_msc`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_query_geom_smoke_msc)

### Main read

On MSC, query-conditioning helps geometry substantially.

- plain `geometry` improves modestly
- `query_conditioned_geometry` is clearly stronger than plain geometry
- `query_conditioned_geometry_keep_compress_drop` becomes competitive with the strongest sparse codecs

### Concrete deltas vs uniform

At `0.20`:

- `geometry`: `-77.737`
- `query_conditioned_geometry`: `-176.541`
- `support_aware_geometry_keep_compress_drop`: `-505.510`
- `query_conditioned_geometry_keep_compress_drop`: `-501.114`
- `semantic`: `-433.905`
- `semantic_keep_compress_drop`: `-479.531`

At `0.35`:

- `geometry`: `-39.630`
- `query_conditioned_geometry`: `-167.923`
- `support_aware_geometry_keep_compress_drop`: `-536.340`
- `query_conditioned_geometry_keep_compress_drop`: `-553.196`
- `semantic`: `-608.159`
- `semantic_keep_compress_drop`: `-645.714`

### Interpretation

MSC says:

1. query-conditioning is a real improvement over ambient geometry
2. query-conditioned geometry-KCD is viable
3. but the strongest semantic codecs still remain at least as strong on this semantic-memory benchmark

So on MSC the new method is a **partial win**, not a lead result.

## LoCoMo smoke result

Study:

- [`paper3_query_geom_smoke_locomo`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_query_geom_smoke_locomo)

### Main read

On LoCoMo, query-conditioning has a split outcome.

At `0.20`, query-conditioned geometry looks strong:

- `query_conditioned_geometry`: `-250.102`
- `query_conditioned_geometry_keep_compress_drop`: `-1124.247`
- `semantic`: `-1114.413`
- `semantic_keep_compress_drop`: `-1058.737`
- `support_aware_geometry_keep_compress_drop`: `-1102.248`

So under the tightest budget, the query-conditioned KCD variant is fully competitive.

But at `0.35`, the plain query-conditioned retention policy collapses:

- `query_conditioned_geometry`: `+881.417`

and the KCD version only recovers to near-neutral:

- `query_conditioned_geometry_keep_compress_drop`: `+9.724`

while the other sparse codecs remain directionally better:

- `semantic`: `-89.849`
- `semantic_keep_compress_drop`: `-37.937`
- `support_aware_geometry_keep_compress_drop`: `-47.430`

### Interpretation

LoCoMo says:

1. query-conditioning can help dramatically under tight scarcity
2. but the current formulation is unstable across budget regimes
3. the KCD wrapper is doing important regularization work

So the current query-conditioned geometry feature is not yet a stable standalone ranking signal.

## What this means

This smoke checkpoint gives a useful answer.

### What succeeded

- The query signal is now expressed in the same tangent-space coordinates as the local geometry.
- Query-conditioned geometry clearly improves over plain ambient geometry on MSC.
- Query-conditioned geometry-KCD can be fully competitive under tight budgets.

### What did not succeed

- Query-conditioned geometry did not clearly beat the strongest semantic codecs.
- On LoCoMo, the plain query-conditioned retention score is unstable at `0.35`.
- So query-conditioned geometry is not yet the replacement for semantic relevance.

## Updated algorithmic conclusion

The right conclusion is:

> query-conditioned geometry is a useful secondary signal, but not yet a sufficient primary signal for semantic-memory benchmarks.

That means the next algorithm should not be:

- “replace semantic with query-conditioned geometry”

It should be:

- “use semantic relevance for candidate discovery, and use query-conditioned geometry as a decoder-faithfulness refinement inside the candidate set”

## Recommended next experiment

The next concrete algorithm to build is:

### Semantic-first, query-conditioned geometry-KCD

1. semantic shortlist
2. support-aware sparse codec inside the shortlist
3. query-conditioned geometry as the fine-grained rank/keep/compress score

This is now better motivated than a pure geometry-first query-conditioned method.
