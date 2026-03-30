# Paper 3 Low-Budget KCD Smoke Checkpoint

This note records the first engineering checkpoint after adding two new low-budget Paper 3 policy families:

- `support_aware_geometry_keep_compress_drop`
- `semantic_filtered_geometry_keep_compress_drop`

These are not yet publication-scale results. They are bounded smoke runs intended to answer one practical question:

> Does making geometry-KCD more support-aware materially improve it over the original `geometry_keep_compress_drop` baseline under tight budgets?

## Implemented changes

The low-budget upgrade path currently has two concrete pieces.

### 1. Support-aware geometry-KCD

The new support-aware policy changes segment compression in two ways:

- it builds a support score that upweights:
  - user turns
  - the latest user turn
  - turns with constraint-like markers such as formatting, retrieval, code, schema, and ordering language
  - semantic and geometry risk
- it uses that support score to preserve a more targeted sparse memory object inside each compressed region:
  - anchor turn
  - best combined risk/support turn
  - latest support-like turn when present

Operationally, this is a heuristic harm proxy, not a learned predictor yet.

### 2. Semantic-filtered geometry-KCD

The new semantic-filtered policy adds a semantic shortlist before support-aware geometry-KCD:

- shortlist old turns with a semantic density criterion under a relaxed budget
- always include the latest user turn in the shortlist when available
- apply support-aware geometry-KCD only inside the shortlisted memory

This is the first implementation of the two-stage idea:

1. broad semantic filtering
2. geometry-aware sparse compression within the candidate set

## Smoke-run setup

These smoke runs were intentionally small and bounded.

- model: `qwen25_05b`
- budgets: `0.20`, `0.35`
- conversations per benchmark: `2`
- target-turn stride: `6`
- max target turns per conversation: `6`
- segment span: `3`

Benchmarks:

- `msc_valid`
- `locomo10`

Because these are smoke runs, the correct use is directional engineering feedback, not final scientific ranking.

## MSC smoke result

Study:

- [`paper3_low_budget_smoke_msc`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_low_budget_smoke_msc)

### Main read

On this tiny MSC slice:

- the new `support_aware_geometry_keep_compress_drop` clearly improves over the original `geometry_keep_compress_drop`
- semantic-heavy policies still dominate the benchmark

### Concrete deltas vs uniform

At `0.20`:

- `geometry_keep_compress_drop`: `Δ logit L2 = -44.152`
- `support_aware_geometry_keep_compress_drop`: `Δ logit L2 = -505.510`
- `semantic_filtered_geometry_keep_compress_drop`: `Δ logit L2 = -392.507`
- `semantic_keep_compress_drop`: `Δ logit L2 = -479.531`

At `0.35`:

- `geometry_keep_compress_drop`: `Δ logit L2 = -43.910`
- `support_aware_geometry_keep_compress_drop`: `Δ logit L2 = -536.340`
- `semantic_filtered_geometry_keep_compress_drop`: `Δ logit L2 = -604.714`
- `semantic_keep_compress_drop`: `Δ logit L2 = -645.714`

### Pairwise signal

The most important pairwise comparison is the direct improvement over the old geometry-KCD baseline.

At `0.20`:

- `support_aware_geometry_keep_compress_drop` vs `geometry_keep_compress_drop`
  - `Δ logit L2 = -461.357`
  - `p = 0.0285`

At `0.35`:

- `support_aware_geometry_keep_compress_drop` vs `geometry_keep_compress_drop`
  - `Δ logit L2 = -492.430`
  - `p = 0.0210`

So the support-aware geometry-KCD variant is not a cosmetic change. On this bounded MSC run it materially improves over the original geometry-KCD.

However, the benchmark is still fundamentally semantic-heavy:

- `semantic_keep_compress_drop` and `semantic_filtered_geometry_keep_compress_drop` remain stronger than the geometry-only KCD variants
- `semantic_filtered_geometry_keep_compress_drop` does not clearly beat `semantic_keep_compress_drop`

Interpretation:

> Support-awareness helps geometry-KCD a lot on MSC, but it does not overturn the basic MSC story that semantic memory signal is strongest.

## LoCoMo smoke result

Study:

- [`paper3_low_budget_smoke_locomo`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_low_budget_smoke_locomo)

### Main read

On this tiny LoCoMo slice:

- the old geometry families are poor under low budget
- support-aware geometry-KCD fixes a large part of that failure
- semantic-heavy policies are still strongest overall

### Concrete deltas vs uniform

At `0.20`:

- `geometry`: `+6.346`
- `geometry_keep_compress_drop`: `+71.946`
- `support_aware_geometry_keep_compress_drop`: `-1102.248`
- `semantic`: `-1114.413`
- `semantic_filtered_geometry_keep_compress_drop`: `-1015.635`

At `0.35`:

- `geometry`: `+300.638`
- `geometry_keep_compress_drop`: `+578.327`
- `support_aware_geometry_keep_compress_drop`: `-47.430`
- `semantic`: `-89.849`
- `semantic_filtered_geometry_keep_compress_drop`: `-153.588`

### Pairwise signal

At `0.20`:

- `support_aware_geometry_keep_compress_drop` vs `geometry_keep_compress_drop`
  - `Δ logit L2 = -1174.194`
  - `p = 0.0027`

At `0.35`:

- `support_aware_geometry_keep_compress_drop` vs `geometry_keep_compress_drop`
  - `Δ logit L2 = -625.757`
  - `p = 0.0907`

This is the clearest engineering sign in the smoke run:

> The support-aware upgrade can turn geometry-KCD from directionally bad into directionally useful on a small LoCoMo slice.

The semantic shortlist variant also looks promising at `0.35`:

- `semantic_filtered_geometry_keep_compress_drop` vs `semantic`
  - `Δ logit L2 = -63.739`
  - `p = 0.0073`

That is only a smoke-scale result, but it is the first hint that a semantic front-end plus geometry-aware sparse compression may be the right low-budget hybrid for richer conversational memory.

## What this means

These runs do not support a new headline claim yet. They do support a concrete design decision.

### Locked-in engineering read

1. `support_aware_geometry_keep_compress_drop` is a real improvement over the old geometry-KCD baseline.
2. On semantic-memory benchmarks, semantic-led policies remain the strongest overall family.
3. The best next policy to test at real scale is now:
   - `support_aware_geometry_keep_compress_drop`
   - `semantic_filtered_geometry_keep_compress_drop`

### Recommended next run order

1. Re-run MSC at publication scale with:
   - `uniform`
   - `semantic`
   - `geometry`
   - `geometry_keep_compress_drop`
   - `support_aware_geometry_keep_compress_drop`
   - `semantic_keep_compress_drop`
   - `semantic_filtered_geometry_keep_compress_drop`
2. Run the same policy set on a bounded but larger LoCoMo slice.
3. If the LoCoMo signal persists, move the two new policies onto the hard stress set.

## New execution entry points

- runner:
  - [`scripts/run_paper3_low_budget_kcd_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_low_budget_kcd_probe.sh)
- notebook:
  - [`notebooks/rt_paper3_low_budget_kcd_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_low_budget_kcd_runner.ipynb)

## Caveat

These are bounded smoke runs on `qwen25_05b`. They should be used as:

- implementation validation
- directional policy selection

and not yet as manuscript-grade evidence.
