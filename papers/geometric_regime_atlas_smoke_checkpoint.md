# Geometric Regime Atlas Smoke Checkpoint

This note records the bounded smoke runs of the geometric regime atlas, the
numerical curvature bug that appeared on long conversations, and the corrected
stabilized rerun.

The goal was not to beat a baseline yet. The goal was to answer a prior
question:

> do different benchmark families occupy different geometric regimes, and can a
> geometry-only atlas start to recover that split without using benchmark labels
> or query semantics?

## Run

Artifacts:

- [`regime_atlas_smoke_v2`](/Users/pranav/Documents/RT/results/regime_atlas/regime_atlas_smoke_v2)
- [`regime_atlas_smoke_v3`](/Users/pranav/Documents/RT/results/regime_atlas/regime_atlas_smoke_v3)
  - diagnostics:
    - `diagnostics/representative_turn_series.png`
    - `diagnostics/representative_turn_series_log.png`
    - `diagnostics/conversation_series_summary.csv`
    - `diagnostics/curvature_saturation_report.md`
- [`regime_atlas_smoke_v4`](/Users/pranav/Documents/RT/results/regime_atlas/regime_atlas_smoke_v4)
  - stabilized rerun using an arclength floor in the curvature proxy
  - diagnostics:
    - `diagnostics/representative_turn_series.png`
    - `diagnostics/representative_turn_series_log.png`
    - `diagnostics/conversation_series_summary.csv`
    - `diagnostics/curvature_saturation_report.md`

Configuration:

- model: `qwen25_05b`
- inputs:
  - bounded `MSC`
  - bounded `LoCoMo10`
  - bounded `LongMemEval-S cleaned`
  - bounded hard stress set
- conversations: `14`
- extracted segments: `208`
- max turns / conversation: `96`
- cluster count: `4`

## What was implemented

The atlas computes segment-level statistics from transported hidden-state
increments:

- mean / std / skew / max curvature
- mean turning angle
- step-norm statistics
- effective local rank
- rank-jump and subspace-shift summaries
- boundary-score summaries
- role-switch rate

It then clusters segments without using benchmark labels.

## Initial atlas result

The original smoke atlas already recovered a meaningful benchmark split.

### Regime 0: `near_stationary_fact_memory`

Family mix:

- `LongMemEval-S cleaned`: `25 / 31` benchmark segments, or `80.6%`
- `LoCoMo10`: `36 / 96`, or `37.5%`
- `MSC`: `10 / 66`, or `15.2%`
- hard stress families: `0`

Interpretation:

- semantically stable fact-bearing regions
- very small local motion norms
- low subspace-shift structure
- broad personal-memory or fact-memory zones rather than exact instruction turns

This is where we expected LongMemEval-style fact memory to land.

### Regime 1: `curvature_spike_transition`

Family mix:

- `retrieval_heavy`: `12 / 12`, or `100%`
- `long_dependency`: `3 / 3`, or `100%`
- `MSC`: `52 / 66`, or `78.8%`
- `LoCoMo10`: `52 / 96`, or `54.2%`

Interpretation:

- short local exchange units
- strong boundary-score activity
- higher subspace shift
- rapid turn-to-turn transitions

This is where the hard stress set initially landed.

### Small residual spike regimes

Two smaller clusters remained:

- one mixed long-segment spike regime shared by `LoCoMo`, `LongMemEval`, and a
  little `MSC`
- one shorter mid-length spike regime shared by `LoCoMo` and `MSC`

These appear to be transition-heavy mixed segments rather than clean new task
families.

## Diagnostic result: the raw curvature bug was real

The new diagnostics show that the atlas result must be interpreted cautiously.

The strongest numerical finding from `v3` was:

- many of the extreme-curvature segments are also near-zero-step segments
- the saturation audit flagged `67 / 208` segments as suspicious under the rule:
  - `mean_step_norm < 1e-3`
  - `mean_curvature > 100`

The concentration is not random:

- `LongMemEval-S cleaned`: `26` suspicious segments
- `LoCoMo10`: `36`
- `MSC`: `5`
- hard stress families: `0`

At the conversation level:

- `LongMemEval` sample conversations had `89-91` interior turns with curvature
  above `1000`, while also having `90-92` step norms below `1e-3`
- `LoCoMo` samples had `63` curvature values above `1000` and `64` step norms
  below `1e-3`
- the hard stress set had none of these pathologies

At the conversation level, the pathology was obvious:

- `LongMemEval` sample conversations had raw mean curvature in the
  `4487-6315` range
- `LoCoMo` samples had raw mean curvature around `3190`
- `MSC` conversation `msc-00000` had raw mean curvature `2521`
- the hard stress set remained in the healthy `2.1-2.9` range

This was not a subtle modeling issue. It was a numerical bug in the raw
curvature proxy on long near-stationary trajectories. When local step norms
collapsed toward zero, the curvature ratio exploded.

## The fix

The atlas now uses a stabilized curvature proxy with an arclength floor:

```text
kappa_stable = turning_angle / max(local_arclength, 0.05)
```

Important boundary:

- legacy raw curvature is still available for historical reproducibility
- the stabilized proxy is used by the atlas and its diagnostics
- this avoids silently changing frozen Paper 1 artifacts while fixing the atlas

## Corrected result: `regime_atlas_smoke_v4`

After the curvature fix, the atlas remains scientifically useful, but the
interpretation changes.

### What changed numerically

The `v4` diagnostics still preserve the raw-bug evidence on purpose. The
saturation audit therefore still reports the old pathology under the raw
formula:

- suspicious segments (`mean_step_norm < 1e-3` and raw `mean_curvature > 100`):
  `67`
- suspicious family counts:
  - `LoCoMo10`: `36`
  - `LongMemEval-S cleaned`: `26`
  - `MSC`: `5`

What changed is the atlas feature path itself:

- segment clustering now uses stabilized curvature, not raw curvature
- the atlas report and segment rows therefore no longer treat those raw
  thousands-level values as the operative geometry signal
- the diagnostics keep both raw and stabilized views side by side so the bug
  remains visible and auditable

And the long-conversation curvature values collapse from nonsense to plausible
levels:

- `msc-00000`: raw mean curvature `2520.867` -> stabilized `17.503`
- `conv-26-qa000`: raw `3190.459` -> stabilized `19.201`
- `e47becba`: raw `6314.673` -> stabilized `26.130`
- `118b2229`: raw `4487.443` -> stabilized `26.459`
- hard stress conversations stay unchanged around `2.168-2.898`

So the bug is fixed in the atlas path. The raw pathology is still recorded for
comparison, but it no longer drives clustering. The long-conversation
fact-memory regime is therefore no longer just an artifact of raw curvature
blow-up.

### What the stabilized atlas still shows

The corrected `v4` atlas still recovers a meaningful family split.

#### Regime 0: `near_stationary_fact_memory`

Family mix:

- `LongMemEval-S cleaned`: `25 / 31`, or `80.6%`
- `LoCoMo10`: `36 / 96`, or `37.5%`
- `MSC`: `10 / 66`, or `15.2%`
- hard stress families: `0`

Centroid summary:

- `mean_curvature = 25.302`
- `std_curvature = 0.000`
- `mean_subspace_shift = 0.008`
- `role_switch_rate = 0.989`

Interpretation:

- sustained low-motion conversational continuation
- fact-memory or haystack-like regions
- almost no local subspace turnover
- strongly associated with LongMemEval and some LoCoMo

#### Spike-heavy regimes

The remaining three clusters are all transition-heavy, but they are not yet
cleanly separated into distinct semantic categories.

- Regime 1:
  - `34` segments
  - mixed `MSC`, `LoCoMo`, `retrieval_heavy`, a little `LongMemEval`,
    `long_dependency`
  - higher subspace shift and lower mean curvature
- Regime 2:
  - only `7` segments
  - a small high-variance spike family
- Regime 3:
  - `96` segments
  - dominated by `MSC` and `LoCoMo`, but also includes half the
    `retrieval_heavy` set

That means the corrected atlas now supports a more careful conclusion:

> the fact-memory side of the atlas is real, but the spike-heavy side is still
> too coarse and still conflates generic dialogue exchange with true
> support-turn or constraint-heavy structure.

## What this means scientifically

After the fix, the atlas supports a cleaner version of the benchmark-reading
hypothesis:

- `LongMemEval` is mostly a **fact-memory / haystack retrieval** benchmark
- the hard stress set is a **support-turn-critical structural** benchmark
- `LoCoMo` spans both
- `MSC` is dominated by fast local dialogue exchange and continuity rather than
  the same regime as LongMemEval

That means the benchmark split is not arbitrary.

The geometry is already revealing at least two broad memory behaviors:

1. near-stationary fact memory
2. transition-heavy structural exchange

## Important limitations

The current atlas is useful but not yet finished.

There are now two clear limitations.

### 1. Stabilized curvature still compresses the spike-heavy family too much

Even after fixing the blow-up, the spike-heavy side is
still too broad:

- it captures the hard stress set correctly
- but it also absorbs a large fraction of `MSC` and `LoCoMo`

So the current feature stack still conflates:

- generic dialogue exchange
- event transitions
- true constraint/support-turn spikes

In other words:

> the atlas is good enough to separate LongMemEval-style fact memory from the
> hard stress set, but not yet good enough to cleanly split all spike-heavy
> conversational structures.

## Why that limitation is useful

It tells us the next atlas upgrade should not be “more semantic scoring.”

It should be a better geometry stack for regime separation, especially:

- role-alternation-corrected transition features
- event-chain vs constraint-spike separation
- decoder-aware or query-aware spike features inside the spike-heavy family

## Best current reading

This smoke run is already enough to justify the new research direction:

> learn the geometric regime first, then choose compression policy by regime.

But it also tells us the current regime vocabulary is still one step too coarse.

The next correct move is:

1. improve the atlas so spike-heavy segments split into cleaner sub-regimes
2. only then derive regime-specific compression policies

That is the right problem-first path.
