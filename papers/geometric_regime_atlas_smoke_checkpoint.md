# Geometric Regime Atlas Smoke Checkpoint

This note records the first bounded smoke run of the geometric regime atlas.

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

## Main result

The smoke atlas already recovers a meaningful benchmark split.

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

This is exactly where we expected LongMemEval to land.

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

This is exactly where the hard stress set lands.

### Small residual spike regimes

Two smaller clusters remained:

- one mixed long-segment spike regime shared by `LoCoMo`, `LongMemEval`, and a
  little `MSC`
- one shorter mid-length spike regime shared by `LoCoMo` and `MSC`

These appear to be transition-heavy mixed segments rather than clean new task
families.

## Diagnostic result

The new diagnostics show that the atlas result must be interpreted cautiously.

The strongest numerical finding is:

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

This means the extreme-curvature regime is not yet trustworthy as a clean
semantic-memory regime. It is at least partly a numerical near-stationary
artifact.

## What this means scientifically

Even after that caution, the atlas still supports a weaker version of the
benchmark-reading hypothesis:

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

### 1. Saturation / near-stationary blow-up

The current curvature proxy can explode when local step norms become extremely
small. That means some of the fact-memory regime is better understood as:

- low-motion near-stationary conversation regions
- plus a curvature formula that becomes numerically unstable there

So before treating those segments as a real geometric regime, the atlas needs a
stabilized curvature-style feature or an explicit near-stationary detector.

### 2. Spike-heavy family still too broad

Even after separating the near-stationary segments, the spike-heavy side is
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
