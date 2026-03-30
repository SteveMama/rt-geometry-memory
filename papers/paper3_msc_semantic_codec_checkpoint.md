# Paper 3 MSC Semantic-Codec Checkpoint

## Run

- study: `paper3_semantic_kcd_msc_valid`
- benchmark: `msc_valid`
- source benchmark dataset: `gonced8/multi-session_chat`
- model: `qwen25_15b`
- conversations: `24`
- evaluations: `3960`
- behavior evaluations: `3960`
- segment span: `2`
- target-turn stride: `4`
- max target turns / conversation: `16`
- budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- policies:
  - `uniform`
  - `semantic`
  - `geometry`
  - `geometry_keep_compress_drop`
  - `semantic_keep_compress_drop`

Tracked outputs:

- [`study_report.md`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_semantic_kcd_msc_valid/study_report.md)
- [`pairwise_report.md`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_semantic_kcd_msc_valid/pairwise_report.md)
- [`study_summary.json`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_semantic_kcd_msc_valid/study_summary.json)
- [`significance_summary.json`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_semantic_kcd_msc_valid/significance_summary.json)
- [`behavior_significance_summary.json`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_semantic_kcd_msc_valid/behavior_significance_summary.json)

## Main Result

MSC does not support `geometry_keep_compress_drop` as the right sparse codec family. It strongly favors semantic signal, and it partially answers the next design question:

> if the signal is changed from geometry to semantic, does sparse codec structure become the right form on conversational-memory benchmarks?

The answer from this run is:

- at `0.20`, `semantic_keep_compress_drop` is competitive with plain `semantic` and slightly better in mean logit distortion, but not significantly so
- at `0.35` and `0.50`, plain `semantic` is the better overall policy, especially on behavior
- so MSC rewards semantic memory preservation more than sparse codec structure

## Aggregate Results

Negative deltas are improvements over `uniform`.

### Budget `0.20`

- `semantic_keep_compress_drop`
  - `Δ logit L2 = -119.966`
  - relative logit L2 `0.894`
  - `Δ answer NLL = -0.6081`
- `semantic`
  - `Δ logit L2 = -105.782`
  - relative logit L2 `0.906`
  - `Δ answer NLL = -0.6033`
- `geometry`
  - `Δ logit L2 = -8.366`
  - `Δ answer NLL = -0.0617`
- `geometry_keep_compress_drop`
  - `Δ logit L2 = +5.730`
  - `Δ answer NLL = -0.0064`

### Budget `0.35`

- `semantic`
  - `Δ logit L2 = -112.257`
  - relative logit L2 `0.900`
  - `Δ answer NLL = -0.6794`
- `semantic_keep_compress_drop`
  - `Δ logit L2 = -98.320`
  - relative logit L2 `0.912`
  - `Δ answer NLL = -0.6400`
- `geometry`
  - `Δ logit L2 = -53.100`
  - `Δ answer NLL = -0.2007`
- `geometry_keep_compress_drop`
  - `Δ logit L2 = +22.591`
  - `Δ answer NLL = -0.0254`

### Budget `0.50`

- `semantic`
  - `Δ logit L2 = -108.342`
  - relative logit L2 `0.899`
  - `Δ answer NLL = -0.6472`
- `semantic_keep_compress_drop`
  - `Δ logit L2 = -98.447`
  - relative logit L2 `0.908`
  - `Δ answer NLL = -0.5948`
- `geometry`
  - `Δ logit L2 = -65.852`
  - `Δ answer NLL = -0.3146`
- `geometry_keep_compress_drop`
  - `Δ logit L2 = +26.961`
  - `Δ answer NLL = -0.1171`

## Pairwise Results

The most important pairwise comparison is `semantic_keep_compress_drop` vs `semantic`.

### Budget `0.20`

- logit:
  - `Δ = -14.184`
  - `95% CI = [-38.598, 9.387]`
  - `p = 0.2562`
- behavior:
  - `Δ answer NLL = -0.0048`
  - `95% CI = [-0.0234, 0.0091]`
  - `p = 0.6015`

Interpretation:

- codec-style semantic compression is directionally better on mean logit distortion
- but there is no statistically credible win over plain semantic retention
- behavior is effectively tied

### Budget `0.35`

- logit:
  - `Δ = +13.936`
  - `95% CI = [-13.882, 40.986]`
  - `p = 0.3475`
- behavior:
  - `Δ answer NLL = +0.0394`
  - `95% CI = [0.0185, 0.0634]`
  - `p = 0.0000`

Interpretation:

- logit is directionally worse, though not significantly so
- behavior is significantly worse than plain semantic retention

### Budget `0.50`

- logit:
  - `Δ = +9.895`
  - `95% CI = [-24.529, 45.073]`
  - `p = 0.5887`
- behavior:
  - `Δ answer NLL = +0.0524`
  - `95% CI = [0.0278, 0.0788]`
  - `p = 0.0000`

Interpretation:

- no evidence of a logit advantage
- clear behavior degradation relative to plain semantic retention

## Strong Negative Result For Geometry-KCD

Against plain `geometry`, the geometry-driven codec remains badly mismatched to MSC:

### `geometry_keep_compress_drop` vs `geometry`

- `0.20`
  - logit `+14.096`
  - behavior `+0.0553`
- `0.35`
  - logit `+75.691`
  - behavior `+0.1753`
- `0.50`
  - logit `+92.812`
  - behavior `+0.1975`

All of these deltas are in the wrong direction, and the `0.35` and `0.50` gaps are strongly significant on both metrics.

This means MSC is not merely neutral to the geometry codec. It is actively hostile to it.

## Scientific Interpretation

MSC is now a clean semantic-memory benchmark in the current project taxonomy.

It appears to reward:

- broad conversational topic continuity
- persona and preference persistence
- dense recall of prior dialogue content

more than:

- sparse preservation of a few support-critical turns
- exact constraint rescue
- local support-turn compression

That is why:

- `semantic` is strongest overall
- `geometry` remains a useful but weaker fallback
- `geometry_keep_compress_drop` is the wrong family
- `semantic_keep_compress_drop` recovers most of the geometry-KCD failure, but still does not surpass plain semantic retention

## What Is Settled

This run settles several questions.

### Settled positive findings

- MSC strongly validates semantic signal.
- Geometry is still useful relative to uniform.
- Replacing geometry signal with semantic signal improves the codec dramatically.

### Settled negative findings

- Geometry-KCD is not a general-purpose conversational-memory codec.
- Sparse codec form is not universally beneficial once the signal is fixed.
- On MSC, plain semantic retention is better than semantic sparse codec at `0.35` and `0.50`.

## Updated Paper 3 Reading

Paper 3 now has three benchmark-dependent regimes:

1. hard support-turn stress sets
   - `geometry_keep_compress_drop` is strong
2. broader public semantic-memory benchmarks such as LongMemEval
   - semantic and segment-style policies are strong
   - KCD becomes competitive only later
3. MSC-style multi-session conversational memory
   - plain semantic retention is the lead family
   - sparse semantic codec is at best competitive only under the tightest scarcity

## Immediate Follow-On

The next decisive run is:

1. `locomo10` with:
   - `uniform`
   - `semantic`
   - `geometry`
   - `geometry_keep_compress_drop`
   - `semantic_keep_compress_drop`

That run will determine whether MSC is representative of the broader semantic-memory regime, or whether MSC is unusually hostile to sparse codec structure.
