# Paper 3 Public Benchmark Checkpoint

## Run

- study: `paper3_public_v1_public_benchmark`
- benchmark: normalized public benchmark JSONL derived from `LongMemEval-S cleaned`
- model: `qwen25_15b`
- conversations: `25`
- sampled target turns: `800`
- evaluation rows: `12000`
- behavior rows: `6525`
- policies:
  - `uniform`
  - `semantic`
  - `geometry`
  - `geometry_segment_actions`
  - `geometry_keep_compress_drop`
- budgets:
  - `0.20`
  - `0.35`
  - `0.50`

Tracked outputs:

- [`study_report.md`](/Users/pranav/Documents/RT/paper3/paper3_public_v1_public_benchmark/study_report.md)
- [`pairwise_report.md`](/Users/pranav/Documents/RT/paper3/paper3_public_v1_public_benchmark/pairwise_report.md)
- [`study_summary.json`](/Users/pranav/Documents/RT/paper3/paper3_public_v1_public_benchmark/study_summary.json)
- [`significance_summary.json`](/Users/pranav/Documents/RT/paper3/paper3_public_v1_public_benchmark/significance_summary.json)
- [`behavior_significance_summary.json`](/Users/pranav/Documents/RT/paper3/paper3_public_v1_public_benchmark/behavior_significance_summary.json)

## Main Reading

This public-benchmark result is strong enough to matter and different enough to change the policy story.

The hard stress set suggested:

- low/mid budgets favor `geometry_keep_compress_drop`
- higher budgets often favor plain `geometry`

This public benchmark instead shows:

- low budget (`0.20`): `semantic` is strongest
- mid budget (`0.35`): `semantic` and `geometry_segment_actions` are strongest
- higher budget (`0.50`): `geometry_keep_compress_drop` becomes strongest

So the benchmark is not saying the geometry program fails. It is saying the policy ranking depends on what kind of memory problem the benchmark emphasizes.

## Logit Results

Negative deltas are improvements over `uniform`.

### Budget `0.20`

- `semantic`: `Δ logit L2 = -122.613`, relative `0.932`, `p=0.0000`
- `geometry`: `-61.574`, relative `0.966`, `p=0.0000`
- `geometry_segment_actions`: `-52.642`, relative `0.971`, `p=0.0000`
- `geometry_keep_compress_drop`: `-26.258`, relative `0.985`, `p=0.0307`

### Budget `0.35`

- `semantic`: `Δ logit L2 = -104.183`, relative `0.943`, `p=0.0000`
- `geometry_segment_actions`: `-97.432`, relative `0.946`, `p=0.0000`
- `geometry`: `-53.148`, relative `0.971`, `p=0.0000`
- `geometry_keep_compress_drop`: `-42.612`, relative `0.977`, `p=0.0003`

### Budget `0.50`

- `geometry_keep_compress_drop`: `Δ logit L2 = -48.466`, relative `0.972`, `p=0.0000`
- `geometry_segment_actions`: `-34.799`, relative `0.980`, `p=0.0013`
- `semantic`: `+12.164`, relative `1.007`, `p=0.3350`
- `geometry`: `+31.038`, relative `1.018`, `p=0.0032`

## Pairwise Results

### Budget `0.20`

- `KCD vs geometry`: `+35.316`, `p=0.0020`
- `KCD vs segment_actions`: `+26.384`, `p=0.0152`

So KCD is significantly worse than both geometry and segment-actions at the low budget.

### Budget `0.35`

- `KCD vs geometry`: `+10.536`, `p=0.3583`
- `KCD vs segment_actions`: `+54.820`, `p=0.0000`
- `segment_actions vs geometry`: `-44.284`, `p=0.0005`

So at the mid budget:

- `segment_actions` significantly beats `geometry`
- `segment_actions` strongly beats `KCD`
- `KCD` is not the leading public-benchmark policy here

### Budget `0.50`

- `KCD vs geometry`: `-79.504`, `p=0.0000`
- `KCD vs segment_actions`: `-13.667`, `p=0.1998`
- `segment_actions vs geometry`: `-65.837`, `p=0.0000`

So at the higher budget:

- KCD significantly beats geometry
- segment-actions also significantly beats geometry
- KCD is directionally better than segment-actions, though not significantly so

## Behavior Results

Behavior is informative but does not crown KCD.

### Budget `0.20`

- `semantic`: `Δ answer NLL = -0.0625`, `p=0.0000`
- `geometry`: `-0.0521`, `p=0.0000`
- `geometry_segment_actions`: `-0.0411`, `p=0.0030`
- `KCD`: `-0.0177`, `p=0.2447`

### Budget `0.35`

- `geometry`: `-0.0385`, `p=0.0000`
- `semantic`: `-0.0323`, `p=0.0020`
- `geometry_segment_actions`: `-0.0321`, `p=0.0015`
- `KCD`: `-0.0084`, `p=0.5555`

### Budget `0.50`

All methods are near-flat on behavior; no policy is a strong answer-level winner.

## Interpretation

This benchmark appears to reward broader semantic retrieval and episode relevance more strongly than the support-turn-faithful sparse compression behavior that made KCD shine on the hard stress set.

That leads to the current split:

- hard stress set: KCD is strongest under scarcity
- LongMemEval public benchmark: semantic and segment-style policies are strongest under scarcity, while KCD becomes strongest only later

This is not a contradiction. It is evidence that different conversational-memory benchmarks emphasize different failure modes.

## Practical Conclusion

The public benchmark strengthens the overall project, but changes the policy story:

- it confirms that geometry-family policies are real outside the custom hard set
- it does not support KCD as the universal low/mid-budget winner
- it makes `semantic` a stronger competitor than before
- it suggests that future codec work should distinguish between:
  - semantic/episode memory preservation
  - support-turn-faithful sparse compression

## Immediate Follow-On

The next fast-iteration benchmarks should therefore be smaller and quicker:

1. `MSC`
2. `LoCoMo`
3. `GapChat` or `REALTALK`

with the first quick policy set kept narrow:

- `uniform`
- `semantic`
- `geometry`
- `geometry_keep_compress_drop`
