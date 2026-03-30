# Paper 3 Semantic KCD Optimization Checkpoint

This note records the next optimization pass after the first low-budget KCD upgrade checkpoint.

The question for this phase was:

> Can semantic-led KCD be improved further so that it more reliably beats the existing semantic and hybrid baselines on semantic-memory benchmarks?

## What was implemented

Two new semantic-led codec policies were added.

### 1. `support_aware_semantic_keep_compress_drop`

This policy keeps semantic signal as the primary importance score, but adds the same support-aware bias that improved geometry-KCD:

- user-turn bonus
- latest-user bonus
- constraint-marker bonus
- support-aware compressed candidate selection inside each segment

Operationally, it is:

- semantic-led
- support-aware
- still sparse keep/compress/drop

### 2. `budget_aware_semantic_keep_compress_drop`

This policy adds budget-conditioned semantic filtering and budget-conditioned segment size:

- low budget:
  - tighter semantic shortlist
  - finer segmentation
- mid budget:
  - moderate shortlist
  - medium segmentation
- higher budget:
  - broader shortlist
  - coarser segmentation

This is the current closest implementation of a budget-conditioned semantic codec without training a separate learned harm model.

## New execution entry points

- runner:
  - [`scripts/run_paper3_semantic_kcd_optimization.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_semantic_kcd_optimization.sh)
- notebook:
  - [`notebooks/rt_paper3_semantic_kcd_optimization_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_semantic_kcd_optimization_runner.ipynb)

## Smoke-run setup

These were bounded engineering runs, not publication-scale studies.

- model: `qwen25_05b`
- budgets: `0.20`, `0.35`
- conversations per benchmark: `2`
- target-turn stride: `6`
- max target turns per conversation: `6`

Studies:

- [`paper3_semantic_kcd_opt_smoke_msc`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_semantic_kcd_opt_smoke_msc)
- [`paper3_semantic_kcd_opt_smoke_locomo`](/Users/pranav/Documents/RT/results/paper3/studies/paper3_semantic_kcd_opt_smoke_locomo)

## MSC smoke result

### Main read

The new semantic-specific variants are viable, but they do not cleanly beat the strongest existing semantic baselines.

At `0.20`:

- `support_aware_semantic_keep_compress_drop`: `Δ logit L2 = -504.228`
- `budget_aware_semantic_keep_compress_drop`: `-467.766`
- `semantic_keep_compress_drop`: `-479.531`
- `semantic`: `-433.905`

At `0.35`:

- `semantic_keep_compress_drop`: `-645.714`
- `semantic`: `-608.159`
- `semantic_filtered_geometry_keep_compress_drop`: `-604.714`
- `budget_aware_semantic_keep_compress_drop`: `-588.882`
- `support_aware_semantic_keep_compress_drop`: `-572.753`

### Interpretation

On this bounded MSC slice:

- `support_aware_semantic_keep_compress_drop` is competitive
- `budget_aware_semantic_keep_compress_drop` is competitive
- but neither one clearly displaces `semantic_keep_compress_drop`

So for MSC, the semantic-led codec story remains:

> semantic KCD is real, but extra hand-designed support-aware/budget-aware structure has not yet created a decisive new winner over the existing semantic codec variants.

## LoCoMo smoke result

### Main read

LoCoMo gives a sharper split.

At `0.20`:

- `budget_aware_semantic_keep_compress_drop`: `Δ logit L2 = -1146.192`
- `semantic`: `-1114.413`
- `support_aware_semantic_keep_compress_drop`: `-1102.248`
- `semantic_keep_compress_drop`: `-1058.737`

At `0.35`:

- `semantic_filtered_geometry_keep_compress_drop`: `-153.588`
- `semantic`: `-89.849`
- `budget_aware_semantic_keep_compress_drop`: `-89.379`
- `support_aware_geometry_keep_compress_drop`: `-47.430`
- `support_aware_semantic_keep_compress_drop`: `-42.299`
- `semantic_keep_compress_drop`: `-37.937`

### Interpretation

On this bounded LoCoMo slice:

- the budget-aware semantic codec is strongest at the tightest budget
- but it does not hold that lead at the mid budget
- the best mid-budget policy remains `semantic_filtered_geometry_keep_compress_drop`

So the semantic-specific upgrades help, but they do not yet dominate the whole semantic-memory regime.

## What is now settled

1. The semantic-led codec family is the correct Paper 3 direction for semantic-memory benchmarks.
2. Support-awareness helps geometry-side codecs a lot.
3. Support-awareness and budget-conditioning can be added to semantic KCD without breaking it.
4. But those additions alone do not yet produce a stable semantic-KCD winner across MSC and LoCoMo.

## What remains to beat the current baselines

The remaining gains are unlikely to come from another small hand-tuned score mix. The next serious improvements are:

### 1. Learned harm predictor

Replace hand-set mixtures with a direct harm model:

- geometry features
- semantic relevance
- role features
- support-turn indicators
- constraint markers
- recency

Target:

- observed logit damage
- answer-NLL damage
- support-turn failure

### 2. Denser compressed memory object

The current codec still mostly keeps selected raw turns. The next upgrade should preserve more structure per retained slot:

- anchor turn
- latest support turn
- constraint slots
- entities / facts
- order / role metadata

### 3. Policy selector over benchmark regimes

At this point, the data already suggests the system may need a selector:

- semantic-led codec for semantic-memory benchmarks
- support-aware geometry codec for support-turn-critical benchmarks

That selector can be driven by pairwise superiority data already being collected.

## Recommended next experimental order

1. Run publication-scale MSC with:
   - `semantic`
   - `semantic_keep_compress_drop`
   - `support_aware_semantic_keep_compress_drop`
   - `budget_aware_semantic_keep_compress_drop`
   - `semantic_filtered_geometry_keep_compress_drop`
2. Run publication-scale LoCoMo with the same set.
3. If none of the new semantic variants clearly dominates, stop adding hand-designed policy variants and build the learned harm predictor.

## Bottom line

The current optimization pass says:

> semantic-led KCD is still the right family for semantic-memory benchmarks, but the next big gain is unlikely to come from another heuristic mixture. The remaining path to beating the strongest baselines is a learned harm signal plus a denser compressed memory object.
