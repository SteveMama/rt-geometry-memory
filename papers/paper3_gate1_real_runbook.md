# Paper 3 Gate 1 Real Runbook

This note defines the first real decision run for the geometry-versus-semantics gap.

## Goal

Answer one narrow question on real semantic-memory benchmarks:

> Inside a semantic shortlist, does geometry refinement improve over semantic-only compression?

This is the first real Gate 1 decision surface. It replaces toy smokes with a focused public-benchmark run.

## Benchmarks

Run exactly these:

- `MSC valid`
- `LongMemEval-S cleaned`

Use the hard stress set only later as a diagnostic backstop, not as the primary semantic-gap decision benchmark.

## Policy Set

Run exactly these four policies:

- `semantic`
- `budget_aware_semantic_keep_compress_drop`
- `semantic_ambient_geometry_keep_compress_drop`
- `semantic_query_conditioned_geometry_keep_compress_drop`

Why this set:

- `semantic` is the plain semantic baseline
- `budget_aware_semantic_keep_compress_drop` is the strongest current semantic codec incumbent
- `semantic_ambient_geometry_keep_compress_drop` tests whether any geometry refinement helps inside the same semantic shortlist
- `semantic_query_conditioned_geometry_keep_compress_drop` tests whether query-conditioned geometry improves over ambient geometry

## Gate 1 Oracle Question

The oracle layer should answer:

> Within a semantic shortlist, do structural features add ranking value over semantic-only?

Use:

- `semantic_score`
- `query_geom_v2_risk`
- `combined_structural_score`

Primary oracle criteria:

- `Δ Kendall tau >= +0.03`, or
- `Δ top-5 oracle recall >= +5 percentage points`

This must happen on at least one of:

- `MSC valid`
- `LongMemEval-S cleaned`

If not, geometry refinement remains diagnostic only on semantic-memory benchmarks.

## Gate 1 Policy Question

The refinement study should answer:

> Does query-conditioned geometry refinement beat the best semantic codec incumbent on a real benchmark?

Primary comparison:

- `semantic_query_conditioned_geometry_keep_compress_drop`
  vs
- `budget_aware_semantic_keep_compress_drop`

Secondary comparison:

- `semantic_query_conditioned_geometry_keep_compress_drop`
  vs
- `semantic_ambient_geometry_keep_compress_drop`

Success condition:

- wins on at least one of `MSC valid` or `LongMemEval-S cleaned`
- preferably at `0.20` or `0.35`
- should improve logit distortion and ideally not regress on answer NLL
- use both row-level and conversation-level significance summaries

If the hybrid ties or loses across both benchmarks, stop hand-designed geometry refinement and move to the learned harm predictor.

## Execution Surface

Focused refinement runner:

- [run_paper3_gate1_refinement_probe.sh](/Users/pranav/Documents/RT/scripts/run_paper3_gate1_refinement_probe.sh)

Full two-benchmark wrapper:

- [run_paper3_gate1_real.sh](/Users/pranav/Documents/RT/scripts/run_paper3_gate1_real.sh)

Existing oracle runner:

- [run_paper3_harm_oracle_probe.sh](/Users/pranav/Documents/RT/scripts/run_paper3_harm_oracle_probe.sh)

## Commands

Run the full Gate 1 decision surface with one command:

```bash
bash scripts/run_paper3_gate1_real.sh \
  /path/to/msc_valid_normalized.jsonl \
  /path/to/longmemeval_s_cleaned_normalized.jsonl \
  qwen25_15b \
  0.20,0.35,0.50 \
  24 \
  24 \
  4 \
  16
```

Run only the focused refinement study on one benchmark:

```bash
bash scripts/run_paper3_gate1_refinement_probe.sh \
  paper3_gate1_refinement_msc_valid \
  /path/to/msc_valid_normalized.jsonl \
  qwen25_15b \
  0.20,0.35,0.50 \
  24 \
  4 \
  16
```

## Outputs

Oracle outputs:

- `results/paper3/harm_oracle/paper3_gate1_oracle_msc_valid`
- `results/paper3/harm_oracle/paper3_gate1_oracle_longmemeval_s_cleaned`

Refinement study outputs:

- `results/paper3/studies/paper3_gate1_refinement_msc_valid`
- `results/paper3/studies/paper3_gate1_refinement_longmemeval_s_cleaned`

## Local Status

The focused Gate 1 run surface is implemented.

At implementation time, the required benchmark JSONLs were not present on this local machine under the project tree or common local dataset paths, so no canonical MSC/LongMemEval Gate 1 result is recorded in this note yet.
