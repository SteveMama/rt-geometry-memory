# Paper 3 Harm Predictor Checkpoint

This note defines the Gate 3 surface for the learned-harm deployment policy.

## Goal

Train a lightweight harm predictor on oracle ablation labels and test whether it stabilizes the semantic-first hybrid enough to beat the best semantic incumbent on a public semantic-memory benchmark.

## Implemented Surface

- runner:
  - [run_paper3_harm_predictor_probe.sh](/Users/pranav/Documents/RT/scripts/run_paper3_harm_predictor_probe.sh)
- notebook:
  - [rt_paper3_harm_predictor_runner.ipynb](/Users/pranav/Documents/RT/notebooks/rt_paper3_harm_predictor_runner.ipynb)
- predictor implementation:
  - [harm_predictor.py](/Users/pranav/Documents/RT/paper3_codec/harm_predictor.py)

## New Public Policy

- `semantic_harm_keep_compress_drop`

## Gate 3

Promote the learned-harm policy only if it:

- beats the heuristic hybrid on oracle ranking quality
- beats the best semantic incumbent on at least one of `MSC valid` or `LongMemEval-S cleaned` at `0.20` or `0.35`
- does not materially regress on the hard stress set

If it only improves oracle ranking but not benchmark outcomes, it should not become the main method.

## Status

Implemented. No canonical promoted learned-harm artifact is recorded in this note yet.
