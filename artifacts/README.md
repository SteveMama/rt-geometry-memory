# Frozen Artifact Bundle

This directory contains the Git-tracked experiment bundle for the repo.

Included:

- `paper1/expanded_v8_final`
  - frozen Paper 1 summaries, plots, and rerun manifest
- `paper2/blazing_study_v3_confidence`
  - Paper 2 mid-budget confidence study
- `paper2/behavior_stress_v1`
  - hard stress-set study with behavior metrics
- `paper2/behavior_stress_qwen_cases`
  - qualitative and memory-critical explanation reports
- `paper3/smoke_paper3_qwen25`
  - minimal Paper 3 sparse-memory smoke run
- `paper3/paper3_pilot_v3_full`
  - current full Paper 3 checkpoint with significance, behavior, and memory-critical reports
- `paper3/paper3_batch_v1_fairness`
  - fairness-controlled Paper 3 sweep with pairwise comparisons and mechanism reports
- `paper3/paper3_batch_v1_3b`
  - first 3B Paper 3 probe with pairwise comparisons and mechanism reports
- `shareable`
  - checkpoint summary figures ready to share externally

The full scratch `results/` tree is intentionally not tracked. Reproduce fresh outputs with the scripts in `/Users/pranav/Documents/RT/scripts`.
