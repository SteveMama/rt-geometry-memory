# Paper 3 Artifacts

Tracked checkpoint bundle for Paper 3.

Included:

- `smoke_paper3_qwen25`
  - the first minimal smoke run that verified the pilot path
- `paper3_pilot_v3_full`
  - the current main Paper 3 checkpoint
  - full study summary and report
  - confidence and significance summaries
  - behavior summaries
  - per-model summaries
  - memory-critical reports for both compressed-memory policy families
- `paper3_batch_v1_fairness`
  - fairness-controlled Paper 3 sweep on `qwen25_15b`
  - direct pairwise policy significance
  - memory-critical reports for both policy families
- `paper3_batch_v1_3b`
  - first 3B Paper 3 probe on `qwen25_3b`
  - direct pairwise policy significance
  - 3B memory-critical reports for both policy families

Current reading:

- `geometry_keep_compress_drop` is now the strongest low-to-mid budget Paper 3 family under fairness control
- plain `geometry` becomes the strongest high-budget logit policy on the 3B probe
- `geometry_segment_actions` remains viable but is no longer the lead low/mid-budget family on the Qwen fairness-controlled runs
- the project is still in-flight; this is a checkpoint, not a terminal result
