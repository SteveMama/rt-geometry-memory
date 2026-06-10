# Supervised Regime Detector

Conversations: 46 across 3 regimes (hardset, longmemeval, msc).

- 5-fold CV accuracy: **1.000** (majority baseline 0.696)
- macro F1: **1.000**

## Confusion matrix (rows = true, cols = predicted)

| | hardset | longmemeval | msc |
|---|---|---|---|
| hardset | 2 | 0 | 0 |
| longmemeval | 0 | 12 | 0 |
| msc | 0 | 0 | 32 |

## Strongest features per regime

### hardset
- `segment_mean_stabilized_curvature__std`: -0.131
- `segment_mean_step_norm__std`: -0.128
- `query_geom_v2_risk__max`: -0.126
- `query_geom_v2_curvature__max`: -0.125
- `segment_rank95__max`: -0.124
- `query_geom_v2_energy__max`: -0.124
- `segment_mean_stabilized_curvature__max`: -0.122
- `user_turn_fraction`: +0.120

### longmemeval
- `token_cost__mean`: +0.494
- `constraint_score__max`: +0.370
- `token_cost__std`: +0.337
- `constraint_score__std`: +0.331
- `token_cost__max`: +0.284
- `geometry_score__mean`: -0.273
- `constraint_score__mean`: +0.262
- `support_score__max`: -0.236

### msc
- `token_cost__mean`: -0.462
- `constraint_score__max`: -0.371
- `token_cost__std`: -0.348
- `constraint_score__std`: -0.347
- `constraint_score__mean`: -0.341
- `geometry_score__mean`: +0.313
- `token_cost__max`: -0.294
- `support_score__max`: +0.241

## Reading

If CV accuracy clearly beats the majority baseline, the paper can claim a
working supervised regime selector (replacing the failed unsupervised
atlas) and the 'signal-conditioned' title is earned. If it does not, add
this as an explicit limitation and consider retitling.