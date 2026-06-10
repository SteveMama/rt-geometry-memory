# Answer-Harm Gate 1 Reanalysis: longmemeval_s_cleaned

Side-by-side Gate 1 ranking value under logit-defined vs answer-NLL-defined
oracle harm. The circularity concern is resolved in favor of the paper only
if the geometry/hybrid delta-tau survives under **answer_harm**.

## View: overall

### Harm target: answer_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | 0.1214 | baseline | — |
| 0.2 | geometry_score | 0.0022 | -0.1192 (p=0.5633 / p=1.0000) | -0.0125 |
| 0.2 | query_geom_v2_risk | -0.0084 | -0.1297 (p=0.5130 / p=0.9477) | +0.0500 |
| 0.2 | combined_structural_score | 0.1943 | +0.0729 (p=0.8405 / p=0.8215) | +0.0250 |
| 0.35 | semantic_score | 0.1214 | baseline | — |
| 0.35 | geometry_score | 0.0022 | -0.1192 (p=0.5727 / p=1.0000) | -0.0125 |
| 0.35 | query_geom_v2_risk | -0.0315 | -0.1529 (p=0.4765 / p=0.9420) | +0.0125 |
| 0.35 | combined_structural_score | 0.1753 | +0.0540 (p=0.9073 / p=0.8888) | +0.0000 |
| 0.5 | semantic_score | 0.1214 | baseline | — |
| 0.5 | geometry_score | 0.0022 | -0.1192 (p=0.5745 / p=1.0000) | -0.0125 |
| 0.5 | query_geom_v2_risk | -0.0218 | -0.1431 (p=0.5008 / p=0.9520) | +0.0250 |
| 0.5 | combined_structural_score | -0.0181 | -0.1395 (p=0.5102 / p=0.9650) | +0.0250 |

### Harm target: logit_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | -0.4465 | baseline | — |
| 0.2 | geometry_score | 0.3739 | +0.8205 (p=0.0000 / p=0.0018) | +0.1352 |
| 0.2 | query_geom_v2_risk | 0.3822 | +0.8287 (p=0.0000 / p=0.0013) | +0.1019 |
| 0.2 | combined_structural_score | 0.3244 | +0.7709 (p=0.0000 / p=0.0005) | +0.0963 |
| 0.35 | semantic_score | -0.4458 | baseline | — |
| 0.35 | geometry_score | 0.3721 | +0.8179 (p=0.0000 / p=0.0000) | +0.1315 |
| 0.35 | query_geom_v2_risk | 0.2016 | +0.6475 (p=0.0000 / p=0.0013) | +0.0574 |
| 0.35 | combined_structural_score | 0.1164 | +0.5622 (p=0.0000 / p=0.0005) | +0.0148 |
| 0.5 | semantic_score | -0.4423 | baseline | — |
| 0.5 | geometry_score | 0.3707 | +0.8130 (p=0.0000 / p=0.0005) | +0.1389 |
| 0.5 | query_geom_v2_risk | 0.3534 | +0.7957 (p=0.0000 / p=0.0005) | +0.0981 |
| 0.5 | combined_structural_score | 0.2531 | +0.6954 (p=0.0000 / p=0.0000) | +0.1019 |

## View: semantic_shortlist

### Harm target: answer_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | 0.1206 | baseline | — |
| 0.2 | geometry_score | 0.0020 | -0.1186 (p=0.5537 / p=1.0000) | -0.0125 |
| 0.2 | query_geom_v2_risk | -0.0083 | -0.1289 (p=0.5085 / p=0.9683) | +0.0500 |
| 0.2 | combined_structural_score | 0.1943 | +0.0737 (p=0.8385 / p=0.8327) | +0.0250 |
| 0.35 | semantic_score | 0.1206 | baseline | — |
| 0.35 | geometry_score | 0.0020 | -0.1186 (p=0.5763 / p=1.0000) | -0.0125 |
| 0.35 | query_geom_v2_risk | -0.0313 | -0.1519 (p=0.4718 / p=0.9367) | +0.0125 |
| 0.35 | combined_structural_score | 0.1752 | +0.0546 (p=0.9070 / p=0.8640) | +0.0000 |
| 0.5 | semantic_score | 0.1206 | baseline | — |
| 0.5 | geometry_score | 0.0020 | -0.1186 (p=0.5715 / p=1.0000) | -0.0125 |
| 0.5 | query_geom_v2_risk | -0.0215 | -0.1421 (p=0.5230 / p=0.9493) | +0.0250 |
| 0.5 | combined_structural_score | -0.0183 | -0.1388 (p=0.5190 / p=0.9640) | +0.0250 |

### Harm target: logit_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | -0.3889 | baseline | — |
| 0.2 | geometry_score | 0.3066 | +0.6955 (p=0.0000 / p=0.0005) | +0.1128 |
| 0.2 | query_geom_v2_risk | 0.3355 | +0.7245 (p=0.0000 / p=0.0005) | +0.1362 |
| 0.2 | combined_structural_score | 0.3484 | +0.7373 (p=0.0000 / p=0.0005) | +0.0979 |
| 0.35 | semantic_score | -0.3811 | baseline | — |
| 0.35 | geometry_score | 0.3024 | +0.6835 (p=0.0000 / p=0.0005) | +0.1074 |
| 0.35 | query_geom_v2_risk | 0.1076 | +0.4888 (p=0.0000 / p=0.0000) | -0.0021 |
| 0.35 | combined_structural_score | 0.1021 | +0.4833 (p=0.0000 / p=0.0003) | -0.0421 |
| 0.5 | semantic_score | -0.3883 | baseline | — |
| 0.5 | geometry_score | 0.3060 | +0.6943 (p=0.0000 / p=0.0003) | +0.1170 |
| 0.5 | query_geom_v2_risk | 0.3373 | +0.7257 (p=0.0000 / p=0.0008) | +0.1298 |
| 0.5 | combined_structural_score | 0.3081 | +0.6964 (p=0.0000 / p=0.0000) | +0.0787 |

## Gate threshold sensitivity (overall view)

Runbook requirement: the +0.03 tau cutoff is heuristic and must be shown stable under perturbation.

### answer_harm

| threshold | passes | passing features |
|---|---|---|
| tau>=0.01 | True | combined_structural_score |
| tau>=0.02 | True | combined_structural_score |
| tau>=0.03 | True | combined_structural_score |
| tau>=0.05 | True | combined_structural_score |
| tau>=0.08 | False | — |
| tau>=0.10 | False | — |

### logit_harm

| threshold | passes | passing features |
|---|---|---|
| tau>=0.01 | True | combined_structural_score, geometry_score, query_geom_v2_risk |
| tau>=0.02 | True | combined_structural_score, geometry_score, query_geom_v2_risk |
| tau>=0.03 | True | combined_structural_score, geometry_score, query_geom_v2_risk |
| tau>=0.05 | True | combined_structural_score, geometry_score, query_geom_v2_risk |
| tau>=0.08 | True | combined_structural_score, geometry_score, query_geom_v2_risk |
| tau>=0.10 | True | combined_structural_score, geometry_score, query_geom_v2_risk |
