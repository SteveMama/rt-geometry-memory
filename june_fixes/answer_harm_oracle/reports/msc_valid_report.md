# Answer-Harm Gate 1 Reanalysis: msc_valid

Side-by-side Gate 1 ranking value under logit-defined vs answer-NLL-defined
oracle harm. The circularity concern is resolved in favor of the paper only
if the geometry/hybrid delta-tau survives under **answer_harm**.

## View: overall

### Harm target: answer_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | -0.0054 | baseline | — |
| 0.2 | geometry_score | 0.0077 | +0.0131 (p=0.7143 / p=0.5717) | -0.0135 |
| 0.2 | query_geom_v2_risk | 0.0097 | +0.0151 (p=0.6915 / p=0.4688) | +0.0000 |
| 0.2 | combined_structural_score | -0.0063 | -0.0009 (p=0.9688 / p=0.7927) | -0.0104 |
| 0.35 | semantic_score | -0.0037 | baseline | — |
| 0.35 | geometry_score | 0.0073 | +0.0110 (p=0.7590 / p=0.5905) | -0.0135 |
| 0.35 | query_geom_v2_risk | 0.0241 | +0.0279 (p=0.4537 / p=0.2940) | +0.0135 |
| 0.35 | combined_structural_score | -0.0025 | +0.0012 (p=0.9593 / p=0.7930) | -0.0031 |
| 0.5 | semantic_score | -0.0048 | baseline | — |
| 0.5 | geometry_score | 0.0091 | +0.0139 (p=0.6927 / p=0.5475) | -0.0125 |
| 0.5 | query_geom_v2_risk | 0.0232 | +0.0280 (p=0.4562 / p=0.3947) | +0.0375 |
| 0.5 | combined_structural_score | 0.0274 | +0.0322 (p=0.3610 / p=0.3757) | +0.0187 |

### Harm target: logit_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | -0.2148 | baseline | — |
| 0.2 | geometry_score | 0.1711 | +0.3859 (p=0.0000 / p=0.0070) | +0.0518 |
| 0.2 | query_geom_v2_risk | 0.1354 | +0.3502 (p=0.0000 / p=0.0075) | +0.0631 |
| 0.2 | combined_structural_score | 0.1048 | +0.3196 (p=0.0000 / p=0.0000) | +0.0321 |
| 0.35 | semantic_score | -0.2188 | baseline | — |
| 0.35 | geometry_score | 0.1731 | +0.3920 (p=0.0000 / p=0.0075) | +0.0577 |
| 0.35 | query_geom_v2_risk | 0.1339 | +0.3528 (p=0.0000 / p=0.0030) | +0.0690 |
| 0.35 | combined_structural_score | 0.0586 | +0.2774 (p=0.0000 / p=0.0008) | +0.0190 |
| 0.5 | semantic_score | -0.2158 | baseline | — |
| 0.5 | geometry_score | 0.1676 | +0.3834 (p=0.0000 / p=0.0083) | +0.0595 |
| 0.5 | query_geom_v2_risk | 0.1637 | +0.3795 (p=0.0000 / p=0.0040) | +0.0619 |
| 0.5 | combined_structural_score | 0.0722 | +0.2880 (p=0.0000 / p=0.0283) | +0.0464 |

## View: semantic_shortlist

### Harm target: answer_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | -0.0177 | baseline | — |
| 0.2 | geometry_score | 0.0086 | +0.0262 (p=0.4650 / p=0.4000) | -0.0052 |
| 0.2 | query_geom_v2_risk | 0.0079 | +0.0255 (p=0.4945 / p=0.3663) | -0.0031 |
| 0.2 | combined_structural_score | -0.0125 | +0.0052 (p=0.8448 / p=0.6635) | -0.0084 |
| 0.35 | semantic_score | -0.0152 | baseline | — |
| 0.35 | geometry_score | 0.0071 | +0.0224 (p=0.5170 / p=0.4417) | -0.0042 |
| 0.35 | query_geom_v2_risk | 0.0208 | +0.0361 (p=0.3242 / p=0.2293) | +0.0167 |
| 0.35 | combined_structural_score | -0.0106 | +0.0046 (p=0.8630 / p=0.7290) | -0.0010 |
| 0.5 | semantic_score | -0.0141 | baseline | — |
| 0.5 | geometry_score | 0.0090 | +0.0231 (p=0.5238 / p=0.4447) | -0.0083 |
| 0.5 | query_geom_v2_risk | 0.0163 | +0.0304 (p=0.4088 / p=0.3458) | +0.0354 |
| 0.5 | combined_structural_score | 0.0183 | +0.0324 (p=0.3800 / p=0.3565) | +0.0177 |

### Harm target: logit_harm

| budget | feature | mean tau | Δtau vs semantic (row p / conv p) | Δtop5 |
|---|---|---|---|---|
| 0.2 | semantic_score | -0.1815 | baseline | — |
| 0.2 | geometry_score | 0.1470 | +0.3285 (p=0.0000 / p=0.0328) | +0.1457 |
| 0.2 | query_geom_v2_risk | 0.1354 | +0.3169 (p=0.0000 / p=0.0375) | +0.1396 |
| 0.2 | combined_structural_score | 0.1443 | +0.3258 (p=0.0000 / p=0.0005) | +0.1232 |
| 0.35 | semantic_score | -0.1904 | baseline | — |
| 0.35 | geometry_score | 0.1484 | +0.3387 (p=0.0000 / p=0.0243) | +0.1488 |
| 0.35 | query_geom_v2_risk | 0.1267 | +0.3170 (p=0.0000 / p=0.0168) | +0.1049 |
| 0.35 | combined_structural_score | 0.1218 | +0.3122 (p=0.0000 / p=0.0008) | +0.0628 |
| 0.5 | semantic_score | -0.1980 | baseline | — |
| 0.5 | geometry_score | 0.1605 | +0.3585 (p=0.0000 / p=0.0192) | +0.1378 |
| 0.5 | query_geom_v2_risk | 0.1930 | +0.3910 (p=0.0000 / p=0.0057) | +0.1720 |
| 0.5 | combined_structural_score | 0.1666 | +0.3646 (p=0.0000 / p=0.0118) | +0.1323 |

## Gate threshold sensitivity (overall view)

Runbook requirement: the +0.03 tau cutoff is heuristic and must be shown stable under perturbation.

### answer_harm

| threshold | passes | passing features |
|---|---|---|
| tau>=0.01 | True | combined_structural_score, geometry_score, query_geom_v2_risk |
| tau>=0.02 | True | combined_structural_score, query_geom_v2_risk |
| tau>=0.03 | True | combined_structural_score |
| tau>=0.05 | False | — |
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
