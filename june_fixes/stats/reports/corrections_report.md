# Multiple-Comparison Correction Report

Tests collected: 1273 across 69 families (family = study x summary kind x test level).

| view | significant at alpha=0.05 |
|---|---|
| raw p | 283 |
| BH within family | 116 |
| Holm within family | 61 |

## Findings killed by BH correction (167)

These are reported as significant in the manuscript pipeline but do not
survive within-family FDR control. Each must be either re-run with more
data, downgraded to 'directional', or dropped from the claims.

| source | test | raw p | BH q | Holm p |
|---|---|---|---|---|
| behavior_stress_v1 | smollm2_17b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0035 | 0.0900 | 0.3150 |
| behavior_stress_v1 | smollm2_17b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0035 | 0.0900 | 0.3150 |
| paper3_batch_v1_fairness | qwen25_15b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0040 | 0.0960 | 0.0960 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_logit_l2 | 0.0043 | 0.0510 | 0.0510 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_logit_l2 | 0.0043 | 0.0510 | 0.0510 |
| paper3_pilot_v3_full | qwen25_15b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0047 | 0.0675 | 0.2565 |
| paper3_pilot_v3_full | qwen25_15b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0047 | 0.0675 | 0.2565 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0050 | 0.0675 | 0.2600 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0050 | 0.0675 | 0.2600 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0053 | 0.0788 | 0.1575 |
| behavior_stress_v1 | qwen25_05b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0053 | 0.0900 | 0.4620 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0053 | 0.0788 | 0.1575 |
| behavior_stress_v1 | qwen25_05b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0053 | 0.0900 | 0.4620 |
| expanded_v8_final | qwen25_15b/geometry_lexical_vs_lexical_only/auprc | 0.0058 | 0.1044 | 0.2088 |
| expanded_v8_final | qwen25_15b/geometry_lexical_vs_lexical_only/auprc | 0.0058 | 0.1044 | 0.2088 |
| behavior_stress_v1 | qwen25_15b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0060 | 0.0900 | 0.5160 |
| behavior_stress_v1 | qwen25_15b/0.50/lexical/delta_answer_avg_neg_logprob | 0.0060 | 0.0900 | 0.5160 |
| paper3_batch_v1_3b | qwen25_3b/0.35/geometry_keep_compress_drop/delta_logit_l2 | 0.0070 | 0.0630 | 0.0630 |
| blazing_study_v4_segment_behavior | smollm2_17b/0.35/geometry/delta_logit_l2 | 0.0077 | 0.1701 | 0.4185 |
| blazing_study_v3_confidence | qwen25_05b/0.50/geometry_lexical/delta_logit_l2 | 0.0088 | 0.2025 | 1.0000 |
| blazing_study_v4_segment_behavior | qwen25_05b/0.50/geometry_lexical/delta_logit_l2 | 0.0088 | 0.1701 | 0.4638 |
| bridge_smoke_qwen | qwen25_05b/0.50/geometry_lexical/delta_logit_l2 | 0.0088 | 0.0900 | 0.1575 |
| bridge_smoke_qwen_v2 | qwen25_05b/0.50/geometry_lexical/delta_logit_l2 | 0.0088 | 0.0900 | 0.1575 |
| blazing_study_v3_confidence | qwen25_05b/0.50/geometry_lexical/delta_logit_l2 | 0.0088 | 0.2025 | 1.0000 |
| behavior_stress_v1 | qwen25_15b/0.20/geometry_lexical/delta_logit_l2 | 0.0092 | 0.0540 | 0.8510 |
| behavior_stress_v1 | qwen25_15b/0.20/geometry_lexical/delta_logit_l2 | 0.0092 | 0.0540 | 0.8510 |
| paper3_batch_v1_fairness | qwen25_15b/0.46/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0095 | 0.1020 | 0.2185 |
| behavior_stress_v1 | qwen25_05b/0.35/lexical/delta_kl | 0.0100 | 0.0540 | 0.9000 |
| blazing_study_v3_confidence | qwen25_05b/0.50/lexical/delta_logit_l2 | 0.0100 | 0.2025 | 1.0000 |
| blazing_study_v4_segment_behavior | qwen25_05b/0.50/lexical/delta_logit_l2 | 0.0100 | 0.1701 | 0.5200 |
| bridge_smoke_qwen | qwen25_05b/0.50/lexical/delta_logit_l2 | 0.0100 | 0.0900 | 0.1700 |
| bridge_smoke_qwen_v2 | qwen25_05b/0.50/lexical/delta_logit_l2 | 0.0100 | 0.0900 | 0.1700 |
| behavior_stress_v1 | qwen25_05b/0.35/lexical/delta_kl | 0.0100 | 0.0540 | 0.9000 |
| blazing_study_v3_confidence | qwen25_05b/0.50/lexical/delta_logit_l2 | 0.0100 | 0.2025 | 1.0000 |
| blazing_study_v3_confidence | smollm2_17b/0.35/geometry/delta_logit_l2 | 0.0107 | 0.2025 | 1.0000 |
| blazing_study_v3_confidence | smollm2_17b/0.35/geometry/delta_logit_l2 | 0.0107 | 0.2025 | 1.0000 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0110 | 0.0660 | 0.1210 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0110 | 0.0660 | 0.1210 |
| behavior_stress_v1 | qwen25_05b/0.20/geometry_lexical/delta_kl | 0.0112 | 0.0552 | 0.9900 |
| blazing_study_v3_confidence | qwen25_15b/0.65/lexical/delta_logit_l2 | 0.0112 | 0.2025 | 1.0000 |
| behavior_stress_v1 | qwen25_05b/0.20/geometry_lexical/delta_kl | 0.0112 | 0.0552 | 0.9900 |
| blazing_study_v3_confidence | qwen25_15b/0.65/lexical/delta_logit_l2 | 0.0112 | 0.2025 | 1.0000 |
| paper3_batch_v1_fairness | qwen25_15b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0127 | 0.1020 | 0.2805 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0130 | 0.1160 | 0.2080 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0135 | 0.0855 | 0.1620 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0135 | 0.1305 | 0.1620 |
| behavior_stress_v1 | qwen25_15b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0140 | 0.1575 | 1.0000 |
| behavior_stress_v1 | qwen25_15b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0140 | 0.1575 | 1.0000 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.35/semantic_filtered_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0143 | 0.0855 | 0.1620 |
| blazing_study_v3_confidence | qwen25_15b/0.50/geometry_lexical/delta_logit_l2 | 0.0152 | 0.2100 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.50/geometry_lexical/delta_logit_l2 | 0.0152 | 0.2100 | 1.0000 |
| blazing_study_v4_segment_behavior | qwen25_15b/0.50/lexical/delta_logit_l2 | 0.0155 | 0.1701 | 0.7905 |
| blazing_study_v4_segment_behavior | qwen25_15b/0.50/geometry_lexical/delta_logit_l2 | 0.0158 | 0.1701 | 0.7905 |
| blazing_study_v3_confidence | qwen25_15b/0.50/lexical/delta_logit_l2 | 0.0175 | 0.2100 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.50/lexical/delta_logit_l2 | 0.0175 | 0.2100 | 1.0000 |
| paper3_pilot_v3_full | qwen25_15b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0175 | 0.1552 | 0.8750 |
| paper3_pilot_v3_full | qwen25_15b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0175 | 0.1552 | 0.8750 |
| paper3_public_v1_medium_public | qwen25_15b/0.20/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0175 | 0.0700 | 0.1750 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.35/support_aware_geometry_keep_compress_drop/delta_logit_l2 | 0.0185 | 0.0740 | 0.1850 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.35/support_aware_geometry_keep_compress_drop/delta_logit_l2 | 0.0185 | 0.0732 | 0.1850 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.35/semantic_filtered_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0187 | 0.1160 | 0.2812 |
| behavior_stress_v1 | smollm2_17b/0.35/geometry/delta_logit_l2 | 0.0200 | 0.0900 | 1.0000 |
| behavior_stress_v1 | smollm2_17b/0.35/geometry/delta_logit_l2 | 0.0200 | 0.0900 | 1.0000 |
| paper3_batch_v1_3b | qwen25_3b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0200 | 0.1260 | 0.1800 |
| paper3_batch_v1_fairness | qwen25_15b/0.38/geometry_keep_compress_drop/delta_logit_l2 | 0.0213 | 0.0729 | 0.3825 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.20/query_conditioned_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0217 | 0.1305 | 0.2392 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.35/budget_aware_semantic_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0222 | 0.1160 | 0.3115 |
| behavior_stress_v1 | smollm2_17b/0.50/geometry/delta_logit_l2 | 0.0230 | 0.0906 | 1.0000 |
| behavior_stress_v1 | smollm2_17b/0.50/geometry/delta_logit_l2 | 0.0230 | 0.0906 | 1.0000 |
| behavior_stress_v1 | smollm2_17b/0.35/lexical/delta_logit_l2 | 0.0235 | 0.0906 | 1.0000 |
| behavior_stress_v1 | smollm2_17b/0.35/lexical/delta_logit_l2 | 0.0235 | 0.0906 | 1.0000 |
| behavior_stress_v1 | qwen25_15b/0.50/uniform_segment_actions/delta_answer_avg_neg_logprob | 0.0238 | 0.1757 | 1.0000 |
| behavior_stress_v1 | qwen25_15b/0.50/uniform_segment_actions/delta_answer_avg_neg_logprob | 0.0238 | 0.1757 | 1.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/uniform_segment_actions/delta_answer_avg_neg_logprob | 0.0243 | 0.1410 | 0.6790 |
| behavior_stress_v1 | qwen25_05b/0.50/uniform_segment_actions/delta_answer_avg_neg_logprob | 0.0243 | 0.1757 | 1.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/uniform_segment_actions/delta_answer_avg_neg_logprob | 0.0243 | 0.1410 | 0.6790 |
| behavior_stress_v1 | qwen25_05b/0.50/uniform_segment_actions/delta_answer_avg_neg_logprob | 0.0243 | 0.1757 | 1.0000 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.35/support_aware_semantic_keep_compress_drop/delta_logit_l2 | 0.0245 | 0.0673 | 0.2940 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.20/query_conditioned_geometry_keep_compress_drop/delta_logit_l2 | 0.0248 | 0.0732 | 0.2228 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.35/support_aware_geometry_keep_compress_drop/delta_logit_l2 | 0.0253 | 0.0673 | 0.2940 |
| paper3_pilot_v3_full | qwen25_15b/0.35/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0262 | 0.1552 | 1.0000 |
| paper3_pilot_v3_full | qwen25_15b/0.35/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0262 | 0.1552 | 1.0000 |
| paper3_batch_v1_fairness | qwen25_15b/0.32/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0270 | 0.1290 | 0.5670 |
| paper3_low_budget_smoke_locomo | qwen25_05b/0.20/semantic/delta_answer_avg_neg_logprob | 0.0272 | 0.0732 | 0.3270 |
| paper3_low_budget_smoke_locomo | qwen25_05b/0.20/semantic_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0272 | 0.0732 | 0.3270 |
| paper3_query_geom_smoke_locomo | qwen25_05b/0.20/query_conditioned_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0272 | 0.0610 | 0.3270 |
| paper3_query_geom_smoke_locomo | qwen25_05b/0.20/semantic_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0272 | 0.0610 | 0.3270 |
| paper3_semantic_kcd_opt_smoke_locomo | qwen25_05b/0.20/semantic_filtered_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0272 | 0.0697 | 0.4360 |
| paper3_semantic_kcd_opt_smoke_locomo | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0275 | 0.0697 | 0.4360 |
| paper3_batch_v1_3b | qwen25_3b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0280 | 0.1260 | 0.2240 |
| blazing_study_v3_confidence | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0285 | 0.2448 | 1.0000 |
| blazing_study_v4_segment_behavior | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0285 | 0.2314 | 1.0000 |
| bridge_smoke_qwen | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0285 | 0.1710 | 0.4560 |
| bridge_smoke_qwen_v2 | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0285 | 0.1710 | 0.4560 |
| blazing_study_v3_confidence | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0285 | 0.2448 | 1.0000 |
| paper3_low_budget_smoke_locomo | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0288 | 0.0732 | 0.3270 |
| paper3_query_geom_smoke_locomo | qwen25_05b/0.20/support_aware_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0288 | 0.0610 | 0.3270 |
| paper3_semantic_kcd_opt_smoke_locomo | qwen25_05b/0.20/semantic_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0288 | 0.0697 | 0.4360 |
| paper3_low_budget_smoke_locomo | qwen25_05b/0.20/geometry/delta_answer_avg_neg_logprob | 0.0290 | 0.0732 | 0.3270 |
| paper3_query_geom_smoke_locomo | qwen25_05b/0.20/geometry/delta_answer_avg_neg_logprob | 0.0290 | 0.0610 | 0.3270 |
| paper3_semantic_kcd_opt_smoke_locomo | qwen25_05b/0.20/budget_aware_semantic_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0290 | 0.0697 | 0.4360 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.20/support_aware_semantic_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0290 | 0.1160 | 0.3770 |
| paper3_batch_v1_fairness | qwen25_15b/0.35/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0293 | 0.1290 | 0.5850 |
| paper3_pilot_v3_full | smollm2_17b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0295 | 0.1552 | 1.0000 |
| paper3_pilot_v3_full | smollm2_17b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0295 | 0.1552 | 1.0000 |
| expanded_v8_final | qwen25_15b/geometry_lexical_vs_geometry_only/auprc | 0.0299 | 0.2394 | 1.0000 |
| expanded_v8_final | qwen25_15b/geometry_lexical_vs_geometry_only/auprc | 0.0299 | 0.2394 | 1.0000 |
| blazing_study_v4_segment_behavior | qwen25_15b/0.35/geometry/delta_logit_l2 | 0.0300 | 0.2314 | 1.0000 |
| behavior_stress_v1 | smollm2_17b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0302 | 0.1757 | 1.0000 |
| paper3_query_geom_smoke_locomo | qwen25_05b/0.35/query_conditioned_geometry/delta_answer_avg_neg_logprob | 0.0302 | 0.0610 | 0.3270 |
| paper3_semantic_kcd_opt_smoke_locomo | qwen25_05b/0.20/geometry/delta_answer_avg_neg_logprob | 0.0302 | 0.0697 | 0.4360 |
| paper3_semantic_kcd_opt_smoke_locomo | qwen25_05b/0.20/support_aware_semantic_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0302 | 0.0697 | 0.4360 |
| behavior_stress_v1 | smollm2_17b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0302 | 0.1757 | 1.0000 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.35/query_conditioned_geometry_keep_compress_drop/delta_logit_l2 | 0.0305 | 0.0732 | 0.2440 |
| paper3_low_budget_smoke_locomo | qwen25_05b/0.20/semantic_filtered_geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0305 | 0.0732 | 0.3270 |
| paper3_query_geom_smoke_locomo | qwen25_05b/0.20/semantic/delta_answer_avg_neg_logprob | 0.0305 | 0.0610 | 0.3270 |
| paper3_semantic_kcd_opt_smoke_locomo | qwen25_05b/0.20/semantic/delta_answer_avg_neg_logprob | 0.0305 | 0.0697 | 0.4360 |
| behavior_stress_qwen_cases | qwen25_05b/0.20/geometry/delta_kl | 0.0312 | 0.0625 | 0.6250 |
| behavior_stress_v1 | qwen25_05b/0.20/geometry/delta_kl | 0.0312 | 0.1072 | 1.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.20/geometry/delta_kl | 0.0312 | 0.0625 | 0.6250 |
| behavior_stress_v1 | qwen25_05b/0.20/geometry/delta_kl | 0.0312 | 0.1072 | 1.0000 |
| behavior_stress_v1 | smollm2_17b/0.35/geometry/delta_kl | 0.0318 | 0.1072 | 1.0000 |
| behavior_stress_v1 | smollm2_17b/0.35/geometry/delta_kl | 0.0318 | 0.1072 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.65/geometry_lexical/delta_logit_l2 | 0.0320 | 0.2448 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.65/geometry_lexical/delta_logit_l2 | 0.0320 | 0.2448 | 1.0000 |
| paper3_batch_v1_fairness | qwen25_15b/0.46/geometry/delta_answer_avg_neg_logprob | 0.0323 | 0.1290 | 0.6128 |
| blazing_study_v3_confidence | qwen25_15b/0.35/geometry/delta_logit_l2 | 0.0325 | 0.2448 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.35/geometry/delta_logit_l2 | 0.0325 | 0.2448 | 1.0000 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.35/semantic_filtered_geometry_keep_compress_drop/delta_logit_l2 | 0.0333 | 0.0760 | 0.3325 |
| paper3_batch_v1_3b | qwen25_3b/0.50/geometry/delta_logit_l2 | 0.0333 | 0.1373 | 0.2660 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0335 | 0.1410 | 0.8710 |
| behavior_stress_v1 | qwen25_05b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0335 | 0.1757 | 1.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0335 | 0.1410 | 0.8710 |
| behavior_stress_v1 | qwen25_05b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0335 | 0.1757 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.65/geometry/delta_logit_l2 | 0.0340 | 0.2448 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.65/geometry/delta_logit_l2 | 0.0340 | 0.2448 | 1.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0355 | 0.1552 | 1.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry/delta_answer_avg_neg_logprob | 0.0355 | 0.1552 | 1.0000 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.35/semantic_filtered_geometry_keep_compress_drop/delta_logit_l2 | 0.0377 | 0.0790 | 0.3397 |
| blazing_study_v4_segment_behavior | qwen25_15b/0.50/geometry/delta_logit_l2 | 0.0380 | 0.2565 | 1.0000 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.35/geometry_keep_compress_drop/delta_logit_l2 | 0.0380 | 0.0790 | 0.3397 |
| behavior_stress_v1 | qwen25_15b/0.50/geometry_lexical/delta_answer_avg_neg_logprob | 0.0382 | 0.1757 | 1.0000 |
| behavior_stress_v1 | qwen25_15b/0.50/geometry_lexical/delta_answer_avg_neg_logprob | 0.0382 | 0.1757 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.50/geometry/delta_logit_l2 | 0.0395 | 0.2585 | 1.0000 |
| paper3_low_budget_smoke_msc | qwen25_05b/0.20/semantic/delta_logit_l2 | 0.0395 | 0.0790 | 0.3397 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/geometry_lexical/delta_answer_avg_neg_logprob | 0.0395 | 0.1410 | 0.9480 |
| behavior_stress_v1 | qwen25_05b/0.50/geometry_lexical/delta_answer_avg_neg_logprob | 0.0395 | 0.1757 | 1.0000 |
| blazing_study_v3_confidence | qwen25_15b/0.50/geometry/delta_logit_l2 | 0.0395 | 0.2585 | 1.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/geometry_lexical/delta_answer_avg_neg_logprob | 0.0395 | 0.1410 | 0.9480 |
| behavior_stress_v1 | qwen25_05b/0.50/geometry_lexical/delta_answer_avg_neg_logprob | 0.0395 | 0.1757 | 1.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0405 | 0.1552 | 1.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0405 | 0.1552 | 1.0000 |
| paper3_query_geom_smoke_msc | qwen25_05b/0.20/semantic/delta_logit_l2 | 0.0425 | 0.0850 | 0.2975 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.20/semantic/delta_logit_l2 | 0.0425 | 0.0773 | 0.3825 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.35/geometry_keep_compress_drop/delta_logit_l2 | 0.0435 | 0.0773 | 0.3825 |
| paper3_batch_v1_3b | qwen25_3b/0.35/geometry_segment_actions/delta_logit_l2 | 0.0457 | 0.1373 | 0.3202 |
| paper3_batch_v1_fairness | qwen25_15b/0.24/geometry_segment_actions/delta_logit_l2 | 0.0457 | 0.1320 | 0.7777 |
| paper3_batch_v1_fairness | qwen25_15b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0465 | 0.1594 | 0.8370 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0470 | 0.1410 | 1.0000 |
| behavior_stress_v1 | qwen25_05b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0470 | 0.1757 | 1.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0470 | 0.1410 | 1.0000 |
| behavior_stress_v1 | qwen25_05b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0470 | 0.1757 | 1.0000 |
| paper3_pilot_v3_full | smollm2_17b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0473 | 0.1552 | 1.0000 |
| paper3_pilot_v3_full | smollm2_17b/0.50/geometry_keep_compress_drop/delta_answer_avg_neg_logprob | 0.0473 | 0.1552 | 1.0000 |
| behavior_stress_v1 | qwen25_15b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0478 | 0.1757 | 1.0000 |
| behavior_stress_v1 | qwen25_15b/0.50/geometry_segment_actions/delta_answer_avg_neg_logprob | 0.0478 | 0.1757 | 1.0000 |
| paper3_semantic_kcd_opt_smoke_msc | qwen25_05b/0.20/geometry/delta_answer_avg_neg_logprob | 0.0493 | 0.1576 | 0.5910 |

## Strongest surviving findings (BH q < 0.01)

| source | test | raw p | BH q |
|---|---|---|---|
| behavior_stress_qwen_cases | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| behavior_stress_v1 | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| behavior_stress_v1 | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| behavior_stress_qwen_cases | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| behavior_stress_v1 | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| behavior_stress_v1 | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.35/geometry_segment_actions/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.35/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_pilot_v3_full | qwen25_05b/0.35/geometry_segment_actions/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_medium_public | qwen25_15b/0.20/semantic/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_medium_public | qwen25_15b/0.50/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.20/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.20/geometry_segment_actions/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.20/semantic/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.35/geometry/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.35/geometry_segment_actions/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.35/semantic/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.50/geometry_keep_compress_drop/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_tiny_public | qwen25_15b/0.35/geometry_keep_compress_drop/delta_logit_l2 | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.20/geometry/delta_answer_avg_neg_logprob | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.20/semantic/delta_answer_avg_neg_logprob | 0.0000 | 0.0000 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.35/geometry/delta_answer_avg_neg_logprob | 0.0000 | 0.0000 |
| paper3_public_v1_tiny_public | qwen25_15b/0.35/geometry/delta_logit_l2 | 0.0003 | 0.0003 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.35/geometry_keep_compress_drop/delta_logit_l2 | 0.0003 | 0.0004 |
| paper3_pilot_v3_full | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_pilot_v3_full | qwen25_05b/0.20/geometry_segment_actions/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_pilot_v3_full | qwen25_05b/0.35/geometry_keep_compress_drop/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry_segment_actions/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_pilot_v3_full | qwen25_05b/0.20/geometry/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_pilot_v3_full | qwen25_05b/0.20/geometry_segment_actions/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_pilot_v3_full | qwen25_05b/0.35/geometry_keep_compress_drop/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_pilot_v3_full | qwen25_05b/0.50/geometry_segment_actions/delta_logit_l2 | 0.0003 | 0.0011 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.50/geometry_segment_actions/delta_logit_l2 | 0.0013 | 0.0017 |
| paper3_pilot_v3_full | smollm2_17b/0.35/geometry_segment_actions/delta_logit_l2 | 0.0010 | 0.0039 |
| paper3_pilot_v3_full | smollm2_17b/0.35/geometry_segment_actions/delta_logit_l2 | 0.0010 | 0.0039 |
| paper3_public_v1_public_benchmark | qwen25_15b/0.50/geometry/delta_logit_l2 | 0.0032 | 0.0039 |
| paper3_public_v1_medium_public | qwen25_15b/0.35/geometry/delta_logit_l2 | 0.0015 | 0.0045 |