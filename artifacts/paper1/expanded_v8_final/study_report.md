# Paper 1 Study: expanded_v8_final

- Input: `paper1_geometry/assets/paper1_study_conversations.jsonl`
- Input files: paper1_geometry/assets/paper1_study_conversations.jsonl, paper1_geometry/assets/paper1_h2_stress_conversations.jsonl
- Models run: qwen25_05b, qwen25_15b, smollm2_17b

## Benchmark Audit

- Mean turns / conversation: 6.083
- Median turns / conversation: 6.000
- Mean candidate boundaries / conversation: 4.083
- Median candidate boundaries / conversation: 4.000
- Mean gold boundaries / conversation: 0.958
- Mean gold boundary density: 0.233
- Zero-gold conversations: 0.375
- Oracle random expected macro exact F1: 0.608
- Oracle random expected macro exact F1 (nonempty only): 0.372
- Oracle random empirical macro exact F1: 0.611
- Oracle random empirical micro exact F1: 0.414

### Caveats

- Boundary metrics are high-variance here because most conversations are short and expose only a few candidate inter-turn boundaries.
- `oracle_random_matched_count` is a chance-reference oracle that uses the gold boundary count; it is not a practical baseline.
- Macro exact F1 is inflated by no-boundary conversations because predicting no boundaries receives a perfect score on those examples.
- The strongest supported Paper 1 claims remain low-rank structure and geometry-to-decoder relevance. Boundary recovery is secondary, mixed, and formulation-sensitive.

## qwen25_05b

# Paper 1 Run Summary

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Device: `mps`
- Dtype: `auto`
- State layer: `-1`
- Transformers: `4.49.0`

## Aggregate

- Conversations: 24
- Mean rank95: 1.049
- Mean curvature: 2.022
- Mean corr(geodesic, logit L2): 0.989
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.579
- Macro boundary F1 exact: 0.292
- Micro boundary F1 exact: 0.453
- Macro boundary F1 tol1: 0.292
- Micro boundary F1 tol1: 0.453
- Macro boundary F1 tol2: 0.528
- Micro boundary F1 tol2: 0.679
- Macro boundary F1 tol3: 0.528
- Micro boundary F1 tol3: 0.679
- Mean nearest boundary distance: 0.694
- Mean WindowDiff: 0.292
- Mean Pk: 0.384
- Mean boundary AUPRC: 0.732
- Mean candidate boundaries / conversation: 4.042
- Mean gold boundary density: 0.235
- Zero-gold conversations: 0.375
- Mean logit L2: 150.655
- Mean KL: 0.000
- Mean corr(geodesic, KL): 0.273

## By Family

### casual_chat
- Conversations: 4
- Mean rank95: 1.000
- Mean curvature: 1.929
- Mean turning angle: 2.846
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.589
- Mean boundary score: 2.474
- Mean boundary prominence: 0.161
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.222
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.222
- Macro boundary F1 tol2: 0.917
- Micro boundary F1 tol2: 0.889
- Macro boundary F1 tol3: 0.917
- Micro boundary F1 tol3: 0.889
- Mean nearest boundary distance: 1.667
- Mean WindowDiff: 0.583
- Mean Pk: 0.625
- Mean boundary AUPRC: 0.438
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.250
- Mean state geodesic error: 0.131
- Mean logit L2: 153.535
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.987
- Mean top-1 agreement: 1.000

### code_conversation
- Conversations: 4
- Mean rank95: 1.208
- Mean curvature: 2.005
- Mean turning angle: 2.805
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.680
- Mean boundary score: 2.357
- Mean boundary prominence: 0.168
- Macro boundary F1 exact: 0.250
- Micro boundary F1 exact: 0.500
- Macro boundary F1 tol1: 0.250
- Micro boundary F1 tol1: 0.500
- Macro boundary F1 tol2: 0.250
- Micro boundary F1 tol2: 0.500
- Macro boundary F1 tol3: 0.250
- Micro boundary F1 tol3: 0.500
- Mean nearest boundary distance: 0.000
- Mean WindowDiff: 0.000
- Mean Pk: 0.417
- Mean boundary AUPRC: 0.938
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.103
- Mean logit L2: 122.459
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.997
- Mean top-1 agreement: 1.000

### long_dependency
- Conversations: 4
- Mean rank95: 1.000
- Mean curvature: 2.251
- Mean turning angle: 2.893
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.382
- Mean boundary score: 2.508
- Mean boundary prominence: 0.115
- Macro boundary F1 exact: 0.500
- Micro boundary F1 exact: 0.500
- Macro boundary F1 tol1: 0.500
- Micro boundary F1 tol1: 0.500
- Macro boundary F1 tol2: 0.667
- Micro boundary F1 tol2: 0.667
- Macro boundary F1 tol3: 0.667
- Micro boundary F1 tol3: 0.667
- Mean nearest boundary distance: 1.167
- Mean WindowDiff: 0.625
- Mean Pk: 0.275
- Mean boundary AUPRC: 0.644
- Mean candidate boundaries / conversation: 4.250
- Mean gold boundary density: 0.475
- Mean state geodesic error: 0.115
- Mean logit L2: 136.500
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.986
- Mean top-1 agreement: 1.000

### multi_topic_chat
- Conversations: 4
- Mean rank95: 1.083
- Mean curvature: 2.003
- Mean turning angle: 2.827
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.653
- Mean boundary score: 2.465
- Mean boundary prominence: 0.144
- Macro boundary F1 exact: 0.417
- Micro boundary F1 exact: 0.545
- Macro boundary F1 tol1: 0.417
- Micro boundary F1 tol1: 0.545
- Macro boundary F1 tol2: 0.917
- Micro boundary F1 tol2: 0.909
- Macro boundary F1 tol3: 0.917
- Micro boundary F1 tol3: 0.909
- Mean nearest boundary distance: 1.167
- Mean WindowDiff: 0.417
- Mean Pk: 0.438
- Mean boundary AUPRC: 0.583
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.312
- Mean state geodesic error: 0.113
- Mean logit L2: 132.806
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.993
- Mean top-1 agreement: 1.000

### reasoning_chat
- Conversations: 4
- Mean rank95: 1.000
- Mean curvature: 1.961
- Mean turning angle: 2.871
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.508
- Mean boundary score: 2.446
- Mean boundary prominence: 0.147
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.333
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.333
- Macro boundary F1 tol2: 0.167
- Micro boundary F1 tol2: 0.333
- Macro boundary F1 tol3: 0.167
- Micro boundary F1 tol3: 0.333
- Mean nearest boundary distance: 0.167
- Mean WindowDiff: 0.125
- Mean Pk: 0.300
- Mean boundary AUPRC: 0.854
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.134
- Mean logit L2: 170.354
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.995
- Mean top-1 agreement: 1.000

### retrieval_heavy
- Conversations: 4
- Mean rank95: 1.000
- Mean curvature: 1.984
- Mean turning angle: 2.822
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.665
- Mean boundary score: 2.448
- Mean boundary prominence: 0.213
- Macro boundary F1 exact: 0.250
- Micro boundary F1 exact: 0.571
- Macro boundary F1 tol1: 0.250
- Micro boundary F1 tol1: 0.571
- Macro boundary F1 tol2: 0.250
- Micro boundary F1 tol2: 0.571
- Macro boundary F1 tol3: 0.250
- Micro boundary F1 tol3: 0.571
- Mean nearest boundary distance: 0.000
- Mean WindowDiff: 0.000
- Mean Pk: 0.250
- Mean boundary AUPRC: 0.938
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.155
- Mean logit L2: 188.278
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.976
- Mean top-1 agreement: 1.000

### Uncertainty

- mean_rank95: estimate 1.049, bootstrap std 0.027, 95% CI [1.000, 1.111]
- mean_corr_geodesic_vs_logit_l2: estimate 0.989, bootstrap std 0.002, 95% CI [0.984, 0.993]
- micro_boundary_f1_exact: estimate 0.453, bootstrap std 0.094, 95% CI [0.267, 0.621]
- micro_boundary_f1_tol2: estimate 0.679, bootstrap std 0.068, 95% CI [0.533, 0.793]
- mean_boundary_auprc: estimate 0.732, bootstrap std 0.059, 95% CI [0.621, 0.844]

### Null Controls

- H1 shuffled turn order: real rank95 1.049, shuffled 1.486, gap 0.438, 95% CI [0.299, 0.576], p=0.0000
- H3 permuted alignment: real corr 0.989, permuted -0.004, gap 0.993, 95% CI [0.984, 1.001], p=0.0000

### Boundary Variant Ablation

- geometry_lexical: micro exact 0.453, micro tol2 0.679, mean AUPRC 0.732, mean nearest distance 0.694
- geometry_only: micro exact 0.298, micro tol2 0.638, mean AUPRC 0.698, mean nearest distance 0.889
- lexical_only: micro exact 0.298, micro tol2 0.638, mean AUPRC 0.878, mean nearest distance 0.889

### Boundary Significance

- geometry_lexical_vs_lexical_only: exact diff 0.097 (p=0.0662), tol2 diff 0.014 (p=1.0000), AUPRC diff -0.146 (p=0.0665)
- geometry_lexical_vs_geometry_only: exact diff 0.097 (p=0.0615), tol2 diff 0.014 (p=1.0000), AUPRC diff 0.035 (p=0.1213)

## qwen25_15b

# Paper 1 Run Summary

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Device: `mps`
- Dtype: `auto`
- State layer: `-1`
- Transformers: `4.49.0`

## Aggregate

- Conversations: 24
- Mean rank95: 1.167
- Mean curvature: 2.934
- Mean corr(geodesic, logit L2): 0.994
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.571
- Macro boundary F1 exact: 0.236
- Micro boundary F1 exact: 0.367
- Macro boundary F1 tol1: 0.236
- Micro boundary F1 tol1: 0.367
- Macro boundary F1 tol2: 0.514
- Micro boundary F1 tol2: 0.653
- Macro boundary F1 tol3: 0.514
- Micro boundary F1 tol3: 0.653
- Mean nearest boundary distance: 0.806
- Mean WindowDiff: 0.347
- Mean Pk: 0.394
- Mean boundary AUPRC: 0.758
- Mean candidate boundaries / conversation: 4.042
- Mean gold boundary density: 0.235
- Zero-gold conversations: 0.375
- Mean logit L2: 186.749
- Mean KL: 0.000
- Mean corr(geodesic, KL): 0.300

## By Family

### casual_chat
- Conversations: 4
- Mean rank95: 1.125
- Mean curvature: 2.975
- Mean turning angle: 2.819
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.653
- Mean boundary score: 2.503
- Mean boundary prominence: 0.209
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.222
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.222
- Macro boundary F1 tol2: 0.917
- Micro boundary F1 tol2: 0.889
- Macro boundary F1 tol3: 0.917
- Micro boundary F1 tol3: 0.889
- Mean nearest boundary distance: 1.667
- Mean WindowDiff: 0.583
- Mean Pk: 0.625
- Mean boundary AUPRC: 0.458
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.250
- Mean state geodesic error: 0.128
- Mean logit L2: 188.810
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.994
- Mean top-1 agreement: 1.000

### code_conversation
- Conversations: 4
- Mean rank95: 1.250
- Mean curvature: 3.184
- Mean turning angle: 2.838
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.475
- Mean boundary score: 2.511
- Mean boundary prominence: 0.080
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.333
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.333
- Macro boundary F1 tol2: 0.167
- Micro boundary F1 tol2: 0.333
- Macro boundary F1 tol3: 0.167
- Micro boundary F1 tol3: 0.333
- Mean nearest boundary distance: 0.167
- Mean WindowDiff: 0.125
- Mean Pk: 0.300
- Mean boundary AUPRC: 0.896
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.121
- Mean logit L2: 181.797
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.993
- Mean top-1 agreement: 1.000

### long_dependency
- Conversations: 4
- Mean rank95: 1.125
- Mean curvature: 2.480
- Mean turning angle: 2.823
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.553
- Mean boundary score: 2.542
- Mean boundary prominence: 0.153
- Macro boundary F1 exact: 0.500
- Micro boundary F1 exact: 0.500
- Macro boundary F1 tol1: 0.500
- Micro boundary F1 tol1: 0.500
- Macro boundary F1 tol2: 0.667
- Micro boundary F1 tol2: 0.667
- Macro boundary F1 tol3: 0.667
- Micro boundary F1 tol3: 0.667
- Mean nearest boundary distance: 1.167
- Mean WindowDiff: 0.625
- Mean Pk: 0.275
- Mean boundary AUPRC: 0.675
- Mean candidate boundaries / conversation: 4.250
- Mean gold boundary density: 0.475
- Mean state geodesic error: 0.148
- Mean logit L2: 221.741
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.994
- Mean top-1 agreement: 1.000

### multi_topic_chat
- Conversations: 4
- Mean rank95: 1.125
- Mean curvature: 3.025
- Mean turning angle: 2.832
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.545
- Mean boundary score: 2.545
- Mean boundary prominence: 0.163
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.222
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.222
- Macro boundary F1 tol2: 0.917
- Micro boundary F1 tol2: 0.889
- Macro boundary F1 tol3: 0.917
- Micro boundary F1 tol3: 0.889
- Mean nearest boundary distance: 1.667
- Mean WindowDiff: 0.625
- Mean Pk: 0.613
- Mean boundary AUPRC: 0.583
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.312
- Mean state geodesic error: 0.126
- Mean logit L2: 186.913
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.994
- Mean top-1 agreement: 1.000

### reasoning_chat
- Conversations: 4
- Mean rank95: 1.375
- Mean curvature: 3.259
- Mean turning angle: 2.825
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.652
- Mean boundary score: 2.465
- Mean boundary prominence: 0.169
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.333
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.333
- Macro boundary F1 tol2: 0.167
- Micro boundary F1 tol2: 0.333
- Macro boundary F1 tol3: 0.167
- Micro boundary F1 tol3: 0.333
- Mean nearest boundary distance: 0.167
- Mean WindowDiff: 0.125
- Mean Pk: 0.300
- Mean boundary AUPRC: 0.938
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.099
- Mean logit L2: 146.729
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.997
- Mean top-1 agreement: 1.000

### retrieval_heavy
- Conversations: 4
- Mean rank95: 1.000
- Mean curvature: 2.680
- Mean turning angle: 2.839
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.549
- Mean boundary score: 2.565
- Mean boundary prominence: 0.113
- Macro boundary F1 exact: 0.250
- Micro boundary F1 exact: 0.571
- Macro boundary F1 tol1: 0.250
- Micro boundary F1 tol1: 0.571
- Macro boundary F1 tol2: 0.250
- Micro boundary F1 tol2: 0.571
- Macro boundary F1 tol3: 0.250
- Micro boundary F1 tol3: 0.571
- Mean nearest boundary distance: 0.000
- Mean WindowDiff: 0.000
- Mean Pk: 0.250
- Mean boundary AUPRC: 1.000
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.130
- Mean logit L2: 194.506
- Mean KL: 0.000
- Mean corr(geodesic, logit L2): 0.994
- Mean top-1 agreement: 1.000

### Uncertainty

- mean_rank95: estimate 1.167, bootstrap std 0.049, 95% CI [1.083, 1.271]
- mean_corr_geodesic_vs_logit_l2: estimate 0.994, bootstrap std 0.001, 95% CI [0.993, 0.996]
- micro_boundary_f1_exact: estimate 0.367, bootstrap std 0.085, 95% CI [0.182, 0.519]
- micro_boundary_f1_tol2: estimate 0.653, bootstrap std 0.063, 95% CI [0.524, 0.764]
- mean_boundary_auprc: estimate 0.758, bootstrap std 0.061, 95% CI [0.644, 0.878]

### Null Controls

- H1 shuffled turn order: real rank95 1.167, shuffled 1.458, gap 0.292, 95% CI [0.153, 0.444], p=0.0008
- H3 permuted alignment: real corr 0.994, permuted -0.001, gap 0.995, 95% CI [0.983, 1.008], p=0.0000

### Boundary Variant Ablation

- geometry_lexical: micro exact 0.367, micro tol2 0.653, mean AUPRC 0.758, mean nearest distance 0.806
- geometry_only: micro exact 0.298, micro tol2 0.638, mean AUPRC 0.687, mean nearest distance 0.889
- lexical_only: micro exact 0.298, micro tol2 0.638, mean AUPRC 0.878, mean nearest distance 0.889

### Boundary Significance

- geometry_lexical_vs_lexical_only: exact diff 0.042 (p=0.4917), tol2 diff 0.000 (p=1.0000), AUPRC diff -0.120 (p=0.0058)
- geometry_lexical_vs_geometry_only: exact diff 0.042 (p=0.4950), tol2 diff 0.000 (p=1.0000), AUPRC diff 0.071 (p=0.0299)

## smollm2_17b

# Paper 1 Run Summary

- Model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Device: `mps`
- Dtype: `auto`
- State layer: `-1`
- Transformers: `4.49.0`

## Aggregate

- Conversations: 24
- Mean rank95: 1.528
- Mean curvature: 2.883
- Mean corr(geodesic, logit L2): 0.989
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.838
- Macro boundary F1 exact: 0.194
- Micro boundary F1 exact: 0.292
- Macro boundary F1 tol1: 0.215
- Micro boundary F1 tol1: 0.333
- Macro boundary F1 tol2: 0.528
- Micro boundary F1 tol2: 0.667
- Macro boundary F1 tol3: 0.528
- Micro boundary F1 tol3: 0.667
- Mean nearest boundary distance: 0.830
- Mean WindowDiff: 0.365
- Mean Pk: 0.430
- Mean boundary AUPRC: 0.821
- Mean candidate boundaries / conversation: 4.042
- Mean gold boundary density: 0.235
- Zero-gold conversations: 0.375
- Mean logit L2: 124.507
- Mean KL: 0.034
- Mean corr(geodesic, KL): 0.579

## By Family

### casual_chat
- Conversations: 4
- Mean rank95: 1.500
- Mean curvature: 3.069
- Mean turning angle: 2.588
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.983
- Mean boundary score: 2.541
- Mean boundary prominence: 0.102
- Macro boundary F1 exact: 0.000
- Micro boundary F1 exact: 0.000
- Macro boundary F1 tol1: 0.000
- Micro boundary F1 tol1: 0.000
- Macro boundary F1 tol2: 1.000
- Micro boundary F1 tol2: 1.000
- Macro boundary F1 tol3: 1.000
- Micro boundary F1 tol3: 1.000
- Mean nearest boundary distance: 2.000
- Mean WindowDiff: 0.667
- Mean Pk: 0.750
- Mean boundary AUPRC: 0.521
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.250
- Mean state geodesic error: 0.133
- Mean logit L2: 132.590
- Mean KL: 0.015
- Mean corr(geodesic, logit L2): 0.990
- Mean top-1 agreement: 1.000

### code_conversation
- Conversations: 4
- Mean rank95: 1.500
- Mean curvature: 3.004
- Mean turning angle: 2.643
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.858
- Mean boundary score: 2.510
- Mean boundary prominence: 0.110
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.333
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.333
- Macro boundary F1 tol2: 0.167
- Micro boundary F1 tol2: 0.333
- Macro boundary F1 tol3: 0.167
- Micro boundary F1 tol3: 0.333
- Mean nearest boundary distance: 0.167
- Mean WindowDiff: 0.125
- Mean Pk: 0.300
- Mean boundary AUPRC: 1.000
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.127
- Mean logit L2: 130.787
- Mean KL: 0.022
- Mean corr(geodesic, logit L2): 0.990
- Mean top-1 agreement: 1.000

### long_dependency
- Conversations: 4
- Mean rank95: 1.542
- Mean curvature: 2.933
- Mean turning angle: 2.661
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.755
- Mean boundary score: 2.528
- Mean boundary prominence: 0.205
- Macro boundary F1 exact: 0.500
- Micro boundary F1 exact: 0.462
- Macro boundary F1 tol1: 0.625
- Micro boundary F1 tol1: 0.615
- Macro boundary F1 tol2: 0.750
- Micro boundary F1 tol2: 0.769
- Macro boundary F1 tol3: 0.750
- Micro boundary F1 tol3: 0.769
- Mean nearest boundary distance: 0.812
- Mean WindowDiff: 0.525
- Mean Pk: 0.317
- Mean boundary AUPRC: 0.758
- Mean candidate boundaries / conversation: 4.250
- Mean gold boundary density: 0.475
- Mean state geodesic error: 0.101
- Mean logit L2: 108.625
- Mean KL: 0.009
- Mean corr(geodesic, logit L2): 0.991
- Mean top-1 agreement: 1.000

### multi_topic_chat
- Conversations: 4
- Mean rank95: 1.625
- Mean curvature: 2.956
- Mean turning angle: 2.644
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.895
- Mean boundary score: 2.588
- Mean boundary prominence: 0.133
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.222
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.222
- Macro boundary F1 tol2: 0.917
- Micro boundary F1 tol2: 0.889
- Macro boundary F1 tol3: 0.917
- Micro boundary F1 tol3: 0.889
- Mean nearest boundary distance: 1.667
- Mean WindowDiff: 0.625
- Mean Pk: 0.613
- Mean boundary AUPRC: 0.688
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.312
- Mean state geodesic error: 0.123
- Mean logit L2: 121.511
- Mean KL: 0.040
- Mean corr(geodesic, logit L2): 0.989
- Mean top-1 agreement: 0.958

### reasoning_chat
- Conversations: 4
- Mean rank95: 1.500
- Mean curvature: 2.515
- Mean turning angle: 2.701
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.744
- Mean boundary score: 2.464
- Mean boundary prominence: 0.189
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.333
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.333
- Macro boundary F1 tol2: 0.167
- Micro boundary F1 tol2: 0.333
- Macro boundary F1 tol3: 0.167
- Micro boundary F1 tol3: 0.333
- Mean nearest boundary distance: 0.167
- Mean WindowDiff: 0.125
- Mean Pk: 0.300
- Mean boundary AUPRC: 1.000
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.119
- Mean logit L2: 124.632
- Mean KL: 0.030
- Mean corr(geodesic, logit L2): 0.986
- Mean top-1 agreement: 1.000

### retrieval_heavy
- Conversations: 4
- Mean rank95: 1.500
- Mean curvature: 2.818
- Mean turning angle: 2.682
- Mean rank-jump score: 0.000
- Mean subspace-shift score: 0.796
- Mean boundary score: 2.505
- Mean boundary prominence: 0.221
- Macro boundary F1 exact: 0.167
- Micro boundary F1 exact: 0.333
- Macro boundary F1 tol1: 0.167
- Micro boundary F1 tol1: 0.333
- Macro boundary F1 tol2: 0.167
- Micro boundary F1 tol2: 0.333
- Macro boundary F1 tol3: 0.167
- Micro boundary F1 tol3: 0.333
- Mean nearest boundary distance: 0.167
- Mean WindowDiff: 0.125
- Mean Pk: 0.300
- Mean boundary AUPRC: 0.958
- Mean candidate boundaries / conversation: 4.000
- Mean gold boundary density: 0.125
- Mean state geodesic error: 0.135
- Mean logit L2: 128.899
- Mean KL: 0.089
- Mean corr(geodesic, logit L2): 0.989
- Mean top-1 agreement: 0.833

### Uncertainty

- mean_rank95: estimate 1.528, bootstrap std 0.022, 95% CI [1.500, 1.576]
- mean_corr_geodesic_vs_logit_l2: estimate 0.989, bootstrap std 0.001, 95% CI [0.988, 0.991]
- micro_boundary_f1_exact: estimate 0.292, bootstrap std 0.076, 95% CI [0.136, 0.433]
- micro_boundary_f1_tol2: estimate 0.667, bootstrap std 0.062, 95% CI [0.533, 0.778]
- mean_boundary_auprc: estimate 0.821, bootstrap std 0.052, 95% CI [0.718, 0.920]

### Null Controls

- H1 shuffled turn order: real rank95 1.528, shuffled 1.799, gap 0.271, 95% CI [0.083, 0.479], p=0.0122
- H3 permuted alignment: real corr 0.989, permuted 0.004, gap 0.985, 95% CI [0.974, 0.995], p=0.0000

### Boundary Variant Ablation

- geometry_lexical: micro exact 0.292, micro tol2 0.667, mean AUPRC 0.821, mean nearest distance 0.830
- geometry_only: micro exact 0.298, micro tol2 0.638, mean AUPRC 0.847, mean nearest distance 0.889
- lexical_only: micro exact 0.298, micro tol2 0.638, mean AUPRC 0.878, mean nearest distance 0.889

### Boundary Significance

- geometry_lexical_vs_lexical_only: exact diff 0.000 (p=1.0000), tol2 diff 0.014 (p=1.0000), AUPRC diff -0.058 (p=0.2020)
- geometry_lexical_vs_geometry_only: exact diff 0.000 (p=1.0000), tol2 diff 0.014 (p=1.0000), AUPRC diff -0.026 (p=0.3730)

## Baselines

### fixed_window

- Macro boundary F1 exact: 0.403
- Micro boundary F1 exact: 0.083
- Macro boundary F1 tol1: 0.403
- Micro boundary F1 tol1: 0.083
- Macro boundary F1 tol2: 0.403
- Micro boundary F1 tol2: 0.083
- Macro boundary F1 tol3: 0.403
- Micro boundary F1 tol3: 0.083
- Mean nearest boundary distance: 0.028
- Mean WindowDiff: 0.535
- Mean Pk: 0.268
- Mean boundary AUPRC: 0.000
- Mean candidate boundaries / conversation: 4.083
- Mean gold boundary density: 0.233
- Mean ordered boundary MAE: 0.083
- Mean oversegmentation rate: 0.000
- Mean miss rate: 0.604

### lexical_shift

- Macro boundary F1 exact: 0.389
- Micro boundary F1 exact: 0.389
- Macro boundary F1 tol1: 0.417
- Micro boundary F1 tol1: 0.444
- Macro boundary F1 tol2: 0.417
- Micro boundary F1 tol2: 0.444
- Macro boundary F1 tol3: 0.417
- Micro boundary F1 tol3: 0.444
- Mean nearest boundary distance: 0.208
- Mean WindowDiff: 0.410
- Mean Pk: 0.299
- Mean boundary AUPRC: 0.875
- Mean candidate boundaries / conversation: 4.083
- Mean gold boundary density: 0.233
- Mean ordered boundary MAE: 0.125
- Mean oversegmentation rate: 0.229
- Mean miss rate: 0.438

### oracle_random_matched_count

- Macro boundary F1 exact: 0.611
- Micro boundary F1 exact: 0.414
- Macro boundary F1 tol1: 0.886
- Micro boundary F1 tol1: 0.843
- Macro boundary F1 tol2: 0.971
- Micro boundary F1 tol2: 0.940
- Macro boundary F1 tol3: 1.000
- Micro boundary F1 tol3: 1.000
- Mean nearest boundary distance: 0.488
- Mean WindowDiff: 0.234
- Mean Pk: 0.260
- Mean boundary AUPRC: 0.000
- Mean candidate boundaries / conversation: 4.083
- Mean gold boundary density: 0.233
- Mean ordered boundary MAE: 0.539
- Mean oversegmentation rate: 0.389
- Mean miss rate: 0.389

### style_shift

- Macro boundary F1 exact: 0.042
- Micro boundary F1 exact: 0.000
- Macro boundary F1 tol1: 0.208
- Micro boundary F1 tol1: 0.308
- Macro boundary F1 tol2: 0.292
- Micro boundary F1 tol2: 0.410
- Macro boundary F1 tol3: 0.292
- Micro boundary F1 tol3: 0.410
- Mean nearest boundary distance: 0.500
- Mean WindowDiff: 0.466
- Mean Pk: 0.607
- Mean boundary AUPRC: 0.678
- Mean candidate boundaries / conversation: 4.083
- Mean gold boundary density: 0.233
- Mean ordered boundary MAE: 0.667
- Mean oversegmentation rate: 0.542
- Mean miss rate: 0.625

## Plots

- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/curvature_traces.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/boundary_score_traces.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/boundary_prominence_traces.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/rank95_by_family.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/family_correlation_heatmap.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/boundary_eval_heatmap.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/geometry_vs_decoder.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/rank_energy_curves.png`
- `/Users/pranav/Documents/RT/results/paper1/studies/expanded_v8_final/plots/baseline_eval_heatmap.png`
