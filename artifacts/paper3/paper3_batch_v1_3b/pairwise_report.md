# Paper 3 Pairwise Policy Analysis

Negative deltas mean the left policy is better than the right policy.

## qwen25_3b

- budget 0.20:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -8.080 [-28.963, 3.224], p=0.8645
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -0.815 [-25.227, 21.389], p=0.9345
  geometry_segment_actions__vs__geometry: Δ logit L2 -7.265 [-24.193, 3.022], p=0.5130
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL 0.0687 [-0.0470, 0.2275], p=0.5005
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.0167 [-0.2302, 0.1941], p=0.8708
    geometry_segment_actions__vs__geometry: Δ answer NLL 0.0854 [0.0000, 0.2506], p=0.5035
- budget 0.35:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -53.675 [-111.239, -9.048], p=0.0272
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -16.216 [-37.639, 1.769], p=0.1345
  geometry_segment_actions__vs__geometry: Δ logit L2 -37.459 [-93.543, 2.970], p=0.2052
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.3616 [-0.7593, -0.0171], p=0.1905
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.2176 [-0.5267, 0.0096], p=0.2542
    geometry_segment_actions__vs__geometry: Δ answer NLL -0.1441 [-0.4466, 0.0143], p=1.0000
- budget 0.50:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 24.357 [7.660, 44.806], p=0.0063
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 7.435 [-20.439, 28.172], p=0.6248
  geometry_segment_actions__vs__geometry: Δ logit L2 16.922 [-8.804, 44.293], p=0.2540
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.1475 [-0.4180, 0.0627], p=0.5050
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.2901 [-0.6118, -0.0286], p=0.0745
    geometry_segment_actions__vs__geometry: Δ answer NLL 0.1427 [-0.0850, 0.4374], p=0.3645
