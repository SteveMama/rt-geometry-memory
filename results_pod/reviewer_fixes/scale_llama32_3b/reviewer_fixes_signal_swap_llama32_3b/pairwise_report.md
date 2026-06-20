# Paper 3 Pairwise Policy Analysis

Negative deltas mean the left policy is better than the right policy.

## llama32_3b

- budget 0.20:
  geometry_keep_compress_drop__vs__geometry: row Δ logit L2 4.784 [-12.745, 27.330], p=0.6987; conversation Δ 4.784 [-15.850, 34.337], p=0.8748
  semantic_keep_compress_drop__vs__semantic: row Δ logit L2 -15.368 [-52.005, 15.119], p=0.4672; conversation Δ -15.368 [-49.157, 10.190], p=0.4898
  semantic_keep_compress_drop__vs__geometry: row Δ logit L2 -14.011 [-55.639, 20.850], p=0.5128; conversation Δ -14.011 [-57.464, 22.922], p=0.5423
  semantic_keep_compress_drop__vs__geometry_keep_compress_drop: row Δ logit L2 -18.796 [-65.514, 18.907], p=0.3952; conversation Δ -18.796 [-59.991, 17.656], p=0.4178
  behavior:
    geometry_keep_compress_drop__vs__geometry: row Δ answer NLL -0.1048 [-0.4488, 0.1095], p=1.0000; conversation Δ -0.1048 [-0.4435, 0.1041], p=1.0000
    semantic_keep_compress_drop__vs__semantic: row Δ answer NLL 0.1716 [-0.0613, 0.5256], p=0.4943; conversation Δ 0.1716 [-0.0603, 0.5220], p=0.4960
    semantic_keep_compress_drop__vs__geometry: row Δ answer NLL 0.0662 [-0.0569, 0.1948], p=0.3510; conversation Δ 0.0662 [-0.0485, 0.1730], p=0.3553
    semantic_keep_compress_drop__vs__geometry_keep_compress_drop: row Δ answer NLL 0.1711 [-0.0658, 0.5374], p=0.4520; conversation Δ 0.1711 [-0.0648, 0.5091], p=0.4470
- budget 0.35:
  geometry_keep_compress_drop__vs__geometry: row Δ logit L2 37.940 [-11.433, 97.947], p=0.1903; conversation Δ 37.940 [-20.852, 135.650], p=0.7600
  semantic_keep_compress_drop__vs__semantic: row Δ logit L2 38.267 [5.823, 83.192], p=0.0442; conversation Δ 38.267 [11.274, 74.927], p=0.0315
  semantic_keep_compress_drop__vs__geometry: row Δ logit L2 -5.866 [-69.933, 58.593], p=0.8670; conversation Δ -5.866 [-62.286, 51.737], p=0.8688
  semantic_keep_compress_drop__vs__geometry_keep_compress_drop: row Δ logit L2 -43.805 [-115.664, 15.001], p=0.2107; conversation Δ -43.805 [-124.547, 26.704], p=0.2963
  behavior:
    geometry_keep_compress_drop__vs__geometry: row Δ answer NLL 0.2339 [-0.0396, 0.6491], p=0.4873; conversation Δ 0.2339 [-0.0547, 0.7643], p=1.0000
    semantic_keep_compress_drop__vs__semantic: row Δ answer NLL 0.2946 [0.0266, 0.6087], p=0.0927; conversation Δ 0.2946 [0.0001, 0.7112], p=0.1827
    semantic_keep_compress_drop__vs__geometry: row Δ answer NLL 0.1436 [-0.3130, 0.5907], p=0.6520; conversation Δ 0.1436 [-0.3326, 0.6524], p=0.7047
    semantic_keep_compress_drop__vs__geometry_keep_compress_drop: row Δ answer NLL -0.0903 [-0.6257, 0.3982], p=0.7840; conversation Δ -0.0903 [-0.6501, 0.4894], p=0.8153
- budget 0.50:
  geometry_keep_compress_drop__vs__geometry: row Δ logit L2 47.282 [-3.031, 106.761], p=0.1505; conversation Δ 47.282 [-15.110, 156.956], p=1.0000
  semantic_keep_compress_drop__vs__semantic: row Δ logit L2 11.236 [-0.068, 25.872], p=0.0988; conversation Δ 11.236 [2.035, 24.660], p=0.0607
  semantic_keep_compress_drop__vs__geometry: row Δ logit L2 2.668 [-22.693, 24.915], p=0.8230; conversation Δ 2.668 [-16.144, 21.565], p=0.7920
  semantic_keep_compress_drop__vs__geometry_keep_compress_drop: row Δ logit L2 -44.614 [-106.485, 2.744], p=0.1570; conversation Δ -44.614 [-156.339, 20.775], p=0.8390
  behavior:
    geometry_keep_compress_drop__vs__geometry: row Δ answer NLL 0.1977 [-0.0160, 0.5873], p=0.4993; conversation Δ 0.1977 [-0.0240, 0.6170], p=1.0000
    semantic_keep_compress_drop__vs__semantic: row Δ answer NLL 0.1969 [0.0226, 0.4492], p=0.0307; conversation Δ 0.1969 [0.0170, 0.5201], p=0.0678
    semantic_keep_compress_drop__vs__geometry: row Δ answer NLL 0.2437 [0.0039, 0.5224], p=0.1182; conversation Δ 0.2437 [-0.0335, 0.6156], p=0.2470
    semantic_keep_compress_drop__vs__geometry_keep_compress_drop: row Δ answer NLL 0.0460 [-0.4549, 0.4303], p=0.8592; conversation Δ 0.0460 [-0.3532, 0.5055], p=0.6940
