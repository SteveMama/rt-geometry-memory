# Paper 3 Pairwise Policy Analysis

Negative deltas mean the left policy is better than the right policy.

## qwen25_15b

- budget 0.24:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -5.574 [-15.079, 0.573], p=0.2420
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -19.441 [-35.953, -6.145], p=0.0057
  geometry_segment_actions__vs__geometry: Δ logit L2 13.867 [2.324, 28.779], p=0.0245
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL 0.0417 [-0.0449, 0.1671], p=0.7545
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL 0.1956 [-0.0091, 0.4408], p=0.1705
    geometry_segment_actions__vs__geometry: Δ answer NLL -0.1540 [-0.3898, 0.0084], p=0.2973
- budget 0.28:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -2.219 [-6.266, 0.400], p=0.4990
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -28.203 [-45.679, -13.122], p=0.0013
  geometry_segment_actions__vs__geometry: Δ logit L2 25.983 [10.456, 43.683], p=0.0020
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.0131 [-0.0478, 0.0056], p=1.0000
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL 0.0051 [-0.1573, 0.2109], p=0.9480
    geometry_segment_actions__vs__geometry: Δ answer NLL -0.0182 [-0.2244, 0.1394], p=0.8998
- budget 0.32:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -0.193 [-7.748, 9.028], p=1.0000
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -24.964 [-44.954, -6.324], p=0.0177
  geometry_segment_actions__vs__geometry: Δ logit L2 24.770 [8.172, 41.966], p=0.0065
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.1908 [-0.4638, 0.0000], p=0.4880
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.2281 [-0.5407, 0.0018], p=0.0695
    geometry_segment_actions__vs__geometry: Δ answer NLL 0.0373 [-0.0413, 0.1265], p=0.3272
- budget 0.35:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -18.080 [-46.256, 2.446], p=0.1308
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -16.628 [-36.275, 3.745], p=0.1175
  geometry_segment_actions__vs__geometry: Δ logit L2 -1.452 [-32.256, 23.755], p=0.9385
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.1908 [-0.5169, 0.0000], p=0.4980
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.2194 [-0.4924, 0.0117], p=0.1410
    geometry_segment_actions__vs__geometry: Δ answer NLL 0.0286 [-0.0550, 0.1201], p=0.5600
- budget 0.38:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -18.999 [-43.578, -2.751], p=0.0325
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -21.013 [-42.030, -0.124], p=0.0545
  geometry_segment_actions__vs__geometry: Δ logit L2 2.014 [-29.185, 30.601], p=0.9177
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.1908 [-0.4904, 0.0000], p=0.5058
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.1043 [-0.4456, 0.2031], p=0.5833
    geometry_segment_actions__vs__geometry: Δ answer NLL -0.0866 [-0.2819, 0.0660], p=0.4690
- budget 0.42:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -14.994 [-42.782, 5.657], p=0.2775
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -9.390 [-34.116, 17.044], p=0.4940
  geometry_segment_actions__vs__geometry: Δ logit L2 -5.603 [-40.082, 24.295], p=0.7650
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.1288 [-0.3281, -0.0071], p=0.1265
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.1616 [-0.4118, 0.0954], p=0.2308
    geometry_segment_actions__vs__geometry: Δ answer NLL 0.0328 [-0.1591, 0.2179], p=0.8685
- budget 0.46:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 -9.777 [-36.519, 9.591], p=0.5680
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 -0.782 [-17.544, 13.484], p=0.9387
  geometry_segment_actions__vs__geometry: Δ logit L2 -8.995 [-37.513, 12.483], p=0.5757
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.1288 [-0.3281, -0.0142], p=0.1265
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.1153 [-0.2528, -0.0206], p=0.0215
    geometry_segment_actions__vs__geometry: Δ answer NLL -0.0134 [-0.2384, 0.1829], p=0.9042
- budget 0.50:
  geometry_keep_compress_drop__vs__geometry: Δ logit L2 8.043 [-11.741, 30.006], p=0.5302
  geometry_keep_compress_drop__vs__geometry_segment_actions: Δ logit L2 3.298 [-19.020, 27.245], p=0.7970
  geometry_segment_actions__vs__geometry: Δ logit L2 4.745 [-8.212, 18.639], p=0.4818
  behavior:
    geometry_keep_compress_drop__vs__geometry: Δ answer NLL -0.1288 [-0.3137, -0.0071], p=0.1273
    geometry_keep_compress_drop__vs__geometry_segment_actions: Δ answer NLL -0.1153 [-0.2526, -0.0206], p=0.0257
    geometry_segment_actions__vs__geometry: Δ answer NLL -0.0134 [-0.2378, 0.1806], p=0.9015
