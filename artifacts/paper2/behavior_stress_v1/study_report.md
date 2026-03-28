# Paper 2 Study: behavior_stress_v1

- Created: 2026-03-27T19:50:03
- Models: qwen25_05b, qwen25_15b, smollm2_17b
- Families: long_dependency, retrieval_heavy, code_conversation
- Budgets: 0.20, 0.35, 0.50

## By Model

### qwen25_05b

- Model name: `Qwen/Qwen2.5-0.5B-Instruct`
- Conversations: 9
- Evaluations: 648
- Behavior evaluations: 324
- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 -65.654, relative logit L2 0.836
  geometry_lexical: delta logit L2 -61.141, relative logit L2 0.847
  geometry_segment_actions: delta logit L2 -67.566, relative logit L2 0.831
  lexical: delta logit L2 -28.836, relative logit L2 0.928
  uniform_segment_actions: delta logit L2 -66.691, relative logit L2 0.833
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -99.503, relative logit L2 0.734
  geometry_lexical: delta logit L2 -76.474, relative logit L2 0.796
  geometry_segment_actions: delta logit L2 -99.116, relative logit L2 0.735
  lexical: delta logit L2 -33.992, relative logit L2 0.909
  uniform_segment_actions: delta logit L2 -97.115, relative logit L2 0.741
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -41.668, relative logit L2 0.862
  geometry_lexical: delta logit L2 -17.953, relative logit L2 0.940
  geometry_segment_actions: delta logit L2 -98.003, relative logit L2 0.675
  lexical: delta logit L2 -4.695, relative logit L2 0.984
  uniform_segment_actions: delta logit L2 -54.379, relative logit L2 0.820
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL -0.0835, delta answer-loss increase -0.0835
  geometry_lexical: delta answer avg NLL 0.0211, delta answer-loss increase 0.0211
  geometry_segment_actions: delta answer avg NLL -0.0490, delta answer-loss increase -0.0490
  lexical: delta answer avg NLL 0.0211, delta answer-loss increase 0.0211
  uniform_segment_actions: delta answer avg NLL -0.0701, delta answer-loss increase -0.0701
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.0912, delta answer-loss increase -0.0912
  geometry_lexical: delta answer avg NLL -0.0122, delta answer-loss increase -0.0122
  geometry_segment_actions: delta answer avg NLL 0.0054, delta answer-loss increase 0.0054
  lexical: delta answer avg NLL -0.0088, delta answer-loss increase -0.0088
  uniform_segment_actions: delta answer avg NLL 0.0019, delta answer-loss increase 0.0019
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.4281, delta answer-loss increase -0.4281
  geometry_lexical: delta answer avg NLL -0.4167, delta answer-loss increase -0.4167
  geometry_segment_actions: delta answer avg NLL -0.3794, delta answer-loss increase -0.3794
  lexical: delta answer avg NLL -0.4859, delta answer-loss increase -0.4859
  uniform_segment_actions: delta answer avg NLL -0.2694, delta answer-loss increase -0.2694

### qwen25_15b

- Model name: `Qwen/Qwen2.5-1.5B-Instruct`
- Conversations: 9
- Evaluations: 648
- Behavior evaluations: 324
- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 -33.647, relative logit L2 0.913
  geometry_lexical: delta logit L2 -30.586, relative logit L2 0.921
  geometry_segment_actions: delta logit L2 -40.078, relative logit L2 0.896
  lexical: delta logit L2 -18.866, relative logit L2 0.951
  uniform_segment_actions: delta logit L2 -39.648, relative logit L2 0.898
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -21.669, relative logit L2 0.935
  geometry_lexical: delta logit L2 -4.908, relative logit L2 0.985
  geometry_segment_actions: delta logit L2 -23.908, relative logit L2 0.929
  lexical: delta logit L2 6.253, relative logit L2 1.019
  uniform_segment_actions: delta logit L2 -21.229, relative logit L2 0.937
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -1.749, relative logit L2 0.993
  geometry_lexical: delta logit L2 -1.547, relative logit L2 0.994
  geometry_segment_actions: delta logit L2 1.420, relative logit L2 1.005
  lexical: delta logit L2 -2.163, relative logit L2 0.992
  uniform_segment_actions: delta logit L2 23.314, relative logit L2 1.088
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL 0.0229, delta answer-loss increase 0.0229
  geometry_lexical: delta answer avg NLL 0.0022, delta answer-loss increase 0.0022
  geometry_segment_actions: delta answer avg NLL -0.0065, delta answer-loss increase -0.0065
  lexical: delta answer avg NLL 0.0022, delta answer-loss increase 0.0022
  uniform_segment_actions: delta answer avg NLL -0.0087, delta answer-loss increase -0.0087
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.1814, delta answer-loss increase -0.1814
  geometry_lexical: delta answer avg NLL -0.0916, delta answer-loss increase -0.0916
  geometry_segment_actions: delta answer avg NLL -0.1542, delta answer-loss increase -0.1542
  lexical: delta answer avg NLL -0.0187, delta answer-loss increase -0.0187
  uniform_segment_actions: delta answer avg NLL -0.1118, delta answer-loss increase -0.1118
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.5090, delta answer-loss increase -0.5090
  geometry_lexical: delta answer avg NLL -0.4562, delta answer-loss increase -0.4562
  geometry_segment_actions: delta answer avg NLL -0.5215, delta answer-loss increase -0.5215
  lexical: delta answer avg NLL -0.6255, delta answer-loss increase -0.6255
  uniform_segment_actions: delta answer avg NLL -0.3896, delta answer-loss increase -0.3896

### smollm2_17b

- Model name: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Conversations: 9
- Evaluations: 648
- Behavior evaluations: 324
- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 -4.105, relative logit L2 0.982
  geometry_lexical: delta logit L2 -5.382, relative logit L2 0.977
  geometry_segment_actions: delta logit L2 -4.105, relative logit L2 0.982
  lexical: delta logit L2 -3.518, relative logit L2 0.985
  uniform_segment_actions: delta logit L2 -13.001, relative logit L2 0.944
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -11.329, relative logit L2 0.948
  geometry_lexical: delta logit L2 -7.654, relative logit L2 0.965
  geometry_segment_actions: delta logit L2 -19.689, relative logit L2 0.909
  lexical: delta logit L2 -9.261, relative logit L2 0.957
  uniform_segment_actions: delta logit L2 -23.072, relative logit L2 0.893
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -19.776, relative logit L2 0.898
  geometry_lexical: delta logit L2 -14.653, relative logit L2 0.925
  geometry_segment_actions: delta logit L2 -25.208, relative logit L2 0.870
  lexical: delta logit L2 -8.081, relative logit L2 0.958
  uniform_segment_actions: delta logit L2 -17.806, relative logit L2 0.908
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL -0.0287, delta answer-loss increase -0.0287
  geometry_lexical: delta answer avg NLL 0.0116, delta answer-loss increase 0.0116
  geometry_segment_actions: delta answer avg NLL -0.0287, delta answer-loss increase -0.0287
  lexical: delta answer avg NLL 0.0100, delta answer-loss increase 0.0100
  uniform_segment_actions: delta answer avg NLL -0.0405, delta answer-loss increase -0.0405
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.0024, delta answer-loss increase -0.0024
  geometry_lexical: delta answer avg NLL -0.0044, delta answer-loss increase -0.0044
  geometry_segment_actions: delta answer avg NLL -0.0024, delta answer-loss increase -0.0024
  lexical: delta answer avg NLL -0.0149, delta answer-loss increase -0.0149
  uniform_segment_actions: delta answer avg NLL -0.0699, delta answer-loss increase -0.0699
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.2822, delta answer-loss increase -0.2822
  geometry_lexical: delta answer avg NLL -0.2597, delta answer-loss increase -0.2597
  geometry_segment_actions: delta answer avg NLL -0.3665, delta answer-loss increase -0.3665
  lexical: delta answer avg NLL -0.4405, delta answer-loss increase -0.4405
  uniform_segment_actions: delta answer avg NLL -0.2709, delta answer-loss increase -0.2709

## Confidence And Significance

### qwen25_05b

- budget 0.20:
  geometry: mean delta logit L2 -65.654 [-116.524, -24.108], p=0.0000
  lexical: mean delta logit L2 -28.836 [-69.631, -0.795], p=0.0948
  geometry_lexical: mean delta logit L2 -61.141 [-109.680, -19.057], p=0.0025
- budget 0.35:
  geometry: mean delta logit L2 -99.503 [-162.757, -46.524], p=0.0000
  lexical: mean delta logit L2 -33.992 [-64.765, -9.769], p=0.0010
  geometry_lexical: mean delta logit L2 -76.474 [-133.347, -27.713], p=0.0043
- budget 0.50:
  geometry: mean delta logit L2 -41.668 [-107.015, 15.240], p=0.2145
  lexical: mean delta logit L2 -4.695 [-52.542, 43.425], p=0.8458
  geometry_lexical: mean delta logit L2 -17.953 [-77.002, 37.365], p=0.5413
- Behavior:
  budget 0.20:
    geometry: mean delta answer avg NLL -0.0835 [-0.3101, 0.0526], p=1.0000
    lexical: mean delta answer avg NLL 0.0211 [0.0000, 0.0596], p=0.4918
    geometry_lexical: mean delta answer avg NLL 0.0211 [0.0000, 0.0596], p=0.5098
    uniform_segment_actions: mean delta answer avg NLL -0.0701 [-0.3121, 0.0713], p=1.0000
    geometry_segment_actions: mean delta answer avg NLL -0.0490 [-0.2926, 0.0970], p=1.0000
  budget 0.35:
    geometry: mean delta answer avg NLL -0.0912 [-0.3348, 0.0625], p=0.7222
    lexical: mean delta answer avg NLL -0.0088 [-0.0542, 0.0210], p=0.8772
    geometry_lexical: mean delta answer avg NLL -0.0122 [-0.0517, 0.0179], p=0.6208
    uniform_segment_actions: mean delta answer avg NLL 0.0019 [-0.0779, 0.0836], p=1.0000
    geometry_segment_actions: mean delta answer avg NLL 0.0054 [-0.0724, 0.0855], p=0.7562
  budget 0.50:
    geometry: mean delta answer avg NLL -0.4281 [-0.8128, -0.1028], p=0.0335
    lexical: mean delta answer avg NLL -0.4859 [-0.8344, -0.1783], p=0.0053
    geometry_lexical: mean delta answer avg NLL -0.4167 [-0.7864, -0.0919], p=0.0395
    uniform_segment_actions: mean delta answer avg NLL -0.2694 [-0.5041, -0.0670], p=0.0243
    geometry_segment_actions: mean delta answer avg NLL -0.3794 [-0.7291, -0.0841], p=0.0470

### qwen25_15b

- budget 0.20:
  geometry: mean delta logit L2 -33.647 [-65.257, -10.096], p=0.0015
  lexical: mean delta logit L2 -18.866 [-49.989, 0.935], p=0.1510
  geometry_lexical: mean delta logit L2 -30.586 [-62.230, -6.131], p=0.0092
- budget 0.35:
  geometry: mean delta logit L2 -21.669 [-58.261, 14.471], p=0.2637
  lexical: mean delta logit L2 6.253 [-17.749, 29.496], p=0.6388
  geometry_lexical: mean delta logit L2 -4.908 [-29.291, 16.792], p=0.7023
- budget 0.50:
  geometry: mean delta logit L2 -1.749 [-29.608, 24.748], p=0.9022
  lexical: mean delta logit L2 -2.163 [-34.466, 28.170], p=0.8918
  geometry_lexical: mean delta logit L2 -1.547 [-28.503, 24.804], p=0.9147
- Behavior:
  budget 0.20:
    geometry: mean delta answer avg NLL 0.0229 [0.0000, 0.0640], p=0.5130
    lexical: mean delta answer avg NLL 0.0022 [-0.0479, 0.0546], p=1.0000
    geometry_lexical: mean delta answer avg NLL 0.0022 [-0.0479, 0.0546], p=1.0000
    uniform_segment_actions: mean delta answer avg NLL -0.0087 [-0.1951, 0.1351], p=1.0000
    geometry_segment_actions: mean delta answer avg NLL -0.0065 [-0.1926, 0.1409], p=1.0000
  budget 0.35:
    geometry: mean delta answer avg NLL -0.1814 [-0.3955, -0.0032], p=0.1343
    lexical: mean delta answer avg NLL -0.0187 [-0.0878, 0.0442], p=0.6322
    geometry_lexical: mean delta answer avg NLL -0.0916 [-0.2410, 0.0179], p=0.2522
    uniform_segment_actions: mean delta answer avg NLL -0.1118 [-0.2934, 0.0000], p=0.5012
    geometry_segment_actions: mean delta answer avg NLL -0.1542 [-0.3216, -0.0264], p=0.1255
  budget 0.50:
    geometry: mean delta answer avg NLL -0.5090 [-0.8803, -0.1536], p=0.0140
    lexical: mean delta answer avg NLL -0.6255 [-1.0867, -0.2255], p=0.0060
    geometry_lexical: mean delta answer avg NLL -0.4562 [-0.8655, -0.1086], p=0.0382
    uniform_segment_actions: mean delta answer avg NLL -0.3896 [-0.7530, -0.1000], p=0.0238
    geometry_segment_actions: mean delta answer avg NLL -0.5215 [-0.9805, -0.1148], p=0.0478

### smollm2_17b

- budget 0.20:
  geometry: mean delta logit L2 -4.105 [-11.622, 2.853], p=0.2878
  lexical: mean delta logit L2 -3.518 [-8.432, -0.047], p=0.0912
  geometry_lexical: mean delta logit L2 -5.382 [-12.523, 0.002], p=0.1145
- budget 0.35:
  geometry: mean delta logit L2 -11.329 [-20.587, -2.331], p=0.0200
  lexical: mean delta logit L2 -9.261 [-17.016, -2.017], p=0.0235
  geometry_lexical: mean delta logit L2 -7.654 [-18.122, 3.850], p=0.2032
- budget 0.50:
  geometry: mean delta logit L2 -19.776 [-36.184, -3.691], p=0.0230
  lexical: mean delta logit L2 -8.081 [-22.261, 5.970], p=0.2993
  geometry_lexical: mean delta logit L2 -14.653 [-30.582, 0.965], p=0.0900
- Behavior:
  budget 0.20:
    geometry: mean delta answer avg NLL -0.0287 [-0.2241, 0.0918], p=1.0000
    lexical: mean delta answer avg NLL 0.0100 [-0.0058, 0.0356], p=1.0000
    geometry_lexical: mean delta answer avg NLL 0.0116 [-0.0038, 0.0373], p=0.7410
    uniform_segment_actions: mean delta answer avg NLL -0.0405 [-0.2311, 0.0817], p=1.0000
    geometry_segment_actions: mean delta answer avg NLL -0.0287 [-0.2102, 0.0940], p=1.0000
  budget 0.35:
    geometry: mean delta answer avg NLL -0.0024 [-0.0297, 0.0281], p=1.0000
    lexical: mean delta answer avg NLL -0.0149 [-0.0361, 0.0011], p=0.2532
    geometry_lexical: mean delta answer avg NLL -0.0044 [-0.0320, 0.0263], p=0.8740
    uniform_segment_actions: mean delta answer avg NLL -0.0699 [-0.2050, 0.0000], p=0.4965
    geometry_segment_actions: mean delta answer avg NLL -0.0024 [-0.0297, 0.0267], p=1.0000
  budget 0.50:
    geometry: mean delta answer avg NLL -0.2822 [-0.5662, -0.0358], p=0.0700
    lexical: mean delta answer avg NLL -0.4405 [-0.7395, -0.1690], p=0.0035
    geometry_lexical: mean delta answer avg NLL -0.2597 [-0.5016, -0.0345], p=0.0688
    uniform_segment_actions: mean delta answer avg NLL -0.2709 [-0.5352, -0.0483], p=0.0508
    geometry_segment_actions: mean delta answer avg NLL -0.3665 [-0.6461, -0.1185], p=0.0302

## Aggregate Policy Means

- qwen25_05b | geometry | budget 0.20: logit L2 334.052 [257.674, 418.264], KL 0.000009, top1 1.000, token fraction 0.541
- qwen25_05b | geometry | budget 0.35: logit L2 275.085 [213.860, 349.481], KL 0.000010, top1 1.000, token fraction 0.640
- qwen25_05b | geometry | budget 0.50: logit L2 259.982 [201.617, 323.987], KL 0.000007, top1 1.000, token fraction 0.730
- qwen25_05b | geometry_lexical | budget 0.20: logit L2 338.565 [264.048, 427.084], KL 0.000009, top1 1.000, token fraction 0.539
- qwen25_05b | geometry_lexical | budget 0.35: logit L2 298.114 [232.682, 372.793], KL 0.000010, top1 1.000, token fraction 0.628
- qwen25_05b | geometry_lexical | budget 0.50: logit L2 283.697 [214.183, 363.016], KL 0.000009, top1 1.000, token fraction 0.721
- qwen25_05b | geometry_segment_actions | budget 0.20: logit L2 332.141 [258.183, 413.756], KL 0.000009, top1 1.000, token fraction 0.542
- qwen25_05b | geometry_segment_actions | budget 0.35: logit L2 275.472 [214.877, 352.735], KL 0.000010, top1 1.000, token fraction 0.647
- qwen25_05b | geometry_segment_actions | budget 0.50: logit L2 203.648 [170.196, 241.569], KL 0.000007, top1 1.000, token fraction 0.727
- qwen25_05b | lexical | budget 0.20: logit L2 370.870 [288.264, 452.073], KL 0.000010, top1 1.000, token fraction 0.542
- qwen25_05b | lexical | budget 0.35: logit L2 340.596 [267.854, 427.931], KL 0.000010, top1 1.000, token fraction 0.627
- qwen25_05b | lexical | budget 0.50: logit L2 296.956 [229.090, 374.875], KL 0.000008, top1 1.000, token fraction 0.716
- qwen25_05b | uniform | budget 0.20: logit L2 399.706 [313.629, 491.732], KL 0.000010, top1 1.000, token fraction 0.547
- qwen25_05b | uniform | budget 0.35: logit L2 374.588 [295.121, 462.487], KL 0.000011, top1 1.000, token fraction 0.632
- qwen25_05b | uniform | budget 0.50: logit L2 301.651 [235.431, 382.514], KL 0.000008, top1 1.000, token fraction 0.680
- qwen25_05b | uniform_segment_actions | budget 0.20: logit L2 333.015 [259.660, 415.076], KL 0.000009, top1 1.000, token fraction 0.543
- qwen25_05b | uniform_segment_actions | budget 0.35: logit L2 277.473 [213.983, 348.930], KL 0.000010, top1 1.000, token fraction 0.648
- qwen25_05b | uniform_segment_actions | budget 0.50: logit L2 247.271 [209.549, 287.880], KL 0.000010, top1 1.000, token fraction 0.706
- qwen25_15b | geometry | budget 0.20: logit L2 353.253 [299.228, 407.397], KL 0.000013, top1 1.000, token fraction 0.540
- qwen25_15b | geometry | budget 0.35: logit L2 313.164 [266.527, 368.807], KL 0.000011, top1 1.000, token fraction 0.638
- qwen25_15b | geometry | budget 0.50: logit L2 263.116 [232.617, 294.502], KL 0.000010, top1 1.000, token fraction 0.726
- qwen25_15b | geometry_lexical | budget 0.20: logit L2 356.314 [301.387, 413.268], KL 0.000013, top1 1.000, token fraction 0.540
- qwen25_15b | geometry_lexical | budget 0.35: logit L2 329.925 [280.033, 381.307], KL 0.000012, top1 1.000, token fraction 0.629
- qwen25_15b | geometry_lexical | budget 0.50: logit L2 263.319 [235.055, 291.294], KL 0.000010, top1 1.000, token fraction 0.721
- qwen25_15b | geometry_segment_actions | budget 0.20: logit L2 346.822 [295.237, 401.743], KL 0.000013, top1 1.000, token fraction 0.542
- qwen25_15b | geometry_segment_actions | budget 0.35: logit L2 310.925 [275.400, 347.454], KL 0.000014, top1 1.000, token fraction 0.646
- qwen25_15b | geometry_segment_actions | budget 0.50: logit L2 266.285 [235.324, 300.155], KL 0.000010, top1 1.000, token fraction 0.727
- qwen25_15b | lexical | budget 0.20: logit L2 368.034 [310.506, 429.720], KL 0.000013, top1 1.000, token fraction 0.542
- qwen25_15b | lexical | budget 0.35: logit L2 341.086 [294.084, 391.002], KL 0.000013, top1 1.000, token fraction 0.627
- qwen25_15b | lexical | budget 0.50: logit L2 262.702 [234.803, 293.074], KL 0.000010, top1 1.000, token fraction 0.716
- qwen25_15b | uniform | budget 0.20: logit L2 386.900 [321.320, 452.018], KL 0.000013, top1 1.000, token fraction 0.547
- qwen25_15b | uniform | budget 0.35: logit L2 334.833 [287.198, 388.929], KL 0.000012, top1 1.000, token fraction 0.632
- qwen25_15b | uniform | budget 0.50: logit L2 264.865 [230.457, 305.680], KL 0.000011, top1 1.000, token fraction 0.680
- qwen25_15b | uniform_segment_actions | budget 0.20: logit L2 347.252 [298.354, 396.267], KL 0.000013, top1 1.000, token fraction 0.543
- qwen25_15b | uniform_segment_actions | budget 0.35: logit L2 313.604 [278.025, 351.322], KL 0.000014, top1 1.000, token fraction 0.648
- qwen25_15b | uniform_segment_actions | budget 0.50: logit L2 288.179 [255.653, 322.759], KL 0.000012, top1 1.000, token fraction 0.706
- smollm2_17b | geometry | budget 0.20: logit L2 228.719 [212.731, 245.400], KL 0.112221, top1 0.806, token fraction 0.549
- smollm2_17b | geometry | budget 0.35: logit L2 204.497 [187.858, 221.419], KL 0.046555, top1 0.889, token fraction 0.625
- smollm2_17b | geometry | budget 0.50: logit L2 174.804 [161.498, 188.403], KL 0.028890, top1 0.889, token fraction 0.705
- smollm2_17b | geometry_lexical | budget 0.20: logit L2 227.443 [212.755, 243.108], KL 0.098898, top1 0.806, token fraction 0.546
- smollm2_17b | geometry_lexical | budget 0.35: logit L2 208.171 [191.695, 227.529], KL 0.069259, top1 0.861, token fraction 0.624
- smollm2_17b | geometry_lexical | budget 0.50: logit L2 179.927 [166.841, 193.309], KL 0.046039, top1 0.833, token fraction 0.704
- smollm2_17b | geometry_segment_actions | budget 0.20: logit L2 228.719 [212.808, 245.701], KL 0.112221, top1 0.806, token fraction 0.549
- smollm2_17b | geometry_segment_actions | budget 0.35: logit L2 196.137 [184.253, 208.054], KL 0.044435, top1 0.889, token fraction 0.638
- smollm2_17b | geometry_segment_actions | budget 0.50: logit L2 169.372 [155.728, 183.170], KL 0.046388, top1 0.861, token fraction 0.721
- smollm2_17b | lexical | budget 0.20: logit L2 229.307 [214.737, 246.236], KL 0.102561, top1 0.806, token fraction 0.545
- smollm2_17b | lexical | budget 0.35: logit L2 206.565 [191.789, 223.113], KL 0.067651, top1 0.889, token fraction 0.628
- smollm2_17b | lexical | budget 0.50: logit L2 186.499 [174.313, 200.748], KL 0.054771, top1 0.806, token fraction 0.716
- smollm2_17b | uniform | budget 0.20: logit L2 232.825 [216.966, 250.348], KL 0.099158, top1 0.750, token fraction 0.552
- smollm2_17b | uniform | budget 0.35: logit L2 215.826 [200.341, 232.524], KL 0.101058, top1 0.806, token fraction 0.633
- smollm2_17b | uniform | budget 0.50: logit L2 194.580 [179.175, 210.383], KL 0.082640, top1 0.806, token fraction 0.682
- smollm2_17b | uniform_segment_actions | budget 0.20: logit L2 219.824 [205.926, 236.398], KL 0.090494, top1 0.833, token fraction 0.547
- smollm2_17b | uniform_segment_actions | budget 0.35: logit L2 192.754 [182.662, 203.654], KL 0.046928, top1 0.861, token fraction 0.648
- smollm2_17b | uniform_segment_actions | budget 0.50: logit L2 176.774 [168.760, 185.315], KL 0.035435, top1 0.833, token fraction 0.708

## Behavior Policy Means

- qwen25_05b | geometry | budget 0.20: answer avg NLL 1.5011 [1.1375, 1.8696], answer-loss increase 0.6930
- qwen25_05b | geometry | budget 0.35: answer avg NLL 1.1229 [0.9078, 1.3611], answer-loss increase 0.3148
- qwen25_05b | geometry | budget 0.50: answer avg NLL 0.8895 [0.7195, 1.0685], answer-loss increase 0.0814
- qwen25_05b | geometry_lexical | budget 0.20: answer avg NLL 1.6056 [1.2387, 1.9644], answer-loss increase 0.7975
- qwen25_05b | geometry_lexical | budget 0.35: answer avg NLL 1.2019 [0.9072, 1.5246], answer-loss increase 0.3938
- qwen25_05b | geometry_lexical | budget 0.50: answer avg NLL 0.9009 [0.7305, 1.0712], answer-loss increase 0.0928
- qwen25_05b | geometry_segment_actions | budget 0.20: answer avg NLL 1.5355 [1.1579, 1.9338], answer-loss increase 0.7274
- qwen25_05b | geometry_segment_actions | budget 0.35: answer avg NLL 1.2196 [0.9239, 1.5382], answer-loss increase 0.4115
- qwen25_05b | geometry_segment_actions | budget 0.50: answer avg NLL 0.9381 [0.7845, 1.1060], answer-loss increase 0.1300
- qwen25_05b | lexical | budget 0.20: answer avg NLL 1.6056 [1.2534, 1.9799], answer-loss increase 0.7975
- qwen25_05b | lexical | budget 0.35: answer avg NLL 1.2054 [0.9055, 1.5512], answer-loss increase 0.3973
- qwen25_05b | lexical | budget 0.50: answer avg NLL 0.8317 [0.6803, 0.9713], answer-loss increase 0.0235
- qwen25_05b | uniform | budget 0.20: answer avg NLL 1.5846 [1.2060, 1.9813], answer-loss increase 0.7765
- qwen25_05b | uniform | budget 0.35: answer avg NLL 1.2142 [0.9264, 1.5227], answer-loss increase 0.4061
- qwen25_05b | uniform | budget 0.50: answer avg NLL 1.3176 [1.0288, 1.6523], answer-loss increase 0.5095
- qwen25_05b | uniform_segment_actions | budget 0.20: answer avg NLL 1.5145 [1.1320, 1.9253], answer-loss increase 0.7063
- qwen25_05b | uniform_segment_actions | budget 0.35: answer avg NLL 1.2161 [0.9179, 1.5572], answer-loss increase 0.4080
- qwen25_05b | uniform_segment_actions | budget 0.50: answer avg NLL 1.0482 [0.8035, 1.3634], answer-loss increase 0.2401
- qwen25_15b | geometry | budget 0.20: answer avg NLL 1.6459 [1.2276, 2.0770], answer-loss increase 1.0472
- qwen25_15b | geometry | budget 0.35: answer avg NLL 1.0900 [0.7493, 1.5146], answer-loss increase 0.4913
- qwen25_15b | geometry | budget 0.50: answer avg NLL 0.8396 [0.6060, 1.1302], answer-loss increase 0.2408
- qwen25_15b | geometry_lexical | budget 0.20: answer avg NLL 1.6252 [1.2030, 2.0715], answer-loss increase 1.0265
- qwen25_15b | geometry_lexical | budget 0.35: answer avg NLL 1.1798 [0.8125, 1.5661], answer-loss increase 0.5811
- qwen25_15b | geometry_lexical | budget 0.50: answer avg NLL 0.8924 [0.6467, 1.1721], answer-loss increase 0.2937
- qwen25_15b | geometry_segment_actions | budget 0.20: answer avg NLL 1.6165 [1.1579, 2.0918], answer-loss increase 1.0178
- qwen25_15b | geometry_segment_actions | budget 0.35: answer avg NLL 1.1173 [0.7443, 1.5287], answer-loss increase 0.5185
- qwen25_15b | geometry_segment_actions | budget 0.50: answer avg NLL 0.8271 [0.5825, 1.1222], answer-loss increase 0.2284
- qwen25_15b | lexical | budget 0.20: answer avg NLL 1.6252 [1.1982, 2.0508], answer-loss increase 1.0265
- qwen25_15b | lexical | budget 0.35: answer avg NLL 1.2527 [0.8356, 1.7140], answer-loss increase 0.6540
- qwen25_15b | lexical | budget 0.50: answer avg NLL 0.7230 [0.4880, 1.0063], answer-loss increase 0.1243
- qwen25_15b | uniform | budget 0.20: answer avg NLL 1.6230 [1.1842, 2.0801], answer-loss increase 1.0243
- qwen25_15b | uniform | budget 0.35: answer avg NLL 1.2714 [0.8741, 1.7134], answer-loss increase 0.6727
- qwen25_15b | uniform | budget 0.50: answer avg NLL 1.3486 [0.9360, 1.8157], answer-loss increase 0.7498
- qwen25_15b | uniform_segment_actions | budget 0.20: answer avg NLL 1.6143 [1.1739, 2.0866], answer-loss increase 1.0155
- qwen25_15b | uniform_segment_actions | budget 0.35: answer avg NLL 1.1596 [0.7828, 1.5430], answer-loss increase 0.5609
- qwen25_15b | uniform_segment_actions | budget 0.50: answer avg NLL 0.9590 [0.6223, 1.3739], answer-loss increase 0.3603
- smollm2_17b | geometry | budget 0.20: answer avg NLL 1.3465 [0.9412, 1.8116], answer-loss increase 0.7892
- smollm2_17b | geometry | budget 0.35: answer avg NLL 1.0524 [0.7348, 1.4029], answer-loss increase 0.4951
- smollm2_17b | geometry | budget 0.50: answer avg NLL 0.8380 [0.5803, 1.1405], answer-loss increase 0.2807
- smollm2_17b | geometry_lexical | budget 0.20: answer avg NLL 1.3868 [0.9614, 1.8328], answer-loss increase 0.8295
- smollm2_17b | geometry_lexical | budget 0.35: answer avg NLL 1.0505 [0.7455, 1.4111], answer-loss increase 0.4932
- smollm2_17b | geometry_lexical | budget 0.50: answer avg NLL 0.8606 [0.5987, 1.1797], answer-loss increase 0.3033
- smollm2_17b | geometry_segment_actions | budget 0.20: answer avg NLL 1.3465 [0.9291, 1.8009], answer-loss increase 0.7892
- smollm2_17b | geometry_segment_actions | budget 0.35: answer avg NLL 1.0524 [0.7500, 1.3984], answer-loss increase 0.4951
- smollm2_17b | geometry_segment_actions | budget 0.50: answer avg NLL 0.7537 [0.5288, 1.0299], answer-loss increase 0.1964
- smollm2_17b | lexical | budget 0.20: answer avg NLL 1.3852 [0.9717, 1.8227], answer-loss increase 0.8279
- smollm2_17b | lexical | budget 0.35: answer avg NLL 1.0399 [0.7199, 1.3952], answer-loss increase 0.4826
- smollm2_17b | lexical | budget 0.50: answer avg NLL 0.6798 [0.5023, 0.8738], answer-loss increase 0.1225
- smollm2_17b | uniform | budget 0.20: answer avg NLL 1.3752 [0.9410, 1.8374], answer-loss increase 0.8179
- smollm2_17b | uniform | budget 0.35: answer avg NLL 1.0549 [0.7243, 1.3987], answer-loss increase 0.4976
- smollm2_17b | uniform | budget 0.50: answer avg NLL 1.1202 [0.7950, 1.4519], answer-loss increase 0.5630
- smollm2_17b | uniform_segment_actions | budget 0.20: answer avg NLL 1.3347 [0.9188, 1.8157], answer-loss increase 0.7774
- smollm2_17b | uniform_segment_actions | budget 0.35: answer avg NLL 0.9850 [0.6712, 1.3226], answer-loss increase 0.4277
- smollm2_17b | uniform_segment_actions | budget 0.50: answer avg NLL 0.8493 [0.5695, 1.1446], answer-loss increase 0.2920
