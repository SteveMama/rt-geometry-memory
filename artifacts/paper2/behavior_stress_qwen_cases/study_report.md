# Paper 2 Study: behavior_stress_qwen_cases

- Created: 2026-03-27T20:03:29
- Models: qwen25_05b
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
