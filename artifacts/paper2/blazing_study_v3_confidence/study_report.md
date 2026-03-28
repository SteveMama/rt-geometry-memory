# Paper 2 Study: blazing_study_v3_confidence

- Created: 2026-03-27T18:53:15
- Models: qwen25_05b, qwen25_15b, smollm2_17b
- Families: long_dependency, retrieval_heavy, code_conversation
- Budgets: 0.20, 0.35, 0.50, 0.65

## By Model

### qwen25_05b

- Model name: `Qwen/Qwen2.5-0.5B-Instruct`
- Conversations: 9
- Evaluations: 288
- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 0.000, relative logit L2 1.000
  geometry_lexical: delta logit L2 0.000, relative logit L2 1.000
  lexical: delta logit L2 0.000, relative logit L2 1.000
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -118.761, relative logit L2 0.664
  geometry_lexical: delta logit L2 1.705, relative logit L2 1.005
  lexical: delta logit L2 0.000, relative logit L2 1.000
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -21.401, relative logit L2 0.920
  geometry_lexical: delta logit L2 76.624, relative logit L2 1.287
  lexical: delta logit L2 59.261, relative logit L2 1.222
- Improvement vs uniform @ 0.65:
  geometry: delta logit L2 4.079, relative logit L2 1.015
  geometry_lexical: delta logit L2 4.079, relative logit L2 1.015
  lexical: delta logit L2 1.699, relative logit L2 1.006

### qwen25_15b

- Model name: `Qwen/Qwen2.5-1.5B-Instruct`
- Conversations: 9
- Evaluations: 288
- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 0.000, relative logit L2 1.000
  geometry_lexical: delta logit L2 0.000, relative logit L2 1.000
  lexical: delta logit L2 0.000, relative logit L2 1.000
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -34.553, relative logit L2 0.871
  geometry_lexical: delta logit L2 -13.757, relative logit L2 0.948
  lexical: delta logit L2 0.000, relative logit L2 1.000
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 20.884, relative logit L2 1.095
  geometry_lexical: delta logit L2 32.740, relative logit L2 1.149
  lexical: delta logit L2 56.601, relative logit L2 1.257
- Improvement vs uniform @ 0.65:
  geometry: delta logit L2 41.277, relative logit L2 1.195
  geometry_lexical: delta logit L2 41.277, relative logit L2 1.195
  lexical: delta logit L2 45.131, relative logit L2 1.213

### smollm2_17b

- Model name: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Conversations: 9
- Evaluations: 288
- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 0.000, relative logit L2 1.000
  geometry_lexical: delta logit L2 0.000, relative logit L2 1.000
  lexical: delta logit L2 0.000, relative logit L2 1.000
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -33.713, relative logit L2 0.840
  geometry_lexical: delta logit L2 -10.358, relative logit L2 0.951
  lexical: delta logit L2 0.000, relative logit L2 1.000
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -2.721, relative logit L2 0.984
  geometry_lexical: delta logit L2 10.460, relative logit L2 1.062
  lexical: delta logit L2 20.818, relative logit L2 1.123
- Improvement vs uniform @ 0.65:
  geometry: delta logit L2 -0.908, relative logit L2 0.994
  geometry_lexical: delta logit L2 -0.908, relative logit L2 0.994
  lexical: delta logit L2 -0.908, relative logit L2 0.994

## Confidence And Significance

### qwen25_05b

- budget 0.20:
  geometry: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  geometry_lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
- budget 0.35:
  geometry: mean delta logit L2 -118.761 [-218.108, -35.578], p=0.0285
  lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  geometry_lexical: mean delta logit L2 1.705 [0.000, 5.114], p=1.0000
- budget 0.50:
  geometry: mean delta logit L2 -21.401 [-110.509, 71.775], p=0.6983
  lexical: mean delta logit L2 59.261 [18.229, 121.479], p=0.0100
  geometry_lexical: mean delta logit L2 76.624 [22.079, 140.602], p=0.0088
- budget 0.65:
  geometry: mean delta logit L2 4.079 [-38.736, 51.073], p=0.8638
  lexical: mean delta logit L2 1.699 [-40.023, 47.077], p=0.9513
  geometry_lexical: mean delta logit L2 4.079 [-39.539, 55.314], p=0.8698

### qwen25_15b

- budget 0.20:
  geometry: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  geometry_lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
- budget 0.35:
  geometry: mean delta logit L2 -34.553 [-68.198, -7.919], p=0.0325
  lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  geometry_lexical: mean delta logit L2 -13.757 [-39.975, 2.041], p=0.4958
- budget 0.50:
  geometry: mean delta logit L2 20.884 [3.301, 43.562], p=0.0395
  lexical: mean delta logit L2 56.601 [22.349, 96.102], p=0.0175
  geometry_lexical: mean delta logit L2 32.740 [9.729, 62.187], p=0.0152
- budget 0.65:
  geometry: mean delta logit L2 41.277 [9.293, 81.235], p=0.0340
  lexical: mean delta logit L2 45.131 [13.673, 83.215], p=0.0112
  geometry_lexical: mean delta logit L2 41.277 [7.109, 81.681], p=0.0320

### smollm2_17b

- budget 0.20:
  geometry: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  geometry_lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
- budget 0.35:
  geometry: mean delta logit L2 -33.713 [-53.761, -14.664], p=0.0107
  lexical: mean delta logit L2 0.000 [0.000, 0.000], p=1.0000
  geometry_lexical: mean delta logit L2 -10.358 [-26.495, 0.000], p=0.4913
- budget 0.50:
  geometry: mean delta logit L2 -2.721 [-10.509, 3.757], p=0.5092
  lexical: mean delta logit L2 20.818 [1.234, 43.734], p=0.0695
  geometry_lexical: mean delta logit L2 10.460 [-5.134, 27.488], p=0.2795
- budget 0.65:
  geometry: mean delta logit L2 -0.908 [-20.368, 17.448], p=0.9315
  lexical: mean delta logit L2 -0.908 [-20.692, 17.200], p=0.9280
  geometry_lexical: mean delta logit L2 -0.908 [-19.988, 16.619], p=0.9317

## Aggregate Policy Means

- qwen25_05b | geometry | budget 0.20: logit L2 431.865 [302.714, 569.258], KL 0.000009, top1 1.000, token fraction 0.580
- qwen25_05b | geometry | budget 0.35: logit L2 234.383 [179.564, 302.806], KL 0.000008, top1 1.000, token fraction 0.669
- qwen25_05b | geometry | budget 0.50: logit L2 245.932 [166.568, 337.784], KL 0.000006, top1 1.000, token fraction 0.757
- qwen25_05b | geometry | budget 0.65: logit L2 281.367 [200.789, 361.367], KL 0.000006, top1 1.000, token fraction 0.821
- qwen25_05b | geometry_lexical | budget 0.20: logit L2 431.865 [300.125, 575.536], KL 0.000009, top1 1.000, token fraction 0.580
- qwen25_05b | geometry_lexical | budget 0.35: logit L2 354.849 [252.243, 455.951], KL 0.000008, top1 1.000, token fraction 0.653
- qwen25_05b | geometry_lexical | budget 0.50: logit L2 343.957 [235.981, 462.861], KL 0.000006, top1 1.000, token fraction 0.754
- qwen25_05b | geometry_lexical | budget 0.65: logit L2 281.367 [204.221, 361.587], KL 0.000006, top1 1.000, token fraction 0.821
- qwen25_05b | lexical | budget 0.20: logit L2 431.865 [302.284, 571.858], KL 0.000009, top1 1.000, token fraction 0.580
- qwen25_05b | lexical | budget 0.35: logit L2 353.144 [248.378, 456.098], KL 0.000008, top1 1.000, token fraction 0.655
- qwen25_05b | lexical | budget 0.50: logit L2 326.593 [220.880, 443.053], KL 0.000006, top1 1.000, token fraction 0.751
- qwen25_05b | lexical | budget 0.65: logit L2 278.987 [198.895, 361.291], KL 0.000006, top1 1.000, token fraction 0.818
- qwen25_05b | uniform | budget 0.20: logit L2 431.865 [308.212, 580.283], KL 0.000009, top1 1.000, token fraction 0.580
- qwen25_05b | uniform | budget 0.35: logit L2 353.144 [247.179, 460.158], KL 0.000008, top1 1.000, token fraction 0.655
- qwen25_05b | uniform | budget 0.50: logit L2 267.333 [175.319, 370.015], KL 0.000005, top1 1.000, token fraction 0.746
- qwen25_05b | uniform | budget 0.65: logit L2 277.288 [194.463, 374.885], KL 0.000005, top1 1.000, token fraction 0.760
- qwen25_15b | geometry | budget 0.20: logit L2 314.438 [245.341, 385.496], KL 0.000007, top1 1.000, token fraction 0.580
- qwen25_15b | geometry | budget 0.35: logit L2 232.460 [190.711, 278.682], KL 0.000006, top1 1.000, token fraction 0.669
- qwen25_15b | geometry | budget 0.50: logit L2 240.865 [196.445, 291.531], KL 0.000007, top1 1.000, token fraction 0.752
- qwen25_15b | geometry | budget 0.65: logit L2 253.390 [207.972, 301.769], KL 0.000007, top1 1.000, token fraction 0.821
- qwen25_15b | geometry_lexical | budget 0.20: logit L2 314.438 [245.221, 388.334], KL 0.000007, top1 1.000, token fraction 0.580
- qwen25_15b | geometry_lexical | budget 0.35: logit L2 253.256 [201.491, 309.765], KL 0.000006, top1 1.000, token fraction 0.655
- qwen25_15b | geometry_lexical | budget 0.50: logit L2 252.721 [203.598, 308.468], KL 0.000007, top1 1.000, token fraction 0.747
- qwen25_15b | geometry_lexical | budget 0.65: logit L2 253.390 [205.358, 302.463], KL 0.000007, top1 1.000, token fraction 0.821
- qwen25_15b | lexical | budget 0.20: logit L2 314.438 [247.353, 384.700], KL 0.000007, top1 1.000, token fraction 0.580
- qwen25_15b | lexical | budget 0.35: logit L2 267.013 [215.467, 323.562], KL 0.000006, top1 1.000, token fraction 0.655
- qwen25_15b | lexical | budget 0.50: logit L2 276.582 [222.298, 329.896], KL 0.000007, top1 1.000, token fraction 0.751
- qwen25_15b | lexical | budget 0.65: logit L2 257.244 [209.892, 307.417], KL 0.000007, top1 1.000, token fraction 0.818
- qwen25_15b | uniform | budget 0.20: logit L2 314.438 [242.478, 386.589], KL 0.000007, top1 1.000, token fraction 0.580
- qwen25_15b | uniform | budget 0.35: logit L2 267.013 [214.726, 323.369], KL 0.000006, top1 1.000, token fraction 0.655
- qwen25_15b | uniform | budget 0.50: logit L2 219.981 [187.516, 260.787], KL 0.000007, top1 1.000, token fraction 0.746
- qwen25_15b | uniform | budget 0.65: logit L2 212.113 [175.985, 250.449], KL 0.000005, top1 1.000, token fraction 0.760
- smollm2_17b | geometry | budget 0.20: logit L2 232.513 [202.795, 266.042], KL 0.123332, top1 0.833, token fraction 0.574
- smollm2_17b | geometry | budget 0.35: logit L2 176.935 [163.353, 191.576], KL 0.077708, top1 0.722, token fraction 0.668
- smollm2_17b | geometry | budget 0.50: logit L2 166.978 [152.484, 182.617], KL 0.052093, top1 0.833, token fraction 0.753
- smollm2_17b | geometry | budget 0.65: logit L2 161.055 [149.809, 172.522], KL 0.023649, top1 0.944, token fraction 0.822
- smollm2_17b | geometry_lexical | budget 0.20: logit L2 232.513 [201.910, 264.226], KL 0.123332, top1 0.833, token fraction 0.574
- smollm2_17b | geometry_lexical | budget 0.35: logit L2 200.289 [174.122, 227.931], KL 0.073030, top1 0.778, token fraction 0.655
- smollm2_17b | geometry_lexical | budget 0.50: logit L2 180.160 [160.741, 203.999], KL 0.057021, top1 0.833, token fraction 0.747
- smollm2_17b | geometry_lexical | budget 0.65: logit L2 161.055 [149.011, 171.902], KL 0.023649, top1 0.944, token fraction 0.822
- smollm2_17b | lexical | budget 0.20: logit L2 232.513 [201.856, 264.846], KL 0.123332, top1 0.833, token fraction 0.574
- smollm2_17b | lexical | budget 0.35: logit L2 210.647 [186.217, 237.644], KL 0.068087, top1 0.833, token fraction 0.661
- smollm2_17b | lexical | budget 0.50: logit L2 190.517 [169.703, 214.602], KL 0.052078, top1 0.889, token fraction 0.754
- smollm2_17b | lexical | budget 0.65: logit L2 161.055 [149.574, 171.833], KL 0.023649, top1 0.944, token fraction 0.822
- smollm2_17b | uniform | budget 0.20: logit L2 232.513 [202.287, 264.292], KL 0.123332, top1 0.833, token fraction 0.574
- smollm2_17b | uniform | budget 0.35: logit L2 210.647 [185.633, 237.670], KL 0.068087, top1 0.833, token fraction 0.661
- smollm2_17b | uniform | budget 0.50: logit L2 169.699 [155.652, 184.654], KL 0.050838, top1 0.889, token fraction 0.740
- smollm2_17b | uniform | budget 0.65: logit L2 161.962 [144.500, 180.477], KL 0.053805, top1 0.833, token fraction 0.752
