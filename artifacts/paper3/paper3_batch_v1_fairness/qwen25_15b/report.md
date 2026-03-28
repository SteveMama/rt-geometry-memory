# Paper 3 Pilot: qwen25_15b

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Budgets: 0.24, 0.28, 0.32, 0.35, 0.38, 0.42, 0.46, 0.50
- Segment span: 2
- Conversations: 9
- Evaluations: 1152
- Behavior evaluations: 576

## Aggregate

### geometry

- budget 0.24: logit L2 343.698, KL 0.000012, token fraction 0.584, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.28: logit L2 321.542, KL 0.000011, token fraction 0.605, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.32: logit L2 306.513, KL 0.000011, token fraction 0.628, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 315.146, KL 0.000011, token fraction 0.638, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.38: logit L2 302.804, KL 0.000011, token fraction 0.654, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.42: logit L2 294.029, KL 0.000010, token fraction 0.673, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.46: logit L2 283.551, KL 0.000010, token fraction 0.695, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 264.774, KL 0.000010, token fraction 0.726, kept/compressed/evicted segments 0.00/0.00/0.00

### geometry_keep_compress_drop

- budget 0.24: logit L2 338.123, KL 0.000012, token fraction 0.572, kept/compressed/evicted segments 0.25/0.75/1.00
- budget 0.28: logit L2 319.323, KL 0.000011, token fraction 0.601, kept/compressed/evicted segments 0.36/0.78/0.86
- budget 0.32: logit L2 306.319, KL 0.000010, token fraction 0.628, kept/compressed/evicted segments 0.39/0.83/0.78
- budget 0.35: logit L2 297.066, KL 0.000010, token fraction 0.647, kept/compressed/evicted segments 0.42/0.83/0.75
- budget 0.38: logit L2 283.805, KL 0.000010, token fraction 0.665, kept/compressed/evicted segments 0.53/0.75/0.72
- budget 0.42: logit L2 279.035, KL 0.000010, token fraction 0.689, kept/compressed/evicted segments 0.53/0.94/0.53
- budget 0.46: logit L2 273.774, KL 0.000011, token fraction 0.707, kept/compressed/evicted segments 0.53/1.11/0.36
- budget 0.50: logit L2 272.817, KL 0.000011, token fraction 0.734, kept/compressed/evicted segments 0.50/1.42/0.08

### geometry_segment_actions

- budget 0.24: logit L2 357.565, KL 0.000014, token fraction 0.584, kept/compressed/evicted segments 0.25/0.86/0.89
- budget 0.28: logit L2 347.526, KL 0.000014, token fraction 0.601, kept/compressed/evicted segments 0.36/0.83/0.81
- budget 0.32: logit L2 331.283, KL 0.000014, token fraction 0.627, kept/compressed/evicted segments 0.39/0.97/0.64
- budget 0.35: logit L2 313.694, KL 0.000014, token fraction 0.646, kept/compressed/evicted segments 0.44/1.00/0.56
- budget 0.38: logit L2 304.818, KL 0.000013, token fraction 0.663, kept/compressed/evicted segments 0.53/0.97/0.50
- budget 0.42: logit L2 288.426, KL 0.000011, token fraction 0.682, kept/compressed/evicted segments 0.67/0.81/0.53
- budget 0.46: logit L2 274.556, KL 0.000010, token fraction 0.707, kept/compressed/evicted segments 0.78/0.75/0.47
- budget 0.50: logit L2 269.519, KL 0.000010, token fraction 0.727, kept/compressed/evicted segments 0.86/0.69/0.44

### uniform

- budget 0.24: logit L2 376.396, KL 0.000013, token fraction 0.579, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.28: logit L2 361.461, KL 0.000013, token fraction 0.595, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.32: logit L2 362.485, KL 0.000013, token fraction 0.617, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 336.441, KL 0.000012, token fraction 0.632, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.38: logit L2 325.686, KL 0.000012, token fraction 0.633, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.42: logit L2 309.218, KL 0.000010, token fraction 0.637, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.46: logit L2 295.787, KL 0.000010, token fraction 0.646, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 265.767, KL 0.000012, token fraction 0.680, kept/compressed/evicted segments 0.00/0.00/0.00

## Behavior Aggregate

### geometry

- budget 0.24: answer avg NLL 1.5572, answer delta 0.9581
- budget 0.28: answer avg NLL 1.3644, answer delta 0.7653
- budget 0.32: answer avg NLL 1.0642, answer delta 0.4651
- budget 0.35: answer avg NLL 1.0899, answer delta 0.4908
- budget 0.38: answer avg NLL 1.0827, answer delta 0.4836
- budget 0.42: answer avg NLL 0.9054, answer delta 0.3064
- budget 0.46: answer avg NLL 0.8402, answer delta 0.2412
- budget 0.50: answer avg NLL 0.8402, answer delta 0.2412

### geometry_keep_compress_drop

- budget 0.24: answer avg NLL 1.5989, answer delta 0.9998
- budget 0.28: answer avg NLL 1.3513, answer delta 0.7522
- budget 0.32: answer avg NLL 0.8734, answer delta 0.2743
- budget 0.35: answer avg NLL 0.8990, answer delta 0.3000
- budget 0.38: answer avg NLL 0.8918, answer delta 0.2928
- budget 0.42: answer avg NLL 0.7767, answer delta 0.1776
- budget 0.46: answer avg NLL 0.7115, answer delta 0.1124
- budget 0.50: answer avg NLL 0.7115, answer delta 0.1124

### geometry_segment_actions

- budget 0.24: answer avg NLL 1.4032, answer delta 0.8042
- budget 0.28: answer avg NLL 1.3462, answer delta 0.7471
- budget 0.32: answer avg NLL 1.1015, answer delta 0.5024
- budget 0.35: answer avg NLL 1.1185, answer delta 0.5194
- budget 0.38: answer avg NLL 0.9961, answer delta 0.3970
- budget 0.42: answer avg NLL 0.9383, answer delta 0.3392
- budget 0.46: answer avg NLL 0.8268, answer delta 0.2277
- budget 0.50: answer avg NLL 0.8268, answer delta 0.2277

### uniform

- budget 0.24: answer avg NLL 1.6332, answer delta 1.0342
- budget 0.28: answer avg NLL 1.5761, answer delta 0.9771
- budget 0.32: answer avg NLL 1.2637, answer delta 0.6646
- budget 0.35: answer avg NLL 1.2728, answer delta 0.6737
- budget 0.38: answer avg NLL 1.1977, answer delta 0.5986
- budget 0.42: answer avg NLL 1.0856, answer delta 0.4865
- budget 0.46: answer avg NLL 1.2387, answer delta 0.6396
- budget 0.50: answer avg NLL 1.3490, answer delta 0.7499

## Improvement Vs Uniform

### budget 0.24

- geometry: delta logit L2 -32.698, relative logit L2 0.913
- geometry_keep_compress_drop: delta logit L2 -38.272, relative logit L2 0.898
- geometry_segment_actions: delta logit L2 -18.831, relative logit L2 0.950

### budget 0.28

- geometry: delta logit L2 -39.919, relative logit L2 0.890
- geometry_keep_compress_drop: delta logit L2 -42.138, relative logit L2 0.883
- geometry_segment_actions: delta logit L2 -13.935, relative logit L2 0.961

### budget 0.32

- geometry: delta logit L2 -55.972, relative logit L2 0.846
- geometry_keep_compress_drop: delta logit L2 -56.165, relative logit L2 0.845
- geometry_segment_actions: delta logit L2 -31.202, relative logit L2 0.914

### budget 0.35

- geometry: delta logit L2 -21.295, relative logit L2 0.937
- geometry_keep_compress_drop: delta logit L2 -39.375, relative logit L2 0.883
- geometry_segment_actions: delta logit L2 -22.747, relative logit L2 0.932

### budget 0.38

- geometry: delta logit L2 -22.882, relative logit L2 0.930
- geometry_keep_compress_drop: delta logit L2 -41.881, relative logit L2 0.871
- geometry_segment_actions: delta logit L2 -20.868, relative logit L2 0.936

### budget 0.42

- geometry: delta logit L2 -15.189, relative logit L2 0.951
- geometry_keep_compress_drop: delta logit L2 -30.183, relative logit L2 0.902
- geometry_segment_actions: delta logit L2 -20.793, relative logit L2 0.933

### budget 0.46

- geometry: delta logit L2 -12.236, relative logit L2 0.959
- geometry_keep_compress_drop: delta logit L2 -22.013, relative logit L2 0.926
- geometry_segment_actions: delta logit L2 -21.232, relative logit L2 0.928

### budget 0.50

- geometry: delta logit L2 -0.993, relative logit L2 0.996
- geometry_keep_compress_drop: delta logit L2 7.050, relative logit L2 1.027
- geometry_segment_actions: delta logit L2 3.752, relative logit L2 1.014

## Behavior Improvement Vs Uniform

### budget 0.24

- geometry: delta answer avg NLL -0.0760, delta answer-loss increase -0.0760
- geometry_keep_compress_drop: delta answer avg NLL -0.0344, delta answer-loss increase -0.0344
- geometry_segment_actions: delta answer avg NLL -0.2300, delta answer-loss increase -0.2300

### budget 0.28

- geometry: delta answer avg NLL -0.2118, delta answer-loss increase -0.2118
- geometry_keep_compress_drop: delta answer avg NLL -0.2249, delta answer-loss increase -0.2249
- geometry_segment_actions: delta answer avg NLL -0.2300, delta answer-loss increase -0.2300

### budget 0.32

- geometry: delta answer avg NLL -0.1995, delta answer-loss increase -0.1995
- geometry_keep_compress_drop: delta answer avg NLL -0.3903, delta answer-loss increase -0.3903
- geometry_segment_actions: delta answer avg NLL -0.1622, delta answer-loss increase -0.1622

### budget 0.35

- geometry: delta answer avg NLL -0.1829, delta answer-loss increase -0.1829
- geometry_keep_compress_drop: delta answer avg NLL -0.3737, delta answer-loss increase -0.3737
- geometry_segment_actions: delta answer avg NLL -0.1543, delta answer-loss increase -0.1543

### budget 0.38

- geometry: delta answer avg NLL -0.1150, delta answer-loss increase -0.1150
- geometry_keep_compress_drop: delta answer avg NLL -0.3058, delta answer-loss increase -0.3058
- geometry_segment_actions: delta answer avg NLL -0.2016, delta answer-loss increase -0.2016

### budget 0.42

- geometry: delta answer avg NLL -0.1801, delta answer-loss increase -0.1801
- geometry_keep_compress_drop: delta answer avg NLL -0.3089, delta answer-loss increase -0.3089
- geometry_segment_actions: delta answer avg NLL -0.1473, delta answer-loss increase -0.1473

### budget 0.46

- geometry: delta answer avg NLL -0.3985, delta answer-loss increase -0.3985
- geometry_keep_compress_drop: delta answer avg NLL -0.5272, delta answer-loss increase -0.5272
- geometry_segment_actions: delta answer avg NLL -0.4119, delta answer-loss increase -0.4119

### budget 0.50

- geometry: delta answer avg NLL -0.5088, delta answer-loss increase -0.5088
- geometry_keep_compress_drop: delta answer avg NLL -0.6376, delta answer-loss increase -0.6376
- geometry_segment_actions: delta answer avg NLL -0.5222, delta answer-loss increase -0.5222
