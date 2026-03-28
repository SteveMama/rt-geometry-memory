# Paper 3 Pilot: qwen25_15b

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Budgets: 0.20, 0.35, 0.50
- Segment span: 2
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216

## Aggregate

### geometry

- budget 0.20: logit L2 354.600, KL 0.000014, token fraction 0.540, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 315.146, KL 0.000011, token fraction 0.638, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 264.774, KL 0.000010, token fraction 0.726, kept/compressed/evicted segments 0.00/0.00/0.00

### geometry_keep_compress_drop

- budget 0.20: logit L2 351.072, KL 0.000013, token fraction 0.537, kept/compressed/evicted segments 0.25/0.47/1.28
- budget 0.35: logit L2 297.066, KL 0.000010, token fraction 0.647, kept/compressed/evicted segments 0.42/0.83/0.75
- budget 0.50: logit L2 272.817, KL 0.000011, token fraction 0.734, kept/compressed/evicted segments 0.50/1.42/0.08

### geometry_segment_actions

- budget 0.20: logit L2 348.401, KL 0.000013, token fraction 0.542, kept/compressed/evicted segments 0.25/0.50/1.25
- budget 0.35: logit L2 313.694, KL 0.000014, token fraction 0.646, kept/compressed/evicted segments 0.44/1.00/0.56
- budget 0.50: logit L2 269.519, KL 0.000010, token fraction 0.727, kept/compressed/evicted segments 0.86/0.69/0.44

### uniform

- budget 0.20: logit L2 388.745, KL 0.000013, token fraction 0.547, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 336.441, KL 0.000012, token fraction 0.632, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 265.767, KL 0.000012, token fraction 0.680, kept/compressed/evicted segments 0.00/0.00/0.00

## Behavior Aggregate

### geometry

- budget 0.20: answer avg NLL 1.6460, answer delta 1.0469
- budget 0.35: answer avg NLL 1.0899, answer delta 0.4908
- budget 0.50: answer avg NLL 0.8402, answer delta 0.2412

### geometry_keep_compress_drop

- budget 0.20: answer avg NLL 1.6437, answer delta 1.0446
- budget 0.35: answer avg NLL 0.8990, answer delta 0.3000
- budget 0.50: answer avg NLL 0.7115, answer delta 0.1124

### geometry_segment_actions

- budget 0.20: answer avg NLL 1.6164, answer delta 1.0173
- budget 0.35: answer avg NLL 1.1185, answer delta 0.5194
- budget 0.50: answer avg NLL 0.8268, answer delta 0.2277

### uniform

- budget 0.20: answer avg NLL 1.6233, answer delta 1.0242
- budget 0.35: answer avg NLL 1.2728, answer delta 0.6737
- budget 0.50: answer avg NLL 1.3490, answer delta 0.7499

## Improvement Vs Uniform

### budget 0.20

- geometry: delta logit L2 -34.144, relative logit L2 0.912
- geometry_keep_compress_drop: delta logit L2 -37.672, relative logit L2 0.903
- geometry_segment_actions: delta logit L2 -40.344, relative logit L2 0.896

### budget 0.35

- geometry: delta logit L2 -21.295, relative logit L2 0.937
- geometry_keep_compress_drop: delta logit L2 -39.375, relative logit L2 0.883
- geometry_segment_actions: delta logit L2 -22.747, relative logit L2 0.932

### budget 0.50

- geometry: delta logit L2 -0.993, relative logit L2 0.996
- geometry_keep_compress_drop: delta logit L2 7.050, relative logit L2 1.027
- geometry_segment_actions: delta logit L2 3.752, relative logit L2 1.014

## Behavior Improvement Vs Uniform

### budget 0.20

- geometry: delta answer avg NLL 0.0227, delta answer-loss increase 0.0227
- geometry_keep_compress_drop: delta answer avg NLL 0.0204, delta answer-loss increase 0.0204
- geometry_segment_actions: delta answer avg NLL -0.0069, delta answer-loss increase -0.0069

### budget 0.35

- geometry: delta answer avg NLL -0.1829, delta answer-loss increase -0.1829
- geometry_keep_compress_drop: delta answer avg NLL -0.3737, delta answer-loss increase -0.3737
- geometry_segment_actions: delta answer avg NLL -0.1543, delta answer-loss increase -0.1543

### budget 0.50

- geometry: delta answer avg NLL -0.5088, delta answer-loss increase -0.5088
- geometry_keep_compress_drop: delta answer avg NLL -0.6376, delta answer-loss increase -0.6376
- geometry_segment_actions: delta answer avg NLL -0.5222, delta answer-loss increase -0.5222
