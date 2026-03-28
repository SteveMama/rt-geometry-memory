# Paper 3 Pilot: qwen25_05b

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Budgets: 0.20, 0.35, 0.50
- Segment span: 2
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216

## Aggregate

### geometry

- budget 0.20: logit L2 335.610, KL 0.000009, token fraction 0.541, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 274.467, KL 0.000010, token fraction 0.640, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 260.394, KL 0.000007, token fraction 0.730, kept/compressed/evicted segments 0.00/0.00/0.00

### geometry_keep_compress_drop

- budget 0.20: logit L2 335.069, KL 0.000010, token fraction 0.542, kept/compressed/evicted segments 0.19/0.53/1.28
- budget 0.35: logit L2 270.702, KL 0.000011, token fraction 0.644, kept/compressed/evicted segments 0.44/0.83/0.72
- budget 0.50: logit L2 273.342, KL 0.000010, token fraction 0.730, kept/compressed/evicted segments 0.50/1.42/0.08

### geometry_segment_actions

- budget 0.20: logit L2 333.491, KL 0.000009, token fraction 0.542, kept/compressed/evicted segments 0.25/0.50/1.25
- budget 0.35: logit L2 275.376, KL 0.000010, token fraction 0.647, kept/compressed/evicted segments 0.44/1.00/0.56
- budget 0.50: logit L2 203.965, KL 0.000007, token fraction 0.727, kept/compressed/evicted segments 0.81/0.81/0.39

### uniform

- budget 0.20: logit L2 402.375, KL 0.000010, token fraction 0.547, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 376.348, KL 0.000011, token fraction 0.632, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 301.308, KL 0.000008, token fraction 0.680, kept/compressed/evicted segments 0.00/0.00/0.00

## Behavior Aggregate

### geometry

- budget 0.20: answer avg NLL 1.5019, answer delta 0.6934
- budget 0.35: answer avg NLL 1.1232, answer delta 0.3147
- budget 0.50: answer avg NLL 0.8896, answer delta 0.0811

### geometry_keep_compress_drop

- budget 0.20: answer avg NLL 1.5015, answer delta 0.6930
- budget 0.35: answer avg NLL 0.9972, answer delta 0.1888
- budget 0.50: answer avg NLL 0.8060, answer delta -0.0025

### geometry_segment_actions

- budget 0.20: answer avg NLL 1.5367, answer delta 0.7282
- budget 0.35: answer avg NLL 1.2190, answer delta 0.4105
- budget 0.50: answer avg NLL 0.9381, answer delta 0.1297

### uniform

- budget 0.20: answer avg NLL 1.5854, answer delta 0.7769
- budget 0.35: answer avg NLL 1.2143, answer delta 0.4058
- budget 0.50: answer avg NLL 1.3172, answer delta 0.5087

## Improvement Vs Uniform

### budget 0.20

- geometry: delta logit L2 -66.765, relative logit L2 0.834
- geometry_keep_compress_drop: delta logit L2 -67.306, relative logit L2 0.833
- geometry_segment_actions: delta logit L2 -68.883, relative logit L2 0.829

### budget 0.35

- geometry: delta logit L2 -101.881, relative logit L2 0.729
- geometry_keep_compress_drop: delta logit L2 -105.647, relative logit L2 0.719
- geometry_segment_actions: delta logit L2 -100.973, relative logit L2 0.732

### budget 0.50

- geometry: delta logit L2 -40.913, relative logit L2 0.864
- geometry_keep_compress_drop: delta logit L2 -27.965, relative logit L2 0.907
- geometry_segment_actions: delta logit L2 -97.342, relative logit L2 0.677

## Behavior Improvement Vs Uniform

### budget 0.20

- geometry: delta answer avg NLL -0.0835, delta answer-loss increase -0.0835
- geometry_keep_compress_drop: delta answer avg NLL -0.0839, delta answer-loss increase -0.0839
- geometry_segment_actions: delta answer avg NLL -0.0487, delta answer-loss increase -0.0487

### budget 0.35

- geometry: delta answer avg NLL -0.0911, delta answer-loss increase -0.0911
- geometry_keep_compress_drop: delta answer avg NLL -0.2170, delta answer-loss increase -0.2170
- geometry_segment_actions: delta answer avg NLL 0.0047, delta answer-loss increase 0.0047

### budget 0.50

- geometry: delta answer avg NLL -0.4276, delta answer-loss increase -0.4276
- geometry_keep_compress_drop: delta answer avg NLL -0.5112, delta answer-loss increase -0.5112
- geometry_segment_actions: delta answer avg NLL -0.3791, delta answer-loss increase -0.3791
