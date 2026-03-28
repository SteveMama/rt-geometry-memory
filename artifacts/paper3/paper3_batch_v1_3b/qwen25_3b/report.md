# Paper 3 Pilot: qwen25_3b

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Budgets: 0.20, 0.35, 0.50
- Segment span: 2
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216

## Aggregate

### geometry

- budget 0.20: logit L2 519.596, KL 1.845409, token fraction 0.542, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 469.259, KL 1.199930, token fraction 0.637, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 381.511, KL 0.234238, token fraction 0.722, kept/compressed/evicted segments 0.00/0.00/0.00

### geometry_keep_compress_drop

- budget 0.20: logit L2 511.516, KL 1.817745, token fraction 0.541, kept/compressed/evicted segments 0.17/0.56/1.28
- budget 0.35: logit L2 415.584, KL 0.409414, token fraction 0.646, kept/compressed/evicted segments 0.44/0.86/0.69
- budget 0.50: logit L2 405.868, KL 0.411406, token fraction 0.723, kept/compressed/evicted segments 0.50/1.42/0.08

### geometry_segment_actions

- budget 0.20: logit L2 512.331, KL 1.832668, token fraction 0.542, kept/compressed/evicted segments 0.25/0.50/1.25
- budget 0.35: logit L2 431.799, KL 0.660958, token fraction 0.646, kept/compressed/evicted segments 0.44/1.00/0.56
- budget 0.50: logit L2 398.433, KL 0.422466, token fraction 0.722, kept/compressed/evicted segments 0.78/0.86/0.36

### uniform

- budget 0.20: logit L2 549.987, KL 2.431669, token fraction 0.547, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 486.581, KL 1.161572, token fraction 0.632, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 428.037, KL 0.569900, token fraction 0.680, kept/compressed/evicted segments 0.00/0.00/0.00

## Behavior Aggregate

### geometry

- budget 0.20: answer avg NLL 2.7447, answer delta 1.6300
- budget 0.35: answer avg NLL 2.0874, answer delta 0.9728
- budget 0.50: answer avg NLL 1.6128, answer delta 0.4981

### geometry_keep_compress_drop

- budget 0.20: answer avg NLL 2.8134, answer delta 1.6988
- budget 0.35: answer avg NLL 1.7257, answer delta 0.6111
- budget 0.50: answer avg NLL 1.4653, answer delta 0.3507

### geometry_segment_actions

- budget 0.20: answer avg NLL 2.8301, answer delta 1.7154
- budget 0.35: answer avg NLL 1.9433, answer delta 0.8287
- budget 0.50: answer avg NLL 1.7554, answer delta 0.6408

### uniform

- budget 0.20: answer avg NLL 2.9082, answer delta 1.7936
- budget 0.35: answer avg NLL 2.1230, answer delta 1.0083
- budget 0.50: answer avg NLL 2.2380, answer delta 1.1233

## Improvement Vs Uniform

### budget 0.20

- geometry: delta logit L2 -30.391, relative logit L2 0.945
- geometry_keep_compress_drop: delta logit L2 -38.471, relative logit L2 0.930
- geometry_segment_actions: delta logit L2 -37.656, relative logit L2 0.932

### budget 0.35

- geometry: delta logit L2 -17.322, relative logit L2 0.964
- geometry_keep_compress_drop: delta logit L2 -70.997, relative logit L2 0.854
- geometry_segment_actions: delta logit L2 -54.781, relative logit L2 0.887

### budget 0.50

- geometry: delta logit L2 -46.526, relative logit L2 0.891
- geometry_keep_compress_drop: delta logit L2 -22.169, relative logit L2 0.948
- geometry_segment_actions: delta logit L2 -29.604, relative logit L2 0.931

## Behavior Improvement Vs Uniform

### budget 0.20

- geometry: delta answer avg NLL -0.1635, delta answer-loss increase -0.1635
- geometry_keep_compress_drop: delta answer avg NLL -0.0948, delta answer-loss increase -0.0948
- geometry_segment_actions: delta answer avg NLL -0.0781, delta answer-loss increase -0.0781

### budget 0.35

- geometry: delta answer avg NLL -0.0356, delta answer-loss increase -0.0356
- geometry_keep_compress_drop: delta answer avg NLL -0.3972, delta answer-loss increase -0.3972
- geometry_segment_actions: delta answer avg NLL -0.1797, delta answer-loss increase -0.1797

### budget 0.50

- geometry: delta answer avg NLL -0.6252, delta answer-loss increase -0.6252
- geometry_keep_compress_drop: delta answer avg NLL -0.7727, delta answer-loss increase -0.7727
- geometry_segment_actions: delta answer avg NLL -0.4825, delta answer-loss increase -0.4825
