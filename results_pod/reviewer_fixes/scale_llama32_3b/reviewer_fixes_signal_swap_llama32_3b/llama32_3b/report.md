# Paper 3 Pilot: llama32_3b

- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Budgets: 0.20, 0.35, 0.50
- Policies: uniform, semantic, geometry, geometry_keep_compress_drop, semantic_keep_compress_drop
- Segment span: 2
- Target-turn stride: 1
- Max target turns / conversation: None
- Max turns / conversation: None
- Conversations: 8
- Evaluations: 540
- Behavior evaluations: 270

## Aggregate

### geometry

- budget 0.20: logit L2 471.702, KL 0.000113, token fraction 0.624, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 415.440, KL 0.000089, token fraction 0.700, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 338.237, KL 0.000096, token fraction 0.789, kept/compressed/evicted segments 0.00/0.00/0.00

### geometry_keep_compress_drop

- budget 0.20: logit L2 476.486, KL 0.000148, token fraction 0.622, kept/compressed/evicted segments 0.22/0.75/1.03
- budget 0.35: logit L2 453.380, KL 0.000091, token fraction 0.707, kept/compressed/evicted segments 0.47/1.11/0.42
- budget 0.50: logit L2 385.519, KL 0.000102, token fraction 0.780, kept/compressed/evicted segments 0.78/1.11/0.11

### semantic

- budget 0.20: logit L2 473.058, KL 0.000109, token fraction 0.627, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 371.307, KL 0.000055, token fraction 0.712, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 329.668, KL 0.000066, token fraction 0.791, kept/compressed/evicted segments 0.00/0.00/0.00

### semantic_keep_compress_drop

- budget 0.20: logit L2 457.690, KL 0.000100, token fraction 0.614, kept/compressed/evicted segments 0.17/0.67/1.17
- budget 0.35: logit L2 409.575, KL 0.000097, token fraction 0.701, kept/compressed/evicted segments 0.44/0.92/0.64
- budget 0.50: logit L2 340.905, KL 0.000067, token fraction 0.773, kept/compressed/evicted segments 0.69/1.03/0.28

### uniform

- budget 0.20: logit L2 449.116, KL 0.000110, token fraction 0.627, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 383.891, KL 0.000076, token fraction 0.718, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 355.471, KL 0.000112, token fraction 0.747, kept/compressed/evicted segments 0.00/0.00/0.00

## Behavior Aggregate

### geometry

- budget 0.20: answer avg NLL 1.8832, answer delta 1.2062
- budget 0.35: answer avg NLL 1.0717, answer delta 0.3948
- budget 0.50: answer avg NLL 0.8283, answer delta 0.1514

### geometry_keep_compress_drop

- budget 0.20: answer avg NLL 1.7783, answer delta 1.1014
- budget 0.35: answer avg NLL 1.3056, answer delta 0.6287
- budget 0.50: answer avg NLL 1.0260, answer delta 0.3490

### semantic

- budget 0.20: answer avg NLL 1.7778, answer delta 1.1008
- budget 0.35: answer avg NLL 0.9207, answer delta 0.2438
- budget 0.50: answer avg NLL 0.8751, answer delta 0.1981

### semantic_keep_compress_drop

- budget 0.20: answer avg NLL 1.9494, answer delta 1.2725
- budget 0.35: answer avg NLL 1.2153, answer delta 0.5384
- budget 0.50: answer avg NLL 1.0720, answer delta 0.3950

### uniform

- budget 0.20: answer avg NLL 1.8903, answer delta 1.2133
- budget 0.35: answer avg NLL 0.8324, answer delta 0.1555
- budget 0.50: answer avg NLL 1.0394, answer delta 0.3624

## Improvement Vs Uniform

### budget 0.20

- geometry: delta logit L2 22.586, relative logit L2 1.050
- geometry_keep_compress_drop: delta logit L2 27.370, relative logit L2 1.061
- semantic: delta logit L2 23.943, relative logit L2 1.053
- semantic_keep_compress_drop: delta logit L2 8.574, relative logit L2 1.019

### budget 0.35

- geometry: delta logit L2 31.549, relative logit L2 1.082
- geometry_keep_compress_drop: delta logit L2 69.488, relative logit L2 1.181
- semantic: delta logit L2 -12.584, relative logit L2 0.967
- semantic_keep_compress_drop: delta logit L2 25.683, relative logit L2 1.067

### budget 0.50

- geometry: delta logit L2 -17.234, relative logit L2 0.952
- geometry_keep_compress_drop: delta logit L2 30.048, relative logit L2 1.085
- semantic: delta logit L2 -25.803, relative logit L2 0.927
- semantic_keep_compress_drop: delta logit L2 -14.567, relative logit L2 0.959

## Behavior Improvement Vs Uniform

### budget 0.20

- geometry: delta answer avg NLL -0.0071, delta answer-loss increase -0.0071
- geometry_keep_compress_drop: delta answer avg NLL -0.1119, delta answer-loss increase -0.1119
- semantic: delta answer avg NLL -0.1125, delta answer-loss increase -0.1125
- semantic_keep_compress_drop: delta answer avg NLL 0.0591, delta answer-loss increase 0.0591

### budget 0.35

- geometry: delta answer avg NLL 0.2393, delta answer-loss increase 0.2393
- geometry_keep_compress_drop: delta answer avg NLL 0.4732, delta answer-loss increase 0.4732
- semantic: delta answer avg NLL 0.0883, delta answer-loss increase 0.0883
- semantic_keep_compress_drop: delta answer avg NLL 0.3830, delta answer-loss increase 0.3830

### budget 0.50

- geometry: delta answer avg NLL -0.2111, delta answer-loss increase -0.2111
- geometry_keep_compress_drop: delta answer avg NLL -0.0134, delta answer-loss increase -0.0134
- semantic: delta answer avg NLL -0.1643, delta answer-loss increase -0.1643
- semantic_keep_compress_drop: delta answer avg NLL 0.0326, delta answer-loss increase 0.0326
