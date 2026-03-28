# Paper 3 Pilot: smollm2_17b

- Model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Budgets: 0.20, 0.35, 0.50
- Segment span: 2
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216

## Aggregate

### geometry

- budget 0.20: logit L2 228.665, KL 0.112962, token fraction 0.549, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 204.396, KL 0.046591, token fraction 0.625, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 174.741, KL 0.029158, token fraction 0.705, kept/compressed/evicted segments 0.00/0.00/0.00

### geometry_keep_compress_drop

- budget 0.20: logit L2 229.306, KL 0.107235, token fraction 0.550, kept/compressed/evicted segments 0.08/0.67/1.25
- budget 0.35: logit L2 196.258, KL 0.040629, token fraction 0.636, kept/compressed/evicted segments 0.42/0.97/0.61
- budget 0.50: logit L2 184.640, KL 0.038737, token fraction 0.704, kept/compressed/evicted segments 0.42/1.47/0.11

### geometry_segment_actions

- budget 0.20: logit L2 228.665, KL 0.112962, token fraction 0.549, kept/compressed/evicted segments 0.11/0.64/1.25
- budget 0.35: logit L2 196.050, KL 0.044637, token fraction 0.638, kept/compressed/evicted segments 0.44/0.89/0.67
- budget 0.50: logit L2 169.314, KL 0.046591, token fraction 0.721, kept/compressed/evicted segments 0.78/0.75/0.47

### uniform

- budget 0.20: logit L2 232.735, KL 0.099064, token fraction 0.552, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.35: logit L2 215.836, KL 0.101613, token fraction 0.633, kept/compressed/evicted segments 0.00/0.00/0.00
- budget 0.50: logit L2 194.510, KL 0.082792, token fraction 0.682, kept/compressed/evicted segments 0.00/0.00/0.00

## Behavior Aggregate

### geometry

- budget 0.20: answer avg NLL 1.3464, answer delta 0.7894
- budget 0.35: answer avg NLL 1.0517, answer delta 0.4947
- budget 0.50: answer avg NLL 0.8371, answer delta 0.2801

### geometry_keep_compress_drop

- budget 0.20: answer avg NLL 1.3464, answer delta 0.7894
- budget 0.35: answer avg NLL 1.0412, answer delta 0.4841
- budget 0.50: answer avg NLL 0.8229, answer delta 0.2658

### geometry_segment_actions

- budget 0.20: answer avg NLL 1.3464, answer delta 0.7894
- budget 0.35: answer avg NLL 1.0517, answer delta 0.4947
- budget 0.50: answer avg NLL 0.7526, answer delta 0.1956

### uniform

- budget 0.20: answer avg NLL 1.3753, answer delta 0.8183
- budget 0.35: answer avg NLL 1.0542, answer delta 0.4972
- budget 0.50: answer avg NLL 1.1198, answer delta 0.5627

## Improvement Vs Uniform

### budget 0.20

- geometry: delta logit L2 -4.070, relative logit L2 0.983
- geometry_keep_compress_drop: delta logit L2 -3.429, relative logit L2 0.985
- geometry_segment_actions: delta logit L2 -4.070, relative logit L2 0.983

### budget 0.35

- geometry: delta logit L2 -11.441, relative logit L2 0.947
- geometry_keep_compress_drop: delta logit L2 -19.579, relative logit L2 0.909
- geometry_segment_actions: delta logit L2 -19.786, relative logit L2 0.908

### budget 0.50

- geometry: delta logit L2 -19.769, relative logit L2 0.898
- geometry_keep_compress_drop: delta logit L2 -9.870, relative logit L2 0.949
- geometry_segment_actions: delta logit L2 -25.196, relative logit L2 0.870

## Behavior Improvement Vs Uniform

### budget 0.20

- geometry: delta answer avg NLL -0.0289, delta answer-loss increase -0.0289
- geometry_keep_compress_drop: delta answer avg NLL -0.0289, delta answer-loss increase -0.0289
- geometry_segment_actions: delta answer avg NLL -0.0289, delta answer-loss increase -0.0289

### budget 0.35

- geometry: delta answer avg NLL -0.0025, delta answer-loss increase -0.0025
- geometry_keep_compress_drop: delta answer avg NLL -0.0131, delta answer-loss increase -0.0131
- geometry_segment_actions: delta answer avg NLL -0.0025, delta answer-loss increase -0.0025

### budget 0.50

- geometry: delta answer avg NLL -0.2827, delta answer-loss increase -0.2827
- geometry_keep_compress_drop: delta answer avg NLL -0.2969, delta answer-loss increase -0.2969
- geometry_segment_actions: delta answer avg NLL -0.3671, delta answer-loss increase -0.3671
