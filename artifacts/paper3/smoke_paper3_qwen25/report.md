# Paper 3 Pilot: qwen25_05b

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Budgets: 0.35
- Conversations: 2
- Evaluations: 32

## Aggregate

### geometry

- budget 0.35: logit L2 311.987, KL 0.000008, token fraction 0.628, kept/compressed/evicted segments 0.00/0.00/0.00

### geometry_keep_compress_drop

- budget 0.35: logit L2 376.338, KL 0.000011, token fraction 0.475, kept/compressed/evicted segments 0.25/0.00/1.00

### geometry_segment_actions

- budget 0.35: logit L2 314.167, KL 0.000008, token fraction 0.621, kept/compressed/evicted segments 0.38/0.88/0.75

### uniform

- budget 0.35: logit L2 465.009, KL 0.000009, token fraction 0.611, kept/compressed/evicted segments 0.00/0.00/0.00

## Improvement Vs Uniform

### budget 0.35

- geometry: delta logit L2 -153.022, relative logit L2 0.671
- geometry_keep_compress_drop: delta logit L2 -88.671, relative logit L2 0.809
- geometry_segment_actions: delta logit L2 -150.842, relative logit L2 0.676
