# Paper 3 Study: paper3_batch_v1_3b

- Created: 2026-03-28T19:14:06
- Models: qwen25_3b
- Families: long_dependency, retrieval_heavy, code_conversation
- Budgets: 0.20, 0.35, 0.50

## qwen25_3b

- Model name: `Qwen/Qwen2.5-3B-Instruct`
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216
- Segment span: 2

- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 -30.391, relative logit L2 0.945
  geometry_keep_compress_drop: delta logit L2 -38.471, relative logit L2 0.930
  geometry_segment_actions: delta logit L2 -37.656, relative logit L2 0.932
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -17.322, relative logit L2 0.964
  geometry_keep_compress_drop: delta logit L2 -70.997, relative logit L2 0.854
  geometry_segment_actions: delta logit L2 -54.781, relative logit L2 0.887
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -46.526, relative logit L2 0.891
  geometry_keep_compress_drop: delta logit L2 -22.169, relative logit L2 0.948
  geometry_segment_actions: delta logit L2 -29.604, relative logit L2 0.931
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL -0.1635, delta answer-loss increase -0.1635
  geometry_keep_compress_drop: delta answer avg NLL -0.0948, delta answer-loss increase -0.0948
  geometry_segment_actions: delta answer avg NLL -0.0781, delta answer-loss increase -0.0781
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.0356, delta answer-loss increase -0.0356
  geometry_keep_compress_drop: delta answer avg NLL -0.3972, delta answer-loss increase -0.3972
  geometry_segment_actions: delta answer avg NLL -0.1797, delta answer-loss increase -0.1797
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.6252, delta answer-loss increase -0.6252
  geometry_keep_compress_drop: delta answer avg NLL -0.7727, delta answer-loss increase -0.7727
  geometry_segment_actions: delta answer avg NLL -0.4825, delta answer-loss increase -0.4825

- Confidence and significance:
  budget 0.20:
    geometry: mean delta logit L2 -30.391 [-79.490, 11.341], p=0.2325
    geometry_segment_actions: mean delta logit L2 -37.656 [-87.561, 3.827], p=0.1230
    geometry_keep_compress_drop: mean delta logit L2 -38.471 [-84.880, -2.197], p=0.0790
  budget 0.35:
    geometry: mean delta logit L2 -17.322 [-70.146, 27.692], p=0.5192
    geometry_segment_actions: mean delta logit L2 -54.781 [-110.881, -5.643], p=0.0457
    geometry_keep_compress_drop: mean delta logit L2 -70.997 [-123.447, -21.824], p=0.0070
  budget 0.50:
    geometry: mean delta logit L2 -46.526 [-89.833, -6.243], p=0.0333
    geometry_segment_actions: mean delta logit L2 -29.604 [-79.691, 21.281], p=0.2712
    geometry_keep_compress_drop: mean delta logit L2 -22.169 [-70.085, 23.971], p=0.3812
  behavior:
    budget 0.20:
      geometry: mean delta answer avg NLL -0.1635 [-0.6057, 0.1171], p=0.6305
      geometry_segment_actions: mean delta answer avg NLL -0.0781 [-0.4958, 0.2589], p=0.8063
      geometry_keep_compress_drop: mean delta answer avg NLL -0.0948 [-0.4961, 0.1603], p=1.0000
    budget 0.35:
      geometry: mean delta answer avg NLL -0.0356 [-0.2262, 0.1156], p=0.7608
      geometry_segment_actions: mean delta answer avg NLL -0.1797 [-0.4728, 0.0467], p=0.3785
      geometry_keep_compress_drop: mean delta answer avg NLL -0.3972 [-0.8192, -0.0504], p=0.0912
    budget 0.50:
      geometry: mean delta answer avg NLL -0.6252 [-1.1254, -0.1909], p=0.0200
      geometry_segment_actions: mean delta answer avg NLL -0.4825 [-0.9660, -0.0563], p=0.0645
      geometry_keep_compress_drop: mean delta answer avg NLL -0.7727 [-1.3748, -0.2112], p=0.0280
