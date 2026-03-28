# Paper 3 Study: paper3_batch_v1_fairness

- Created: 2026-03-28T18:50:23
- Models: qwen25_15b
- Families: long_dependency, retrieval_heavy, code_conversation
- Budgets: 0.24, 0.28, 0.32, 0.35, 0.38, 0.42, 0.46, 0.50

## qwen25_15b

- Model name: `Qwen/Qwen2.5-1.5B-Instruct`
- Conversations: 9
- Evaluations: 1152
- Behavior evaluations: 576
- Segment span: 2

- Improvement vs uniform @ 0.24:
  geometry: delta logit L2 -32.698, relative logit L2 0.913
  geometry_keep_compress_drop: delta logit L2 -38.272, relative logit L2 0.898
  geometry_segment_actions: delta logit L2 -18.831, relative logit L2 0.950
- Improvement vs uniform @ 0.28:
  geometry: delta logit L2 -39.919, relative logit L2 0.890
  geometry_keep_compress_drop: delta logit L2 -42.138, relative logit L2 0.883
  geometry_segment_actions: delta logit L2 -13.935, relative logit L2 0.961
- Improvement vs uniform @ 0.32:
  geometry: delta logit L2 -55.972, relative logit L2 0.846
  geometry_keep_compress_drop: delta logit L2 -56.165, relative logit L2 0.845
  geometry_segment_actions: delta logit L2 -31.202, relative logit L2 0.914
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -21.295, relative logit L2 0.937
  geometry_keep_compress_drop: delta logit L2 -39.375, relative logit L2 0.883
  geometry_segment_actions: delta logit L2 -22.747, relative logit L2 0.932
- Improvement vs uniform @ 0.38:
  geometry: delta logit L2 -22.882, relative logit L2 0.930
  geometry_keep_compress_drop: delta logit L2 -41.881, relative logit L2 0.871
  geometry_segment_actions: delta logit L2 -20.868, relative logit L2 0.936
- Improvement vs uniform @ 0.42:
  geometry: delta logit L2 -15.189, relative logit L2 0.951
  geometry_keep_compress_drop: delta logit L2 -30.183, relative logit L2 0.902
  geometry_segment_actions: delta logit L2 -20.793, relative logit L2 0.933
- Improvement vs uniform @ 0.46:
  geometry: delta logit L2 -12.236, relative logit L2 0.959
  geometry_keep_compress_drop: delta logit L2 -22.013, relative logit L2 0.926
  geometry_segment_actions: delta logit L2 -21.232, relative logit L2 0.928
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -0.993, relative logit L2 0.996
  geometry_keep_compress_drop: delta logit L2 7.050, relative logit L2 1.027
  geometry_segment_actions: delta logit L2 3.752, relative logit L2 1.014
- Behavior improvement vs uniform @ 0.24:
  geometry: delta answer avg NLL -0.0760, delta answer-loss increase -0.0760
  geometry_keep_compress_drop: delta answer avg NLL -0.0344, delta answer-loss increase -0.0344
  geometry_segment_actions: delta answer avg NLL -0.2300, delta answer-loss increase -0.2300
- Behavior improvement vs uniform @ 0.28:
  geometry: delta answer avg NLL -0.2118, delta answer-loss increase -0.2118
  geometry_keep_compress_drop: delta answer avg NLL -0.2249, delta answer-loss increase -0.2249
  geometry_segment_actions: delta answer avg NLL -0.2300, delta answer-loss increase -0.2300
- Behavior improvement vs uniform @ 0.32:
  geometry: delta answer avg NLL -0.1995, delta answer-loss increase -0.1995
  geometry_keep_compress_drop: delta answer avg NLL -0.3903, delta answer-loss increase -0.3903
  geometry_segment_actions: delta answer avg NLL -0.1622, delta answer-loss increase -0.1622
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.1829, delta answer-loss increase -0.1829
  geometry_keep_compress_drop: delta answer avg NLL -0.3737, delta answer-loss increase -0.3737
  geometry_segment_actions: delta answer avg NLL -0.1543, delta answer-loss increase -0.1543
- Behavior improvement vs uniform @ 0.38:
  geometry: delta answer avg NLL -0.1150, delta answer-loss increase -0.1150
  geometry_keep_compress_drop: delta answer avg NLL -0.3058, delta answer-loss increase -0.3058
  geometry_segment_actions: delta answer avg NLL -0.2016, delta answer-loss increase -0.2016
- Behavior improvement vs uniform @ 0.42:
  geometry: delta answer avg NLL -0.1801, delta answer-loss increase -0.1801
  geometry_keep_compress_drop: delta answer avg NLL -0.3089, delta answer-loss increase -0.3089
  geometry_segment_actions: delta answer avg NLL -0.1473, delta answer-loss increase -0.1473
- Behavior improvement vs uniform @ 0.46:
  geometry: delta answer avg NLL -0.3985, delta answer-loss increase -0.3985
  geometry_keep_compress_drop: delta answer avg NLL -0.5272, delta answer-loss increase -0.5272
  geometry_segment_actions: delta answer avg NLL -0.4119, delta answer-loss increase -0.4119
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.5088, delta answer-loss increase -0.5088
  geometry_keep_compress_drop: delta answer avg NLL -0.6376, delta answer-loss increase -0.6376
  geometry_segment_actions: delta answer avg NLL -0.5222, delta answer-loss increase -0.5222

- Confidence and significance:
  budget 0.24:
    geometry: mean delta logit L2 -32.698 [-59.812, -10.217], p=0.0088
    geometry_segment_actions: mean delta logit L2 -18.831 [-40.123, -2.187], p=0.0457
    geometry_keep_compress_drop: mean delta logit L2 -38.272 [-73.573, -12.625], p=0.0022
  budget 0.28:
    geometry: mean delta logit L2 -39.919 [-64.704, -16.844], p=0.0010
    geometry_segment_actions: mean delta logit L2 -13.935 [-28.517, -1.358], p=0.0522
    geometry_keep_compress_drop: mean delta logit L2 -42.138 [-68.685, -19.840], p=0.0008
  budget 0.32:
    geometry: mean delta logit L2 -55.972 [-93.575, -19.401], p=0.0032
    geometry_segment_actions: mean delta logit L2 -31.202 [-63.807, -3.907], p=0.0550
    geometry_keep_compress_drop: mean delta logit L2 -56.165 [-94.162, -21.102], p=0.0030
  budget 0.35:
    geometry: mean delta logit L2 -21.295 [-60.241, 14.452], p=0.2710
    geometry_segment_actions: mean delta logit L2 -22.747 [-61.795, 11.767], p=0.2455
    geometry_keep_compress_drop: mean delta logit L2 -39.375 [-81.138, -0.457], p=0.0735
  budget 0.38:
    geometry: mean delta logit L2 -22.882 [-52.724, 5.438], p=0.1467
    geometry_segment_actions: mean delta logit L2 -20.868 [-59.144, 12.446], p=0.2878
    geometry_keep_compress_drop: mean delta logit L2 -41.881 [-77.088, -8.643], p=0.0213
  budget 0.42:
    geometry: mean delta logit L2 -15.189 [-42.120, 10.323], p=0.2792
    geometry_segment_actions: mean delta logit L2 -20.793 [-55.970, 11.182], p=0.2510
    geometry_keep_compress_drop: mean delta logit L2 -30.183 [-65.422, 0.585], p=0.0915
  budget 0.46:
    geometry: mean delta logit L2 -12.236 [-36.586, 7.521], p=0.3270
    geometry_segment_actions: mean delta logit L2 -21.232 [-54.186, 6.064], p=0.2065
    geometry_keep_compress_drop: mean delta logit L2 -22.013 [-54.846, 4.806], p=0.1785
  budget 0.50:
    geometry: mean delta logit L2 -0.993 [-28.534, 25.389], p=0.9463
    geometry_segment_actions: mean delta logit L2 3.752 [-21.964, 30.165], p=0.7748
    geometry_keep_compress_drop: mean delta logit L2 7.050 [-25.775, 40.920], p=0.6730
  behavior:
    budget 0.24:
      geometry: mean delta answer avg NLL -0.0760 [-0.2211, 0.0418], p=0.3862
      geometry_segment_actions: mean delta answer avg NLL -0.2300 [-0.4751, -0.0198], p=0.0920
      geometry_keep_compress_drop: mean delta answer avg NLL -0.0344 [-0.1200, 0.0481], p=0.5282
    budget 0.28:
      geometry: mean delta answer avg NLL -0.2118 [-0.4927, 0.0043], p=0.1870
      geometry_segment_actions: mean delta answer avg NLL -0.2300 [-0.4687, -0.0152], p=0.0927
      geometry_keep_compress_drop: mean delta answer avg NLL -0.2249 [-0.5078, 0.0050], p=0.1290
    budget 0.32:
      geometry: mean delta answer avg NLL -0.1995 [-0.4213, -0.0064], p=0.0935
      geometry_segment_actions: mean delta answer avg NLL -0.1622 [-0.3547, -0.0161], p=0.1278
      geometry_keep_compress_drop: mean delta answer avg NLL -0.3903 [-0.7071, -0.1123], p=0.0270
    budget 0.35:
      geometry: mean delta answer avg NLL -0.1829 [-0.3908, -0.0100], p=0.1207
      geometry_segment_actions: mean delta answer avg NLL -0.1543 [-0.3344, -0.0161], p=0.1263
      geometry_keep_compress_drop: mean delta answer avg NLL -0.3737 [-0.6843, -0.1027], p=0.0293
    budget 0.38:
      geometry: mean delta answer avg NLL -0.1150 [-0.3371, 0.1217], p=0.3443
      geometry_segment_actions: mean delta answer avg NLL -0.2016 [-0.5162, 0.0242], p=0.2545
      geometry_keep_compress_drop: mean delta answer avg NLL -0.3058 [-0.6330, 0.0057], p=0.0793
    budget 0.42:
      geometry: mean delta answer avg NLL -0.1801 [-0.4862, 0.1448], p=0.3220
      geometry_segment_actions: mean delta answer avg NLL -0.1473 [-0.5503, 0.1932], p=0.4803
      geometry_keep_compress_drop: mean delta answer avg NLL -0.3089 [-0.5902, -0.0827], p=0.0678
    budget 0.46:
      geometry: mean delta answer avg NLL -0.3985 [-0.7366, -0.0978], p=0.0323
      geometry_segment_actions: mean delta answer avg NLL -0.4119 [-0.8861, -0.0250], p=0.0940
      geometry_keep_compress_drop: mean delta answer avg NLL -0.5272 [-0.9466, -0.1553], p=0.0095
    budget 0.50:
      geometry: mean delta answer avg NLL -0.5088 [-0.8871, -0.1638], p=0.0127
      geometry_segment_actions: mean delta answer avg NLL -0.5222 [-0.9861, -0.1191], p=0.0465
      geometry_keep_compress_drop: mean delta answer avg NLL -0.6376 [-1.0783, -0.2603], p=0.0040
