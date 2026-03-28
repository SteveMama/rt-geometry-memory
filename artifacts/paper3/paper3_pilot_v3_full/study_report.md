# Paper 3 Study: paper3_pilot_v3_full

- Created: 2026-03-28T04:43:05
- Models: qwen25_05b, qwen25_15b, smollm2_17b
- Families: long_dependency, retrieval_heavy, code_conversation
- Budgets: 0.20, 0.35, 0.50

## qwen25_05b

- Model name: `Qwen/Qwen2.5-0.5B-Instruct`
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216
- Segment span: 2

- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 -66.765, relative logit L2 0.834
  geometry_keep_compress_drop: delta logit L2 -67.306, relative logit L2 0.833
  geometry_segment_actions: delta logit L2 -68.883, relative logit L2 0.829
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -101.881, relative logit L2 0.729
  geometry_keep_compress_drop: delta logit L2 -105.647, relative logit L2 0.719
  geometry_segment_actions: delta logit L2 -100.973, relative logit L2 0.732
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -40.913, relative logit L2 0.864
  geometry_keep_compress_drop: delta logit L2 -27.965, relative logit L2 0.907
  geometry_segment_actions: delta logit L2 -97.342, relative logit L2 0.677
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL -0.0835, delta answer-loss increase -0.0835
  geometry_keep_compress_drop: delta answer avg NLL -0.0839, delta answer-loss increase -0.0839
  geometry_segment_actions: delta answer avg NLL -0.0487, delta answer-loss increase -0.0487
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.0911, delta answer-loss increase -0.0911
  geometry_keep_compress_drop: delta answer avg NLL -0.2170, delta answer-loss increase -0.2170
  geometry_segment_actions: delta answer avg NLL 0.0047, delta answer-loss increase 0.0047
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.4276, delta answer-loss increase -0.4276
  geometry_keep_compress_drop: delta answer avg NLL -0.5112, delta answer-loss increase -0.5112
  geometry_segment_actions: delta answer avg NLL -0.3791, delta answer-loss increase -0.3791

- Confidence and significance:
  budget 0.20:
    geometry: mean delta logit L2 -66.765 [-115.766, -25.056], p=0.0003
    geometry_segment_actions: mean delta logit L2 -68.883 [-119.947, -28.077], p=0.0003
    geometry_keep_compress_drop: mean delta logit L2 -67.306 [-120.233, -24.026], p=0.0067
  budget 0.35:
    geometry: mean delta logit L2 -101.881 [-165.748, -47.175], p=0.0000
    geometry_segment_actions: mean delta logit L2 -100.973 [-165.722, -48.822], p=0.0000
    geometry_keep_compress_drop: mean delta logit L2 -105.647 [-168.297, -51.034], p=0.0003
  budget 0.50:
    geometry: mean delta logit L2 -40.913 [-106.393, 19.883], p=0.2185
    geometry_segment_actions: mean delta logit L2 -97.342 [-160.666, -42.459], p=0.0003
    geometry_keep_compress_drop: mean delta logit L2 -27.965 [-91.948, 35.208], p=0.3862
  behavior:
    budget 0.20:
      geometry: mean delta answer avg NLL -0.0835 [-0.3133, 0.0533], p=1.0000
      geometry_segment_actions: mean delta answer avg NLL -0.0487 [-0.2933, 0.1018], p=1.0000
      geometry_keep_compress_drop: mean delta answer avg NLL -0.0839 [-0.3133, 0.0616], p=1.0000
    budget 0.35:
      geometry: mean delta answer avg NLL -0.0911 [-0.3428, 0.0636], p=0.7100
      geometry_segment_actions: mean delta answer avg NLL 0.0047 [-0.0744, 0.0869], p=0.7568
      geometry_keep_compress_drop: mean delta answer avg NLL -0.2170 [-0.5194, 0.0181], p=0.2160
    budget 0.50:
      geometry: mean delta answer avg NLL -0.4276 [-0.7799, -0.1175], p=0.0355
      geometry_segment_actions: mean delta answer avg NLL -0.3791 [-0.7321, -0.0931], p=0.0405
      geometry_keep_compress_drop: mean delta answer avg NLL -0.5112 [-0.8760, -0.2082], p=0.0050

## qwen25_15b

- Model name: `Qwen/Qwen2.5-1.5B-Instruct`
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216
- Segment span: 2

- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 -34.144, relative logit L2 0.912
  geometry_keep_compress_drop: delta logit L2 -37.672, relative logit L2 0.903
  geometry_segment_actions: delta logit L2 -40.344, relative logit L2 0.896
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -21.295, relative logit L2 0.937
  geometry_keep_compress_drop: delta logit L2 -39.375, relative logit L2 0.883
  geometry_segment_actions: delta logit L2 -22.747, relative logit L2 0.932
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -0.993, relative logit L2 0.996
  geometry_keep_compress_drop: delta logit L2 7.050, relative logit L2 1.027
  geometry_segment_actions: delta logit L2 3.752, relative logit L2 1.014
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL 0.0227, delta answer-loss increase 0.0227
  geometry_keep_compress_drop: delta answer avg NLL 0.0204, delta answer-loss increase 0.0204
  geometry_segment_actions: delta answer avg NLL -0.0069, delta answer-loss increase -0.0069
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.1829, delta answer-loss increase -0.1829
  geometry_keep_compress_drop: delta answer avg NLL -0.3737, delta answer-loss increase -0.3737
  geometry_segment_actions: delta answer avg NLL -0.1543, delta answer-loss increase -0.1543
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.5088, delta answer-loss increase -0.5088
  geometry_keep_compress_drop: delta answer avg NLL -0.6376, delta answer-loss increase -0.6376
  geometry_segment_actions: delta answer avg NLL -0.5222, delta answer-loss increase -0.5222

- Confidence and significance:
  budget 0.20:
    geometry: mean delta logit L2 -34.144 [-66.635, -10.651], p=0.0018
    geometry_segment_actions: mean delta logit L2 -40.344 [-72.421, -14.378], p=0.0053
    geometry_keep_compress_drop: mean delta logit L2 -37.672 [-71.111, -10.658], p=0.0130
  budget 0.35:
    geometry: mean delta logit L2 -21.295 [-58.524, 14.483], p=0.2810
    geometry_segment_actions: mean delta logit L2 -22.747 [-62.984, 12.401], p=0.2505
    geometry_keep_compress_drop: mean delta logit L2 -39.375 [-80.604, 1.631], p=0.0645
  budget 0.50:
    geometry: mean delta logit L2 -0.993 [-30.281, 25.959], p=0.9440
    geometry_segment_actions: mean delta logit L2 3.752 [-21.626, 29.088], p=0.7873
    geometry_keep_compress_drop: mean delta logit L2 7.050 [-26.598, 40.741], p=0.6855
  behavior:
    budget 0.20:
      geometry: mean delta answer avg NLL 0.0227 [0.0000, 0.0635], p=0.4973
      geometry_segment_actions: mean delta answer avg NLL -0.0069 [-0.2017, 0.1534], p=1.0000
      geometry_keep_compress_drop: mean delta answer avg NLL 0.0204 [0.0000, 0.0611], p=1.0000
    budget 0.35:
      geometry: mean delta answer avg NLL -0.1829 [-0.3863, -0.0072], p=0.1237
      geometry_segment_actions: mean delta answer avg NLL -0.1543 [-0.3195, -0.0161], p=0.1235
      geometry_keep_compress_drop: mean delta answer avg NLL -0.3737 [-0.6758, -0.0936], p=0.0262
    budget 0.50:
      geometry: mean delta answer avg NLL -0.5088 [-0.8953, -0.1567], p=0.0175
      geometry_segment_actions: mean delta answer avg NLL -0.5222 [-1.0047, -0.0949], p=0.0517
      geometry_keep_compress_drop: mean delta answer avg NLL -0.6376 [-1.0943, -0.2511], p=0.0047

## smollm2_17b

- Model name: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Conversations: 9
- Evaluations: 432
- Behavior evaluations: 216
- Segment span: 2

- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 -4.070, relative logit L2 0.983
  geometry_keep_compress_drop: delta logit L2 -3.429, relative logit L2 0.985
  geometry_segment_actions: delta logit L2 -4.070, relative logit L2 0.983
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 -11.441, relative logit L2 0.947
  geometry_keep_compress_drop: delta logit L2 -19.579, relative logit L2 0.909
  geometry_segment_actions: delta logit L2 -19.786, relative logit L2 0.908
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -19.769, relative logit L2 0.898
  geometry_keep_compress_drop: delta logit L2 -9.870, relative logit L2 0.949
  geometry_segment_actions: delta logit L2 -25.196, relative logit L2 0.870
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL -0.0289, delta answer-loss increase -0.0289
  geometry_keep_compress_drop: delta answer avg NLL -0.0289, delta answer-loss increase -0.0289
  geometry_segment_actions: delta answer avg NLL -0.0289, delta answer-loss increase -0.0289
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL -0.0025, delta answer-loss increase -0.0025
  geometry_keep_compress_drop: delta answer avg NLL -0.0131, delta answer-loss increase -0.0131
  geometry_segment_actions: delta answer avg NLL -0.0025, delta answer-loss increase -0.0025
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.2827, delta answer-loss increase -0.2827
  geometry_keep_compress_drop: delta answer avg NLL -0.2969, delta answer-loss increase -0.2969
  geometry_segment_actions: delta answer avg NLL -0.3671, delta answer-loss increase -0.3671

- Confidence and significance:
  budget 0.20:
    geometry: mean delta logit L2 -4.070 [-11.635, 2.822], p=0.2775
    geometry_segment_actions: mean delta logit L2 -4.070 [-11.582, 2.444], p=0.2890
    geometry_keep_compress_drop: mean delta logit L2 -3.429 [-10.768, 3.115], p=0.3757
  budget 0.35:
    geometry: mean delta logit L2 -11.441 [-21.610, -2.556], p=0.0220
    geometry_segment_actions: mean delta logit L2 -19.786 [-31.579, -9.184], p=0.0010
    geometry_keep_compress_drop: mean delta logit L2 -19.579 [-31.679, -8.502], p=0.0020
  budget 0.50:
    geometry: mean delta logit L2 -19.769 [-35.629, -3.682], p=0.0257
    geometry_segment_actions: mean delta logit L2 -25.196 [-42.512, -7.916], p=0.0088
    geometry_keep_compress_drop: mean delta logit L2 -9.870 [-25.176, 4.921], p=0.2182
  behavior:
    budget 0.20:
      geometry: mean delta answer avg NLL -0.0289 [-0.2196, 0.0934], p=1.0000
      geometry_segment_actions: mean delta answer avg NLL -0.0289 [-0.2194, 0.0958], p=1.0000
      geometry_keep_compress_drop: mean delta answer avg NLL -0.0289 [-0.2229, 0.0931], p=1.0000
    budget 0.35:
      geometry: mean delta answer avg NLL -0.0025 [-0.0296, 0.0268], p=1.0000
      geometry_segment_actions: mean delta answer avg NLL -0.0025 [-0.0306, 0.0268], p=1.0000
      geometry_keep_compress_drop: mean delta answer avg NLL -0.0131 [-0.0331, 0.0020], p=0.4995
    budget 0.50:
      geometry: mean delta answer avg NLL -0.2827 [-0.5683, -0.0365], p=0.0730
      geometry_segment_actions: mean delta answer avg NLL -0.3671 [-0.6777, -0.1268], p=0.0295
      geometry_keep_compress_drop: mean delta answer avg NLL -0.2969 [-0.5754, -0.0555], p=0.0473
