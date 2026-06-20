# Paper 3 Study: reviewer_fixes_signal_swap_llama32_3b

- Created: 2026-06-20T03:43:34
- Models: llama32_3b
- Families: long_dependency, retrieval_heavy, code_conversation
- Budgets: 0.20, 0.35, 0.50
- Policies: uniform, semantic, geometry, geometry_keep_compress_drop, semantic_keep_compress_drop

## llama32_3b

- Model name: `meta-llama/Llama-3.2-3B-Instruct`
- Conversations: 8
- Evaluations: 540
- Behavior evaluations: 270
- Segment span: 2
- Target-turn stride: 1
- Max target turns / conversation: None
- Max turns / conversation: None

- Improvement vs uniform @ 0.20:
  geometry: delta logit L2 22.586, relative logit L2 1.050
  geometry_keep_compress_drop: delta logit L2 27.370, relative logit L2 1.061
  semantic: delta logit L2 23.943, relative logit L2 1.053
  semantic_keep_compress_drop: delta logit L2 8.574, relative logit L2 1.019
- Improvement vs uniform @ 0.35:
  geometry: delta logit L2 31.549, relative logit L2 1.082
  geometry_keep_compress_drop: delta logit L2 69.488, relative logit L2 1.181
  semantic: delta logit L2 -12.584, relative logit L2 0.967
  semantic_keep_compress_drop: delta logit L2 25.683, relative logit L2 1.067
- Improvement vs uniform @ 0.50:
  geometry: delta logit L2 -17.234, relative logit L2 0.952
  geometry_keep_compress_drop: delta logit L2 30.048, relative logit L2 1.085
  semantic: delta logit L2 -25.803, relative logit L2 0.927
  semantic_keep_compress_drop: delta logit L2 -14.567, relative logit L2 0.959
- Behavior improvement vs uniform @ 0.20:
  geometry: delta answer avg NLL -0.0071, delta answer-loss increase -0.0071
  geometry_keep_compress_drop: delta answer avg NLL -0.1119, delta answer-loss increase -0.1119
  semantic: delta answer avg NLL -0.1125, delta answer-loss increase -0.1125
  semantic_keep_compress_drop: delta answer avg NLL 0.0591, delta answer-loss increase 0.0591
- Behavior improvement vs uniform @ 0.35:
  geometry: delta answer avg NLL 0.2393, delta answer-loss increase 0.2393
  geometry_keep_compress_drop: delta answer avg NLL 0.4732, delta answer-loss increase 0.4732
  semantic: delta answer avg NLL 0.0883, delta answer-loss increase 0.0883
  semantic_keep_compress_drop: delta answer avg NLL 0.3830, delta answer-loss increase 0.3830
- Behavior improvement vs uniform @ 0.50:
  geometry: delta answer avg NLL -0.2111, delta answer-loss increase -0.2111
  geometry_keep_compress_drop: delta answer avg NLL -0.0134, delta answer-loss increase -0.0134
  semantic: delta answer avg NLL -0.1643, delta answer-loss increase -0.1643
  semantic_keep_compress_drop: delta answer avg NLL 0.0326, delta answer-loss increase 0.0326

- Confidence and significance:
  budget 0.20:
    geometry: row Δ logit L2 22.586 [0.576, 50.591], p=0.1242; conversation Δ 22.586 [-2.142, 51.706], p=0.2485
    geometry_keep_compress_drop: row Δ logit L2 27.370 [1.121, 57.808], p=0.0752; conversation Δ 27.370 [-6.470, 66.171], p=0.2782
    semantic: row Δ logit L2 23.943 [-2.434, 53.873], p=0.1365; conversation Δ 23.943 [0.589, 50.642], p=0.1860
    semantic_keep_compress_drop: row Δ logit L2 8.574 [-35.798, 43.493], p=0.6905; conversation Δ 8.574 [-36.457, 44.473], p=0.7292
  budget 0.35:
    geometry: row Δ logit L2 31.549 [-5.215, 74.153], p=0.1502; conversation Δ 31.549 [-9.264, 75.328], p=0.2107
    geometry_keep_compress_drop: row Δ logit L2 69.488 [13.380, 136.285], p=0.0385; conversation Δ 69.488 [-8.930, 167.085], p=0.1872
    semantic: row Δ logit L2 -12.584 [-36.488, 12.515], p=0.3100; conversation Δ -12.584 [-35.177, 10.349], p=0.2980
    semantic_keep_compress_drop: row Δ logit L2 25.683 [-16.518, 75.679], p=0.3247; conversation Δ 25.683 [-12.248, 68.794], p=0.2915
  budget 0.50:
    geometry: row Δ logit L2 -17.234 [-76.300, 34.242], p=0.5550; conversation Δ -17.234 [-95.720, 41.006], p=0.7788
    geometry_keep_compress_drop: row Δ logit L2 30.048 [-41.869, 99.876], p=0.4400; conversation Δ 30.048 [-73.650, 144.681], p=0.5485
    semantic: row Δ logit L2 -25.803 [-81.014, 18.535], p=0.3845; conversation Δ -25.803 [-99.596, 25.884], p=0.6727
    semantic_keep_compress_drop: row Δ logit L2 -14.567 [-71.952, 28.971], p=0.6120; conversation Δ -14.567 [-87.145, 44.625], p=0.8070
  behavior:
    budget 0.20:
      geometry: row Δ answer avg NLL -0.0071 [-0.0494, 0.0281], p=1.0000; conversation Δ -0.0071 [-0.0494, 0.0281], p=1.0000
      geometry_keep_compress_drop: row Δ answer avg NLL -0.1119 [-0.4494, 0.1064], p=0.8177; conversation Δ -0.1119 [-0.4630, 0.1082], p=0.8117
      semantic: row Δ answer avg NLL -0.1125 [-0.4630, 0.1175], p=0.8127; conversation Δ -0.1125 [-0.4643, 0.1081], p=0.8045
      semantic_keep_compress_drop: row Δ answer avg NLL 0.0591 [-0.0507, 0.1693], p=0.3890; conversation Δ 0.0591 [-0.0553, 0.1733], p=0.3967
    budget 0.35:
      geometry: row Δ answer avg NLL 0.2393 [0.0453, 0.5003], p=0.0330; conversation Δ 0.2393 [0.0470, 0.4799], p=0.0340
      geometry_keep_compress_drop: row Δ answer avg NLL 0.4732 [0.1150, 0.9119], p=0.0620; conversation Δ 0.4732 [0.0840, 1.0449], p=0.1288
      semantic: row Δ answer avg NLL 0.0883 [-0.0222, 0.2700], p=0.4680; conversation Δ 0.0883 [-0.0233, 0.2614], p=0.4730
      semantic_keep_compress_drop: row Δ answer avg NLL 0.3830 [0.0917, 0.7297], p=0.0343; conversation Δ 0.3830 [0.0547, 0.7723], p=0.0665
    budget 0.50:
      geometry: row Δ answer avg NLL -0.2111 [-0.4505, -0.0293], p=0.1253; conversation Δ -0.2111 [-0.4284, -0.0472], p=0.1172
      geometry_keep_compress_drop: row Δ answer avg NLL -0.0134 [-0.3990, 0.4532], p=1.0000; conversation Δ -0.0134 [-0.3787, 0.4806], p=1.0000
      semantic: row Δ answer avg NLL -0.1643 [-0.4688, 0.0813], p=0.3830; conversation Δ -0.1643 [-0.4321, 0.1035], p=0.3740
      semantic_keep_compress_drop: row Δ answer avg NLL 0.0326 [-0.2569, 0.3288], p=0.8462; conversation Δ 0.0326 [-0.2104, 0.2469], p=0.7917
