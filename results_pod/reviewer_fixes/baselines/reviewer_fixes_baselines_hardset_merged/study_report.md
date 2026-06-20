# Paper 3 Study: reviewer_fixes_baselines_hardset_merged

- Created: 2026-06-20T04:22:11
- Models: qwen25_15b
- Families: code_conversation, long_dependency, retrieval_heavy
- Budgets: 0.20, 0.35, 0.50
- Policies: lexical_tfidf, longllmlingua, recency, recency_keep_compress_drop, uniform

## qwen25_15b

- Model name: `Qwen/Qwen2.5-1.5B-Instruct`
- Conversations: 9
- Evaluations: 540
- Behavior evaluations: 270
- Segment span: None
- Target-turn stride: 1
- Max target turns / conversation: None
- Max turns / conversation: None

- Improvement vs uniform @ 0.20:
  lexical_tfidf: delta logit L2 -5.463, relative logit L2 0.986
  longllmlingua: delta logit L2 -1.772, relative logit L2 0.995
  recency: delta logit L2 -39.017, relative logit L2 0.899
  recency_keep_compress_drop: delta logit L2 -39.017, relative logit L2 0.899
- Improvement vs uniform @ 0.35:
  lexical_tfidf: delta logit L2 -14.933, relative logit L2 0.956
  longllmlingua: delta logit L2 34.409, relative logit L2 1.102
  recency: delta logit L2 -11.670, relative logit L2 0.965
  recency_keep_compress_drop: delta logit L2 -26.147, relative logit L2 0.923
- Improvement vs uniform @ 0.50:
  lexical_tfidf: delta logit L2 8.427, relative logit L2 1.031
  longllmlingua: delta logit L2 101.299, relative logit L2 1.377
  recency: delta logit L2 6.180, relative logit L2 1.023
  recency_keep_compress_drop: delta logit L2 22.489, relative logit L2 1.084
- Behavior improvement vs uniform @ 0.20:
  lexical_tfidf: delta answer avg NLL -0.0742, delta answer-loss increase -0.0742
  longllmlingua: delta answer avg NLL -0.3753, delta answer-loss increase -0.3753
  recency: delta answer avg NLL -0.0069, delta answer-loss increase -0.0069
  recency_keep_compress_drop: delta answer avg NLL -0.0069, delta answer-loss increase -0.0069
- Behavior improvement vs uniform @ 0.35:
  lexical_tfidf: delta answer avg NLL -0.2682, delta answer-loss increase -0.2682
  longllmlingua: delta answer avg NLL -0.5135, delta answer-loss increase -0.5135
  recency: delta answer avg NLL -0.1567, delta answer-loss increase -0.1567
  recency_keep_compress_drop: delta answer avg NLL -0.1567, delta answer-loss increase -0.1567
- Behavior improvement vs uniform @ 0.50:
  lexical_tfidf: delta answer avg NLL -0.5521, delta answer-loss increase -0.5521
  longllmlingua: delta answer avg NLL -0.6563, delta answer-loss increase -0.6563
  recency: delta answer avg NLL -0.3264, delta answer-loss increase -0.3264
  recency_keep_compress_drop: delta answer avg NLL -0.3935, delta answer-loss increase -0.3935

- Confidence and significance:
  budget 0.20:
    lexical_tfidf: row Δ logit L2 -5.463 [-22.175, 6.892], p=0.6240; conversation Δ -5.463 [-21.428, 6.990], p=0.6947
    longllmlingua: row Δ logit L2 -1.772 [-28.633, 27.507], p=0.9133; conversation Δ -1.772 [-33.474, 29.252], p=0.9387
    recency: row Δ logit L2 -39.017 [-68.673, -14.713], p=0.0043; conversation Δ -39.017 [-56.865, -20.228], p=0.0032
    recency_keep_compress_drop: row Δ logit L2 -39.017 [-68.683, -14.495], p=0.0032; conversation Δ -39.017 [-57.196, -21.421], p=0.0025
  budget 0.35:
    lexical_tfidf: row Δ logit L2 -14.933 [-57.100, 23.165], p=0.4665; conversation Δ -14.933 [-36.208, 9.507], p=0.2527
    longllmlingua: row Δ logit L2 34.409 [9.150, 60.824], p=0.0077; conversation Δ 34.409 [13.003, 60.553], p=0.0168
    recency: row Δ logit L2 -11.670 [-43.564, 15.813], p=0.4718; conversation Δ -11.670 [-33.842, 10.706], p=0.3785
    recency_keep_compress_drop: row Δ logit L2 -26.147 [-66.697, 7.790], p=0.1885; conversation Δ -26.147 [-46.726, -3.880], p=0.0573
  budget 0.50:
    lexical_tfidf: row Δ logit L2 8.427 [-14.090, 30.109], p=0.4577; conversation Δ 8.427 [-15.742, 30.540], p=0.5068
    longllmlingua: row Δ logit L2 101.299 [70.559, 135.970], p=0.0000; conversation Δ 101.299 [79.149, 124.553], p=0.0035
    recency: row Δ logit L2 6.180 [-15.598, 27.684], p=0.5900; conversation Δ 6.180 [-11.099, 22.693], p=0.5062
    recency_keep_compress_drop: row Δ logit L2 22.489 [-5.341, 49.627], p=0.1065; conversation Δ 22.489 [-7.291, 52.975], p=0.2185
  behavior:
    budget 0.20:
      lexical_tfidf: row Δ answer avg NLL -0.0742 [-0.2225, 0.0000], p=1.0000; conversation Δ -0.0742 [-0.2225, 0.0000], p=1.0000
      longllmlingua: row Δ answer avg NLL -0.3753 [-0.5826, -0.1711], p=0.0015; conversation Δ -0.3753 [-0.5194, -0.2272], p=0.0077
      recency: row Δ answer avg NLL -0.0069 [-0.2013, 0.1491], p=1.0000; conversation Δ -0.0069 [-0.2003, 0.1463], p=1.0000
      recency_keep_compress_drop: row Δ answer avg NLL -0.0069 [-0.2069, 0.1498], p=1.0000; conversation Δ -0.0069 [-0.1916, 0.1475], p=1.0000
    budget 0.35:
      lexical_tfidf: row Δ answer avg NLL -0.2682 [-0.5342, -0.0259], p=0.1155; conversation Δ -0.2682 [-0.5078, -0.0602], p=0.1190
      longllmlingua: row Δ answer avg NLL -0.5135 [-0.8875, -0.1345], p=0.0227; conversation Δ -0.5135 [-0.7893, -0.2126], p=0.0222
      recency: row Δ answer avg NLL -0.1567 [-0.3248, -0.0176], p=0.1225; conversation Δ -0.1567 [-0.3007, -0.0272], p=0.1247
      recency_keep_compress_drop: row Δ answer avg NLL -0.1567 [-0.3210, -0.0176], p=0.1338; conversation Δ -0.1567 [-0.3115, -0.0417], p=0.1300
    budget 0.50:
      lexical_tfidf: row Δ answer avg NLL -0.5521 [-1.0055, -0.1716], p=0.0285; conversation Δ -0.5521 [-0.8868, -0.2333], p=0.0283
      longllmlingua: row Δ answer avg NLL -0.6563 [-1.0359, -0.3007], p=0.0025; conversation Δ -0.6563 [-0.9605, -0.3616], p=0.0083
      recency: row Δ answer avg NLL -0.3264 [-0.7224, 0.0091], p=0.1148; conversation Δ -0.3264 [-0.6607, -0.0019], p=0.1163
      recency_keep_compress_drop: row Δ answer avg NLL -0.3935 [-0.7556, -0.1015], p=0.0270; conversation Δ -0.3935 [-0.6834, -0.1166], p=0.0235
