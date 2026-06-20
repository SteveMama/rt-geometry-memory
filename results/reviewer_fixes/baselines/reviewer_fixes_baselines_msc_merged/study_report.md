# Paper 3 Study: reviewer_fixes_baselines_msc_merged

- Created: 2026-06-20T04:22:16
- Models: qwen25_15b
- Families: all
- Budgets: 0.20, 0.35, 0.50
- Policies: lexical_tfidf, longllmlingua, recency, recency_keep_compress_drop, uniform

## qwen25_15b

- Model name: `Qwen/Qwen2.5-1.5B-Instruct`
- Conversations: 1000
- Evaluations: 163755
- Behavior evaluations: 158115
- Segment span: None
- Target-turn stride: 2
- Max target turns / conversation: 16
- Max turns / conversation: None

- Improvement vs uniform @ 0.20:
  lexical_tfidf: delta logit L2 -11.710, relative logit L2 0.985
  longllmlingua: delta logit L2 94.363, relative logit L2 1.118
  recency: delta logit L2 -25.094, relative logit L2 0.969
  recency_keep_compress_drop: delta logit L2 -27.384, relative logit L2 0.966
- Improvement vs uniform @ 0.35:
  lexical_tfidf: delta logit L2 -6.869, relative logit L2 0.991
  longllmlingua: delta logit L2 120.919, relative logit L2 1.155
  recency: delta logit L2 -45.030, relative logit L2 0.942
  recency_keep_compress_drop: delta logit L2 -30.664, relative logit L2 0.961
- Improvement vs uniform @ 0.50:
  lexical_tfidf: delta logit L2 -9.495, relative logit L2 0.988
  longllmlingua: delta logit L2 157.951, relative logit L2 1.208
  recency: delta logit L2 -64.254, relative logit L2 0.916
  recency_keep_compress_drop: delta logit L2 -45.401, relative logit L2 0.940
- Behavior improvement vs uniform @ 0.20:
  lexical_tfidf: delta answer avg NLL -0.1457, delta answer-loss increase -0.1457
  longllmlingua: delta answer avg NLL -0.1769, delta answer-loss increase -0.1769
  recency: delta answer avg NLL -0.1586, delta answer-loss increase -0.1586
  recency_keep_compress_drop: delta answer avg NLL -0.1733, delta answer-loss increase -0.1733
- Behavior improvement vs uniform @ 0.35:
  lexical_tfidf: delta answer avg NLL -0.2445, delta answer-loss increase -0.2445
  longllmlingua: delta answer avg NLL -0.1246, delta answer-loss increase -0.1246
  recency: delta answer avg NLL -0.2661, delta answer-loss increase -0.2661
  recency_keep_compress_drop: delta answer avg NLL -0.2883, delta answer-loss increase -0.2883
- Behavior improvement vs uniform @ 0.50:
  lexical_tfidf: delta answer avg NLL -0.3180, delta answer-loss increase -0.3180
  longllmlingua: delta answer avg NLL -0.0742, delta answer-loss increase -0.0742
  recency: delta answer avg NLL -0.3400, delta answer-loss increase -0.3400
  recency_keep_compress_drop: delta answer avg NLL -0.3727, delta answer-loss increase -0.3727

- Confidence and significance:
  budget 0.20:
    lexical_tfidf: row Δ logit L2 -11.710 [-14.867, -8.653], p=0.0000; conversation Δ -9.799 [-12.628, -7.128], p=0.0000
    longllmlingua: row Δ logit L2 94.363 [92.423, 96.210], p=0.0000; conversation Δ 103.228 [101.063, 105.251], p=0.0000
    recency: row Δ logit L2 -25.094 [-28.224, -22.059], p=0.0000; conversation Δ -21.835 [-24.735, -19.299], p=0.0000
    recency_keep_compress_drop: row Δ logit L2 -27.384 [-30.250, -24.486], p=0.0000; conversation Δ -22.270 [-25.055, -19.576], p=0.0000
  budget 0.35:
    lexical_tfidf: row Δ logit L2 -6.869 [-11.118, -2.488], p=0.0018; conversation Δ -3.235 [-7.035, 0.502], p=0.0963
    longllmlingua: row Δ logit L2 120.919 [118.545, 123.245], p=0.0000; conversation Δ 127.257 [124.709, 130.009], p=0.0000
    recency: row Δ logit L2 -45.030 [-49.114, -40.946], p=0.0000; conversation Δ -41.116 [-44.900, -37.357], p=0.0000
    recency_keep_compress_drop: row Δ logit L2 -30.664 [-34.817, -26.628], p=0.0000; conversation Δ -22.827 [-26.769, -19.013], p=0.0000
  budget 0.50:
    lexical_tfidf: row Δ logit L2 -9.495 [-14.957, -4.059], p=0.0008; conversation Δ -2.237 [-6.978, 3.004], p=0.3847
    longllmlingua: row Δ logit L2 157.951 [154.833, 160.944], p=0.0000; conversation Δ 160.910 [157.647, 164.105], p=0.0000
    recency: row Δ logit L2 -64.254 [-69.331, -59.515], p=0.0000; conversation Δ -58.783 [-64.007, -53.788], p=0.0000
    recency_keep_compress_drop: row Δ logit L2 -45.401 [-50.543, -40.300], p=0.0000; conversation Δ -31.871 [-37.377, -26.323], p=0.0000
  behavior:
    budget 0.20:
      lexical_tfidf: row Δ answer avg NLL -0.1457 [-0.1563, -0.1351], p=0.0000; conversation Δ -0.1121 [-0.1222, -0.1021], p=0.0000
      longllmlingua: row Δ answer avg NLL -0.1769 [-0.1831, -0.1706], p=0.0000; conversation Δ -0.2471 [-0.2594, -0.2351], p=0.0000
      recency: row Δ answer avg NLL -0.1586 [-0.1697, -0.1475], p=0.0000; conversation Δ -0.1255 [-0.1363, -0.1148], p=0.0000
      recency_keep_compress_drop: row Δ answer avg NLL -0.1733 [-0.1842, -0.1626], p=0.0000; conversation Δ -0.1424 [-0.1527, -0.1326], p=0.0000
    budget 0.35:
      lexical_tfidf: row Δ answer avg NLL -0.2445 [-0.2582, -0.2309], p=0.0000; conversation Δ -0.1949 [-0.2073, -0.1824], p=0.0000
      longllmlingua: row Δ answer avg NLL -0.1246 [-0.1323, -0.1166], p=0.0000; conversation Δ -0.1991 [-0.2130, -0.1844], p=0.0000
      recency: row Δ answer avg NLL -0.2661 [-0.2797, -0.2516], p=0.0000; conversation Δ -0.2189 [-0.2321, -0.2065], p=0.0000
      recency_keep_compress_drop: row Δ answer avg NLL -0.2883 [-0.3012, -0.2746], p=0.0000; conversation Δ -0.2487 [-0.2608, -0.2372], p=0.0000
    budget 0.50:
      lexical_tfidf: row Δ answer avg NLL -0.3180 [-0.3338, -0.3027], p=0.0000; conversation Δ -0.2409 [-0.2571, -0.2265], p=0.0000
      longllmlingua: row Δ answer avg NLL -0.0742 [-0.0840, -0.0644], p=0.0000; conversation Δ -0.1581 [-0.1745, -0.1408], p=0.0000
      recency: row Δ answer avg NLL -0.3400 [-0.3563, -0.3239], p=0.0000; conversation Δ -0.2698 [-0.2849, -0.2547], p=0.0000
      recency_keep_compress_drop: row Δ answer avg NLL -0.3727 [-0.3887, -0.3571], p=0.0000; conversation Δ -0.3113 [-0.3252, -0.2977], p=0.0000
