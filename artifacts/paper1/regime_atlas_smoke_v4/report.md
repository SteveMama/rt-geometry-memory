# Geometric Regime Atlas Report

- Model: `qwen25_05b`
- Input paths: /tmp/atlas_msc_smoke.jsonl, /tmp/atlas_locomo_smoke.jsonl, /tmp/atlas_longmem_smoke.jsonl, /tmp/atlas_stress_smoke.jsonl
- Segments: 208
- Requested clusters: 4
- Segment length bounds: 3..8
- Curvature metric: stabilized curvature with arclength floor `0.050`

## Regimes

### Regime 0: near_stationary_fact_memory

- Segments: 71
- Family counts: {'locomo10': 36, 'longmemeval_s_cleaned': 25, 'msc_valid': 10}
- Centroid stats:
  - mean_curvature=25.302
  - std_curvature=0.000
  - max_curvature=25.302
  - rank95=1.507
  - mean_rank_jump=0.000
  - mean_subspace_shift=0.008
  - role_switch_rate=0.989

- Example segments:
  - `msc_valid` `msc-00000` turns 33-40: assistant: I often take it out when I am having a rough day at work...it is mindless fun.  Almost as much as the stunt double work  || user: That sounds like an incredible amount of fun, but anything but mindless! What was your last stunt gig? || assistant: Only the real stars get that treatment.  The stunt doubles get old coffee and donuts in the trailer. || user: Do you have any new stunt double jobs coming  up?
  - `msc_valid` `msc-00000` turns 40-47: user: Do you have any new stunt double jobs coming  up? || assistant: No, I don't have any at this time. || user: Of course, I'll watch or read anything true crime related.  My kindle is just filled with crime books.  What was your ex || assistant: It was so creepy it still enters my dreams.  I see you like Steven King movies, but what about his books?
  - `msc_valid` `msc-00000` turns 47-54: assistant: It was so creepy it still enters my dreams.  I see you like Steven King movies, but what about his books? || user: I love the books more!  I'll never watch a Steven King movie until I've read the book.  I think the IT movies may have r || assistant: I haven't yet. I have been a little busy, but I should be able to finish it by next week! || user: I need to read it so we can talk about it when you are done. Have you read Fatal Charm?

### Regime 1: curvature_spike_transition

- Segments: 34
- Family counts: {'locomo10': 12, 'msc_valid': 11, 'retrieval_heavy': 6, 'longmemeval_s_cleaned': 3, 'long_dependency': 2}
- Centroid stats:
  - mean_curvature=1.138
  - std_curvature=0.185
  - max_curvature=1.378
  - rank95=1.324
  - mean_rank_jump=0.000
  - mean_subspace_shift=0.572
  - role_switch_rate=1.000

- Example segments:
  - `msc_valid` `msc-00000` turns 0-1: user: Hello what are doing today? || assistant: I am good, I just got off work and tired, I have two jobs.
  - `msc_valid` `msc-00000` turns 24-26: user: Oh ok. I have used other type of apps where it syncs up with the library number to check books out. I really love that o || assistant: Yeah it does! I am so happy to have someone to talk about books with! Not many people that I know appreciate them as muc || user: You never know, with everything you have done in your past, I might read a book about YOU eventually! I can't even imagi
  - `msc_valid` `1` turns 0-1: user: Hi! How are you doing tonight? || assistant: I'm doing great. Just relaxing with my two dogs.

### Regime 2: curvature_spike_transition

- Segments: 7
- Family counts: {'locomo10': 4, 'longmemeval_s_cleaned': 2, 'msc_valid': 1}
- Centroid stats:
  - mean_curvature=18.822
  - std_curvature=11.772
  - max_curvature=27.802
  - rank95=1.571
  - mean_rank_jump=0.000
  - mean_subspace_shift=0.678
  - role_switch_rate=0.918

- Example segments:
  - `msc_valid` `msc-00000` turns 26-33: user: You never know, with everything you have done in your past, I might read a book about YOU eventually! I can't even imagi || assistant: I mean, you said you like Stephen King. And Shawshank. I read that book. It was pretty boring. But it was also amazing.  || user: Well my son is probably too old for it at this point, but I might volunteer to read it for the kids group at the library || assistant: I often take it out when I am having a rough day at work...it is mindless fun.  Almost as much as the stunt double work 
  - `locomo10` `conv-26-qa000` turns 29-36: assistant: I chose them 'cause they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me. || user: That's great, Caroline! Loving the inclusivity and support. Anything you're excited for in the adoption process? || user: Hey Melanie! How's it going? I wanted to tell you about my school event last week. It was awesome! I talked about my tra || assistant: Hey Caroline! Great to hear from you. Sounds like your event was amazing! I'm so proud of you for spreading awareness an
  - `locomo10` `conv-26-qa001` turns 29-36: assistant: I chose them 'cause they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me. || user: That's great, Caroline! Loving the inclusivity and support. Anything you're excited for in the adoption process? || user: Hey Melanie! How's it going? I wanted to tell you about my school event last week. It was awesome! I talked about my tra || assistant: Hey Caroline! Great to hear from you. Sounds like your event was amazing! I'm so proud of you for spreading awareness an

### Regime 3: curvature_spike_transition

- Segments: 96
- Family counts: {'locomo10': 44, 'msc_valid': 44, 'retrieval_heavy': 6, 'long_dependency': 1, 'longmemeval_s_cleaned': 1}
- Centroid stats:
  - mean_curvature=2.097
  - std_curvature=0.050
  - max_curvature=2.149
  - rank95=1.000
  - mean_rank_jump=0.000
  - mean_subspace_shift=0.383
  - role_switch_rate=1.000

- Example segments:
  - `msc_valid` `msc-00000` turns 1-3: assistant: I am good, I just got off work and tired, I have two jobs. || user: I just got done watching a horror movie || assistant: I rather read, I've read about 20 books this year.
  - `msc_valid` `msc-00000` turns 3-5: assistant: I rather read, I've read about 20 books this year. || user: Wow! I do love a good horror movie. Loving this cooler weather || assistant: But a good movie is always good.
  - `msc_valid` `msc-00000` turns 5-7: assistant: But a good movie is always good. || user: Yes! My son is in junior high and I just started letting him watch them too || assistant: I work in the movies as well.

## Family Regime Distribution

| Family | Regime | Count | Fraction |
| --- | --- | ---: | ---: |
| locomo10 | 0 near_stationary_fact_memory | 36 | 0.375 |
| locomo10 | 1 curvature_spike_transition | 12 | 0.125 |
| locomo10 | 2 curvature_spike_transition | 4 | 0.042 |
| locomo10 | 3 curvature_spike_transition | 44 | 0.458 |
| long_dependency | 1 curvature_spike_transition | 2 | 0.667 |
| long_dependency | 3 curvature_spike_transition | 1 | 0.333 |
| longmemeval_s_cleaned | 0 near_stationary_fact_memory | 25 | 0.806 |
| longmemeval_s_cleaned | 1 curvature_spike_transition | 3 | 0.097 |
| longmemeval_s_cleaned | 2 curvature_spike_transition | 2 | 0.065 |
| longmemeval_s_cleaned | 3 curvature_spike_transition | 1 | 0.032 |
| msc_valid | 0 near_stationary_fact_memory | 10 | 0.152 |
| msc_valid | 1 curvature_spike_transition | 11 | 0.167 |
| msc_valid | 2 curvature_spike_transition | 1 | 0.015 |
| msc_valid | 3 curvature_spike_transition | 44 | 0.667 |
| retrieval_heavy | 1 curvature_spike_transition | 6 | 0.500 |
| retrieval_heavy | 3 curvature_spike_transition | 6 | 0.500 |
