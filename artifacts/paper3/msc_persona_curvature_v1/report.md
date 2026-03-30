# MSC Persona Curvature Check

- Benchmark: `msc_valid`
- Model: `qwen25_05b` (Qwen/Qwen2.5-0.5B-Instruct)
- Device: `mps`
- Conversations: 5
- Mean support-filler curvature delta: 0.1220
- Median support-filler curvature delta: 0.0104
- Positive deltas: 3
- Negative deltas: 2
- Mean support percentile: 0.6138
- Mean filler percentile: 0.6087

## Per-conversation summary

| Conversation | Support mean curvature | Filler mean curvature | Delta | Support pct | Filler pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| msc-00000 | 1.0656 | 0.0000 | 1.0656 | 0.7583 | 0.6333 |
| 1 | 0.5025 | 1.0048 | -0.5023 | 0.5167 | 0.5500 |
| 2 | 0.9594 | 0.5345 | 0.4249 | 0.5792 | 0.5958 |
| 6 | 0.6757 | 1.0642 | -0.3885 | 0.5484 | 0.6573 |
| 8 | 2.0098 | 1.9993 | 0.0104 | 0.6667 | 0.6071 |

## Reading

Positive delta means the manually labeled support/persona turns had higher stabilized curvature than the chosen filler turns in that conversation.

