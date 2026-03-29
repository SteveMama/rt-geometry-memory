# Quick Benchmark Plan

This file records the fast-iteration benchmark order for Paper 3 after the first
LongMemEval public-benchmark run.

## Order

1. `MSC`
2. `LoCoMo`
3. `GapChat` or `REALTALK`

## Why This Order

- `MSC` is the fastest useful sanity check for multi-session conversational memory.
- `LoCoMo` is the strongest small-but-rich long-memory benchmark.
- `GapChat` adds time-gap structure.
- `REALTALK` adds realism, but is heavier for the first debug loop.

## Recommended First Runs

### MSC

- subset size: `20-50` conversations
- policies:
  - `uniform`
  - `semantic`
  - `geometry`
  - `geometry_keep_compress_drop`
- budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- objective:
  - quickest multi-session sanity check for conversational fact retention and preference recall

### LoCoMo

- subset size: `5-10` conversations
- policies:
  - `uniform`
  - `semantic`
  - `geometry`
  - `geometry_keep_compress_drop`
- budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- objective:
  - richer support-turn and sparse-codec stress than MSC

## Success Criteria

The quick-benchmark loop is informative if it answers at least one of:

- does `semantic` still win under scarcity?
- does `geometry_keep_compress_drop` recover at looser budgets?
- does the hard-set KCD story transfer to a lighter public benchmark?
- is the benchmark more aligned with broad semantic retrieval than support-turn faithful sparse compression?

## Operational Notes

- prefer `qwen25_15b` for the first quick public benchmark
- keep `segment_actions` out of the first quick loop unless needed
- use bounded target-turn sampling from the start
- treat these as comparative checkpoint runs, not final publication runs
