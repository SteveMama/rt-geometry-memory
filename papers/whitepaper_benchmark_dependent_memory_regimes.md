# White Paper: Benchmark-Dependent Memory Regimes In Geometry-Guided Conversation Compression

## Executive Summary

The current RT checkpoint shows that conversation compression does not admit a single universally best policy family.

Instead, the evidence now supports a benchmark-dependent regime view:

- **support-turn-critical tasks** reward sparse, support-faithful codec policies such as `geometry_keep_compress_drop`
- **semantic conversational-memory tasks** reward broader semantic retention policies, with sparse codec structure helping weakly or not at all

This is not a weakness in the project. It is the main scientific clarification produced by the current checkpoint.

The central question has shifted:

> the problem is no longer whether hidden-state geometry matters, but which compression signal and memory form match which conversational-memory regime

## Program Context

The RT program now has a stable three-stage structure.

### Paper 1

Paper 1 established that conversation-state trajectories are:

- strongly low-rank
- geometrically structured
- decoder-relevant

The main settled claim is that geometric distortion is predictive of decoder drift.

### Paper 2

Paper 2 established that geometry can be used as a memory-control signal:

- geometry-aware retention beats uniform under conversational memory scarcity
- the mechanism is not black-box:
  - geometry preserves memory-critical support turns more often than uniform

This made a Paper 3 codec stage scientifically justified.

### Paper 3

Paper 3 began with geometry-driven sparse codec policies and later expanded to semantic public-benchmark comparisons.

At this point, Paper 3 no longer supports a single universal winner. Instead, it supports a regime split.

## The Core Distinction

The current data separate conversational-memory benchmarks into two broad families.

### Regime A: support-turn-critical memory

These benchmarks reward:

- exact preservation of key support turns
- rescue of hard constraints
- retention of code, retrieval, or formatting requirements that later turns depend on precisely

The hard stress set used in the project belongs here.

In this regime:

- `geometry_keep_compress_drop` is strong
- plain geometry is a good fallback
- the sparse codec form is genuinely useful

This regime supports the original sparse-codec hypothesis.

### Regime B: semantic conversational memory

These benchmarks reward:

- broad episode continuity
- persona and preference recall
- recovery of earlier topic content in a semantically smooth way

MSC clearly belongs here. LongMemEval appears to overlap with this regime, though with a somewhat more mixed profile.

In this regime:

- `semantic` is often strongest
- `geometry` remains useful
- sparse codec structure is weakly helpful or harmful
- geometry-driven sparse codec is specifically mismatched

This regime says that the right signal is semantic rather than geometric, and that sparse codec form is not automatically the correct representation.

## What The New MSC Result Adds

The MSC run with `semantic_keep_compress_drop` answers a key design question:

> if a benchmark appears to prefer semantic memory, can sparse codec form recover performance when paired with semantic signal?

The answer is:

- **at `0.20`**:
  - `semantic_keep_compress_drop` is numerically best on mean logit distortion
  - but not significantly better than plain `semantic`
  - behavior is effectively tied
- **at `0.35` and `0.50`**:
  - plain `semantic` is better overall
  - behavior is significantly better under plain semantic retention

So the semantic codec result is not a codec victory. It is a signal-isolation result.

It shows that:

- changing from geometry to semantic fixes most of the codec mismatch
- but sparse codec form still does not beat plain semantic retention on MSC

That is a valuable negative result.

## Updated Interpretation Of Each Policy Family

### `semantic`

This is now the best available baseline for semantic-memory benchmarks.

It should be treated as:

- the leading policy on MSC-style multi-session recall
- a serious competitor on LongMemEval-style public benchmarks

### `geometry`

Plain geometry remains important.

It is:

- clearly useful relative to uniform
- often the best fallback/control policy when the sparse codec is not well matched
- still central to the overall program, because it remains the strongest mechanism for support-turn-aware control

### `geometry_keep_compress_drop`

This policy should no longer be framed as a general conversational-memory codec.

It should now be framed more precisely as:

> the sparse codec family for support-turn-faithful memory under scarcity

That is narrower, but scientifically stronger.

### `semantic_keep_compress_drop`

This new policy is informative even though it does not win.

It demonstrates that:

- codec structure can inherit much of the benefit of semantic signal
- but on MSC, sparse compression is still not the best form for semantic conversational memory

This clarifies that the open problem is not just signal choice. It is also memory-object choice.

## What Is Now Scientifically Settled

The current checkpoint supports the following claims.

### Settled Claim 1

Hidden-state geometry is real and decoder-relevant.

This is Paper 1.

### Settled Claim 2

Geometry is a useful control signal for conversation compression under scarcity.

This is Paper 2.

### Settled Claim 3

Sparse codec structure is beneficial for support-turn-critical conversational memory, but not universally.

This is the strongest current Paper 3 claim.

### Settled Claim 4

Semantic conversational-memory benchmarks favor semantic signal and often plain semantic retention over sparse codec structure.

This is the new contribution of the MSC result.

## What Is Not Yet Settled

Several questions remain open.

### Open Question 1

Does LoCoMo behave like MSC or like the hard stress set?

This is the next most important benchmark question.

### Open Question 2

Can a memory object be designed that preserves support-turn fidelity without sacrificing broad semantic continuity?

This is the core codec-design question now.

### Open Question 3

Can a regime-aware policy choose between sparse codec and dense semantic retention automatically from the benchmark structure or the conversation trace?

That is the natural unification question for later Paper 3 work.

## Research Strategy Going Forward

The strongest next move is not to search for one universal winning policy by heuristic tuning.

The stronger strategy is:

1. confirm the semantic-memory regime on `LoCoMo`
2. analyze mechanism differences between:
   - `semantic`
   - `geometry`
   - `semantic_keep_compress_drop`
3. define a more explicit benchmark taxonomy:
   - support-turn-critical
   - semantic-memory
4. only then design a unified or regime-aware codec

## Strategic Conclusion

The project is stronger now than before the MSC result.

Before MSC, the risk was overgeneralizing from the hard stress set and treating sparse codec behavior as universally desirable.

After MSC, the project can say something sharper and more credible:

> conversational memory compression is regime-dependent; the right signal and the right memory form depend on whether the task requires exact support-turn fidelity or broad semantic continuity

That statement is stronger than a single benchmark win. It is the beginning of a real theory of conversational memory compression.
