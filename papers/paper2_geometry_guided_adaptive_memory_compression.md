# Paper 2

## Working Title

**Geometry-Guided Adaptive Memory Compression for Long-Context LLM Inference**

## Goal

Use curvature and local rank geometry as a policy signal to improve memory or KV compression.

## Core Question

Can geometry be used as a control law for deciding where compression is safe and where it is dangerous?

## Core Hypothesis

- `H4`: geometry-guided compression beats uniform compression at equal budget

More concretely:

- low-curvature, low-rank segments can be compressed more aggressively
- high-curvature or decoder-sensitive segments should be preserved more carefully

## System Idea

For each segment or memory block, define a geometry score:

`g_j = alpha * mean_curvature_j + beta * effective_rank_j + gamma * sensitivity_j`

Map that score to a local compression policy:

`pi_j = compression_level(g_j)`

Examples:

- low score: stronger quantization, lower retention, more aggressive eviction
- high score: denser storage, more bits, preserve keyframes, avoid eviction

## Optimization Framing

Constrained objective:

`min_pi E[output_drift under pi]`

subject to:

`E[memory_cost under pi] <= B`

Equivalent Lagrangian form:

`L(pi) = E[KL(p_theta, p_theta_pi)] + lambda * E[M(pi)]`

## Target Statement

- Proposition 3: if decoder drift is monotone in a geometry-derived sensitivity score and distortion is convex in compression severity, prioritized budget allocation weakly dominates uniform allocation at fixed budget.

## Algorithms To Compare

- cache retention and eviction
- selective KV preservation
- adaptive quantization
- selective summarization of old segments
- hybrid preserve/summarize/compress policies

## Experiments

Benchmarks:

- long conversation QA
- conversation recall
- instruction following with early-turn dependencies
- multi-hop reasoning with old facts
- code assistance over long dialogue

Metrics:

- answer correctness
- old-turn factual recall
- token-level KL or logit drift
- memory footprint
- latency
- throughput

## Contribution

- first geometry-aware compression controller
- practical improvement over flat heuristics
- bridge from representation geometry to systems performance

## Current Empirical Status

- strongest result: geometry-aware retention beats uniform retention under constrained budgets on the harder long-context stress set
- clearest regime: low-to-mid budgets, especially around `0.20` to `0.35`
- behavior metrics are now informative on the hard set, but the answer-level gains remain mixed rather than a clean geometry-only win
- lexical retention is now a credible comparator on the hard set, but geometry still gives the cleanest overall control signal
- segmentwise keep/compress/evict is implemented and usable as a bridge to Paper 3, but it is not yet a stronger headline controller than turn-level geometry retention

## Current Claim Boundary

Paper 2 should currently claim:

- geometry-aware retention improves decoder stability relative to uniform allocation under constrained budgets on the hard long-context stress set
- the clearest and most reliable gains occur in the lower-to-mid budget regime
- answer-level effects are directionally supportive but mixed

Paper 2 should not currently claim:

- that geometry is uniquely best on answer-level behavior
- that segment-action control is already a stronger controller than turn-level geometry retention
- that hybrid geometry-plus-lexical policies are the main direction

## Explanation Layer

The current explanation layer is now concrete enough to support the bridge to Paper 3:

- on the hard stress set, geometry often preserves fact-bearing or constraint-bearing earlier user turns that uniform drops
- in the first focused `qwen25_05b` case analysis at budget `0.35`, geometry retained more support user turns than uniform in `14/36` evaluated cases, while uniform did so in `5/36`
- geometry kept the latest support user turn while uniform dropped it in `13/36` cases
- the strongest rescued turn type is the support-constraint turn, not the assistant acknowledgement turn

This is the first quantitative sign that the controller is protecting memory-critical structure rather than only redistributing tokens.

## Bridge To Paper 3

Paper 2 should hand over:

- a validated geometry risk score
- hard examples where geometry preserves memory-critical turns better than uniform
- a segmentwise action space: keep, compress, evict
- a concrete memory object candidate for compressed segments

The current bridge asset list is:

- hard stress-set case reports that show which earlier constraints geometry preserves
- memory-critical support-turn analyses that quantify where geometry helps
- a working segment-action controller that can keep, compress, or evict blocks
- a Paper 3 memory object centered on segment anchor plus sparse support memory

## Dependency

Paper 2 should only make strong claims after Paper 1 shows that geometry is stable and decoder-relevant.
