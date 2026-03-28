# Paper 3

## Working Title

**Manifold Memory Codecs for Conversational State Compression in LLMs**

## Goal

Build a decoder-compatible compressed latent memory surrogate from manifold structure.

## Core Question

Can part of old conversational memory be replaced with a decoder-compatible compressed latent representation reconstructed from manifold dynamics?

## Precise Memory Object

Do not define the target as the whole conversation. Define a memory state `m_t in M`, where `m_t` is one of:

- a learned latent memory state
- a compressed cross-layer summary
- a decoder-compatible surrogate for older KV blocks
- a residual memory-token state

## Current Working Commitment

For the current Paper 2 to Paper 3 bridge, use a segment-level memory object:

`m_j = (a_j, s_j, meta_j)`

where:

- `a_j` is a segment anchor state or keyframe summary
- `s_j` is a sparse support memory for the segment
- `meta_j` stores role/order metadata needed for reconstruction or injection

In the current Paper 2 scaffold:

- `keep` means preserve the full segment turns
- `compress` means preserve only sparse support turns for the segment
- `evict` means drop the segment entirely

Paper 3 should generalize `s_j` from sparse retained turns into a decoder-compatible latent summary or surrogate memory state.

## Current Pilot Status

- current strongest baseline: `geometry_segment_actions`
- current minimal codec policy: `geometry_keep_compress_drop`
- first failure mode was degenerate action selection with zero compressed segments
- current fixed policy now produces nonzero compressed segments and responds to budget
- on the verified `qwen25_05b` hard-set rerun, `geometry_keep_compress_drop` is competitive with plain `geometry` at `0.20` and `0.35`, but still weaker than `geometry_segment_actions` at `0.50`

So the current Paper 3 claim boundary is:

- segment-level memory actions are promising
- sparse keep/compress/drop is now a real policy
- but `geometry_segment_actions` remains the stronger bridge baseline until the compressed representation improves

## Codec Structure

For each segment `j`:

1. store keyframe `q_j`
2. store low-rank basis `B_j`
3. store coefficient sequence `c_t`
4. reconstruct by transported tangent dynamics

Informally:

`m_hat_{t+1} = Exp_{m_hat_t}(PT_{q_j -> m_hat_t}(B_j c_t))`

## Honest Storage Accounting

Use:

`storage = Jd + sum_j d r_j + sum_t r_{s(t)} + metadata`

not optimistic shorthand.

## Target Theorem Direction

If:

- segment curvature is bounded
- subspace approximation error is bounded
- the decoder map is locally Lipschitz in a decoder-aware metric

then:

- memory reconstruction error is bounded
- decoder output drift is bounded by the induced manifold distortion

## Variants

- sphere-only codec on normalized summaries
- piecewise tangent low-rank codec
- learned manifold codec trained to preserve logits or attention behavior

## Experiments

Compare against:

- summarization-based memory
- rolling window truncation
- learned memory tokens
- KV compression baselines
- the Paper 2 controller

Tasks:

- long dialogue memory
- early-turn retrieval
- consistency over long interaction
- code/debug assistance with old state
- multi-session memory if feasible

Metrics:

- exact match or QA scores
- faithfulness to earlier turns
- token-level KL
- long-horizon degradation curves
- compression ratio
- memory saved per retained accuracy

## Contribution If It Works

- theoretical codec
- practical latent-memory system
- decoder-aware reconstruction guarantee
- strong memory/quality tradeoff

## Dependency

Paper 3 should happen only if Paper 1 and Paper 2 both look real enough to justify the codec attempt.
