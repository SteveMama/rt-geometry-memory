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
- on the original three-model hard-set checkpoint, `geometry_keep_compress_drop` was competitive with plain `geometry` at `0.20` and `0.35`, but still weaker than `geometry_segment_actions` at `0.50`
- after the fairness-controlled `qwen25_15b` sweep and the `qwen25_3b` probe, the picture is sharper:
  - `geometry_keep_compress_drop` is now the strongest low-to-mid budget codec family
  - plain `geometry` becomes the strongest high-budget logit policy on the 3B probe
  - `geometry_segment_actions` remains viable, but is no longer the leading low/mid-budget Paper 3 family on the Qwen fairness-controlled runs

So the current Paper 3 claim boundary is:

- segment-level memory actions are promising
- sparse keep/compress/drop is now a real policy
- the codec story is now real enough to justify direct codec-specific optimization rather than only bridge-policy tuning

## Current Quantitative Checkpoint

Use these quantities:

- `Delta_logit(policy) = E[||ell_policy - ell_full||_2] - E[||ell_uniform - ell_full||_2]`
- `R_logit(policy) = E[||ell_policy - ell_full||_2] / E[||ell_uniform - ell_full||_2]`
- `Delta_NLL(policy) = E[NLL_policy - NLL_uniform]`

Negative `Delta_logit` and `Delta_NLL` are better. `R_logit < 1` is better.

### Fairness Sweep

Tracked artifact:

- [`artifacts/paper3/paper3_batch_v1_fairness`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_batch_v1_fairness)

On `qwen25_15b`, `geometry_keep_compress_drop` is the strongest logit policy at every budget from `0.24` through `0.46`, with particularly strong low-to-mid budget results:

- `0.24`: `Delta_logit = -38.272`, `R_logit = 0.898`, `p = 0.0022`
- `0.28`: `Delta_logit = -42.138`, `R_logit = 0.883`, `p = 0.0008`
- `0.32`: `Delta_logit = -56.165`, `R_logit = 0.845`, `p = 0.0030`
- `0.38`: `Delta_logit = -41.881`, `R_logit = 0.871`, `p = 0.0213`

This is specifically important because the realized token fractions are tightly matched there:

- `budget 0.24`: `uniform 0.579`, `KCD 0.572`
- `budget 0.28`: `uniform 0.595`, `KCD 0.601`
- `budget 0.32`: `uniform 0.617`, `KCD 0.628`

So the low-budget Paper 3 win survives fairness control. At `0.24`, the codec wins while using less realized token fraction than uniform.

Behavior is also strongest for the codec family:

- `budget 0.32`: `Delta_NLL = -0.3903`, `p = 0.0270`
- `budget 0.35`: `Delta_NLL = -0.3737`, `p = 0.0293`
- `budget 0.46`: `Delta_NLL = -0.5272`, `p = 0.0095`
- `budget 0.50`: `Delta_NLL = -0.6376`, `p = 0.0040`

### Pairwise Policy Structure

Tracked artifact:

- [`artifacts/paper3/paper3_batch_v1_fairness/pairwise_report.md`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_batch_v1_fairness/pairwise_report.md)

On the fairness sweep:

- `KCD` beats `segment_actions` on logit at `0.24`, `0.28`, and `0.32`
- `segment_actions` is significantly worse than plain `geometry` on logit at `0.24`, `0.28`, and `0.32`
- `KCD` beats `segment_actions` on behavior at `0.46` and `0.50`

So the current low/mid-budget ordering on `qwen25_15b` is:

`geometry_keep_compress_drop` > `geometry` > `geometry_segment_actions`

while the high-budget ordering flattens and eventually reverses toward geometry retention.

### 3B Probe

Tracked artifact:

- [`artifacts/paper3/paper3_batch_v1_3b`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_batch_v1_3b)

On `qwen25_3b`, the regime split is clean:

- at `0.35`, `geometry_keep_compress_drop` is strongest:
  - `Delta_logit = -70.997`
  - `R_logit = 0.854`
  - `p = 0.0070`
- at `0.50`, plain `geometry` is strongest:
  - `Delta_logit = -46.526`
  - `R_logit = 0.891`
  - `p = 0.0333`

Pairwise:

- `KCD vs geometry @ 0.35`: `Delta = -53.675`, `p = 0.0272`
- `KCD vs geometry @ 0.50`: `Delta = +24.357`, `p = 0.0063`

This is the strongest evidence yet that the codec family is best under scarcity, while direct geometry retention retakes the lead once budget loosens.

### Mechanism

Tracked artifacts:

- [`artifacts/paper3/paper3_batch_v1_fairness/memory_critical_qwen25_15b_keep_compress_drop_b035.md`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_batch_v1_fairness/memory_critical_qwen25_15b_keep_compress_drop_b035.md)
- [`artifacts/paper3/paper3_batch_v1_3b/memory_critical_qwen25_3b_keep_compress_drop_b035.md`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_batch_v1_3b/memory_critical_qwen25_3b_keep_compress_drop_b035.md)

At budget `0.35`, the sparse codec retains more support user turns than uniform:

- `qwen25_15b`: `17/36` better, `2/36` worse, latest support rescued in `17/36`
- `qwen25_3b`: `14/36` better, `6/36` worse, latest support rescued in `13/36`

Compression itself is not destroying the mechanism:

- `qwen25_15b`: `27/29` compressed cases are not worse than uniform on support retention
- `qwen25_3b`: `23/29` compressed cases are not worse than uniform on support retention

The rescued support objects remain the same ones found earlier:

- support constraints
- base-memory turns
- exact code and retrieval requirements

## Current Scientific Reading

The Paper 3 picture is now:

- `geometry_keep_compress_drop` is the strongest low-to-mid budget sparse codec family
- plain `geometry` is the strongest high-budget logit policy
- `geometry_segment_actions` is still viable, but is no longer the main low/mid-budget candidate on the fairness-controlled Qwen runs
- the next design step should focus on making the compressed memory object even more support-aware rather than inventing many more policy families

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
