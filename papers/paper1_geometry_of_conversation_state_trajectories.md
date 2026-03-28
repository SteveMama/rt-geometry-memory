# Paper 1

## Working Title

**Geometry of Conversation-State Trajectories in Large Language Models**

## Goal

Characterize whether conversation-state dynamics have exploitable geometric structure, and determine which parts of that geometry are decoder-relevant.

## Core Question

Do conversation states evolve as a piecewise-smooth, low-rank manifold trajectory in a way that is relevant to decoder behavior?

## Central Hypotheses

- `H1`: transported increments are low-rank within segments
- `H2`: segment curvature is bounded except at regime changes
- `H3`: small reconstruction error implies small output drift

## Current Reading

- `H1` is strong: segment-level rank-95 is consistently very small across tested local models.
- `H3` is strongest: geometric distortion is tightly correlated with logit drift.
- `H2` is mixed but real: geometry carries regime-boundary signal, but the evidence currently supports approximate localization rather than exact segmentation.

## Mathematical Setup

For a multi-turn conversation `x_{1:T}`, define a turn-level state summary:

`z_t = f_theta(x_{1:t}) in R^d`

Normalize it onto the sphere:

`h_t = z_t / ||z_t|| in S^{d-1}`

Partition the trajectory into segments `S_1, ..., S_J`, with segment reference point `q_j`. Inside a segment, define the transported increment:

`u_t = PT_{h_t -> q_j}(log_{h_t}(h_{t+1})) in T_{q_j} S^{d-1}`

Then test whether:

`u_t ~= B_j c_t + e_t`

with `B_j in R^{d x r}` and `r << d`.

Use a discrete curvature proxy such as:

`kappa_t ~= ||u_t - u_{t-1}|| / Delta s_t`

For decoder relevance, compare original and reconstructed states with:

- logit distance
- KL divergence
- top-1 agreement

## Target Statements

- Proposition 1: transported increments within a segment admit a common tangent-space representation, making low-rank analysis well-defined.
- Proposition 2: with bounded discrete geodesic curvature and bounded subspace error, piecewise-geodesic reconstruction error scales like curvature times segment-length-squared plus subspace error.

## Experiments

Models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen2.5-1.5B-Instruct`

Conversation families:

- casual chat
- multi-topic chat
- retrieval-heavy chat
- long-dependency reasoning chat
- code conversations

Measurements:

- consecutive geodesic angle
- curvature time series
- intrinsic rank of transported increment matrices
- rank-vs-energy curves
- geometric distance vs logit drift correlation
- geometric distance vs KL correlation
- reconstruction quality under low-rank segment approximation

## Success Criteria

- low within-segment geodesic distance
- visible curvature spikes at regime changes
- rank-95 small enough to be compressive
- geometric error meaningfully correlated with decoder drift

Current interpretation:

- the paper headline should be low-rank, decoder-relevant geometry
- regime-boundary recovery should remain a secondary analysis
- tolerance-aware localization and ranking metrics are more informative than exact F1 alone

## Value If Negative

Still publishable if the result is:

- geometry exists but is not sufficiently compressible
- or geometry is descriptive but not predictive of decoder behavior

## Current Code Anchor

The bootstrap implementation lives in:

- [`paper1_geometry/run_paper1.py`](/Users/pranav/Documents/RT/paper1_geometry/run_paper1.py)
- [`paper1_geometry/analysis.py`](/Users/pranav/Documents/RT/paper1_geometry/analysis.py)

Current benchmark anchors:

- audited baseline study: [`expanded_v5_audit`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v5_audit/study_report.md)
- changepoint follow-up: [`expanded_v6_changepoint`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v6_changepoint/study_report.md)
