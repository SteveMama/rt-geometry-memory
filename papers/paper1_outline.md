# Paper 1 Outline

## Working Title

**Low-Rank and Decoder-Relevant Geometry in Conversation-State Trajectories**

## Claim Hierarchy

1. Conversation-state trajectories are extremely low-rank.
2. Geometric distortion strongly predicts decoder drift.
3. Geometry contributes modest marginal value for approximate regime localization, but not a robust exact-boundary segmentation win.

## Section Outline

### 1. Introduction

- Motivate long-conversation memory as a structural, not purely token-budget, problem.
- Pose the main question: do turn-level conversation states evolve along a compact geometric trajectory?
- State the core result hierarchy:
  - low-rank structure
  - decoder relevance
  - partial boundary signal
- Close with why this matters for geometry-guided compression and control.

### 2. Setup

- Define turn-level hidden-state summaries.
- Normalize onto the sphere.
- Define transported increments in a common tangent space.
- Define segment-level low-rank approximation.
- Define decoder-aware distortion metrics: logit `L2`, KL, top-1 agreement.

### 3. Methods

#### 3.1 State Geometry Pipeline

- conversation to hidden states
- hidden states to normalized trajectory
- transported increment computation
- rank and reconstruction analysis

#### 3.2 Boundary Analysis

- turning-angle and subspace-shift features
- changepoint-style decoding
- geometry-only, lexical-only, geometry-plus-lexical ablations

#### 3.3 Controls and Statistical Protocol

- H1 null control: shuffled turn order
- H3 null control: turnwise permutation for geometry/logit alignment
- bootstrap confidence intervals
- paired significance tests for H2 ablations

### 4. Experimental Setup

- models
- conversation families
- boundary annotations
- study outputs and evaluation metrics

### 5. Results

#### 5.1 H1: Conversation-State Geometry Is Extremely Low-Rank

- report rank95 by model and family
- compare against shuffled-order null control
- show rank-energy curves

#### 5.2 H3: Geometric Distortion Strongly Predicts Decoder Drift

- report correlation between geodesic error and logit drift
- compare against permutation null control
- show scatter/regression plots

#### 5.3 H2: Geometry Carries Approximate Localization Signal

- report exact, tol-2, AUPRC, and nearest-boundary distance
- compare geometry-only, lexical-only, and geometry-plus-lexical
- state clearly that the result is approximate localization, not exact segmentation

### 6. Discussion

- summarize the stable scientific story:
  - H1 strong
  - H3 strongest
  - H2 secondary and modest
- explain why approximate localization is a reasonable target for conversations
- explain why Paper 2 should use continuous geometry-derived risk, not hard segmentation

### 7. Limitations

- small labeled boundary dataset
- short-conversation benchmark regime
- lexical baselines remain strong for score ranking
- current rank-jump feature is weak; subspace-shift is the more credible geometry-side boundary feature

### 8. Conclusion

- geometry is compact
- geometry is decoder-relevant
- geometry contains partial boundary information
- these results motivate geometry-guided compression/control

## Figures

1. Pipeline figure: conversation to states to transported increments to rank/drift/boundary outputs
2. Rank95 / rank-energy figure
3. Geodesic error vs logit drift figure
4. H2 ablation figure: geometry-only vs lexical-only vs geometry-plus-lexical
5. Qualitative boundary trace figure on a few conversations

## Tables

1. dataset and model summary
2. H1/H3 main results with uncertainty and null controls
3. H2 ablation summary with significance tests
4. benchmark audit summary

## Writing Order

1. Results
2. Methods
3. Discussion
4. Introduction
5. Abstract
