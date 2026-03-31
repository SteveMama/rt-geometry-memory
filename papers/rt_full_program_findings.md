# RT Full Program Findings

This document is the single end-to-end synthesis of the RT program.

It is not a checkpoint paper and not a raw derivation ledger. It is the
program-level readout of:

- what we expected at each step
- what mathematical object or controller we used
- what data and benchmarks actually looked like
- what results we obtained
- why the math fit or failed
- what the accumulated evidence now supports

Canonical companions:

- deep derivation ledger:
  - [`rt_derivation_ledger.md`](rt_derivation_ledger.md)
- manuscript checkpoint:
  - [`../manuscript/paper_checkpoint.tex`](../manuscript/paper_checkpoint.tex)
- benchmark structure note:
  - [`benchmark_memory_type_analysis.md`](benchmark_memory_type_analysis.md)
- phase log:
  - [`../CHECKPOINT_LOG.md`](../CHECKPOINT_LOG.md)

## 1. Executive Summary

The RT program produced three strong scientific outcomes and several valuable
falsifications.

### What held up

1. **Paper 1 is real and strong.**
   Conversation-state trajectories in compact LLMs are extremely low-rank after
   spherical normalization and tangent transport, and manifold distortion is
   tightly coupled to decoder drift.

2. **Paper 2 is real and useful.**
   Geometry-guided retention beats uniform retention on the hard stress set and
   does so by protecting memory-critical earlier user turns, especially support
   constraints and exact retrieval/code requirements.

3. **Paper 3 is real, but benchmark-dependent.**
   Sparse keep/compress/drop codecs are legitimate. They help most on
   support-turn-critical tasks under scarcity. On broader semantic-memory
   benchmarks, semantic-led policies are usually stronger than geometry-led
   ones.

### What failed

1. Geometry did **not** cleanly separate support/persona turns from filler
   inside `MSC`.
2. A geometry-only regime atlas did **not** recover a usable benchmark
   taxonomy without semantics, though it did expose an important curvature bug.
3. State-update supersession detection via simple geometric sign tests failed on
   synthetic data under compact models.

### Current overall conclusion

The right high-level reading is now:

- **geometry is a real representation-level control signal**
- **semantics is the stronger primary signal on semantic-memory benchmarks**
- **the winning memory controller depends on the memory type**

That is not a collapse of the geometry program. It is the actual boundary of
what the evidence supports.

## 2. Core Mathematical Program

The mathematical spine of the project stayed consistent even as the downstream
claims changed.

For a conversation `x_{1:T}`, define a turn-level hidden state summary:

`z_t = f_theta(x_{1:t}) in R^d`

Normalize onto the sphere:

`h_t = z_t / ||z_t|| in S^{d-1}`

Inside a segment with reference point `q_j`, define the transported increment:

`u_t = PT_{h_t -> q_j}(log_{h_t}(h_{t+1})) in T_{q_j} S^{d-1}`

The program asked three main questions:

1. **Paper 1**
   - Are the `u_t` low-rank and decoder-relevant?
2. **Paper 2**
   - Can geometry-derived risk guide budgeted retention better than flat
     baselines?
3. **Paper 3**
   - Can sparse keep/compress/drop memory objects use geometry to outperform
     simpler retention policies under budget?

The central decoder-facing distortion metrics were:

- logit `L2`
- `KL`
- top-1 agreement
- assistant answer average negative log-probability

The control framing stayed the same throughout:

`min E[decoder_drift]` subject to a memory budget.

What changed over time was not the overall objective. What changed was the
belief about which signal is best for ranking history on different benchmarks.

## 3. What The Data Actually Tested

One of the most important lessons in the project was that the benchmarks were
not all testing the same memory object.

### Hard stress set

Source:

- [`../paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl`](../paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl)

Structure:

- short, dense, adversarial dialogues
- exact launch packets
- itineraries
- medication schedules
- SQL/code/formatting constraints

What memory it tests:

- exact support constraints
- exact retrieval packets
- exact formatting/schema requirements

Where the answer lives:

- a few earlier critical user turns

Predicted winner:

- geometry or support-aware structural memory

Observed winner:

- geometry-led controllers and sparse support-aware codecs

This benchmark matched the original geometry story.

### MSC

Benchmark reading:

- [`benchmark_memory_type_analysis.md`](benchmark_memory_type_analysis.md)
- [`paper3_msc_semantic_codec_checkpoint.md`](paper3_msc_semantic_codec_checkpoint.md)

Structure:

- multi-session dialogue
- persona continuity
- preference continuity
- topic continuation
- light emotional / plan tracking

What memory it tests:

- semantic conversational continuity
- not exact support-turn rescue

Where the answer lives:

- one or two earlier persona or preference mentions
- often in a broader topical cluster, not a uniquely critical support turn

Predicted winner:

- semantic retrieval

Observed winner:

- semantic retrieval and semantic-led codec variants

MSC is the clearest evidence that semantic memory and structural support memory
are not the same task.

### LoCoMo

Benchmark reading:

- [`benchmark_memory_type_analysis.md`](benchmark_memory_type_analysis.md)
- [`paper3_query_conditioned_geometry_smoke_checkpoint.md`](paper3_query_conditioned_geometry_smoke_checkpoint.md)
- [`paper3_semantic_kcd_optimization_checkpoint.md`](paper3_semantic_kcd_optimization_checkpoint.md)

Structure:

- very long conversations
- explicit questions at the end
- biography, relationships, plans, temporal facts, event chains

What memory it tests:

- mixed semantic-biographical memory
- mixed temporal-event memory

Predicted winner:

- mixed: semantic should do well overall, structure may help on event chains

Observed winner:

- semantic-led methods overall
- geometry/query-conditioned variants sometimes competitive under tight budgets

LoCoMo is mixed, but still more semantic than the hard stress set.

### LongMemEval public

Checkpoint:

- [`paper3_public_benchmark_checkpoint.md`](paper3_public_benchmark_checkpoint.md)

Structure:

- longer public benchmark slice
- many sampled target turns
- broader episode-style retrieval

What memory it tests:

- distributed earlier relevance
- semantic or episodic retrieval
- some long-range structure

Observed split:

- `0.20`: semantic strongest
- `0.35`: semantic and segment-actions strongest
- `0.50`: `geometry_keep_compress_drop` strongest

This benchmark was the first hard evidence that policy ranking depends on the
kind of memory the benchmark emphasizes.

## 4. Paper 1: Geometry Characterization

Reference:

- [`paper1_geometry_of_conversation_state_trajectories.md`](paper1_geometry_of_conversation_state_trajectories.md)
- canonical artifact:
  - [`../artifacts/paper1/expanded_v8_final`](../artifacts/paper1/expanded_v8_final)

### Expectation

We expected that normalized conversation states would trace a structured
trajectory rather than behaving as arbitrary hidden-state noise.

More concretely:

- within local segments, transported increments should be low-rank
- manifold distortion should predict decoder distortion
- curvature should mark regime changes approximately

### Method

Used:

- spherical normalization of turn summaries
- log map + parallel transport into shared tangent spaces
- singular-value analysis of transported increment matrices
- correlation between geometric reconstruction error and decoder drift

### Findings

This phase was the strongest clean win in the whole project.

Canonical reading from the ledger:

- `qwen25_15b`: `rank95 = 1.167`
- geometry-logit correlation: `0.994`

What that means:

- local conversation motion is extremely low-rank
- the low-rank approximation is not just descriptive
- manifold distortion is almost perfectly aligned with decoder drift

### Why the math fit

Paper 1 succeeded because it asked a descriptive question that the geometry is
well-suited to answer:

- what does the hidden-state trajectory look like?
- how compressible is it locally?
- is that geometry decoder-relevant?

It did **not** require geometry to solve a benchmark-specific memory retrieval
problem. It only required geometry to faithfully describe the model’s internal
dynamics.

### Resulting claim

Paper 1 is the frozen foundation:

- the geometry exists
- it is low-rank
- it is decoder-coupled

## 5. Paper 2: Geometry-Guided Adaptive Memory Control

Reference:

- [`paper2_geometry_guided_adaptive_memory_compression.md`](paper2_geometry_guided_adaptive_memory_compression.md)
- canonical artifacts:
  - [`../artifacts/paper2/behavior_stress_v1`](../artifacts/paper2/behavior_stress_v1)
  - [`../artifacts/paper2/behavior_stress_qwen_cases`](../artifacts/paper2/behavior_stress_qwen_cases)

### Expectation

If geometry predicts decoder sensitivity, then it should be possible to use a
geometry-derived risk score to retain dangerous earlier turns more often than
uniform retention at the same budget.

### Method

Built:

- geometry risk scores from curvature, state error, and local subspace shift
- budgeted retention controllers
- comparisons against uniform and lexical alternatives
- answer-level and logit-level evaluation on the custom hard stress set

### Findings

This phase produced the first true systems result.

The strongest mechanism finding was not just aggregate logit improvement. It
was structural:

- on focused `qwen25_05b` hard-set analysis at budget `0.35`, geometry retained
  more support user turns than uniform in `14/36` cases, while uniform did so
  in `5/36`
- geometry kept the latest support user turn while uniform dropped it in
  `13/36` cases

The explanation layer mattered:

- geometry was rescuing support constraints
- base-memory turns
- exact code / retrieval requirements

### Why the math fit

Paper 2 succeeded because the hard stress set was explicitly built around
memory objects that are structurally sparse and support-turn-critical.

In those tasks:

- one missed earlier constraint can destroy the answer
- exact support turns matter more than broad topical similarity

That is exactly the environment where a geometry-derived retention controller
has leverage.

### Limits already visible in Paper 2

Even at this stage, answer-level gains were mixed rather than universally
dominant.

That was the first sign that:

- geometry was a **real control signal**
- but not necessarily the only or always-best signal

## 6. Paper 3 Early Hard-Set Codec Results

Reference:

- [`paper3_manifold_memory_codecs.md`](paper3_manifold_memory_codecs.md)
- artifacts:
  - [`../artifacts/paper3/paper3_pilot_v3_full`](../artifacts/paper3/paper3_pilot_v3_full)
  - [`../artifacts/paper3/paper3_batch_v1_fairness`](../artifacts/paper3/paper3_batch_v1_fairness)
  - [`../artifacts/paper3/paper3_batch_v1_3b`](../artifacts/paper3/paper3_batch_v1_3b)

### Expectation

The codec idea was:

- divide history into segments
- choose `keep`, `compress`, or `evict`
- represent compressed segments by sparse support memory objects

We expected this to outperform simple retention under scarcity, especially on
the same hard-set tasks where geometry control already helped.

### Method

The Paper 3 memory object was:

`m_j = (anchor, sparse_support, metadata)`

Operationally:

- `keep` = retain all turns in the segment
- `compress` = retain only sparse support turns
- `evict` = drop the segment

### Findings

On the hard set, this was a real success.

Fairness-controlled `qwen25_15b` results:

- `0.24`: `geometry_keep_compress_drop`, `Δ logit = -38.272`, `p = 0.0022`
- `0.28`: `-42.138`, `p = 0.0008`
- `0.32`: `-56.165`, `p = 0.0030`
- `0.38`: `-41.881`, `p = 0.0213`

Behavior also supported the codec family:

- `0.32`: `Δ answer NLL = -0.3903`, `p = 0.0270`
- `0.35`: `-0.3737`, `p = 0.0293`
- `0.46`: `-0.5272`, `p = 0.0095`
- `0.50`: `-0.6376`, `p = 0.0040`

3B probe:

- at `0.35`, KCD strongest:
  - `Δ logit = -70.997`
  - `R_logit = 0.854`
  - `p = 0.0070`
- at `0.50`, plain geometry retook the lead:
  - `Δ logit = -46.526`
  - `R_logit = 0.891`
  - `p = 0.0333`

This gave the clean regime split:

- **under scarcity**: sparse codec wins
- **once budget loosens**: plain geometry retention can retake the lead

### Why the math fit

Again, the hard set was the key.

The sparse codec worked because:

- compressed memory objects still preserved the right support turns
- those support turns were the actual answer-bearing objects
- the budget was tight enough that exact retention of all earlier turns was not
  possible

Mechanism evidence supported the codec story directly:

- at budget `0.35`, the codec retained more support user turns than uniform in
  `17/36` `qwen25_15b` cases and `14/36` `qwen25_3b` cases
- `27/29` compressed `qwen25_15b` cases and `23/29` compressed `qwen25_3b`
  cases were not worse than uniform on support retention

This is the strongest positive evidence for geometry-aware sparse memory in the
whole program.

## 7. Public Benchmark Transfer

Reference:

- [`paper3_public_benchmark_checkpoint.md`](paper3_public_benchmark_checkpoint.md)

### Expectation

The early hope was that the hard-set KCD story might generalize directly to
public benchmarks.

### Method

Ran `uniform`, `semantic`, `geometry`, `geometry_segment_actions`, and
`geometry_keep_compress_drop` on a normalized `LongMemEval-S cleaned` slice.

### Findings

This was the first real benchmark-dependent split.

At `0.20`:

- `semantic`: `Δ logit L2 = -122.613`, `p = 0.0000`
- `geometry_keep_compress_drop`: `-26.258`, `p = 0.0307`

At `0.35`:

- `semantic`: `-104.183`, `p = 0.0000`
- `geometry_segment_actions`: `-97.432`, `p = 0.0000`
- `geometry_keep_compress_drop`: `-42.612`, `p = 0.0003`

At `0.50`:

- `geometry_keep_compress_drop`: `-48.466`, `p = 0.0000`
- `semantic`: `+12.164`, `p = 0.3350`

So:

- low budget favored semantic
- mid budget favored semantic and segment-actions
- high budget favored KCD

### Why the original expectation failed

The hard-set codec success did transfer in one sense:

- geometry-family policies were real outside the custom benchmark

But it did not transfer in the stronger sense:

- KCD was **not** the universal low-budget winner

The likely reason is benchmark structure:

- LongMemEval rewards broader episode relevance and semantic retrieval more than
  exact support-turn rescue under tight budgets

This was the point where Paper 3 stopped being a universal codec paper and
became a benchmark-dependent memory-regime paper.

## 8. MSC Semantic Checkpoint

Reference:

- [`paper3_msc_semantic_codec_checkpoint.md`](paper3_msc_semantic_codec_checkpoint.md)

### Expectation

After LongMemEval, the next question became:

> if the problem is not sparse codec structure itself, but the signal, does a
> semantic-led codec beat geometry-led codecs and perhaps beat plain semantic?

### Method

Compared:

- `uniform`
- `semantic`
- `geometry`
- `geometry_keep_compress_drop`
- `semantic_keep_compress_drop`

on `MSC valid`.

### Findings

MSC was the cleanest semantic-memory result in the project.

At `0.20`:

- `semantic_keep_compress_drop`: `Δ logit = -119.966`
- `semantic`: `-105.782`
- pairwise vs semantic:
  - `Δ = -14.184`
  - `p = 0.2562`

At `0.35`:

- `semantic`: `-112.257`
- `semantic_keep_compress_drop`: `-98.320`
- behavior pairwise:
  - `Δ answer NLL = +0.0394`
  - `p = 0.0000`

At `0.50`:

- `semantic`: `-108.342`
- `semantic_keep_compress_drop`: `-98.447`
- behavior pairwise:
  - `Δ answer NLL = +0.0524`
  - `p = 0.0000`

Meanwhile geometry-KCD was badly mismatched:

- at `0.35`, `geometry_keep_compress_drop`: `+22.591`
- at `0.50`, `+26.961`

### Why the math failed here

MSC is primarily a semantic continuity benchmark:

- preferences
- persona
- emotional stance
- topic continuation

That is not the kind of memory object that curvature- or support-driven sparse
selection naturally excels at.

The benchmark did not reward exact sparse rescue of a few critical support
turns. It rewarded recovering the right persona/topic cluster.

### Resulting claim

MSC settled two things:

1. geometry-KCD is not a general-purpose conversational-memory codec
2. sparse semantic codec is viable, but plain semantic retention remains the
   stronger baseline overall on MSC

## 9. Low-Budget KCD Repairs

Reference:

- [`paper3_low_budget_kcd_smoke_checkpoint.md`](paper3_low_budget_kcd_smoke_checkpoint.md)

### Expectation

Maybe geometry-KCD was failing on semantic-memory benchmarks because it was too
coarse and not support-aware enough.

### Method

Added:

- `support_aware_geometry_keep_compress_drop`
- `semantic_filtered_geometry_keep_compress_drop`

### Findings

These were real engineering repairs, not cosmetic changes.

MSC smoke:

- `support_aware_geometry_keep_compress_drop` vs old `geometry_keep_compress_drop`
  - `0.20`: `Δ = -461.357`, `p = 0.0285`
  - `0.35`: `Δ = -492.430`, `p = 0.0210`

LoCoMo smoke:

- `support_aware_geometry_keep_compress_drop` vs old `geometry_keep_compress_drop`
  - `0.20`: `Δ = -1174.194`, `p = 0.0027`

### Why this helped but did not solve the benchmark gap

The support-aware fix improved the **codec object** and the **local selection
mechanism**.

It did **not** change the more fundamental issue:

- on semantic-memory benchmarks, semantic relevance is often the right
  front-door signal

So this phase proved:

- old geometry-KCD was genuinely flawed
- it could be repaired materially
- but even repaired geometry-KCD did not erase the semantic-memory gap

## 10. Query-Conditioned Geometry

Reference:

- [`paper3_query_conditioned_geometry_smoke_checkpoint.md`](paper3_query_conditioned_geometry_smoke_checkpoint.md)

### Expectation

One plausible reason geometry was losing on semantic-memory benchmarks was that
it was query-agnostic. The idea was:

- keep geometry
- but make it query-aware in the same tangent-space frame as local motion

### Method

Built query-conditioned features:

- query-projected curvature
- query-projected local subspace energy

and deployed them as:

- `query_conditioned_geometry`
- `query_conditioned_geometry_keep_compress_drop`

### Findings

MSC:

- query-conditioning clearly improved geometry over ambient geometry
- query-conditioned KCD became competitive with strong sparse codecs

LoCoMo:

- at `0.20`, query-conditioned KCD was fully competitive
- at `0.35`, plain query-conditioned retention collapsed badly

### Why this only partially worked

The new math fixed a real issue:

- the query signal was finally represented in the correct local coordinates

But the downstream benchmark problem remained:

- query-conditioned geometry was still not as stable as semantic relevance
  itself on semantic-memory tasks

So the correct reading became:

- query-conditioned geometry is a **useful refinement signal**
- it is not yet a stable standalone primary selector

## 11. Semantic-KCD Optimization

Reference:

- [`paper3_semantic_kcd_optimization_checkpoint.md`](paper3_semantic_kcd_optimization_checkpoint.md)

### Expectation

If semantic is the right front door, maybe better semantic-KCD engineering
could consistently beat plain semantic.

### Method

Added:

- `support_aware_semantic_keep_compress_drop`
- `budget_aware_semantic_keep_compress_drop`

### Findings

These variants were viable and sometimes strong, but they did not create a
stable new winner.

MSC smoke:

- new variants were competitive
- neither cleanly displaced `semantic_keep_compress_drop`

LoCoMo smoke:

- `budget_aware_semantic_keep_compress_drop` was strongest at `0.20`
- but it did not hold the lead at `0.35`

### Why this did not finish the problem

This was the point where the project learned that heuristic score mixing was
probably near its limit.

The remaining gap looked less like:

- “find one more clever hand-tuned formula”

and more like:

- “learn a harm predictor directly from ablation damage”

## 12. Geometry-Only Regime Atlas

Reference:

- [`geometric_regime_atlas_smoke_checkpoint.md`](geometric_regime_atlas_smoke_checkpoint.md)

### Expectation

A tempting idea was that geometry might self-diagnose benchmark regime:

- fact-memory
- support-turn-critical
- transition-heavy
- event-chain

If true, one could choose a compression policy from geometry alone.

### Method

Built a segment-level atlas over:

- MSC
- LoCoMo
- LongMemEval
- hard stress set

using features such as:

- curvature stats
- turning angle
- step norm
- local rank
- subspace shift
- role-switch rate

### Findings

This branch produced one important numerical discovery and one important
negative result.

#### Important discovery

The raw curvature feature was broken on long near-stationary conversations.

Examples:

- `msc-00000`: raw mean curvature `2520.867` -> stabilized `17.503`
- `conv-26-qa000`: raw `3190.459` -> stabilized `19.201`
- `e47becba`: raw `6314.673` -> stabilized `26.130`

#### Negative result

After fixing curvature with an arclength floor, the atlas still did **not**
cleanly separate the spike-heavy families into a useful semantic taxonomy.

It could distinguish:

- near-stationary fact memory

from

- broad spike-heavy transition regimes

But it could not recover the clean benchmark or task split that would justify a
geometry-only adaptive controller.

### Why the math failed

The segment statistics were too coarse relative to the semantic distinctions
the benchmarks actually cared about.

Geometry could describe:

- how much motion was happening
- whether motion was near-stationary or transition-heavy

It could not, from those statistics alone, infer:

- whether a segment contained a persona fact, a support constraint, or a casual
  topic continuation

This killed the strongest version of the geometry-only “self-diagnosing
compression policy” story.

## 13. MSC Persona Curvature Falsification

Reference:

- [`msc_persona_curvature_check.md`](msc_persona_curvature_check.md)

### Expectation

One last hope for geometry on MSC was:

- maybe inside a semantic topic cluster, the important persona/support turns
  still have higher curvature than filler

### Method

Manually labeled `5` MSC conversations with:

- support/persona turns
- nearby filler turns

and compared stabilized curvature.

### Findings

The result was negative.

Across the five conversations:

- mean support minus filler curvature delta: `0.1220`
- median delta: `0.0104`
- positive deltas: `3 / 5`
- negative deltas: `2 / 5`
- mean support percentile: `0.6139`
- mean filler percentile: `0.6087`

This is not a robust separator.

### Why the math failed

MSC support/persona turns are usually semantically salient, but not necessarily
geometrically spiky.

The benchmark often asks for:

- who the person is
- what they like
- what plan they mentioned

Those facts can be inserted into the dialogue without creating a strong local
geometric signature distinguishable from nearby filler.

This ruled out a strong claim that curvature is the missing within-topic signal
on MSC.

## 14. State-Update Supersession Falsification

Reference:

- [`state_update_alignment_smoke_checkpoint.md`](state_update_alignment_smoke_checkpoint.md)

### Expectation

This was the most ambitious late-stage theorem idea:

- semantics is symmetric
- state updates are asymmetric
- therefore geometry might detect when a later turn supersedes an earlier
  same-topic turn

### Method

Built a synthetic benchmark with explicit updates such as:

- old job -> new job
- old plan -> new plan

Tested:

1. same-sign directional alignment:
   - `A(s,t) = cos(u_s, u_t)`
2. state-position / update-entry cross-term

### Findings

Both clean sign-based formulations failed.

Same-sign alignment:

- mean directional alignment: `0.9128`
- negative alignments: `0 / 10`
- alignments below `-0.2`: `0 / 10`

Cross-term:

- mean update cross: `0.9707`
- mean all-control cross: `0.9780`
- negative update crosses: `0 / 10`

There was only a weak ranking effect, not a usable sign change.

### Why the math failed

The most plausible interpretation is representational resolution.

At compact model scale:

- “user states a fact”
- and
- “user updates a fact”

produce similarly oriented local motions

The model appears to encode both as:

- “absorb new user information”

rather than creating a crisp geometric reversal signal.

This branch is currently dead as an algorithmic claim.

## 15. What The Program Actually Established

The current evidence supports the following program-level findings.

### Finding 1: Geometry is real inside the model

This is the strongest and cleanest result.

- conversation trajectories are low-rank
- manifold distortion predicts decoder distortion

This is a genuine representation-level result.

### Finding 2: Geometry is useful for structural memory control

On support-turn-critical tasks:

- geometry-aware retention beats uniform
- sparse geometry-aware codecs work under scarcity
- the mechanism is real and interpretable

### Finding 3: Semantic signal dominates semantic-memory benchmarks

On benchmarks such as MSC and much of LoCoMo:

- semantic retrieval is usually the right front door
- geometry-only or geometry-first approaches do not dominate

### Finding 4: The benchmark split is principled, not noise

Different benchmarks test different memory objects:

- semantic continuity
- episodic relevance
- event-chain structure
- exact support constraints

The signal winner changes with the memory object.

### Finding 5: Several plausible geometry extensions do not work

The project now has strong negative evidence against:

- geometry-only benchmark regime classification
- raw curvature as MSC support detector
- same-sign state-update alignment as a clean supersession detector

These failures are part of the result, not an embarrassment.

## 16. Current Best System Reading

The best current RT reading is:

- **Paper 1**:
  geometry characterization is frozen and publishable
- **Paper 2**:
  geometry-guided adaptive memory control is stable and publishable on the hard
  stress set
- **Paper 3**:
  sparse codecs are real, but the winning signal is benchmark-dependent

The right system story is now:

- semantic front door for semantic-memory benchmarks
- geometry/support-aware control for structural support-turn memory
- learned harm prediction as the likely next serious step if geometry is to add
  value inside a semantic shortlist

## 17. Current Best-Backed Claims

These are the strongest claims the evidence currently supports.

1. Compact LLM conversation trajectories have low-rank, decoder-relevant
   manifold structure.
2. Geometry-aware retention improves constrained-budget memory control on
   support-turn-critical tasks.
3. Sparse keep/compress/drop memory is a real codec family.
4. The best memory signal depends on the task’s memory type.
5. Semantic baselines are not “mysteriously good”; they are well matched to
   persona, preference, and topic-continuity benchmarks.
6. Several attractive geometry-only extensions were tested directly and failed.

## 18. Final Bottom Line

The full RT program did **not** show that geometry universally beats semantic
retrieval for conversational memory.

It did show something more precise and more defensible:

- representation geometry is real and decoder-relevant
- geometry is the right signal for some memory regimes
- semantic signal is the right signal for others
- the job of the system is to know which problem it is solving

That is the real outcome of the project so far.
