# RT Project: Complete State of Play

> Generated: 2026-04-01  
> Based on: all `.md` files modified since 2026-03-30, derivation ledger, full program findings, gate runbooks

---

## What This Project Is, Precisely

A three-paper program asking: **can the geometric structure of an LLM's hidden states tell you which conversation turns to keep, compress, or drop?**

The unifying math starts here. For a conversation with turns `x_1 ... x_T`, extract the last-token hidden state at each turn:

```
z_t = f_θ(x_{1:t})   ∈ R^d
```

Normalize onto the unit sphere:

```
h_t = z_t / ‖z_t‖   ∈ S^{d-1}
```

Inside a local segment with reference point `q_j`, compute the **transported increment** — one-step motion in a shared tangent frame:

```
u_t = PT_{h_t → q_j}( log_{h_t}(h_{t+1}) )   ∈ T_{q_j} S^{d-1}
```

Where:
- `log_{h_t}(h_{t+1})` is the **Riemannian log map** — the tangent vector pointing from `h_t` to `h_{t+1}` on the sphere
- `PT_{h_t → q_j}(·)` is **parallel transport** — moves that tangent vector into a shared coordinate frame at `q_j`

This gives a matrix of increments `U = [u_1, u_2, ..., u_T]` whose **rank** measures how structured/compressible local conversation motion is.

---

## Paper 1: What Was Established — FROZEN ✓

**Question:** Are the increments `u_t` low-rank? Does geometric distortion predict decoder distortion?

**What was measured:**
- `rank95` = singular values needed to explain 95% of energy in `U`
- Pearson correlation between geometric reconstruction error and logit L2 drift

**Result (Qwen2.5-1.5B):**
- `rank95 = 1.167` — conversation motion is essentially **rank-1 locally**
- Geometry–logit correlation: **0.994**

**Why this is strong:** This is a purely descriptive result. You are not claiming geometry helps some task — you are claiming the model's internal dynamics have near-perfect low-rank structure, and that structure is directly coupled to how much the decoder drifts. It does not matter what benchmark you use. The geometry is just *there*, in the weights.

**Settled claims:**
- Conversation-state trajectories in compact LLMs are extremely low-rank after spherical normalization and tangent transport
- Manifold distortion is tightly coupled to decoder drift
- Null controls collapse the effect

**Status: Frozen. No contested claims. This is the bedrock.**

---

## Paper 2: What Was Established — STABLE ✓

**Question:** Can geometry-derived risk scores guide which turns to retain under a fixed token budget?

**The control problem framing:**
```
min E[ decoder_drift(π) ]   s.t.   E[ memory_cost(π) ] ≤ B
```

Where `π` is a retention policy (which turns to keep).

**Geometry risk score per turn:**
```
risk(t) = α · ‖h_t - ĥ_t‖  +  β · curvature(t)  +  γ · subspace_shift(t)
```

- `‖h_t - ĥ_t‖` — state reconstruction error from low-rank approximation
- `curvature(t) = ‖u_t − u_{t−1}‖ / Δs_t` — turning angle over arclength
- `subspace_shift` — how much the local basis changes across turns

**Result (Qwen2.5-0.5B, budget 0.35):**
- Geometry vs. uniform: `Δ logit L2 = -99.503, p = 0.0000`

**Mechanism result:**
- Geometry retained more support-critical user turns in **14/36** conversations
- Uniform retained more in only **5/36**
- Geometry kept the **latest** support user turn in **13/36** cases where uniform dropped it

**Why this worked:** The hard stress benchmark was designed with exact constraints (launch packets, medication schedules, SQL schemas). These are structurally sparse — one missed support turn destroys the answer. Geometry preferentially preserves turns where the model's hidden state moves a lot, which tracks these exact constraint insertions.

**Status: Stable, publishable, mechanism story is clear.**

---

## Paper 3: The Codec Idea — ACTIVE

### The Sparse Codec Object

Instead of just selecting which turns to retain, encode history into segments with three actions:

```
m_j = (anchor_turn, sparse_support_turns, action ∈ {keep, compress, evict})
```

- `keep` — retain all turns in segment
- `compress` — retain only anchor + sparse support (highest-risk) turns
- `evict` — drop segment entirely

### Hard-set results (Qwen2.5-1.5B, fairness-controlled)

| Budget | Policy | Δ logit L2 | p-value |
|--------|--------|-----------|---------|
| 0.24 | geometry_KCD | -38.272 | 0.0022 |
| 0.28 | geometry_KCD | -42.138 | 0.0008 |
| 0.32 | geometry_KCD | -56.165 | 0.0030 |
| 0.38 | geometry_KCD | -41.881 | 0.0213 |

### Behavior results (answer NLL):

| Budget | Δ answer NLL | p-value |
|--------|-------------|---------|
| 0.32 | -0.3903 | 0.0270 |
| 0.35 | -0.3737 | 0.0293 |
| 0.46 | -0.5272 | 0.0095 |
| 0.50 | -0.6376 | 0.0040 |

### 3B probe regime split (Qwen2.5-3B):

- At `0.35`: KCD strongest — `Δ logit = -70.997, p = 0.0070`
- At `0.50`: plain geometry retook the lead — `Δ logit = -46.526, p = 0.0333`

**Clean regime:** under scarcity, sparse codec wins. At looser budgets, plain geometry retention can retake the lead.

**Status: Real on the hard set. Benchmark-dependent everywhere else.**

---

## The Mega Gap

This is the central unresolved problem of the program. When you move off the hard stress benchmark, the geometry story breaks down.

### MSC (Multi-Session Chat, semantic continuity):

| Budget | Policy | Δ logit L2 |
|--------|--------|-----------|
| 0.35 | semantic | -112.257 |
| 0.35 | semantic_KCD | -98.320 |
| 0.35 | **geometry_KCD** | **+22.591** ← worse than no compression |
| 0.50 | **geometry_KCD** | **+26.961** ← deteriorating at looser budget |

**Geometry-KCD is actively harmful on MSC** — not just weaker, but degrading model output.

### Why:

MSC asks questions like: *"What does Alice do for work? What are her hobbies?"* The answers live in earlier persona statements scattered across a long dialogue. These statements are **semantically salient but geometrically flat** — inserting "I'm a nurse who loves hiking" does not create a large local curvature spike. It just goes in. The geometry has nothing to grab onto.

The geometry measures **how the model moves** through state space. On semantic-memory benchmarks, the answer does not live in the turns that caused the biggest movements — it lives in the turns that introduced the right *content*, which may have arrived quietly with minimal geometric signature.

### Benchmark dependency table:

| Benchmark | Memory type | Winner |
|-----------|-------------|--------|
| Hard stress set | Exact support constraints | Geometry-KCD |
| MSC | Persona / preference continuity | Plain semantic |
| LoCoMo | Mixed biographical + event chains | Semantic-led, mixed |
| LongMemEval 0.20 | Episodic retrieval | Semantic |
| LongMemEval 0.35 | Episodic retrieval | Semantic / segment-actions |
| LongMemEval 0.50 | Episodic retrieval | Geometry-KCD |

---

## The Three Falsifications

### 1. State-update supersession — mathematical failure

**Hypothesis:** If turn `t` supersedes earlier turn `s` on the same topic (old job → new job), transported increments should point in opposite directions:
```
A(s,t) = cos(u_s, u_t)  should be negative
```

**Observed:**
- Mean alignment: `0.9128` (strongly positive)
- Negative alignments: `0 / 10`
- Cross-term variant (`cos(log_q(h_s), u_t^entry)`): update `0.9707` vs. control `0.9780` — indistinguishable

**Why it broke:** At compact model scale, "user states a fact" and "user updates a fact" produce similarly oriented local motions. The model encodes both as "absorb new user information." No geometric reversal occurs. The representational resolution does not exist at small scale.

**Design consequence:** Retire direct same-sign supersession rules for compact-model work.

---

### 2. Geometry-only regime atlas — benchmark transfer failure

**Hypothesis:** Cluster segment-level geometry stats to auto-detect memory regime (fact-memory vs. support-turn-critical vs. event-chain), then pick compression policy automatically.

**Observed:** After fixing a curvature blow-up bug (raw curvature on near-stationary segments hit `2520` → stabilized `17`), the atlas still could not separate retrieval-heavy hard-set segments from MSC and LoCoMo. Segment statistics were too coarse.

**Curvature bug discovered along the way:**
- `msc-00000`: raw mean curvature `2520.867` → stabilized `17.503`
- `conv-26-qa000`: raw `3190.459` → stabilized `19.201`
- `e47becba`: raw `6314.673` → stabilized `26.130`

The fix: `stabilized_curvature = turning_angle / max(local_arclength, floor)`

**Design consequence:** Geometry cannot self-diagnose benchmark regime. Do not use geometry-only clustering as a standalone policy selector.

---

### 3. MSC persona curvature — failed data fit

**Hypothesis:** Inside MSC, earlier persona/support turns should have higher stabilized curvature than nearby filler turns.

**Observed (5 manually labeled conversations):**

| Metric | Value |
|--------|-------|
| Mean support – filler curvature delta | 0.1220 |
| Median delta | 0.0104 |
| Positive deltas | 3 / 5 |
| Negative deltas | 2 / 5 |
| Mean support percentile | 0.6139 |
| Mean filler percentile | 0.6087 |

**Completely indistinguishable.** MSC support/persona turns are semantically salient but not geometrically spiky. The benchmark rewards recovering the right persona/topic cluster, not rescuing a few structurally critical turns.

**Design consequence:** Do not use curvature as the hidden support detector inside semantic-memory benchmarks.

---

## Attempted Fixes and Why They Stalled

### Fix 1: Support-aware geometry KCD

Added bonus scores to turns that look like support/constraint anchors.

| Benchmark | Budget | Old geometry-KCD | Support-aware KCD | Δ |
|-----------|--------|-----------------|-------------------|---|
| MSC smoke | 0.20 | -44.152 | -505.510 | -461.357 |
| MSC smoke | 0.35 | — | — | -492.430 |
| LoCoMo smoke | 0.20 | — | — | -1174.194 |

A real engineering repair. But semantic baselines still dominated.

---

### Fix 2: Query-conditioned geometry

Project geometry into the query's tangent frame:
```
query_projected_curvature = proj_{q_embed}(u_t) · curvature_weight
```

| Benchmark | Budget | Result |
|-----------|--------|--------|
| MSC | 0.20 | query-conditioned KCD competitive (-501.114 vs plain geometry -77.737) |
| LoCoMo | 0.35 | **plain query-conditioned geometry = +881.417** — catastrophic collapse |

Improved geometry on MSC. Did not stabilize across benchmarks or budget levels. Acts as a refinement feature, not a primary selector.

---

### Fix 3: Semantic-KCD optimization

Added `support_aware_semantic_KCD` and `budget_aware_semantic_KCD`.

Result: viable and sometimes strong, but no variant cleanly displaced plain `semantic` on MSC or LoCoMo across budgets.

**The wall:** Heuristic score mixing appears to be near its limit. The remaining gap looks less like "find one more clever hand-tuned formula" and more like "learn a harm predictor directly from ablation damage."

---

## What Is Being Attempted Now: Gate 1 → 2 → 3

### Gate 1 — Oracle probe (just implemented, not yet run)

**Question:** Inside a semantic shortlist, do geometric features add *any* ranking value?

**Oracle criteria (must hit at least one):**
- Δ Kendall tau ≥ +0.03
- Δ top-5 oracle recall ≥ +5 percentage points

On: `MSC valid` and `LongMemEval-S cleaned`

**Policy set:**
- `semantic` (baseline)
- `budget_aware_semantic_KCD` (incumbent)
- `semantic_ambient_geometry_KCD` (does ambient geometry help inside the shortlist?)
- `semantic_query_conditioned_geometry_KCD` (does query-aware geometry help?)

**Current status:** Scripts written and committed. Benchmark JSONLs (`msc_valid_normalized.jsonl`, `longmemeval_s_cleaned_normalized.jsonl`) now present on disk as untracked files. **No canonical Gate 1 results recorded yet.**

---

### Gate 2 — Conditional on Gate 1 passing

Primary comparison:
```
semantic_query_conditioned_geometry_KCD
  vs.
budget_aware_semantic_KCD
```

Success: wins on at least one benchmark at budget 0.20 or 0.35, improves logit, does not regress answer NLL.

If it ties or loses across both benchmarks → stop hand-designed geometry refinement entirely.

---

### Gate 3 — If heuristics stay stuck

Replace heuristic risk with a **learned harm predictor** trained on oracle ablation damage (which turns, when dropped, hurt answer NLL most).

**New method:** `semantic_object_harm_KCD`

**Features exported for the predictor:**
- `object_size`
- `object_recency`
- `object_memory_score`
- `is_object_anchor`
- `is_object_freshest`
- One-hot object-type indicators: `persona`, `event`, `constraint`, `update`, `generic`

---

### Newest idea (committed 2026-04-01): Semantic Object Codec

The working hypothesis is that the selection unit is wrong — individual turns are too granular for semantic-memory benchmarks. The proposal is to work at **memory object** level:

**Object types:**
- persona bundle
- event bundle
- constraint bundle
- update / freshest-state bundle

**New methods:**
- `semantic_object_keep_compress_drop` — builds semantic objects from the shortlist, compresses at object level
- `semantic_object_harm_keep_compress_drop` — same decomposition, but uses learned predicted harm instead of heuristic object risk

**Success criterion:** Object-level compression must improve *answer behavior* (answer NLL), not just logit distortion. If it only improves logit and not answer NLL, the object decomposition is still not matching the benchmark's true memory object.

---

## Full Program Status Map

| Branch | Status | What is settled |
|--------|--------|----------------|
| Paper 1 geometry characterization | **frozen** | Low-rank trajectories, 0.994 decoder correlation |
| Paper 2 geometry-aware control | **stable** | Geometry beats uniform on hard stress set, mechanism is support-turn rescue |
| Paper 3 hard-set sparse codec | **stable** | geometry_KCD is real, wins under scarcity |
| Paper 3 3B hard-set probe | **stable** | Regime split confirmed at larger scale |
| Paper 3 public benchmark transfer | **stable** | Geometry transfers, winner is budget-dependent |
| Paper 3 semantic-memory checkpoint | **stable** | Semantic-led wins on MSC |
| Low-budget KCD repairs | **provisional** | Support-aware geometry repairs old KCD, doesn't close the gap |
| Semantic-KCD refinements | **provisional** | Viable variants, no stable new winner |
| Query-conditioned geometry | **provisional** | Refinement feature only, not a stable standalone |
| Gate 1 oracle + refinement | **just implemented** | No canonical results yet |
| Semantic object codec | **just designed** | Implementation committed, unevaluated |
| Geometry-only regime atlas | **failed** | Too coarse for benchmark taxonomy |
| MSC persona curvature discriminator | **failed** | No separation between support and filler |
| State-update supersession | **failed** | Both sign-based formulations fail in compact models |

---

## What Is Settled, What Is Not

### Settled
1. Compact LLM conversation trajectories have low-rank, decoder-relevant manifold structure
2. Geometry-aware retention improves constrained-budget memory control on support-turn-critical tasks
3. Sparse keep/compress/drop memory is a real codec family
4. The best memory signal depends on the task's memory type
5. Semantic baselines are well-matched to persona, preference, and topic-continuity benchmarks — they are not mysteriously good
6. Several attractive geometry-only extensions were tested directly and failed

### Not settled
- Whether geometry adds anything inside a semantic shortlist (Gate 1 is unanswered)
- Whether memory-object level compression bridges the semantic gap
- Whether a learned harm predictor can stably combine both signals
- Whether failed compact-model branches become viable at larger representational scale

### Retired
- Raw curvature without arclength stabilization for long-conversation work
- Geometry-only regime classification as a standalone policy selector
- Same-sign supersession detection by `cos(u_s, u_t)`
- Sign-based state/update cross-term detection in compact models
- Any claim that geometry alone beats semantic baselines on semantic-memory benchmarks

---

## The Bottom Line in One Paragraph

The full RT program did not show that geometry universally beats semantic retrieval for conversational memory. It showed something more precise and more defensible: representation geometry is real and decoder-relevant; geometry is the right signal for memory tasks that depend on structurally sparse, exact support turns; semantic signal is the right signal for persona, preference, and topic-continuity tasks; and the job of a complete memory system is to know which problem it is solving. The open frontier is whether geometry can add value *inside* a semantic shortlist, and whether working at the level of memory objects (bundles of related turns) rather than individual turns is the right unit for semantic-memory compression.
