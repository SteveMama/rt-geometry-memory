# RT Derivation Ledger

This document is the canonical deep-dive companion to the checkpoint manuscript:

- manuscript: [`../manuscript/paper_checkpoint.tex`](../manuscript/paper_checkpoint.tex)
- experiment inventory: [`generated/rt_experiment_matrix.csv`](generated/rt_experiment_matrix.csv)
- negative-result inventory: [`generated/rt_negative_result_matrix.csv`](generated/rt_negative_result_matrix.csv)

It consolidates the full RT program into one place:

- the mathematical objects we defined
- how those objects were implemented
- what data structures and benchmarks we actually used
- which experiments succeeded and which failed
- why the math fit some tasks and broke on others

## 1. Program Status Map

| Branch | Status | What is settled | Canonical evidence |
| --- | --- | --- | --- |
| Paper 1 geometry characterization | `frozen` | Conversation-state trajectories are extremely low-rank and strongly decoder-coupled. | [`../artifacts/paper1/expanded_v8_final`](../artifacts/paper1/expanded_v8_final), [`../manuscript/paper_checkpoint.tex`](../manuscript/paper_checkpoint.tex) |
| Paper 2 geometry-aware control | `stable` | Geometry-aware retention beats uniform on the hard stress set and rescues support turns more often. | [`../artifacts/paper2/behavior_stress_v1`](../artifacts/paper2/behavior_stress_v1), [`../artifacts/paper2/behavior_stress_qwen_cases`](../artifacts/paper2/behavior_stress_qwen_cases) |
| Paper 3 hard-set sparse codec | `stable` | `geometry_keep_compress_drop` is a real codec and is strongest under scarcity on the hard set. | [`../artifacts/paper3/paper3_pilot_v3_full`](../artifacts/paper3/paper3_pilot_v3_full), [`../artifacts/paper3/paper3_batch_v1_fairness`](../artifacts/paper3/paper3_batch_v1_fairness) |
| Paper 3 3B hard-set probe | `stable` | KCD wins at `0.35`; plain geometry retakes the lead at `0.50`. | [`../artifacts/paper3/paper3_batch_v1_3b`](../artifacts/paper3/paper3_batch_v1_3b) |
| Paper 3 public benchmark transfer | `stable` | Geometry-family policies transfer, but the winner depends on the benchmark and budget. | [`paper3_public_benchmark_checkpoint.md`](paper3_public_benchmark_checkpoint.md), [`../paper3/paper3_public_v1_public_benchmark`](../paper3/paper3_public_v1_public_benchmark) |
| Paper 3 semantic-memory checkpoint | `stable` | Semantic-led methods win on MSC-style conversational memory. | [`paper3_msc_semantic_codec_checkpoint.md`](paper3_msc_semantic_codec_checkpoint.md) |
| Low-budget KCD repairs | `provisional` | Support-aware geometry repairs materially improve old geometry-KCD, but do not remove the semantic-memory gap. | [`paper3_low_budget_kcd_smoke_checkpoint.md`](paper3_low_budget_kcd_smoke_checkpoint.md), [`../results/paper3/studies/paper3_low_budget_smoke_msc`](../results/paper3/studies/paper3_low_budget_smoke_msc) |
| Semantic-KCD refinements | `provisional` | Semantic-led codec variants are viable, but do not yet dominate the strongest semantic baselines. | [`paper3_semantic_kcd_optimization_checkpoint.md`](paper3_semantic_kcd_optimization_checkpoint.md), [`../results/paper3/studies/paper3_semantic_kcd_opt_smoke_msc`](../results/paper3/studies/paper3_semantic_kcd_opt_smoke_msc) |
| Query-conditioned geometry | `provisional` | Query conditioning improves geometry, but does not produce a stable standalone winner. | [`paper3_query_conditioned_geometry_smoke_checkpoint.md`](paper3_query_conditioned_geometry_smoke_checkpoint.md), [`../results/paper3/studies/paper3_query_geom_smoke_msc`](../results/paper3/studies/paper3_query_geom_smoke_msc) |
| Geometry-only regime atlas | `failed` | Geometry-only clustering is too coarse to recover a reliable benchmark taxonomy. | [`geometric_regime_atlas_smoke_checkpoint.md`](geometric_regime_atlas_smoke_checkpoint.md), [`../artifacts/paper1/regime_atlas_smoke_v4`](../artifacts/paper1/regime_atlas_smoke_v4) |
| MSC support/persona curvature discriminator | `failed` | Stabilized curvature does not cleanly separate earlier support/persona turns from filler in MSC. | [`msc_persona_curvature_check.md`](msc_persona_curvature_check.md), [`../artifacts/paper3/msc_persona_curvature_v1`](../artifacts/paper3/msc_persona_curvature_v1) |
| State-update supersession by geometry | `failed` | Both same-sign alignment and state/increment cross-term formulations fail as clean detectors in compact models. | [`state_update_alignment_smoke_checkpoint.md`](state_update_alignment_smoke_checkpoint.md), [`../artifacts/paper3/state_update_alignment_smoke_qwen05b`](../artifacts/paper3/state_update_alignment_smoke_qwen05b), [`../artifacts/paper3/state_update_cross_control_qwen05b`](../artifacts/paper3/state_update_cross_control_qwen05b) |

## 2. Mathematical Object Ledger

| Object | Definition | Intended interpretation | Implementation | Experimental use | Empirical status |
| --- | --- | --- | --- | --- | --- |
| Turn state `z_t` | `z_t = f_\theta(x_{1:t})` | Turn-level hidden-state summary before normalization. | [`../paper1_geometry/modeling.py`](../paper1_geometry/modeling.py): `ConversationStateExtractor.score_messages`, `extract_conversation` | All three papers | Held up. This is the primary latent state used everywhere. |
| Normalized state `h_t` | `h_t = z_t / ||z_t||` | Conversation state on the sphere `S^{d-1}`. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `normalize_rows` | Papers 1–3 | Held up. Normalization is foundational and stable. |
| Geodesic distance | `d_geo(h_s,h_t)=arccos(<h_s,h_t>)` | Natural distance on the sphere between conversation states. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `sphere_distance` | Paper 1 error metrics, Paper 2 risk components | Held up. Strongly coupled to decoder drift in Paper 1. |
| Log map | `log_x(y)` | Tangent-space representation of local motion from `x` to `y`. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `sphere_log_map` | Papers 1–3 | Held up. Required for all tangent-space constructions. |
| Parallel transport | `PT_{x->y}(v)` | Moves tangent vectors into a shared tangent frame. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `sphere_parallel_transport` | Papers 1–3 | Held up mathematically. Antipodal instability is guarded numerically. |
| Transported increment `u_t` | `PT_{h_t->q_j}(log_{h_t}(h_{t+1}))` | One-step conversation motion in a common tangent space. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `transported_increment_matrix` | Paper 1 low-rank analysis; Paper 3 query-conditioned and failed state-update branches | Held up for low-rank geometry. Failed as a clean state-update supersession detector. |
| Segment reference `q_j` | Local tangent-space anchor for segment `S_j` | Shared coordinate system for transported steps. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `segment_reference` | Paper 1, query-conditioned geometry, state-update checks | Held up. Gauge choice is stable enough for current use. |
| Low-rank basis / `rank95` | Rank needed to explain 95% singular-value energy | Compressibility of local conversation motion. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `low_rank_project`, `effective_rank`; [`../paper1_geometry/analysis.py`](../paper1_geometry/analysis.py) | Paper 1 main result | Held up strongly. This is the strongest mathematical result in the program. |
| Curvature proxy | `||u_t-u_{t-1}|| / Δs_t` | Local geometric instability / regime shift signal. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `curvature_series` | Paper 1 boundary hypothesis; Paper 2 risk; Paper 1/Paper 3 atlas branch | Mixed. Useful as a soft risk signal; too fragile for regime classification without stabilization. |
| Stabilized curvature | `turning_angle / max(local_arclength, floor)` | Curvature proxy that avoids long-conversation blow-ups. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `stabilized_curvature_series` | Regime atlas rerun, MSC persona/filler falsification | Fixed a real numerical bug. Did not rescue regime classification or MSC support detection. |
| Boundary score / prominence | Hybridized local boundary magnitude and local peak strength | Approximate regime boundary localization. | [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py): `boundary_score_series`, `boundary_prominence_series`, `hybrid_boundary_score_series` | Paper 1 secondary boundary claim | Held only as an approximate, secondary signal. |
| Decoder distortion metrics | `L2`, `KL`, top-1 match, answer NLL | Decoder-facing objective under compression. | [`../paper1_geometry/modeling.py`](../paper1_geometry/modeling.py): `project_logits`, `score_assistant_response`; [`../paper2_memory/run_paper2.py`](../paper2_memory/run_paper2.py); [`../paper3_codec/run_paper3.py`](../paper3_codec/run_paper3.py) | Papers 1–3 | Held up. `L2` is the main stable metric; answer NLL is informative but noisier. |
| Budget objective / Lagrangian | `E[D(π)] + λ E[M(π)]` | Control problem: decoder fidelity under memory budget. | Implemented operationally in budgeted selectors rather than symbolic optimization. See [`../paper2_memory/policies.py`](../paper2_memory/policies.py), [`../paper3_codec/policies.py`](../paper3_codec/policies.py) | Papers 2–3 | Held up as the design frame. No learned optimizer yet. |
| Paper 2 geometry risk score | Normalized state error + turning + subspace shift expansion | Turn-level retention priority. | [`../paper2_memory/policies.py`](../paper2_memory/policies.py): `turn_geometry_risk` | Paper 2 geometry controller, Paper 3 geometry baselines | Held up on the hard stress set; not sufficient for semantic-memory benchmarks. |
| Lexical / hybrid Paper 2 risk | Lexical boundary scores and geometry+lexical blend | Comparator and ablation for Paper 2 | [`../paper2_memory/policies.py`](../paper2_memory/policies.py): `turn_lexical_risk`, `turn_hybrid_risk` | Paper 2 competitor matrix | Useful baseline, not the lead result. |
| Semantic turn risk | Cosine to target-turn state | Query/target relevance proxy. | [`../paper2_memory/policies.py`](../paper2_memory/policies.py): `turn_semantic_risk` | Paper 2 semantic baseline, most Paper 3 semantic-led policies | Held up strongly on MSC and public benchmarks. |
| Segment-action bridge policy | Keep/compress/evict by segment under dynamic programming | First bridge from turn retention into segment compression. | [`../paper2_memory/policies.py`](../paper2_memory/policies.py): `select_segment_actions` | Paper 2 bridge, Paper 3 `geometry_segment_actions` | Held up as a viable family, but not the final hard-set winner. |
| Paper 3 sparse memory object | `(segment_start, segment_end, anchor_turn_index, support_turn_indices, retained_turn_indices, risk, action)` | Explicit sparse codec representation for one compressed region. | [`../paper3_codec/policies.py`](../paper3_codec/policies.py): `SparseSegmentMemory` | Paper 3 core | Held up. This is the main codec object. |
| Paper 3 keep/compress/drop selector | Segment DP over `keep`, `compress`, `evict` | Sparse codec decision policy. | [`../paper3_codec/policies.py`](../paper3_codec/policies.py): `_select_sparse_segment_memory_core`, `select_sparse_segment_memory` | Paper 3 hard-set pilot and fairness sweep | Held up on support-turn-critical memory. |
| Support-aware geometry variants | Geometry risk plus support score bonuses and support-aware compressed candidates | Repair geometry-KCD under scarcity. | [`../paper3_codec/policies.py`](../paper3_codec/policies.py): `_support_aware_candidates`, `select_support_aware_sparse_segment_memory`; [`../paper3_codec/run_paper3.py`](../paper3_codec/run_paper3.py) | Low-budget KCD smoke studies | Partially held. Strong engineering repair, but not a semantic-memory winner. |
| Semantic-led codec variants | `semantic_keep_compress_drop`, support-aware semantic KCD, budget-aware semantic KCD | Make the codec semantic-first instead of geometry-first. | [`../paper3_codec/run_paper3.py`](../paper3_codec/run_paper3.py): policy dispatch for `semantic_keep_compress_drop`, `support_aware_semantic_keep_compress_drop`, `budget_aware_semantic_keep_compress_drop`; [`../paper3_codec/policies.py`](../paper3_codec/policies.py): `select_semantic_filtered_sparse_segment_memory` | MSC/LoCoMo semantic-memory studies | Held up as the right family on semantic-memory benchmarks, but no single semantic-KCD variant has separated decisively from plain semantic. |
| Query-conditioned geometry variants | Query-projected curvature + query-projected subspace energy in the same tangent frame as local motion | Give geometry a query-aware signal instead of ambient curvature only. | [`../paper3_codec/query_geometry.py`](../paper3_codec/query_geometry.py): `query_conditioned_turn_risk`; [`../paper3_codec/run_paper3.py`](../paper3_codec/run_paper3.py): `query_conditioned_geometry`, `query_conditioned_geometry_keep_compress_drop` | Query-conditioned geometry smoke studies | Mixed. Improves geometry on MSC; unstable on LoCoMo at `0.35`. |
| State-update same-sign alignment | `A(s,t)=cos(u_s,u_t)` | Detect later turns that supersede earlier same-topic turns. | [`../scripts/run_state_update_alignment_check.py`](../scripts/run_state_update_alignment_check.py): same-sign alignment branch | Synthetic state-update benchmark | Failed math. Same-topic updates remained strongly positive, not negative. |
| State/increment cross-term | `cos(log_q(h_s), u_t^entry)` | Geometrically cleaner update detector comparing earlier state position to later update direction. | [`../scripts/run_state_update_alignment_check.py`](../scripts/run_state_update_alignment_check.py): state/update cross-control branch | Synthetic state-update benchmark | Failed math. Slight ranking margin only; no sign-based separation. |

### Why the failed math broke

| Failed object | Why it broke |
| --- | --- |
| Raw curvature without stabilization | Long conversations produced near-zero step norms, so the ratio blew up numerically and created fake regimes. |
| Curvature as MSC support detector | Persona/support turns and filler turns in MSC did not differ enough in stabilized curvature to create a clean separator. |
| Same-sign update alignment | The model processes “user states a fact” and “user updates a fact” as similarly oriented increments; the turn-to-turn motion did not reverse. |
| State/increment cross-term | The theoretically cleaner cross-term stayed positive for updates and controls alike; the sign rule was too strong for the actual representation geometry. |
| Query-conditioned geometry as a standalone policy | Query projection helped on some budgets but did not stabilize across benchmarks, so it acts more like a refinement signal than a primary selector. |

## 3. Data and Benchmark Ledger

| Dataset / family | Source / path | Structure | Memory type | Where answers live | Theoretical winner | Actual winner | Mismatch reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper 1 study conversations | [`../paper1_geometry/assets/paper1_study_conversations.jsonl`](../paper1_geometry/assets/paper1_study_conversations.jsonl) | Short labeled multi-family dialogues with boundary annotations | Geometric characterization, not downstream memory retrieval | Entire short trajectory | Geometry | Geometry | No major mismatch. This benchmark exists to test the geometry itself. |
| Paper 1 H2 stress conversations | [`../paper1_geometry/assets/paper1_h2_stress_conversations.jsonl`](../paper1_geometry/assets/paper1_h2_stress_conversations.jsonl) | Short stress conversations with sharper local shifts | Boundary / local instability hypothesis | Near the abrupt shift | Geometry | Geometry, but only as approximate localization | Boundary signal is real but formulation-sensitive and not exact segmentation. |
| Paper 2 `long_dependency` | [`../paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl`](../paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl) | Custom hard-set family with late-turn questions depending on early facts | Support-turn-critical memory | Early user turns | Geometry | Geometry | No mismatch. This is where geometry control is supposed to help. |
| Paper 2 `retrieval_heavy` | same as above | Retrieval packets, logistics, exact field recall | Support-turn-critical memory | Earlier precise packet turns | Geometry | Geometry / KCD family | No mismatch. Support-turn rescue is clearest here. |
| Paper 2 `code_conversation` | same as above | Short code / SQL / formatting constraint dialogues | Constraint memory and exact support preservation | Earlier exact user constraints | Geometry | Geometry / KCD family | No mismatch. These are the strongest mechanism cases. |
| LongMemEval public | [`../paper3/paper3_public_v1_public_benchmark`](../paper3/paper3_public_v1_public_benchmark), [`../papers/paper3_public_benchmark_checkpoint.md`](paper3_public_benchmark_checkpoint.md) | Longer public benchmark slice with many sampled target turns | Broader episode memory and semantic retrieval | Distributed, often earlier but semantically recoverable | Mixed; semantic or segment-style under scarcity, possible KCD later | Semantic at `0.20`, semantic/segment at `0.35`, KCD at `0.50` | Geometry transfers, but the winner depends on budget and memory type. |
| MSC | [`../benchmarks/sample_packets.md`](../benchmarks/sample_packets.md), [`benchmark_memory_type_analysis.md`](benchmark_memory_type_analysis.md), [`paper3_msc_semantic_codec_checkpoint.md`](paper3_msc_semantic_codec_checkpoint.md) | Multi-session chat with persona, preferences, and continuity | Semantic / persona continuity | Earlier session facts and conversational state | Semantic | Semantic | Geometry did not provide a strong within-topic support signal here. |
| LoCoMo | [`../benchmarks/sample_packets.md`](../benchmarks/sample_packets.md), [`benchmark_memory_type_analysis.md`](benchmark_memory_type_analysis.md), [`paper3_query_conditioned_geometry_smoke_checkpoint.md`](paper3_query_conditioned_geometry_smoke_checkpoint.md) | Few but very long conversations with event chains and temporal evolution | Mixed temporal-semantic memory | Earlier event chains and biographical states | Mixed | Semantic-led families in current bounded probes | Geometry helps sometimes, but does not yet stabilize as the main signal. |
| Synthetic state-update benchmark | [`../benchmarks/state_update_synthetic_conversations.jsonl`](../benchmarks/state_update_synthetic_conversations.jsonl), [`../benchmarks/state_update_synthetic_labels.json`](../benchmarks/state_update_synthetic_labels.json) | Explicit “old fact -> updated fact” conversations with controlled queries | State supersession / current-state tracking | Earlier original fact and later update turn | Geometry, if update direction truly reversed earlier state | No geometry-based sign detector succeeded | The math overestimated the representational resolution of compact models. |

### Benchmark memory-type interpretation

- MSC mostly tests preference, persona, and conversational continuity.
- LongMemEval public mixes longer-range relevance and episode retrieval; it is not just “constraint recall.”
- The hard stress set was deliberately built to magnify support-turn dependence, exact retrieval, and formatting/code constraints.
- LoCoMo is mixed: semantic biography memory plus event-order or chain structure.
- The synthetic update benchmark is not a natural benchmark; it is a falsification harness for a specific geometric theorem direction.

## 4. Implementation Mapping

### Paper 1 geometry extraction and analysis

- Hidden-state extraction:
  - [`../paper1_geometry/modeling.py`](../paper1_geometry/modeling.py)
  - `ConversationStateExtractor.extract_conversation`
  - `ConversationStateExtractor.score_messages`
- Core geometry:
  - [`../paper1_geometry/geometry.py`](../paper1_geometry/geometry.py)
  - `normalize_rows`
  - `sphere_distance`
  - `sphere_log_map`
  - `sphere_parallel_transport`
  - `segment_reference`
  - `transported_increment_matrix`
  - `curvature_series`
  - `stabilized_curvature_series`
  - `boundary_score_series`
  - `boundary_prominence_series`
  - `low_rank_project`
  - `effective_rank`
- Study and reporting:
  - [`../paper1_geometry/study.py`](../paper1_geometry/study.py)
  - [`../paper1_geometry/analysis.py`](../paper1_geometry/analysis.py)
  - [`../paper1_geometry/reporting.py`](../paper1_geometry/reporting.py)

Implicit assumptions in code:

- one hidden-state summary per turn is enough to characterize conversation motion
- normalized last-token summaries preserve the relevant geometry
- segment references can be chosen locally without destabilizing low-rank conclusions

Math-to-code limitation:

- the code uses one specific turn summary and one specific gauge choice; the paper’s conclusions are therefore about this operational geometry, not every possible hidden-state geometry.

### Paper 2 controller policies and mechanism analysis

- Risk construction:
  - [`../paper2_memory/policies.py`](../paper2_memory/policies.py)
  - `turn_geometry_risk`
  - `turn_lexical_risk`
  - `turn_hybrid_risk`
  - `turn_semantic_risk`
- Budgeted selection:
  - `select_turns`
  - `select_segment_actions`
- Execution:
  - [`../paper2_memory/run_paper2.py`](../paper2_memory/run_paper2.py)
  - [`../paper2_memory/study.py`](../paper2_memory/study.py)
- Mechanism analysis:
  - [`../paper2_memory/memory_critical_analysis.py`](../paper2_memory/memory_critical_analysis.py)
  - [`../paper2_memory/case_analysis.py`](../paper2_memory/case_analysis.py)
  - [`../paper2_memory/cross_model_memory_summary.py`](../paper2_memory/cross_model_memory_summary.py)

Implicit assumptions in code:

- decoder harm can be approximated by normalized geometric risk
- recent-window retention is always hard-preserved
- additive token-like costs are an adequate proxy for budgeted memory use

Math-to-code limitation:

- the budget objective is implemented as a heuristic selector, not a learned or solved Lagrangian optimizer
- support-turn dependence is measured operationally through hand-identified earlier user turns, not latent causal graphs

### Paper 3 codec policies, pairwise analysis, and mechanism analysis

- Sparse memory object and selectors:
  - [`../paper3_codec/policies.py`](../paper3_codec/policies.py)
  - `SparseSegmentMemory`
  - `CodecSelection`
  - `select_sparse_segment_memory`
  - `select_support_aware_sparse_segment_memory`
  - `select_semantic_filtered_sparse_segment_memory`
  - `semantic_shortlist_mask`
- Runtime integration and policy dispatch:
  - [`../paper3_codec/run_paper3.py`](../paper3_codec/run_paper3.py)
- Query-conditioned geometry:
  - [`../paper3_codec/query_geometry.py`](../paper3_codec/query_geometry.py)
  - `query_conditioned_turn_risk`
- Pairwise reporting:
  - [`../paper3_codec/pairwise_analysis.py`](../paper3_codec/pairwise_analysis.py)
- Mechanism analysis:
  - [`../paper3_codec/memory_critical_analysis.py`](../paper3_codec/memory_critical_analysis.py)

Implicit assumptions in code:

- sparse compression can be modeled by retaining a small anchor/support subset
- support-like turns are approximable using risk plus simple support bonuses
- semantic shortlist + structure-aware compression is a viable decomposition

Math-to-code limitation:

- the codec theorem uses an assumed harm predictor, but the code still implements heuristic proxies
- compressed memory is represented by retained turn indices rather than a learned latent codec

### Benchmark normalization and benchmark runners

- Public benchmark download / normalization:
  - [`../scripts/download_public_benchmark.py`](../scripts/download_public_benchmark.py)
  - [`../scripts/prepare_public_benchmark_jsonl.py`](../scripts/prepare_public_benchmark_jsonl.py)
- Benchmark runners:
  - [`../scripts/run_paper3_quick_benchmark.sh`](../scripts/run_paper3_quick_benchmark.sh)
  - [`../scripts/run_paper3_low_budget_kcd_probe.sh`](../scripts/run_paper3_low_budget_kcd_probe.sh)
  - [`../scripts/run_paper3_query_conditioned_geometry_probe.sh`](../scripts/run_paper3_query_conditioned_geometry_probe.sh)
  - [`../scripts/run_paper3_semantic_kcd_optimization.sh`](../scripts/run_paper3_semantic_kcd_optimization.sh)
  - [`../scripts/run_msc_persona_curvature_check.py`](../scripts/run_msc_persona_curvature_check.py)
  - [`../scripts/run_state_update_alignment_check.py`](../scripts/run_state_update_alignment_check.py)

Implicit assumptions in code:

- all downstream experiments can be reduced to the same normalized conversation JSONL format
- benchmark slices and bounded target-turn sampling are representative enough for engineering checkpoints

Math-to-code limitation:

- the mathematical program is benchmark-agnostic in abstraction, but the experiments are highly sensitive to how conversations are normalized and sampled

### Manuscript asset generation

- [`../scripts/build_manuscript_assets.py`](../scripts/build_manuscript_assets.py)
- [`../scripts/build_research_ledger_assets.py`](../scripts/build_research_ledger_assets.py)

What is implemented:

- tables and figures built from tracked JSON summaries
- current program overview, regime map, negative-result table, and status bridge
- machine-readable experiment and negative-result inventories for this ledger

Implicit assumption:

- tracked artifacts and committed notes are the canonical evidence base

Limitation:

- the manuscript build is only as complete as the tracked checkpoint bundles; untracked local experiments are intentionally excluded

## 5. Experiment Ledger

The canonical machine-readable summary of this section is:

- [`generated/rt_experiment_matrix.csv`](generated/rt_experiment_matrix.csv)

### Paper 1 main

- Goal:
  - establish whether conversation trajectories are low-rank and decoder-relevant
- Models:
  - `qwen25_05b`
  - `qwen25_15b`
  - `smollm2_17b`
- Datasets:
  - Paper 1 multi-family characterization conversations
- Policies:
  - none; this is a measurement study
- Primary metrics:
  - `rank95`
  - correlation between geometric distortion and decoder logit drift
  - shuffled/permuted nulls
- Key numerical result:
  - `qwen25_15b`: `rank95 = 1.167`, geometry-logit correlation `0.994`
- Interpretation:
  - the geometry is real, compact, and decoder-facing
- Artifact path:
  - [`../artifacts/paper1/expanded_v8_final`](../artifacts/paper1/expanded_v8_final)

### Paper 2 main

- Goal:
  - test geometry-aware memory control under budget scarcity
- Models:
  - `qwen25_05b`
  - `qwen25_15b`
  - `smollm2_17b`
- Budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- Datasets:
  - hard stress set across `long_dependency`, `retrieval_heavy`, `code_conversation`
- Policies:
  - `uniform`
  - `geometry`
  - comparators
- Primary metrics:
  - `delta_logit_l2`
  - `delta_kl`
  - answer NLL
- Key numerical result:
  - `qwen25_05b @ 0.35`: `delta_logit_l2 = -99.503`, `p = 0.0000`
- Interpretation:
  - geometry-aware retention is a real systems result on the hard stress set
- Artifact path:
  - [`../artifacts/paper2/behavior_stress_v1`](../artifacts/paper2/behavior_stress_v1)

### Paper 2 mechanism

- Goal:
  - explain why geometry helps
- Model:
  - `qwen25_05b`
- Budget:
  - `0.35`
- Dataset:
  - hard stress set
- Primary metric:
  - support-turn rescue counts
- Key numerical result:
  - geometry better than uniform in `14/36` cases
  - uniform better in `5/36`
  - geometry rescues the latest support turn in `13/36`
- Interpretation:
  - the gain comes from preserving exact earlier support turns, not generic recency
- Artifact path:
  - [`../artifacts/paper2/behavior_stress_qwen_cases`](../artifacts/paper2/behavior_stress_qwen_cases)

### Paper 3 hard-set pilot

- Goal:
  - test whether a true sparse codec can be active and beneficial
- Models:
  - `qwen25_05b`
  - `qwen25_15b`
  - `smollm2_17b`
- Budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- Policies:
  - `geometry`
  - `geometry_segment_actions`
  - `geometry_keep_compress_drop`
- Primary metric:
  - `delta_logit_l2`
- Key numerical result:
  - `qwen25_05b @ 0.35`, KCD: `-105.647`, `p = 0.0003`
- Interpretation:
  - the codec is real, active, and not just a degenerate policy shell
- Artifact path:
  - [`../artifacts/paper3/paper3_pilot_v3_full`](../artifacts/paper3/paper3_pilot_v3_full)

### Fairness sweep

- Goal:
  - test whether KCD survives tighter realized-token control
- Model:
  - `qwen25_15b`
- Budgets:
  - dense sweep from `0.24` to `0.50`
- Policies:
  - `uniform`
  - `geometry`
  - `geometry_segment_actions`
  - `geometry_keep_compress_drop`
- Primary metric:
  - `delta_logit_l2`
- Key numerical result:
  - `0.32`, KCD: `-56.165`, `p = 0.0030`
- Interpretation:
  - KCD’s scarcity win survives fairness control
- Artifact path:
  - [`../artifacts/paper3/paper3_batch_v1_fairness`](../artifacts/paper3/paper3_batch_v1_fairness)

### 3B probe

- Goal:
  - test whether the Paper 3 regime split survives at larger model scale
- Model:
  - `qwen25_3b`
- Budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- Policies:
  - `geometry`
  - `geometry_segment_actions`
  - `geometry_keep_compress_drop`
- Primary metrics:
  - pairwise `delta_logit_l2`
- Key numerical result:
  - KCD vs geometry: `-53.675` at `0.35`, `+24.357` at `0.50`
- Interpretation:
  - sparse geometry codec under scarcity, plain geometry at looser budget
- Artifact path:
  - [`../artifacts/paper3/paper3_batch_v1_3b`](../artifacts/paper3/paper3_batch_v1_3b)

### LongMemEval public benchmark

- Goal:
  - test public benchmark transfer
- Model:
  - `qwen25_15b`
- Budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- Policies:
  - `uniform`
  - `semantic`
  - `geometry`
  - `geometry_segment_actions`
  - `geometry_keep_compress_drop`
- Primary metric:
  - `delta_logit_l2`
- Key numerical result:
  - `0.20 semantic = -122.613`
  - `0.35 geometry_segment_actions = -97.432`
  - `0.50 KCD = -48.466`
- Interpretation:
  - geometry transfers, but the ranking is benchmark-dependent
- Canonical paths:
  - [`paper3_public_benchmark_checkpoint.md`](paper3_public_benchmark_checkpoint.md)
  - [`../paper3/paper3_public_v1_public_benchmark`](../paper3/paper3_public_v1_public_benchmark)

### MSC semantic-codec checkpoint

- Goal:
  - test whether semantic-led codecs or geometry-led codecs dominate semantic-memory benchmarks
- Model:
  - `qwen25_15b`
- Budgets:
  - `0.20`
  - `0.35`
  - `0.50`
- Policies:
  - `semantic`
  - `semantic_keep_compress_drop`
  - geometry family controls
- Primary metrics:
  - `delta_logit_l2`
  - behavior pairwise comparisons
- Key result:
  - semantic wins overall; semantic-KCD is competitive only at `0.20`
- Interpretation:
  - MSC is a semantic-memory benchmark, not a geometry-first benchmark
- Canonical path:
  - [`paper3_msc_semantic_codec_checkpoint.md`](paper3_msc_semantic_codec_checkpoint.md)

### Low-budget KCD upgrades

- Goal:
  - repair old geometry-KCD under scarcity
- Model:
  - `qwen25_05b`
- Budgets:
  - `0.20`
  - `0.35`
- Policies:
  - `support_aware_geometry_keep_compress_drop`
  - `semantic_filtered_geometry_keep_compress_drop`
  - older geometry baselines
- Key numerical result:
  - on MSC smoke at `0.20`, support-aware geometry-KCD `-505.510` vs old KCD `-44.152`
- Interpretation:
  - support-awareness is a real engineering repair, even though semantic-led families still dominate
- Canonical paths:
  - [`paper3_low_budget_kcd_smoke_checkpoint.md`](paper3_low_budget_kcd_smoke_checkpoint.md)
  - [`../results/paper3/studies/paper3_low_budget_smoke_msc`](../results/paper3/studies/paper3_low_budget_smoke_msc)

### Semantic-KCD optimization

- Goal:
  - see whether semantic-led KCD can clearly surpass strong semantic baselines
- Model:
  - `qwen25_05b`
- Budgets:
  - `0.20`
  - `0.35`
- Policies:
  - `semantic`
  - `semantic_keep_compress_drop`
  - `support_aware_semantic_keep_compress_drop`
  - `budget_aware_semantic_keep_compress_drop`
- Key numerical result:
  - at `0.35` on MSC smoke, semantic-KCD remains strongest among semantic codec variants
- Interpretation:
  - the next likely gains require a learned harm model, not more heuristic mixing
- Canonical paths:
  - [`paper3_semantic_kcd_optimization_checkpoint.md`](paper3_semantic_kcd_optimization_checkpoint.md)
  - [`../results/paper3/studies/paper3_semantic_kcd_opt_smoke_msc`](../results/paper3/studies/paper3_semantic_kcd_opt_smoke_msc)

### Query-conditioned geometry

- Goal:
  - make geometry query-aware in tangent-space-consistent coordinates
- Model:
  - `qwen25_05b`
- Budgets:
  - `0.20`
  - `0.35`
- Policies:
  - `query_conditioned_geometry`
  - `query_conditioned_geometry_keep_compress_drop`
- Key numerical result:
  - on MSC smoke at `0.20`, query-conditioned geometry-KCD `-501.114` vs plain geometry `-77.737`
- Interpretation:
  - query conditioning materially helps geometry on MSC
- Limiting result:
  - LoCoMo `0.35` plain query-conditioned geometry collapses to `+881.417`
- Canonical paths:
  - [`paper3_query_conditioned_geometry_smoke_checkpoint.md`](paper3_query_conditioned_geometry_smoke_checkpoint.md)
  - [`../results/paper3/studies/paper3_query_geom_smoke_msc`](../results/paper3/studies/paper3_query_geom_smoke_msc)
  - [`../results/paper3/studies/paper3_query_geom_smoke_locomo`](../results/paper3/studies/paper3_query_geom_smoke_locomo)

### Regime atlas

- Goal:
  - discover conversation regimes from geometry alone
- Model:
  - `qwen25_05b`
- Data:
  - smoke atlas over MSC, LoCoMo, LongMemEval, and hard stress segments
- Primary metric:
  - family / cluster separation
- Key result:
  - after stabilization, retrieval-heavy hard-set segments are still mixed into the spike-heavy family rather than separating cleanly
- Interpretation:
  - the atlas idea was mathematically interesting, but did not recover a reliable standalone policy taxonomy
- Canonical paths:
  - [`geometric_regime_atlas_smoke_checkpoint.md`](geometric_regime_atlas_smoke_checkpoint.md)
  - [`../artifacts/paper1/regime_atlas_smoke_v4`](../artifacts/paper1/regime_atlas_smoke_v4)

### MSC persona-curvature falsification

- Goal:
  - test whether curvature separates earlier support/persona turns from filler within MSC
- Model:
  - `qwen25_05b`
- Data:
  - five manually labeled MSC conversations
- Key numerical result:
  - mean support-filler delta `0.1220`, with `3/5` positive and `2/5` negative
- Interpretation:
  - no robust within-topic support discriminator
- Canonical paths:
  - [`msc_persona_curvature_check.md`](msc_persona_curvature_check.md)
  - [`../artifacts/paper3/msc_persona_curvature_v1`](../artifacts/paper3/msc_persona_curvature_v1)

### State-update falsification

- Goal:
  - test whether state supersession creates a directional-reversal signal
- Model:
  - `qwen25_05b`
- Data:
  - synthetic explicit state-update benchmark
- Same-sign result:
  - mean alignment `0.9128`, `0/10` negative
- Cross-term result:
  - update cross `0.9707`, control `0.9780`, `0/10` negative
- Interpretation:
  - both tested compact-model formulations fail as clean update detectors
- Canonical paths:
  - [`state_update_alignment_smoke_checkpoint.md`](state_update_alignment_smoke_checkpoint.md)
  - [`../artifacts/paper3/state_update_alignment_smoke_qwen05b`](../artifacts/paper3/state_update_alignment_smoke_qwen05b)
  - [`../artifacts/paper3/state_update_cross_control_qwen05b`](../artifacts/paper3/state_update_cross_control_qwen05b)

## 6. Failure and Mismatch Ledger

The canonical machine-readable summary of this section is:

- [`generated/rt_negative_result_matrix.csv`](generated/rt_negative_result_matrix.csv)

### Failure template

Each failed branch is recorded as:

- hypothesis
- formula / signal
- benchmark
- observed result
- mathematical mismatch
- design consequence

### 6.1 MSC support/persona vs filler curvature

- Hypothesis:
  - earlier support/persona turns in MSC should have higher curvature than filler turns
- Formula / signal:
  - stabilized curvature on manually labeled support vs filler turns
- Benchmark:
  - five-conversation manual MSC check
- Observed result:
  - mean support-filler delta `0.1220`
  - `3/5` positive, `2/5` negative
- Mathematical mismatch:
  - MSC persona facts are often semantically important without producing geometrically distinctive local spikes
- Failure type:
  - failed data fit
- Design consequence:
  - do not use curvature alone as the hidden support detector inside semantic-memory benchmarks

### 6.2 Geometry-only regime atlas

- Hypothesis:
  - segment-level geometric statistics should separate conversation types into usable compression regimes
- Formula / signal:
  - regime clustering from curvature, step norms, and related segment summaries
- Benchmark:
  - MSC + LoCoMo + LongMemEval + hard stress smoke atlas
- Observed result:
  - after stabilization, retrieval-heavy hard-set segments were still mixed with large parts of MSC and LoCoMo
- Mathematical mismatch:
  - the segment-level summary statistics were too coarse relative to the within-benchmark variation
- Failure type:
  - failed benchmark transfer
- Design consequence:
  - geometry-only self-diagnosing compression is not justified by the current atlas

### 6.3 Same-sign state-update alignment

- Hypothesis:
  - if turn `t` supersedes earlier same-topic turn `s`, then `cos(u_s, u_t)` should be negative
- Formula / signal:
  - `A(s,t)=cos(u_s,u_t)`
- Benchmark:
  - synthetic state-update benchmark
- Observed result:
  - mean alignment `0.9128`
  - `0/10` negative
- Mathematical mismatch:
  - “user states a fact” and “user updates a fact” produced similarly oriented increments in compact models rather than geometric reversals
- Failure type:
  - failed math
- Design consequence:
  - retire the direct same-sign supersession rule for compact-model work

### 6.4 State-vs-increment cross-term

- Hypothesis:
  - comparing the earlier state position against the later update direction should produce a negative update detector
- Formula / signal:
  - `cos(log_q(h_s), u_t^{entry})`
- Benchmark:
  - synthetic state-update benchmark
- Observed result:
  - update cross `0.9707`
  - mean control cross `0.9780`
  - `0/10` negative updates
- Mathematical mismatch:
  - the cleaner geometric construction weakened to a tiny ranking effect rather than a sign change
- Failure type:
  - failed math
- Design consequence:
  - treat the cross-term as, at most, a weak comparative feature, not an algorithm

### 6.5 Query-conditioned geometry instability

- Hypothesis:
  - making geometry query-aware should yield a stable standalone ranking signal
- Formula / signal:
  - query-projected curvature + query-projected subspace energy in the same tangent frame as conversation motion
- Benchmark:
  - bounded LoCoMo smoke
- Observed result:
  - at `0.35`, `query_conditioned_geometry = +881.417`
  - KCD wrapper only recovered to `+9.724`
- Mathematical mismatch:
  - the signal improved alignment to the query but did not stabilize across benchmark regimes or budget levels
- Failure type:
  - failed benchmark transfer
- Design consequence:
  - keep query-conditioned geometry as a refinement feature only, not a primary standalone selector

### 6.6 Curvature blow-up before stabilization

- Hypothesis:
  - the original curvature proxy could be used unchanged on long conversations
- Formula / signal:
  - `||u_t-u_{t-1}|| / Δs_t` with no arclength floor
- Benchmark:
  - regime atlas smoke prior to stabilization
- Observed result:
  - many near-zero-step segments exploded to artificial curvature in the thousands
- Mathematical mismatch:
  - a denominator-level numerical instability dominated the intended geometry
- Failure type:
  - failed numerical implementation
- Design consequence:
  - raw curvature remains auditable, but the operative pipeline must use stabilized curvature

## 7. Progress and Frontier

### What is already publishable

- Paper 1 as a characterization paper:
  - low-rank conversation trajectories
  - near-perfect geometry-logit coupling
  - null controls that collapse the effect
- Paper 2 as a systems/control paper:
  - geometry-aware retention beats uniform under scarcity
  - mechanism is support-turn rescue
- The current checkpoint manuscript:
  - benchmark-dependent Paper 3 story
  - explicit negative results that sharpen claim boundaries

### What is scientifically settled

- Conversation-state geometry is real in compact models.
- Geometric distortion is tightly coupled to decoder drift.
- Geometry helps on support-turn-critical memory control.
- Sparse codecs are real on hard support-turn benchmarks.
- There is no universal best compression signal across all conversational-memory benchmarks.
- Several plausible geometry-only extensions fail or remain too weak in compact models.

### What is still worth exploring

- learned selector models that combine:
  - semantic discovery
  - structural / support-aware refinement
  - budget awareness
- larger-model checks:
  - whether some failed compact-model branches become viable at higher representational resolution
- learned harm predictors for Paper 3:
  - replacing heuristic support or geometry proxies with model-derived distortion estimators

### What should be retired

- raw curvature without stabilization for long-conversation regime work
- geometry-only regime classification as a standalone policy selector
- same-sign supersession detection by `cos(u_s,u_t)`
- sign-based state/update cross-term detection in compact models
- any claim that geometry alone beats semantic baselines on semantic-memory benchmarks

### Noncanonical Local Probes

These are intentionally excluded from canonical claims unless promoted into tracked artifacts.

| Probe | Status | Committed runner | Reproducible command | Saved summary |
| --- | --- | --- | --- | --- |
| Role-conditioned residual novelty check | `pending / noncanonical` | [`../scripts/run_role_residual_check.py`](../scripts/run_role_residual_check.py) | `python scripts/download_public_benchmark.py --benchmark msc_valid --output benchmarks/msc_valid_raw.jsonl && python scripts/prepare_public_benchmark_jsonl.py --format msc --input benchmarks/msc_valid_raw.jsonl --output benchmarks/msc_valid_normalized.jsonl --family msc_valid && python scripts/run_role_residual_check.py --input-path benchmarks/msc_valid_normalized.jsonl --benchmark-name msc_valid --limit-conversations 5 --model-key qwen25_05b --max-input-tokens 768 --output-dir results/role_residual_check/msc_valid_qwen05b` | local-only summary at `../results/role_residual_check/msc_valid_qwen05b/summary.json` |

Current reading of that local probe:

- it is not part of the main claim set
- on the five-conversation local MSC pass, role-level residual separation is weak and mixed
- therefore it remains only an exploratory branch until promoted to tracked evidence
