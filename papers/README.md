# RT Papers

This folder contains the three-paper research program for RT.

For the current manuscript-grade checkpoint paper, see:

- [`manuscript/paper_checkpoint.tex`](/Users/pranav/Documents/RT/manuscript/paper_checkpoint.tex)
- [`manuscript/README.md`](/Users/pranav/Documents/RT/manuscript/README.md)

For the canonical math-to-code-to-artifact synthesis, see:

- [`rt_derivation_ledger.md`](/Users/pranav/Documents/RT/papers/rt_derivation_ledger.md)
- [`generated/rt_experiment_matrix.csv`](/Users/pranav/Documents/RT/papers/generated/rt_experiment_matrix.csv)
- [`generated/rt_negative_result_matrix.csv`](/Users/pranav/Documents/RT/papers/generated/rt_negative_result_matrix.csv)

## Sequence

1. [`paper1_geometry_of_conversation_state_trajectories.md`](/Users/pranav/Documents/RT/papers/paper1_geometry_of_conversation_state_trajectories.md)
2. [`paper1_outline.md`](/Users/pranav/Documents/RT/papers/paper1_outline.md)
3. [`paper2_geometry_guided_adaptive_memory_compression.md`](/Users/pranav/Documents/RT/papers/paper2_geometry_guided_adaptive_memory_compression.md)
4. [`paper3_manifold_memory_codecs.md`](/Users/pranav/Documents/RT/papers/paper3_manifold_memory_codecs.md)
5. [`paper3_public_benchmark_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_public_benchmark_checkpoint.md)
6. [`paper3_msc_semantic_codec_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_msc_semantic_codec_checkpoint.md)
7. [`paper3_low_budget_kcd_smoke_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_low_budget_kcd_smoke_checkpoint.md)
8. [`paper3_semantic_kcd_optimization_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_semantic_kcd_optimization_checkpoint.md)
9. [`paper3_query_conditioned_geometry_smoke_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_query_conditioned_geometry_smoke_checkpoint.md)
10. [`whitepaper_benchmark_dependent_memory_regimes.md`](/Users/pranav/Documents/RT/papers/whitepaper_benchmark_dependent_memory_regimes.md)
11. [`benchmark_memory_type_analysis.md`](/Users/pranav/Documents/RT/papers/benchmark_memory_type_analysis.md)
12. [`geometric_regime_atlas_smoke_checkpoint.md`](/Users/pranav/Documents/RT/papers/geometric_regime_atlas_smoke_checkpoint.md)
13. [`msc_persona_curvature_check.md`](/Users/pranav/Documents/RT/papers/msc_persona_curvature_check.md)
14. [`state_update_alignment_smoke_checkpoint.md`](/Users/pranav/Documents/RT/papers/state_update_alignment_smoke_checkpoint.md)

## Logic

- Paper 1 validates whether the geometry is real and decoder-relevant.
- Paper 2 uses that geometry as a control signal for practical compression.
- Paper 3 attempts a true manifold-memory codec only if the first two stages justify it.

## Current Checkpoint

- Paper 1: frozen characterization result
- Paper 2: strong control result with a mechanism story
- Paper 3: active codec stage with a fairness-controlled low/mid-budget winner and a positive 3B validation run
- The master derivation ledger is now the canonical deep-dive companion to the manuscript, covering definitions, code mappings, benchmark structure, experiment history, and failure analysis
- Paper 3 now also has a semantic-memory checkpoint showing that benchmark family changes the winning policy form
- Paper 3 now has a low-budget upgrade smoke checkpoint showing that support-aware geometry-KCD materially improves over the original geometry-KCD baseline, even though semantic-led policies still dominate semantic-memory benchmarks
- Paper 3 now has a semantic-KCD optimization checkpoint showing that the next likely gains require a learned harm signal and a denser compressed memory object rather than more heuristic score mixing
- Paper 3 now has a query-conditioned geometry smoke checkpoint showing that tangent-space query conditioning improves geometry, but still does not displace the strongest semantic codecs on semantic-memory benchmarks
- The project now also has a benchmark-reading note plus a first geometric regime atlas smoke checkpoint, which turn the benchmark split into an explicit data-structure hypothesis rather than a post-hoc result description
- The project now also has a targeted MSC persona-curvature falsification check showing that manually labeled support/persona turns do not separate cleanly from filler by stabilized curvature alone
- The project now also has a synthetic state-update falsification note showing that both same-sign increment alignment and a state/increment cross-term fail as clean supersession detectors in compact models
- The project now also has a synthetic state-update falsification check showing that the straightforward same-sign directional-alignment supersession hypothesis fails cleanly, while a mixed entry/exit diagnostic exposes a real mathematical ambiguity in how turn-associated update vectors should be defined
