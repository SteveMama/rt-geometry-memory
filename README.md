# RT

RT is the standalone research project for the three-paper program:

- Paper 1: measurement
- Paper 2: systems
- Paper 3: codec

This repository is a checkpoint, not the end of the project. The current state is:

- Paper 1 is frozen as a characterization result
- Paper 2 has a real geometry-aware control result plus a mechanism story
- Paper 3 is alive with two viable compressed-memory policy families, but no final winner yet

This checkpoint now also includes a manuscript-grade paper bundle:

- IEEE-style manuscript source: [`manuscript/paper_checkpoint.tex`](/Users/pranav/Documents/RT/manuscript/paper_checkpoint.tex)
- manuscript build entry point: [`manuscript/build.sh`](/Users/pranav/Documents/RT/manuscript/build.sh)
- manuscript asset generator: [`scripts/build_manuscript_assets.py`](/Users/pranav/Documents/RT/scripts/build_manuscript_assets.py)
- manuscript README: [`manuscript/README.md`](/Users/pranav/Documents/RT/manuscript/README.md)

This project is separate from the CARP/KV-compression repo. The current codebase implements the Paper 1 bootstrap:

- extract one hidden-state summary per conversation turn
- normalize conversation states onto the sphere
- measure consecutive angles, discrete curvature, and segment-wise low-rank structure
- reconstruct trajectories from transported low-rank increments
- compare geometric reconstruction error against decoder drift through logit `L2`, KL, and top-1 agreement

Current Paper 1 framing:

- main claim: conversation-state geometry is low-rank and decoder-relevant
- secondary claim: geometry carries partial regime-boundary information, but boundary recovery is formulation-sensitive on short conversations

Current Paper 1 checkpoint:

- strongest result: geometric distortion strongly predicts decoder drift
- status: frozen evidence bundle in [`artifacts/paper1/expanded_v8_final`](/Users/pranav/Documents/RT/artifacts/paper1/expanded_v8_final)

Current Paper 2 framing:

- main systems question: can geometry-derived risk reduce output drift under a fixed memory budget
- current main result: geometry-aware retention beats uniform allocation on the hard long-context stress set, with the clearest gains at lower-to-mid budgets
- current controller family: `uniform`, `lexical`, `geometry`, plus segment-action bridge policies
- current evaluation target: budgeted logit drift, KL drift, top-1 stability, and answer-level negative log-likelihood on long-dependency, retrieval-heavy, and code-conversation families

Current Paper 2 checkpoint:

- strongest result: geometry-aware control beats uniform on the hard stress set
- mechanism result: geometry preserves memory-critical support turns more often than uniform
- status: best tracked artifacts in [`artifacts/paper2`](/Users/pranav/Documents/RT/artifacts/paper2)
- next experiment focus: semantic competitor baseline, one 3B probe, token-matched fairness sweep, and tighter Paper 3 head-to-head runs

Current Paper 3 framing:

- current goal: a minimal sparse segment memory pilot driven by the Paper 2 geometry mechanism
- current memory object: segment anchor plus sparse support turns
- current comparison: `uniform`, `geometry`, `geometry_segment_actions`, and `geometry_keep_compress_drop`

Current Paper 3 checkpoint:

- `geometry_keep_compress_drop` is now a real compression policy, not a degenerate prototype
- under fairness control, `geometry_keep_compress_drop` is the strongest low-to-mid budget Paper 3 family on `qwen25_15b`
- the 3B probe shows a clean regime split: `geometry_keep_compress_drop` wins at `0.35`, while plain `geometry` retakes the lead at `0.50`
- the MSC semantic-codec checkpoint shows that semantic-memory benchmarks favor `semantic`, while sparse codec form is benchmark-dependent rather than universal
- status: latest tracked checkpoint in [`artifacts/paper3/paper3_pilot_v3_full`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_pilot_v3_full)
- new tracked extensions: [`artifacts/paper3/paper3_batch_v1_fairness`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_batch_v1_fairness) and [`artifacts/paper3/paper3_batch_v1_3b`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_batch_v1_3b)
- first full public-benchmark report: [`paper3_public_benchmark_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_public_benchmark_checkpoint.md)
- MSC semantic-codec report: [`paper3_msc_semantic_codec_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_msc_semantic_codec_checkpoint.md)
- low-budget KCD smoke report: [`paper3_low_budget_kcd_smoke_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_low_budget_kcd_smoke_checkpoint.md)
- semantic-KCD optimization report: [`paper3_semantic_kcd_optimization_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_semantic_kcd_optimization_checkpoint.md)
- query-conditioned geometry smoke report: [`paper3_query_conditioned_geometry_smoke_checkpoint.md`](/Users/pranav/Documents/RT/papers/paper3_query_conditioned_geometry_smoke_checkpoint.md)
- research synthesis white paper: [`whitepaper_benchmark_dependent_memory_regimes.md`](/Users/pranav/Documents/RT/papers/whitepaper_benchmark_dependent_memory_regimes.md)
- benchmark-memory reading note: [`benchmark_memory_type_analysis.md`](/Users/pranav/Documents/RT/papers/benchmark_memory_type_analysis.md)
- geometric regime atlas smoke report: [`geometric_regime_atlas_smoke_checkpoint.md`](/Users/pranav/Documents/RT/papers/geometric_regime_atlas_smoke_checkpoint.md)
- MSC persona-curvature falsification check: [`msc_persona_curvature_check.md`](/Users/pranav/Documents/RT/papers/msc_persona_curvature_check.md)
- next experiment focus: semantic-first, query-conditioned geometry-KCD rather than pure geometry-first query conditioning
- new geometry-atlas direction: classify segment regimes from geometry first, then learn regime-specific compression rather than forcing one codec across all memory types

## Start Here

- Read [`artifacts/README.md`](/Users/pranav/Documents/RT/artifacts/README.md) for the frozen checkpoint bundle
- Read [`papers/README.md`](/Users/pranav/Documents/RT/papers/README.md) for the research-program logic
- Read [`CHECKPOINT_LOG.md`](/Users/pranav/Documents/RT/CHECKPOINT_LOG.md) for the cumulative execution log and result history
- Read [`manuscript/README.md`](/Users/pranav/Documents/RT/manuscript/README.md) for the checkpoint paper and PDF build path
- Read [`benchmarks/quick_benchmark_plan.md`](/Users/pranav/Documents/RT/benchmarks/quick_benchmark_plan.md) for the next small-benchmark loop
- Use [`notebooks/rt_colab_pro_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_colab_pro_runner.ipynb) for Colab runs
- Use [`notebooks/rt_next_phase_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_next_phase_runner.ipynb) for the next competitor-aware Colab suite
- Use [`notebooks/rt_paper3_next_batch_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_next_batch_runner.ipynb) for the fairness-plus-3B Paper 3 batch
- Use [`notebooks/rt_paper3_public_solidification_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_public_solidification_runner.ipynb) for the public-benchmark solidification path
- Use [`notebooks/rt_paper3_quick_benchmarks_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_quick_benchmarks_runner.ipynb) for one-notebook benchmark fetch/normalize/run loops across `MSC`, `LoCoMo10`, `LongMemEval`, and manual-source fallbacks
- Use [`notebooks/rt_paper3_low_budget_kcd_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_low_budget_kcd_runner.ipynb) for the new low-budget KCD variants on `MSC` and `LoCoMo`
- Use [`notebooks/rt_paper3_semantic_kcd_optimization_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_semantic_kcd_optimization_runner.ipynb) for the next semantic-led codec optimization pass on `MSC`, `LoCoMo`, and `LongMemEval`
- Use [`notebooks/rt_paper3_query_conditioned_geometry_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_query_conditioned_geometry_runner.ipynb) for the tangent-space query-conditioned geometry comparison on `MSC` and `LoCoMo`
- Use [`scripts/README.md`](/Users/pranav/Documents/RT/scripts/README.md) for command-line reruns

## Recommended Local Models

Start with these Mac-friendly presets:

- `qwen25_05b` = `Qwen/Qwen2.5-0.5B-Instruct`
- `qwen25_15b` = `Qwen/Qwen2.5-1.5B-Instruct`
- `qwen25_3b` = `Qwen/Qwen2.5-3B-Instruct`
- `qwen3_06b` = `Qwen/Qwen3-0.6B`
- `smollm2_17b` = `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Why these:

- `Qwen2.5-0.5B-Instruct` is the safest first baseline for Apple Silicon.
- `Qwen2.5-1.5B-Instruct` gives a stronger same-family comparison without leaving the local-first regime.
- `Qwen2.5-3B-Instruct` is the first larger-model checkpoint and is intended for Colab or a discrete GPU.
- `Qwen3-0.6B` is attractive for Paper 1, but it requires `transformers>=4.51.0`.
- `SmolLM2-1.7B-Instruct` gives one compact non-Qwen control model.

List the built-in presets with metadata:

```bash
cd /Users/pranav/Documents/RT
python -m paper1_geometry.run_paper1 --list-models --detailed-models
```

## Project Layout

- [`paper1_geometry/run_paper1.py`](/Users/pranav/Documents/RT/paper1_geometry/run_paper1.py)
- [`paper1_geometry/modeling.py`](/Users/pranav/Documents/RT/paper1_geometry/modeling.py)
- [`paper1_geometry/geometry.py`](/Users/pranav/Documents/RT/paper1_geometry/geometry.py)
- [`paper1_geometry/analysis.py`](/Users/pranav/Documents/RT/paper1_geometry/analysis.py)
- [`paper1_geometry/plotting.py`](/Users/pranav/Documents/RT/paper1_geometry/plotting.py)
- [`paper1_geometry/reporting.py`](/Users/pranav/Documents/RT/paper1_geometry/reporting.py)
- [`paper1_geometry/study.py`](/Users/pranav/Documents/RT/paper1_geometry/study.py)
- [`paper1_geometry/assets/sample_conversations.jsonl`](/Users/pranav/Documents/RT/paper1_geometry/assets/sample_conversations.jsonl)
- [`paper1_geometry/assets/paper1_study_conversations.jsonl`](/Users/pranav/Documents/RT/paper1_geometry/assets/paper1_study_conversations.jsonl)
- [`paper1_geometry/assets/paper1_h2_stress_conversations.jsonl`](/Users/pranav/Documents/RT/paper1_geometry/assets/paper1_h2_stress_conversations.jsonl)
- [`papers`](/Users/pranav/Documents/RT/papers)
- [`results/paper1`](/Users/pranav/Documents/RT/results/paper1)
- [`paper2_memory/run_paper2.py`](/Users/pranav/Documents/RT/paper2_memory/run_paper2.py)
- [`paper2_memory/study.py`](/Users/pranav/Documents/RT/paper2_memory/study.py)
- [`paper2_memory/policies.py`](/Users/pranav/Documents/RT/paper2_memory/policies.py)
- [`paper2_memory/plotting.py`](/Users/pranav/Documents/RT/paper2_memory/plotting.py)
- [`paper2_memory/case_analysis.py`](/Users/pranav/Documents/RT/paper2_memory/case_analysis.py)
- [`paper2_memory/memory_critical_analysis.py`](/Users/pranav/Documents/RT/paper2_memory/memory_critical_analysis.py)
- [`paper2_memory/cross_model_memory_summary.py`](/Users/pranav/Documents/RT/paper2_memory/cross_model_memory_summary.py)
- [`paper3_codec/run_paper3.py`](/Users/pranav/Documents/RT/paper3_codec/run_paper3.py)
- [`paper3_codec/policies.py`](/Users/pranav/Documents/RT/paper3_codec/policies.py)
- [`paper3_codec/study.py`](/Users/pranav/Documents/RT/paper3_codec/study.py)
- [`paper3_codec/memory_critical_analysis.py`](/Users/pranav/Documents/RT/paper3_codec/memory_critical_analysis.py)
- [`manuscript`](/Users/pranav/Documents/RT/manuscript)
- [`scripts`](/Users/pranav/Documents/RT/scripts)
- [`notebooks`](/Users/pranav/Documents/RT/notebooks)
- [`results/paper2`](/Users/pranav/Documents/RT/results/paper2)
- [`artifacts`](/Users/pranav/Documents/RT/artifacts)

## Quick Start

```bash
cd /Users/pranav/Documents/RT
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m paper1_geometry.run_paper1 --list-models --detailed-models
```

Run a small smoke test:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.run_paper1 \
  --model-key qwen25_05b \
  --limit-conversations 2 \
  --max-turns 6 \
  --max-input-tokens 1024
```

Outputs are written to [`results/paper1`](/Users/pranav/Documents/RT/results/paper1).

## Manuscript Build

Generate manuscript assets directly from tracked artifact JSON:

```bash
cd /Users/pranav/Documents/RT
python scripts/build_manuscript_assets.py
```

Compile the IEEE-style checkpoint PDF:

```bash
cd /Users/pranav/Documents/RT
bash manuscript/build.sh
```

This writes:

- [`manuscript/build/paper_checkpoint.pdf`](/Users/pranav/Documents/RT/manuscript/build/paper_checkpoint.pdf)

The manuscript is intentionally a checkpoint paper built only from stable, completed result bundles. Ongoing public-benchmark runs should be incorporated only after they finish and are reviewed.

## Paper 2 Quick Start

Run one token-budgeted controller experiment:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper2_memory.run_paper2 \
  --run-name blazing_v2_token_budget \
  --model-key qwen25_05b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets 0.20,0.35,0.50,0.65 \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768
```

This writes:

- per-evaluation outputs in `results/paper2/<run-name>/evaluation_rows.csv`
- aggregate summary in `results/paper2/<run-name>/summary.json`
- markdown report in `results/paper2/<run-name>/report.md`

Run the first multi-model Paper 2 study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper2_memory.study \
  --study-name blazing_study_v1 \
  --model-keys qwen25_05b,qwen25_15b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets 0.20,0.35,0.50,0.65 \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768
```

The Paper 2 study writes:

- per-model run directories under `results/paper2/studies/<study-name>/`
- combined study summary in `study_summary.json`
- bootstrap uncertainty in `confidence_summary.json`
- paired policy-vs-uniform tests in `significance_summary.json`
- behavior bootstrap uncertainty in `behavior_confidence_summary.json`
- behavior paired policy-vs-uniform tests in `behavior_significance_summary.json`
- policy-budget table in `policy_budget_summary.csv`
- behavior policy-budget table in `behavior_policy_budget_summary.csv`
- combined evaluation rows in `evaluation_rows.csv`
- combined behavior rows in `behavior_rows.csv`
- plots under `results/paper2/studies/<study-name>/plots/`

Use the harder Paper 2 behavioral stress set:

- [`paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl`](/Users/pranav/Documents/RT/paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl)
- 9 conversations across `long_dependency`, `retrieval_heavy`, and `code_conversation`
- every late-turn query is designed to depend on earlier facts, formatting rules, or code constraints

Run the hard-set study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper2_memory.study \
  --study-name behavior_stress_v1 \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets 0.20,0.35,0.50 \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768
```

Generate the qualitative and memory-critical bridge reports:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper2_memory.case_analysis \
  --evaluation-csv results/paper2/studies/behavior_stress_qwen_cases/evaluation_rows.csv \
  --behavior-csv results/paper2/studies/behavior_stress_qwen_cases/behavior_rows.csv \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --model-key qwen25_05b \
  --budget-fraction 0.35 \
  --top-n 5 \
  --output-path results/paper2/studies/behavior_stress_qwen_cases/case_analysis_qwen25_05b_b035.md

python -m paper2_memory.memory_critical_analysis \
  --evaluation-csv results/paper2/studies/behavior_stress_qwen_cases/evaluation_rows.csv \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --model-key qwen25_05b \
  --budget-fraction 0.35 \
  --output-path results/paper2/studies/behavior_stress_qwen_cases/memory_critical_qwen25_05b_b035.md
```

## Next Phase On Colab

The next checkpoint is experiment-driven, not architecture-sprawl-driven. The recommended Colab path is:

1. run the Paper 2 competitor matrix with the semantic baseline
2. run the Paper 2 `qwen25_3b` probe
3. run the Paper 2 fairness sweep
4. run the Paper 3 cross-model head-to-head
5. publish the resulting run directories back into tracked `artifacts/`

Use these entry points:

- [`scripts/run_paper2_competitor_matrix.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_competitor_matrix.sh)
- [`scripts/run_paper2_3b_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_3b_probe.sh)
- [`scripts/run_paper2_fairness_sweep.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_fairness_sweep.sh)
- [`scripts/run_paper2_custom_benchmark.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_custom_benchmark.sh)
- [`scripts/run_paper3_head_to_head.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_head_to_head.sh)
- [`scripts/run_paper3_fairness_sweep.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_fairness_sweep.sh)
- [`scripts/run_paper3_3b_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_3b_probe.sh)
- [`scripts/run_paper3_pairwise_report.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_pairwise_report.sh)
- [`scripts/run_paper3_span_ablation.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_span_ablation.sh)
- [`scripts/run_paper3_next_batch.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_next_batch.sh)
- [`scripts/prepare_public_benchmark_jsonl.py`](/Users/pranav/Documents/RT/scripts/prepare_public_benchmark_jsonl.py)
- [`scripts/download_public_benchmark.py`](/Users/pranav/Documents/RT/scripts/download_public_benchmark.py)
- [`scripts/run_paper3_public_benchmark.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_public_benchmark.sh)
- [`scripts/run_paper3_nonqwen_3b_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_nonqwen_3b_probe.sh)
- [`scripts/run_paper3_crossover_sweep.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_crossover_sweep.sh)
- [`scripts/run_paper3_public_solidification.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_public_solidification.sh)
- [`scripts/publish_artifact.sh`](/Users/pranav/Documents/RT/scripts/publish_artifact.sh)
- [`scripts/run_next_phase_suite.sh`](/Users/pranav/Documents/RT/scripts/run_next_phase_suite.sh)
- [`notebooks/rt_next_phase_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_next_phase_runner.ipynb)
- [`notebooks/rt_paper3_next_batch_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_next_batch_runner.ipynb)
- [`notebooks/rt_paper3_public_solidification_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_public_solidification_runner.ipynb)

## Colab Pro

The repo includes one-shot shell runners for Colab:

- `scripts/colab_setup.sh`
- `scripts/run_paper1_final.sh`
- `scripts/run_paper2_hardset.sh`
- `scripts/run_paper2_logged_mechanism.sh`
- `scripts/run_paper3_pilot.sh`

The notebook entry point is:

- [`notebooks/rt_colab_pro_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_colab_pro_runner.ipynb)

The intended Colab flow is:

1. clone the GitHub repo
2. run `scripts/colab_setup.sh`
3. run the Paper 2 hard-set study
4. run the logged mechanism reruns
5. run the Paper 3 pilot

The Paper 3 study now writes:

- `evaluation_rows.csv`
- `behavior_rows.csv`
- `study_summary.json`
- `study_report.md`
- `confidence_summary.json`
- `significance_summary.json`
- `behavior_confidence_summary.json`
- `behavior_significance_summary.json`
- per-model memory-critical reports for `geometry_keep_compress_drop` and `geometry_segment_actions`

Summarize a run directory into JSON and Markdown:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.reporting \
  --results-dir results/paper1 \
  --output-json results/paper1/report.json \
  --output-md results/paper1/report.md
```

Run a small study with aggregation and plots:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name smoke_qwen25 \
  --model-keys qwen25_05b \
  --max-turns 6 \
  --max-input-tokens 768
```

This writes:

- model-level outputs under `results/paper1/studies/<study-name>/<model-key>/`
- study summaries in `study_summary.json`, `study_report.md`, and `conversation_summary.csv`
- plots in `results/paper1/studies/<study-name>/plots/`

Run the first expanded multi-model study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v1 \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --max-turns 6 \
  --max-input-tokens 768
```

The expanded dataset currently contains 18 conversations:

- 3 `casual_chat`
- 3 `multi_topic_chat`
- 3 `retrieval_heavy`
- 3 `reasoning_chat`
- 3 `code_conversation`
- 3 `long_dependency`

For H2 boundary stress tests, use the additional regime-shift pack:

- [`paper1_geometry/assets/paper1_h2_stress_conversations.jsonl`](/Users/pranav/Documents/RT/paper1_geometry/assets/paper1_h2_stress_conversations.jsonl)
- 6 additional conversations with explicit topic, task, or formatting regime changes
- both datasets now support `boundary_indices` so segmentation can be scored against annotated regime changes

Run the current H2-focused multi-model study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v2 \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
```

Current key outputs:

- study report: [`study_report.md`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v2/study_report.md)
- study summary: [`study_summary.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v2/study_summary.json)
- conversation table: [`conversation_summary.csv`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v2/conversation_summary.csv)
- plots: [`results/paper1/studies/expanded_v2/plots`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v2/plots)

Run the current boundary-evaluated study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v3 \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
```

Boundary-evaluation outputs:

- report: [`study_report.md`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v3/study_report.md)
- summary: [`study_summary.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v3/study_summary.json)
- table: [`conversation_summary.csv`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v3/conversation_summary.csv)
- plots: [`results/paper1/studies/expanded_v3/plots`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v3/plots)

Run the current geometry-vs-baseline comparison study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v4 \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
```

Comparison outputs:

- report: [`study_report.md`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v4/study_report.md)
- model summary: [`study_summary.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v4/study_summary.json)
- geometry table: [`conversation_summary.csv`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v4/conversation_summary.csv)
- baseline table: [`baseline_conversation_summary.csv`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v4/baseline_conversation_summary.csv)
- baseline summary: [`baseline_summary.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v4/baseline_summary.json)
- baseline plot: [`baseline_eval_heatmap.png`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v4/plots/baseline_eval_heatmap.png)

Run the benchmark-audited study with corrected macro vs micro reporting:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v5_audit \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
```

Audit outputs:

- report: [`study_report.md`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v5_audit/study_report.md)
- summary: [`study_summary.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v5_audit/study_summary.json)
- benchmark audit: [`benchmark_audit.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v5_audit/benchmark_audit.json)

The `expanded_v5_audit` result is the current trustworthy baseline for Paper 1:

- `H1` is strong: low-rank structure persists across all tested models
- `H3` is strongest: geometric distortion strongly predicts logit distortion
- `H2` is mixed: geometry contains boundary information, but exact localization is brittle and benchmark-sensitive

Run the changepoint/hybrid follow-up study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v6_changepoint \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
```

This version replaces local thresholding with a changepoint-style decoder over low-rank segment cost and hybrid boundary scores, and it adds:

- tolerance-2 and tolerance-3 F1
- nearest-boundary distance
- WindowDiff
- `P_k`
- boundary AUPRC

Run the geometry-vs-lexical ablation study:

```bash
cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v7_ablation \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
```

This study adds a local subspace-shift geometry feature and reports changepoint ablations for:

- `geometry_only`
- `lexical_only`
- `geometry_lexical`

Key outputs:

- report: [`study_report.md`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v7_ablation/study_report.md)
- summary: [`study_summary.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v7_ablation/study_summary.json)
- variant ablation: [`variant_summary.json`](/Users/pranav/Documents/RT/results/paper1/studies/expanded_v7_ablation/variant_summary.json)

## Papers

The paper program is stored in:

- [`papers/README.md`](/Users/pranav/Documents/RT/papers/README.md)
- [`papers/paper1_geometry_of_conversation_state_trajectories.md`](/Users/pranav/Documents/RT/papers/paper1_geometry_of_conversation_state_trajectories.md)
- [`papers/paper2_geometry_guided_adaptive_memory_compression.md`](/Users/pranav/Documents/RT/papers/paper2_geometry_guided_adaptive_memory_compression.md)
- [`papers/paper3_manifold_memory_codecs.md`](/Users/pranav/Documents/RT/papers/paper3_manifold_memory_codecs.md)

## Immediate Next Steps

The current project has moved past the first bootstrap. The next useful steps are:

1. Expand the labeled conversation pack so boundary-local metrics stabilize beyond the current short-horizon setting.
2. Compare changepoint decoding against the audited `expanded_v5_audit` baseline, not against the old macro-only `expanded_v4` interpretation.
3. Add a turn-embedding changepoint baseline if we want one more H2 check; otherwise stop iterating H2 after `expanded_v7_ablation`.
4. Run `qwen3_06b` after upgrading `transformers` and check whether the boundary signal transfers.
5. Start drafting Paper 1 around H1 and H3 as the main results, with H2 presented as a careful secondary analysis and the ablation used to show marginal geometry value.
