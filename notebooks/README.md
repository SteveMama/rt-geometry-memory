# Notebooks

This folder contains notebook entry points for shared runs.

Current notebook:

- [`rt_colab_pro_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_colab_pro_runner.ipynb)
- [`rt_next_phase_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_next_phase_runner.ipynb)
- [`rt_paper3_next_batch_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_next_batch_runner.ipynb)
- [`rt_paper3_public_solidification_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_public_solidification_runner.ipynb)
- [`rt_paper3_quick_benchmarks_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_quick_benchmarks_runner.ipynb)
- [`rt_paper3_low_budget_kcd_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_low_budget_kcd_runner.ipynb)
- [`rt_paper3_semantic_kcd_optimization_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_semantic_kcd_optimization_runner.ipynb)
- [`rt_paper3_query_conditioned_geometry_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_query_conditioned_geometry_runner.ipynb)

Purpose:

- clone the GitHub repo on Colab
- install the project
- run Paper 1, Paper 2, and Paper 3 checkpoint studies
- leave outputs in the run directory for download or later inspection
- run the next competitor-aware Paper 2 and Paper 3 suite
- run the surgical Paper 3 fairness and 3B validation batch
- run the public-benchmark, non-Qwen 3B, and crossover-sweep Paper 3 solidification batch
  - default public benchmark: `LongMemEval-S cleaned`
  - includes `tiny smoke`, `medium subset`, and `full bounded run` sections
- run smaller and quicker benchmark loops for `MSC` and `LoCoMo`-style iteration
- fetch, normalize, and run the auto-supported public benchmarks in one place:
  - `MSC`
  - `LoCoMo10`
  - `LongMemEval`
  - plus manual-source fallback paths for `GapChat`, `REALTALK`, and `EvolMem`
- run the new support-aware and semantic-filtered low-budget KCD variants on `MSC` and `LoCoMo`
- run the semantic-led codec optimization pass with support-aware and budget-aware semantic-KCD variants on `MSC`, `LoCoMo`, and `LongMemEval`
- run the tangent-space query-conditioned geometry comparison on `MSC` and `LoCoMo`
- publish tracked artifacts back into the repository after a Colab run
