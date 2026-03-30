# Paper 3 Codec

This package contains the current Paper 3 pilot code.

Core responsibilities:

- sparse keep/compress/drop memory policy
- segment-action baseline comparison
- behavior evaluation on the hard stress set
- confidence, significance, and mechanism analysis

Main entry points:

- [`run_paper3.py`](/Users/pranav/Documents/RT/paper3_codec/run_paper3.py)
- [`study.py`](/Users/pranav/Documents/RT/paper3_codec/study.py)
- [`policies.py`](/Users/pranav/Documents/RT/paper3_codec/policies.py)
- [`memory_critical_analysis.py`](/Users/pranav/Documents/RT/paper3_codec/memory_critical_analysis.py)
- [`pairwise_analysis.py`](/Users/pranav/Documents/RT/paper3_codec/pairwise_analysis.py)

Current checkpoint:

- `geometry_keep_compress_drop` is now budget-responsive and genuinely uses compression
- `geometry_segment_actions` remains a strong competing family
- next experiment track: fairness sweep, direct pairwise policy significance, and a 3B Paper 3 probe
- the latest tracked Paper 3 checkpoint is [`artifacts/paper3/paper3_pilot_v3_full`](/Users/pranav/Documents/RT/artifacts/paper3/paper3_pilot_v3_full)

Current solidification track:

- recommended first public benchmark: `LongMemEval-S cleaned`
- one normalized public benchmark on Qwen using `uniform`, `semantic`, `geometry`, `geometry_segment_actions`, and `geometry_keep_compress_drop`
- one semantic-codec comparison using `uniform`, `semantic`, `geometry`, `geometry_keep_compress_drop`, and `semantic_keep_compress_drop`
- one low-budget upgrade comparison using:
  - `support_aware_geometry_keep_compress_drop`
  - `semantic_filtered_geometry_keep_compress_drop`
- one non-Qwen 3B hard-set probe using `uniform`, `geometry`, and `geometry_keep_compress_drop`
- one dense crossover sweep over the hard set plus the public benchmark using `uniform`, `semantic`, `geometry`, and `geometry_keep_compress_drop`

Use:

- [`benchmarks/README.md`](/Users/pranav/Documents/RT/benchmarks/README.md)
- [`benchmarks/quick_benchmark_plan.md`](/Users/pranav/Documents/RT/benchmarks/quick_benchmark_plan.md)
- [`scripts/download_public_benchmark.py`](/Users/pranav/Documents/RT/scripts/download_public_benchmark.py)
- [`scripts/prepare_public_benchmark_jsonl.py`](/Users/pranav/Documents/RT/scripts/prepare_public_benchmark_jsonl.py)
- [`scripts/run_paper3_quick_benchmark.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_quick_benchmark.sh)
- [`scripts/run_paper3_low_budget_kcd_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_low_budget_kcd_probe.sh)
- [`scripts/run_paper3_public_benchmark.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_public_benchmark.sh)
- [`scripts/run_paper3_nonqwen_3b_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_nonqwen_3b_probe.sh)
- [`scripts/run_paper3_crossover_sweep.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_crossover_sweep.sh)
- [`scripts/run_paper3_public_solidification.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_public_solidification.sh)
- [`notebooks/rt_paper3_public_solidification_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_public_solidification_runner.ipynb)
- [`notebooks/rt_paper3_quick_benchmarks_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_quick_benchmarks_runner.ipynb)
- [`notebooks/rt_paper3_low_budget_kcd_runner.ipynb`](/Users/pranav/Documents/RT/notebooks/rt_paper3_low_budget_kcd_runner.ipynb)

Notebook run order:

1. `tiny smoke`
2. `medium public-benchmark subset`
3. `full bounded solidification batch`

Checkpoint manuscript:

- [`manuscript/paper_checkpoint.tex`](/Users/pranav/Documents/RT/manuscript/paper_checkpoint.tex)
- [`manuscript/build/paper_checkpoint.pdf`](/Users/pranav/Documents/RT/manuscript/build/paper_checkpoint.pdf)
