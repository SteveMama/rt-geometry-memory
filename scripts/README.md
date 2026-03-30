# Scripts

These shell scripts are the primary rerun entry points for local machines and Colab.

Included:

- [`colab_setup.sh`](/Users/pranav/Documents/RT/scripts/colab_setup.sh)
  - install the project and verify the runtime
- [`colab_commit_push.sh`](/Users/pranav/Documents/RT/scripts/colab_commit_push.sh)
  - configure repo-local git identity on Colab and push tracked artifacts with a GitHub token
- [`run_paper1_final.sh`](/Users/pranav/Documents/RT/scripts/run_paper1_final.sh)
  - rerun the frozen Paper 1 study
- [`run_paper2_hardset.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_hardset.sh)
  - run the hard stress-set Paper 2 study
- [`run_paper2_logged_mechanism.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_logged_mechanism.sh)
  - rerun logged Paper 2 mechanism analyses and cross-model summaries
- [`run_paper3_pilot.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_pilot.sh)
  - run the full Paper 3 pilot study and emit memory-critical reports
- [`run_paper2_competitor_matrix.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_competitor_matrix.sh)
  - run the hard stress-set competitor matrix with `semantic`, `lexical`, `geometry`, and segment-action policies
- [`run_paper2_3b_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_3b_probe.sh)
  - run the first larger-model Paper 2 probe on `qwen25_3b`
- [`run_paper2_fairness_sweep.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_fairness_sweep.sh)
  - run a tighter budget sweep for token-fraction fairness analysis
- [`run_paper2_custom_benchmark.sh`](/Users/pranav/Documents/RT/scripts/run_paper2_custom_benchmark.sh)
  - run the Paper 2 study against any compatible benchmark JSONL without changing code
- [`run_paper3_head_to_head.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_head_to_head.sh)
  - rerun the Paper 3 cross-model head-to-head and emit memory-critical reports
- [`run_paper3_pairwise_report.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_pairwise_report.sh)
  - compute direct pairwise significance between `geometry`, `geometry_segment_actions`, and `geometry_keep_compress_drop`
- [`run_paper3_fairness_sweep.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_fairness_sweep.sh)
  - run the Paper 3 fairness sweep and emit pairwise and mechanism reports
- [`run_paper3_3b_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_3b_probe.sh)
  - run the first 3B Paper 3 probe and emit pairwise and mechanism reports
- [`run_paper3_span_ablation.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_span_ablation.sh)
  - run a segment-span ablation for the current Paper 3 families
- [`run_paper3_next_batch.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_next_batch.sh)
  - run the next surgical Paper 3 batch: fairness sweep plus 3B probe
- [`prepare_public_benchmark_jsonl.py`](/Users/pranav/Documents/RT/scripts/prepare_public_benchmark_jsonl.py)
  - normalize MSC-, LoCoMo-, LongMemEval-, or already-normalized benchmark files into the project's JSONL conversation format
- [`download_public_benchmark.py`](/Users/pranav/Documents/RT/scripts/download_public_benchmark.py)
  - download the auto-supported public benchmarks: `LongMemEval`, `MSC`, and `LoCoMo10`
- [`run_paper3_quick_benchmark.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_quick_benchmark.sh)
  - run a bounded quick benchmark loop for smaller datasets such as MSC or LoCoMo
- [`run_paper3_semantic_codec_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_semantic_codec_probe.sh)
  - run the semantic-codec Paper 3 comparison with `uniform`, `semantic`, `geometry`, `geometry_keep_compress_drop`, and `semantic_keep_compress_drop`
- [`build_manuscript_assets.py`](/Users/pranav/Documents/RT/scripts/build_manuscript_assets.py)
  - generate manuscript tables and packaged figures directly from tracked artifact JSON
- [`run_paper3_public_benchmark.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_public_benchmark.sh)
  - run the Paper 3 policy family on one normalized public benchmark with a Qwen model
- [`run_paper3_nonqwen_3b_probe.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_nonqwen_3b_probe.sh)
  - run the non-Qwen 3B hard-set confirmation with `uniform`, `geometry`, and `geometry_keep_compress_drop`
- [`run_paper3_crossover_sweep.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_crossover_sweep.sh)
  - run a dense crossover sweep over the hard set plus one public benchmark using `uniform`, `semantic`, `geometry`, and `geometry_keep_compress_drop`
- [`run_paper3_public_solidification.sh`](/Users/pranav/Documents/RT/scripts/run_paper3_public_solidification.sh)
  - run the full recommended Paper 3 solidification order: public benchmark, non-Qwen 3B, and crossover sweep
- [`publish_artifact.sh`](/Users/pranav/Documents/RT/scripts/publish_artifact.sh)
  - copy an ignored `results/` run into tracked `artifacts/` and optionally commit/push it
- [`run_next_phase_suite.sh`](/Users/pranav/Documents/RT/scripts/run_next_phase_suite.sh)
  - run the full next-phase suite on Colab with one command

Recommended usage:

- use the notebook in [`notebooks`](/Users/pranav/Documents/RT/notebooks) on Colab
- use these scripts directly when you want a reproducible command-line rerun
- normalize public benchmarks into one JSONL before using the new public-benchmark runners
- generate manuscript assets before building the checkpoint PDF in [`manuscript`](/Users/pranav/Documents/RT/manuscript)
