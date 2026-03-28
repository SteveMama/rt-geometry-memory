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
- [`publish_artifact.sh`](/Users/pranav/Documents/RT/scripts/publish_artifact.sh)
  - copy an ignored `results/` run into tracked `artifacts/` and optionally commit/push it
- [`run_next_phase_suite.sh`](/Users/pranav/Documents/RT/scripts/run_next_phase_suite.sh)
  - run the full next-phase suite on Colab with one command

Recommended usage:

- use the notebook in [`notebooks`](/Users/pranav/Documents/RT/notebooks) on Colab
- use these scripts directly when you want a reproducible command-line rerun
