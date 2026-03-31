# Manuscript

This directory packages the current RT checkpoint as a manuscript-grade bundle.

Main entry points:

- [`paper_checkpoint.tex`](/Users/pranav/Documents/RT/manuscript/paper_checkpoint.tex): IEEE-style checkpoint manuscript
- [`generated`](/Users/pranav/Documents/RT/manuscript/generated): tables generated directly from tracked artifact JSON
- [`figures`](/Users/pranav/Documents/RT/manuscript/figures): packaged manuscript figures
- [`build.sh`](/Users/pranav/Documents/RT/manuscript/build.sh): one-command local build
- [`../papers/rt_derivation_ledger.md`](/Users/pranav/Documents/RT/papers/rt_derivation_ledger.md): canonical deep-dive companion for derivations, benchmark structure, implementation mapping, and falsifications
- [`../papers/generated/rt_experiment_matrix.csv`](/Users/pranav/Documents/RT/papers/generated/rt_experiment_matrix.csv): machine-readable experiment inventory
- [`../papers/generated/rt_negative_result_matrix.csv`](/Users/pranav/Documents/RT/papers/generated/rt_negative_result_matrix.csv): machine-readable negative-result inventory

## Build

Generate manuscript assets from tracked results:

```bash
cd /Users/pranav/Documents/RT
python scripts/build_manuscript_assets.py
```

Generate the companion ledger inventories:

```bash
cd /Users/pranav/Documents/RT
python scripts/build_research_ledger_assets.py
```

Compile the PDF:

```bash
cd /Users/pranav/Documents/RT
bash manuscript/build.sh
```

The expected output is:

- [`paper_checkpoint.pdf`](/Users/pranav/Documents/RT/manuscript/build/paper_checkpoint.pdf)

## Scope

This manuscript is intentionally a checkpoint paper, not a final trilogy submission.
It packages the strongest stable results so far:

- Paper 1 frozen characterization
- Paper 2 geometry-aware memory control
- Paper 3 fairness-controlled sparse codec plus 3B validation

Ongoing public-benchmark runs should be added only after they finish and are reviewed.
The appendix bridge in the manuscript points into the ledger rather than duplicating the full derivation history inside the PDF.
