# June Fixes: Reviewer-Driven Repairs for the ACL Submission

This directory contains one fix per issue raised in the internal review of
`ACL_manuscript/acl_findings_submission.tex`. Every fix is either a new
runnable experiment (GPU), a CPU-only reanalysis of artifacts you already
have, or a concrete manuscript change.

All GPU work is shardable across multiple GPUs using the same
filesystem-queue pattern as `scripts/run_paper3_gate1_scaleup_multigpu.sh`
(atomic-claim job files, per-GPU `CUDA_VISIBLE_DEVICES` pinning, preflight
checks, resume via `progress.json`). The single entry point for cloud runs is
[`multigpu/run_june_fixes_multigpu.sh`](multigpu/run_june_fixes_multigpu.sh)
and the notebook
[`notebooks/june_fixes_vast_runner.ipynb`](notebooks/june_fixes_vast_runner.ipynb).

## Issue → Fix map

| # | Review issue | Fix | Cost | Where |
|---|--------------|-----|------|-------|
| 1 | No task accuracy anywhere (only logit L2 / NLL) | Generation-based QA accuracy (EM / token-F1 / contains) for every policy×budget, replayed from tracked `evaluation_rows.csv` selections | GPU | `qa_accuracy/` |
| 2 | Oracle Gate 1 circularity (logit-defined harm favors geometry features) | Recompute all Gate 1 ranking metrics against **answer-NLL-defined harm only**, side-by-side with the logit-harm view | CPU | `answer_harm_oracle/` |
| 3 | No external / trivial baselines (no recency, no published compressor) | `recency`, `random`, `lexical_tfidf`, and optional `llmlingua2` policies emitting the standard `evaluation_rows.csv` schema | GPU | `baselines/` |
| 4 | p-values near 0.05, no multiple-comparison correction | Benjamini–Hochberg + Holm sweep over every `*significance_summary.json`, with a survives/dies table | CPU | `stats/` |
| 5 | Core signal-swap ablation only on one model family (Qwen) | Signal-swap study on `smollm2_17b` and `llama32_3b` (non-Qwen) | GPU | `crossfamily/` |
| 6 | LongMemEval slice too small / truncated at 40 turns | Scale-up wrapper: more conversations, 80-turn truncation, via the existing multi-GPU scale-up script | GPU | `longmemeval_scaleup/` |
| 7 | Title promises a regime *selector* the paper doesn't deliver | Supervised regime detector trained on oracle candidate features, with cross-validated accuracy and confusion matrix | CPU | `regime_detector/` |
| 8 | Missing implementation details (semantic signal is hidden-state cosine, no embedding model; shortlist factor; risk weights), sign-convention flips, rank-95 defined late, single figure, missing related work, no data statement | Generated implementation-details appendix, regime-map figure, concrete tex edit list, bib additions, anonymized supplementary bundle | CPU | `manuscript/` |

## Execution order (cloud GPU box)

```bash
# 0. setup (repo root, with .venv active and benchmarks prepared)
pip install -e .

# 1. all GPU work, sharded across every visible GPU, resumable
bash june_fixes/multigpu/run_june_fixes_multigpu.sh

# 2. CPU reanalyses (run after GPU studies exist; all are idempotent)
python -m june_fixes.answer_harm_oracle.answer_harm_gate1 \
  --candidate-rows results/paper3/harm_oracle/<oracle-study>/candidate_rows.csv \
  --benchmark-name msc_valid \
  --output-dir results/june_fixes/answer_harm_oracle/msc_valid

python -m june_fixes.stats.multiple_comparisons \
  --search-roots results,artifacts,paper3_gate1_scaleup_multigpu_merged_results \
  --output-dir results/june_fixes/multiple_comparisons

python -m june_fixes.regime_detector.regime_detector \
  --labeled-csv hardset=<hardset-candidate_rows.csv> \
  --labeled-csv msc=<msc-candidate_rows.csv> \
  --labeled-csv longmemeval=<lme-candidate_rows.csv> \
  --output-dir results/june_fixes/regime_detector

# 3. manuscript assets
python june_fixes/manuscript/implementation_details_appendix.py
python june_fixes/manuscript/regime_map_figure.py
bash june_fixes/manuscript/make_anonymous_bundle.sh
```

Then work through [`manuscript/manuscript_edits.md`](manuscript/manuscript_edits.md),
which lists the editorial changes (sign conventions, rank-95 definition,
section retitles, data statement, title options) keyed to line numbers in
`acl_findings_submission.tex`.

## Results already obtained on this machine (2026-06-10, CPU fixes)

- **Fix 2 (circularity) — CONFIRMED, action required.** Under logit-defined
  harm, geometry's Gate 1 gain is huge (MSC Δτ +0.32–0.39, LME +0.56–0.83,
  conv-level p < 0.01). Under **answer-NLL-defined harm** the gain collapses:
  MSC Δτ +0.01–0.04 (p = 0.23–0.97, fails the +0.03 gate at every budget on
  conversation level), and on LME geometry's Δτ turns **negative**
  (−0.12 to −0.15). The manuscript's Gate 1 oracle claims (§4.4, §6, abstract)
  measure metric alignment, not memory value, and must be rewritten before
  submission. See `results/june_fixes/answer_harm_oracle/*/answer_harm_gate1_report.md`.
- **Fix 4 (multiplicity):** 1,273 tests collected across all tracked
  summaries; 283 raw-significant → 116 survive within-family BH at q<0.05
  (167 killed). Check the manuscript's headline cells against
  `results/june_fixes/multiple_comparisons/corrected_pvalues.csv`.
- **Fix 7 (regime detector):** 5-fold CV accuracy 1.000 vs 0.696 majority
  baseline on 46 conversations — but the hardset class has only 2
  conversations (smoke oracle). Re-run the hardset oracle probe at full scale
  (`scripts/run_paper3_harm_oracle_probe.sh` on the 36-conversation stress
  set) and retrain before citing this number.
- Fix 8 assets generated: `ACL_manuscript/generated/table_implementation_details.tex`,
  `ACL_manuscript/figures/regime_map.png`.

## Notes and caveats

- **QA accuracy replays tracked selections.** `qa_accuracy_study.py` does not
  re-run policies; it reads `retained_turn_indices` from an existing study's
  `evaluation_rows.csv` and generates answers under exactly those retained
  contexts. This guarantees the QA numbers describe the same runs the paper
  already reports. It evaluates every behavior-eligible target turn (user turn
  followed by a gold assistant turn), matching how `behavior_rows.csv` is built.
- **The answer-harm oracle fix needs no GPU.** `candidate_rows.csv` from the
  Gate 1 scale-up already contains `delta_answer_avg_neg_logprob_delta` and
  `has_behavior_label` per row, so the circularity question is answerable from
  artifacts you already have.
- **`llama32_3b` is gated on Hugging Face** — export `HF_TOKEN` before the
  cross-family job, or it will be skipped with a warning.
- **`llmlingua2` baseline is optional** (`pip install llmlingua`). If the
  import fails the policy is skipped and noted in the summary, never fatal.
- Every GPU study honors `EXTRACT_CACHE_ROOT` and `EXTRACT_BATCH_SIZE` and
  writes `progress.json` checkpoints, so killed instances resume cleanly.
