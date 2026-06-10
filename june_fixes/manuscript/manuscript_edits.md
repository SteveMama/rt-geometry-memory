# Manuscript Edits for `ACL_manuscript/acl_findings_submission.tex`

Concrete editorial fixes from the internal review, keyed to current line
numbers. Apply after the experiment fixes land so the new numbers can be
filled in.

## 1. Title (line 28) — the paper has no automatic selector

Current title promises *signal-conditioned* compression, but regime selection
is experimenter-chosen and the unsupervised regime atlas is a documented
negative result.

- **If the regime detector (`june_fixes/regime_detector`) clearly beats the
  majority baseline:** keep the title, add a short "Regime Detection"
  subsection citing its CV accuracy, and cite the detector as the conditioning
  mechanism.
- **If it does not:** retitle. Options:
  - "Which Signal Should Score Conversational Memory? Geometry, Semantics,
    and Budgeted Keep/Compress/Drop Compression"
  - "Geometric and Semantic Signals for Budgeted Conversational Memory
    Compression: An Empirical Regime Study"

## 2. Define rank-95 at first use (Section 3.1, ~line 88)

"rank-95" is used in §3.1 but only loosely defined in §8/Appendix A. Insert
after the sentence ending "…measured in a common frame." (line 86):

> Throughout, *rank-95* of a segment denotes the smallest number of singular
> values of the transported-increment matrix $U=[u_1,\dots]$ whose cumulative
> squared singular values reach 95\% of $\lVert U\rVert_F^2$.

## 3. Specify the semantic signal — and own the consequence (Section 3.3, line 105)

The paper never says what `semantic` scoring is. It is **cosine similarity in
the model's own hidden-state space** (`paper2_memory/policies.py:
turn_semantic_risk`) — there is no external embedding model. Reviewers will
assume a sentence-transformer; correcting that assumption late looks bad.
Insert into §3.3 (and cross-reference the new implementation appendix):

> The semantic score of turn $t$ for target $t^\star$ is the cosine
> similarity $\cos(z_t, z_{t^\star})$ between last-token hidden states; no
> external embedding model is used. This keeps all signals in the same
> representation space, but it also means our semantic baseline is not a
> retrieval-grade text embedding; the lexical TF--IDF baseline
> (\S\ref{sec:baselines}) controls for this.

(The TF-IDF baseline is produced by `june_fixes/baselines/`.)

## 4. Unify sign conventions (Tables 2, 5, 7)

Table 2 (`tab:signal_swap`, line 198) says "Positive $\Delta$ means
semantic\_KCD is worse"; Table 5 (`tab:msc_policy`, line 268) says "Negative
deltas favor the left policy"; Table 7 (`tab:3b`, line 313) flips back.
Standardize on **$\Delta$ = comparison policy $-$ reference policy; negative
favors the comparison policy**, restate each table's pair in its caption, and
flip the signs in Tables 2 and 7 accordingly (i.e., report
geometry$-$semantic so improvements are negative everywhere).

## 5. Soften the Table 1 prose (line 173)

"geometry-aware retention improves over uniform retention across compact
models" — but `qwen25_15b` @0.35 has $p=0.26$ and all three NLL deltas are
non-significant. Rewrite:

> On the hard stress set, geometry-aware retention improves logit-space
> fidelity over uniform retention at most model-budget combinations, with
> the largest and most reliable gains on \texttt{qwen25\_05b}; answer-level
> NLL deltas at this stage are directionally favorable but not individually
> significant.

## 6. Add the QA accuracy results (new subsection in §4/§5/§6)

After `june_fixes/qa_accuracy` runs, add a table of EM / token-F1 / contains
per policy x budget for the hard set and MSC, plus a sentence in each results
section tying NLL deltas to generation accuracy. Reviewer objection #1 is
"does NLL improvement translate into correct answers" — answer it with data
either way. Hook: `results/june_fixes/qa_accuracy/<study>/qa_report.md`.

## 7. Address the oracle circularity head-on (Section 4.4, line 165)

After the Gate 1 description, add the answer-harm robustness result from
`june_fixes/answer_harm_oracle`:

> Because logit-defined harm could structurally favor geometric features
> (geometric distortion correlates 0.989--0.994 with logit drift), we
> recompute all ranking metrics against an answer-NLL-defined harm target,
> restricted to candidates with behavior labels. [Report whether
> $\Delta\tau$ survives; cite Table X.]

Also add the threshold sensitivity table (same script) to satisfy the
runbook's own requirement (`papers/paper3_gate1_real_runbook.md:57`).

## 8. Multiple-comparison statement (Section 4.5, line 168)

Append to the Statistical Protocol paragraph:

> Across the full grid of budgets, policies, and metrics we additionally
> report Benjamini--Hochberg q-values within each study x metric x level
> family; results flagged $\dagger$ do not survive FDR control at
> $q<0.05$ and are treated as directional.

Then flag the affected cells using
`results/june_fixes/multiple_comparisons/corrected_pvalues.csv` (the
near-threshold ones: signal-swap 0.35 logit $p=0.079$; 3B 0.20/0.35
$p=0.048/0.050$; MSC query-conditioned $p=0.047$).

## 9. Cross-family ablation (Section 7, line 308)

After `june_fixes/crossfamily` runs, extend the 3B section: report the
signal swap on `smollm2_17b` and `llama32_3b`, and retitle the section
"Model-Scale and Model-Family Validation". If Llama results differ, say so
plainly — the current single-family scope is in the limitations anyway.

## 10. LongMemEval honesty (Section 6.2, line 285)

If the `longmemeval_scaleup` run (40 conversations, 80-turn cap) replaces the
12x40 slice, update Table 6 and drop the "truncated slice" caveats that no
longer apply. If not run in time, change the §6.2 framing from "second public
check" to "preliminary check" and move the 12-conversation caveat into the
first sentence rather than the last.

## 11. Add the regime-map figure (Section 9.1, line 350)

Insert `figures/regime_map.png` (from `june_fixes/manuscript/
regime_map_figure.py`) as Figure 2, referenced from Main Takeaways. The
regime map is the paper's thesis; it should not be prose-only.

## 12. Add missing related work (Section 2)

Add to §2 using `related_work_additions.bib`:

- **Budgeted retention:** StreamingLLM / attention sinks (`xiao2024streaming`)
  next to Scissorhands/H2O (line 67).
- **Representation geometry:** intrinsic-dimension and hidden-state geometry
  literature (`ansuini2019intrinsic`, `valeriani2023geometry`,
  `park2024linear`) — §3/§8 currently float with no positioning against prior
  geometry-of-representations work.
- **Memory benchmarks already cited** are fine.

## 13. Data statement + reproducibility (after Limitations, line 359)

Add a short section/paragraph:

> All experiments use publicly released benchmarks (MSC, LoCoMo,
> LongMemEval) under their original licenses. The hard stress set
> (36 synthetic conversations), all policy implementations, and evaluation
> code are included in the anonymized supplementary material and will be
> released under an open license.

Build the bundle with `june_fixes/manuscript/make_anonymous_bundle.sh` and
attach it to the ARR submission. Also fill in the Responsible NLP checklist
item on data licenses.

## 14. Include the implementation appendix

`june_fixes/manuscript/implementation_details_appendix.py` writes
`ACL_manuscript/generated/table_implementation_details.tex`. Add
`\input{generated/table_implementation_details}` as a new appendix section
"Implementation Details", and cite it from §3.3 and §4.1.
