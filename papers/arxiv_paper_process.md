# arXiv Paper — Complete Process Document
**Project:** Signal-Conditioned Memory Compression for Multi-Turn Conversations (KCD Codec)  
**Target categories:** cs.CL (primary), cs.LG (cross-list)  
**Date written:** 2026-06-20

---

## Part 1 — What an arXiv / NLP Paper Needs

### 1.1 Mandatory sections (hard failures without these)

| Section | Purpose | Notes |
|---|---|---|
| Abstract | Self-contained summary: motivation → gap → method → findings → implication | Write last. ≤250 words. ≤1920 chars for arXiv metadata field. |
| Introduction | Why this problem, why now, what we do, what we find, roadmap | Answer "why" before "how". State 3–4 numbered contributions explicitly. |
| Related Work | Theme-grouped, not a list; must identify the gap your work fills | Never plagiarize. Group: prompt compression, hidden-state geometry, memory benchmarks. |
| Method | Mathematical formulation of KCD codec, three signals, how sem_qcg works | Self-contained. Define all notation. Running example helps. |
| Experiments | Datasets, models, baselines, metrics, all hyperparameters | Must be reproducible. Report ALL results including unflattering ones. |
| Results | Tables + narrative. State what is significant, at what level. | Use signflip p-values. Row-level vs conversation-level clearly labeled. |
| Analysis / Discussion | Why results look the way they do. Mechanism. Scope conditions. | This is where geometry-behavior divergence and scope conditions go. |
| Limitations | Explicit section, not buried. Required for ARR/ACL; good practice for arXiv. | Architecture-dependence, length-dependence, oracle metric circularity. |
| Conclusion | Short. Restate main finding. No new claims. | 1–2 paragraphs. |
| References | All cited work. Published versions preferred over preprints. | Use ACL Anthology for NLP papers. |

### 1.2 Strongly recommended (avoid desk rejection at any venue)

- **Ethics / Broader Impact statement** — brief; confirm no harmful use, dataset license compliance
- **Reproducibility checklist** (ACL/EMNLP require it in submission; good to include in arXiv preprint too)
- **Appendix** — hyperparameters, extended tables, proof sketches, additional ablations
- **Code/data availability statement** — even if code isn't released yet, say so
- **Acknowledgments** — compute credits (RunPod), funding if any

### 1.3 ARR reviewer scoring axes (what matters for conference track)

Three scores: **Soundness** (technical rigor, reproducibility), **Excitement** (novelty, interest), **Overall** (composite). Soundness is the hard gate — a low soundness score cannot be rescued by excitement.

What kills soundness:
- M2: Reproducibility details missing (hyperparams, seeds, dataset splits)
- M4: Unmotivated model/dataset selection
- R1: No statistical significance testing
- R2: Overclaimed generalizations beyond what experiments support
- R5: Missing significance tests

What kills excitement:
- G1: Unclear research question (what exactly is the claim?)
- G3: Missing or misrepresented related work
- Contribution is presented as bigger than it is (overselling)

### 1.4 Style rules (NLP community conventions)

- Lowercase "large language models", "transformer", "attention"
- Tables use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`)
- `\citep{}` for parenthetical, `\citet{}` for textual citations
- Figures: vector PDF preferred; no raster for diagrams
- Short sentences. Split anything over 30 words.
- No "novel", "distinct", "utilize" — say "new", "different", "use"
- Bold best result per row in result tables
- Report: mean ± std or CI, not just point estimates

---

## Part 2 — Our Paper's Specific Structure

### 2.1 Title options

1. **Signal-Conditioned Memory Compression for Multi-Turn Conversations** (current)
2. **Geometry-Behavior Divergence in Conversational Memory Compression**
3. **Keep, Compress, or Drop: Signal-Conditioned Turn Selection for Conversational Memory**

Option 3 is most concrete and searchable. Option 2 foregrounds the key finding.

### 2.2 Abstract (draft template)

> Multi-turn conversation compression must balance two distinct objectives: preserving the geometric structure of the model's internal representation (geometric fidelity) and maintaining behavioral output quality. We show these objectives can diverge — methods optimizing only answer quality (e.g., LongLLMLingua) incur logit L2 penalties of +103–161 units relative to uniform sampling while providing only marginal NLL improvement. We introduce the Keep/Compress/Drop (KCD) codec, a sparse ternary turn-selection framework, and study three scoring signals: geometry-based, semantic-based, and query-conditioned geometry within a semantic shortlist (sem_qcg). On Multi-Session Conversations (MSC, 1,000 conversations, 163k evaluation rows), sem_qcg achieves lower logit L2 than uniform sampling at budget 0.50 — more faithful than random retention — while simultaneously improving answer NLL (p<0.001). On a hard constraint-critical stress set, geometry-based scoring beats semantic scoring by +1.1 nats NLL (p<0.001 at 3B scale). We identify two scope conditions: the geometry advantage is architecture-dependent (present on Qwen2.5 but reversed on Llama-3.2-3B) and length-dependent (degrades on 400–600 turn conversations). These results establish that turn scoring signal is a first-class design decision in conversational memory compression.

### 2.3 Contributions (numbered list in Introduction)

1. We demonstrate geometry-behavior divergence at scale: methods optimizing NLL can catastrophically degrade hidden-state geometric fidelity, motivating dual-objective evaluation.
2. We introduce sem_qcg, a query-conditioned geometry signal within a semantic shortlist, and show it is the only KCD variant achieving simultaneously best geometric fidelity AND behavioral quality on MSC.
3. We characterize two scope conditions (architecture-dependence, length-dependence) that bound where geometry-based signals add value.
4. We provide a support-turn rescue mechanism account explaining *why* geometry works on constraint-critical memory: user instructions produce larger hidden-state displacement than semantically similar assistant echoes.

### 2.4 Full section plan

```
1. Introduction
   1.1 The compression design space
   1.2 Geometry-behavior divergence (the core observation)
   1.3 This paper's approach and contributions

2. Background
   2.1 Conversation state geometry (Rank-95, transported motion)
   2.2 The KCD codec (ternary action space, budget constraint)
   2.3 Scoring signals: geometry, semantic, query-conditioned

3. Geometry-Behavior Divergence
   3.1 Experimental setup (MSC 1000-conv, Qwen2.5-1.5B)
   3.2 Results: LongLLMLingua degrades geometry, improves NLL
   3.3 Implication: dual-objective evaluation is necessary

4. Signal Variant Comparison
   4.1 Setup (MSC 50-conv, three KCD variants)
   4.2 Results: sem_qcg wins both metrics simultaneously at budget ≥ 0.35
   4.3 Mechanism: query-conditioned geometry explains the improvement

5. Constraint-Critical Memory: Hard Stress Set
   5.1 Setup and motivation for synthetic stress set
   5.2 Signal-swap results (geometry vs semantic, 1.5B and 3B)
   5.3 Support-turn rescue analysis (mechanism validation)

6. Scope Conditions
   6.1 Architecture dependence: Llama-3.2-3B reversal
   6.2 Length dependence: LongMemEval-S 100-conv
   6.3 When geometry adds value (and when it doesn't)

7. Related Work
   7.1 Prompt and context compression
   7.2 Conversational memory and long-context evaluation
   7.3 Hidden-state geometry in language models

8. Discussion
   8.1 Implications for memory compression system design
   8.2 Geometry as a complementary signal, not a replacement

9. Limitations

10. Conclusion
```

### 2.5 Key tables to include

| Table | Content | Location |
|---|---|---|
| Table 1 | Geometry characterization (Rank-95, Corr, 3 models) | Background / §2.1 |
| Table 2 | MSC 1000-conv: all baselines, logit L2 + NLL, 3 budgets | §3.2 |
| Table 3 | MSC 50-conv: KCD signal variants, logit L2 + NLL, 3 budgets | §4.2 |
| Table 4 | Pairwise: sem_qcg vs semantic_KCD, all budgets, row+conv p-values | §4.2 |
| Table 5 | Hard stress set: signal-swap, 1.5B + 3B, all budgets | §5.2 |
| Table 6 | Support-turn rescue counts (geometry vs semantic, 36 conversations) | §5.3 |
| Table 7 | LME 100-conv: all KCD variants + pairwise vs sem | §6.2 |
| Table 8 | Llama-3.2-3B: semantic vs geometry, logit L2 + NLL, all budgets | §6.1 |

### 2.6 Key figures to include

| Figure | What it shows | Format |
|---|---|---|
| Fig 1 | Geometry-behavior divergence schematic: two objectives pulling apart | Conceptual diagram |
| Fig 2 | Rank-95 trajectory comparison (real vs shuffled) for one conversation | Line plot |
| Fig 3 | Budget vs logit L2 curves for all MSC signal variants | Multi-line plot |
| Fig 4 | Budget vs answer NLL curves for all MSC signal variants | Multi-line plot |
| Fig 5 | Support-turn rescue bar chart: geometry vs semantic | Bar chart |
| Fig 6 | Scope condition map: 2×2 grid (architecture × conversation length) | Heatmap / schematic |

---

## Part 3 — Related Work to Cite

### 3.1 Prompt and context compression

| Paper | Key claim | Cite for |
|---|---|---|
| LongLLMLingua (Jiang et al., 2023) arXiv:2310.06839 | Perplexity-guided token-level compression for long context | Our adversarial baseline; geometry-behavior divergence |
| LLMLingua-2 (Pan et al., 2024) | Data distillation for task-agnostic prompt compression | Semantic compression baseline family |
| R³Mem (arXiv:2502.15957) | Reversible compression balancing retention vs retrieval | Related approach to compression trade-offs |
| Pretraining Context Compressor (ACL 2025) | Trained token-level compressor | Trained compression baseline |

### 3.2 Conversational memory and benchmarks

| Paper | Key claim | Cite for |
|---|---|---|
| LongMemEval (Wu et al., 2024) arXiv:2410.10813 — ICLR 2025 | 500-question benchmark for 5 long-term memory abilities | Our cross-benchmark dataset |
| MSC / Multi-Session Conversations (Xu et al., 2021) | Persona-grounded multi-session dialogue dataset | Our main training/eval dataset |
| LoCoMo (Maharana et al., 2024) | Very-long-term conversational memory evaluation | Related benchmark |
| Memory for LLM Agents survey (Luo et al., 2026) arXiv:2603.07670 | Survey: context-resident, RAG, reflective, hierarchical | Background / related work overview |
| HyMem (arXiv:2602.13933) | Hybrid memory architecture with dynamic retrieval | Related memory architecture |

### 3.3 Hidden-state geometry and representation

| Paper | Key claim | Cite for |
|---|---|---|
| Platonic Representation Hypothesis (Huh et al., 2024) | Convergence of representation across models | Motivation for hidden-state geometry as signal |
| Geometry of LLM representations (various) | Anisotropy, low intrinsic dimension of transformer activations | Background §2.1 |
| Attention sink papers (Xiao et al., 2023) | First few tokens accumulate disproportionate attention mass | Sink-corrected attention weights in our extractor |

---

## Part 4 — arXiv Submission Process (Step by Step)

### Step 1: Account and endorsement
- Register at arxiv.org if not already done
- First-time cs.CL submitters may need endorsement from an established author
- Institutional email (university/research lab) often auto-endorses
- Endorsement request: find a recent cs.CL author and ask; provide title + abstract

### Step 2: Choose LaTeX template
- For standalone arXiv preprint: use `\documentclass[11pt]{article}` with `\usepackage{arxiv}` or `neurips_2025` style (commonly used for arXiv preprints in ML)
- For ACL Findings track: use official ACL style files from acl-org GitHub
- Template decision: **use ACL 2025 style** since we're targeting ACL Findings — arXiv preprint can use the same compiled PDF

### Step 3: Write the paper (see Part 2 above)
- Order of writing: Method → Results → Experiments → Discussion → Related Work → Introduction → Abstract
- **Never write the abstract or introduction first**
- Draft each section, let it sit 24h, revise
- Get feedback from at least one person outside the project before submitting

### Step 4: Local compilation
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
This generates the `.bbl` file arXiv needs. **Include the `.bbl` in the submission package.**

### Step 5: Package for upload
Create a zip:
```
kcd_arxiv_submission.zip
├── main.tex
├── main.bbl          # compiled bibliography — required
├── acl_natbib.bst    # or whichever .bst you use
├── acl.sty           # style file if not system-standard
├── figures/
│   ├── divergence_schematic.pdf
│   ├── rank95_trajectory.pdf
│   ├── budget_curves.pdf
│   └── ...
└── appendix.tex      # if separate
```
Do NOT include: `.aux`, `.log`, `.pdf`, `.synctex.gz`

Size limit: 50MB total

### Step 6: Metadata on arXiv
Fill in:
- **Title:** exactly as in the paper
- **Authors:** all authors in order (name, affiliation in parentheses; city+country only, no full address)
- **Abstract:** ≤1920 characters (arXiv truncates beyond this — paste into a character counter first)
- **Categories:** Primary `cs.CL`, cross-list `cs.LG`
- **License:** recommend CC BY 4.0 (most open, compatible with ACL)
- **Comments field:** "X pages, Y figures, Z tables. Submitted to [venue] [year]" (optional but useful)
- **MSC/ACM class:** optional for cs.CL

### Step 7: Submit and review
- Upload zip → arXiv compiles server-side → review the generated PDF carefully
- Common issues to check:
  - Figures missing or wrong size
  - Font substitution (use Type 1 / Type 3 fonts, not bitmap)
  - Bibliography entries missing
  - Page count / margin violations
- Click "Submit" — paper goes into a moderation queue
- arXiv announces daily at 20:00 ET Sun–Thu; submitted before 14:00 ET appears that night

### Step 8: After submission
- Paper gets arXiv ID (e.g., `2406.XXXXX`)
- Share the URL; update any slides or docs
- If submitting to a venue with double-blind review: check anonymity policy — most ACL venues now allow arXiv posting at any time (policy updated early 2024)
- If revisions needed: submit a v2 via "Replace" — all versions remain accessible

---

## Part 5 — Our Writing Plan (Ordered Tasks)

### Phase 1: Figures (do first — figures drive the narrative)
- [ ] Fig 1: geometry-behavior divergence conceptual diagram (draw in TikZ or Inkscape)
- [ ] Fig 2: Rank-95 trajectory plot (Python, data already in results)
- [ ] Fig 3/4: budget curves for MSC signal variants (matplotlib, data in pairwise_summary.json)
- [ ] Fig 5: support-turn rescue bar chart
- [ ] Fig 6: scope condition summary schematic

### Phase 2: Core sections (write in this order)
- [ ] §2 Background — codec formulation, three signals (math-heavy, write carefully)
- [ ] §3 Geometry-behavior divergence — Table 2 + narrative
- [ ] §4 Signal variants — Table 3 + Table 4 + mechanism explanation
- [ ] §5 Hard stress set — Table 5 + Table 6 + mechanism
- [ ] §6 Scope conditions — Table 7 + Table 8 + 2×2 scope map
- [ ] §7 Related work — grouped thematic review
- [ ] §8 Discussion — implications
- [ ] §9 Limitations — explicit, honest, 3 paragraphs

### Phase 3: Frame sections (write after core is stable)
- [ ] §1 Introduction — motivation, gap, contributions (4 numbered)
- [ ] Abstract — write from Introduction + Results, compress to ≤250 words

### Phase 4: Polish and checklist
- [ ] Run reproducibility checklist (all hyperparameters reported?)
- [ ] Statistical significance: every Δ in results tables has a p-value
- [ ] Every claim in Introduction has a pointer to a result
- [ ] Bold best result per row in every table
- [ ] All figures have captions that are self-contained (figure understandable without reading body)
- [ ] Limitations section present and honest
- [ ] Ethics/broader impact statement
- [ ] Code/data availability statement
- [ ] Proofread for "novel", "utilize", long sentences
- [ ] Character-count the abstract (≤1920 for arXiv)
- [ ] Local compile clean (no warnings)

### Phase 5: Submit
- [ ] Package zip with .bbl
- [ ] Set categories cs.CL + cs.LG
- [ ] Review server-compiled PDF
- [ ] Submit

---

## Part 6 — Scope and Claims — What We Can and Cannot Say

### What the paper CAN claim (evidence is strong)
1. Geometry-behavior divergence is real and measurable at scale (MSC 1000 conv, p<0.001)
2. sem_qcg is the only KCD variant winning both metrics simultaneously on MSC (p<0.001 conv)
3. On constraint-critical memory, geometry beats semantic as a scoring signal (p=0.028 at 1.5B, p=0.048 at 3B)
4. The advantage grows with model scale (NLL Δ +0.628 at 1.5B → +1.126 at 3B, both p<0.001)
5. Support-turn rescue rate is 17/36 (geometry) vs 7/36 (semantic) — mechanism is real

### What the paper CANNOT claim (scope conditions apply)
- That geometry is universally better than semantic — **falsified by Llama-3.2-3B**
- That sem_qcg works on very long conversations — **falsified by LME at 0.35 budget (p=0.011)**
- That the Gate 1 oracle validates geometry on answer-NLL-defined harm — **it fails on LME**
- That QA accuracy on LME improves — **QA proxy metric is invalid**

### Framing strategy
Present the scope conditions as **contributions, not failures**. The paper contributes:
- A dual-objective evaluation framework that caught what single-metric evaluation misses
- A characterization of when geometry-based signals add value (turn length ≤100, Qwen-family)
- Two informative negatives (learned harm predictor, object-level) that identify failure modes

---

## Appendix: Quick Reference — Key Numbers

| Claim | Number | Significance |
|---|---|---|
| LongLLMLingua logit L2 above uniform @0.50 | +161 units | p<0.001 |
| sem_qcg logit L2 below uniform @0.50 (MSC) | −46.4 units | p<0.001 |
| sem_qcg Δ L2 vs semantic_KCD @0.50 (MSC conv) | −113.7 | p<0.001 |
| sem_qcg Δ NLL vs semantic_KCD @0.50 (MSC conv) | −0.056 nats | p=0.004 |
| Geometry vs semantic Δ NLL @0.50 (hardset 3B) | +1.126 nats | p=0.0003 |
| sem_qcg worse than semantic on LME @0.35 (conv) | +50.3 L2 | p=0.011 |
| Semantic Δ NLL vs geometry (Llama-3.2-3B @0.35) | −0.099 nats | p=0.026 |
| Rank-95 real vs shuffled (qwen25_15b) | 1.167 vs 1.458 | p=0.0008 |
| Corr(d_geo, Δℓ) (qwen25_15b) | 0.994 | p<0.001 |
