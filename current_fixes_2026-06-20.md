# ACL Paper Reviewer Fixes — Session Log
**Date:** 2026-06-20  
**Paper:** Signal-Conditioned Memory Compression for Multi-Turn Conversations (KCD Codec)  
**Venue target:** ACL Findings

---

## What We Have

### Completed Results
| Dataset | Conversations | Eval Rows | Status |
|---|---|---|---|
| MSC valid | 1,000 | 163,755 | ✅ Complete, pushed to GitHub |
| Hardset stress set | 9 | 540 | ✅ Complete (too small to be primary result) |
| Llama-3.2-3B scale | 8 | 540 | ✅ Complete (preliminary/underpowered) |
| LongMemEval-S full | 500 | 0 | ❌ Incomplete — disk full (20GB volume exhausted by .npz hidden state cache) |

### Key MSC Findings (in our favor)
- `recency_keep_compress_drop` beats all baselines at every budget (0.20, 0.35, 0.50), p<0.0001
- LongLLMLingua: +94 to +158 logit L2 vs uniform (catastrophically worse on geometric fidelity), p<0.0001
- LongLLMLingua improves behavior NLL (−0.07 to −0.18) — divergence from geometry metric
- `recency_keep_compress_drop` wins behavior NLL at 0.35 and 0.50 budgets

### Key Llama-3.2-3B Findings (mixed)
- `geometry_keep_compress_drop` WORSE than uniform at budget 0.35 (p=0.04)
- Too small (8 convos) to conclude anything — must be labeled "preliminary"

---

## Critical Gaps (What Reviewers Will Ask)

1. **Metric gap**: logit L2 is NOT a recognized metric. LongMemEval uses GPT-judged QA accuracy. Must add behavioral QA eval.
2. **LME gap**: Full LongMemEval-S (500 conversations) never ran — reviewer specifically flagged the 40-turn truncation.
3. **Scale gap**: Llama-3.2-3B result is underpowered (8 conversations). Need 50+ for credibility.
4. **One benchmark**: Only MSC completed at scale. ACL Findings needs at least 2 benchmarks.

---

## What the Research Found (deep-research, 109 agents, 2.2M tokens)

### Confirmed (high confidence)
- LongMemEval primary metric = GPT-judged QA accuracy (3-0 unanimous)
- ACL 2025 Findings comparable paper (DAST) uses F1/exact match — logit L2 unknown to reviewers
- LongLLMLingua uses contrastive perplexity scoring (3-0) — designed to optimize NLL not geometry
- ACL ARR: statistical significance is "ideal" not mandatory (3-0)
- SAE/quantization paper: behavioral metrics can mask representational damage (INT7 improves perplexity, degrades 18.7% SAE features) — supports geometry-behavior divergence framing

### Key insight for framing
> LongLLMLingua is behaviorally adequate but representationally lossy. KCD targets geometric fidelity as a distinct, complementary objective. This divergence is expected and theoretically grounded — not a weakness.

### Sample size guidance
- LoCoMo (ACL 2024 main): 50 conversations, accepted
- LongMemEval-S: 500 curated questions — don't need all 500 conversations to cover them
- **Minimum credible LME run: 100-150 conversations with QA eval**

---

## Infrastructure Lessons Learned

### Bugs Fixed This Session
| Commit | Bug | Fix |
|---|---|---|
| `6aed986` | Bash glob `2>/dev/null` in for-loop | Removed |
| `46f8cc1` | Model key not resolved in ConversationStateExtractor | Pass `spec.model_name` not `model_key` |
| `ab3bde2` | KCD policies passed to baseline_study.py | Use only supported policies |
| `b11c775` | LME workload missing `--families` flag | Added `--families longmemeval_s_full` |
| `e472388` | 1 worker per GPU, 75% VRAM idle | Added `WORKERS_PER_GPU` parameter |

### RunPod Setup
- **Working image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **GPUs**: L40S × 7 ($6.93/hr) — NOT RTX 5090 (sm_120 incompatible with stable PyTorch)
- **Disk failure**: 20GB volume filled by `.npz` hidden state cache during LME extraction
- **Next run**: Use 100GB+ disk, mount separate cache volume
- **Cache location**: `results/paper3/extract_cache/` — set `EXTRACT_CACHE_ROOT` to persistent volume
- **No tmux/screen available** on RunPod by default — use `nohup` for long runs
- **GitHub push from pod**: Token expired/revoked mid-session — push from local Mac instead
- **WORKERS_PER_GPU=3**: Uses ~34GB of 46GB VRAM, runs all 21 shards simultaneously

### Optimal Run Command (next time)
```bash
nohup bash -c 'cd /workspace/RT && \
GITHUB_USER=SteveMama \
GITHUB_TOKEN=<new_token> \
HF_TOKEN=<hf_token> \
GPU_COUNT=7 \
JOB_MULTIPLIER=3 \
WORKERS_PER_GPU=3 \
RUN_LLAMA32_3B=0 \
SKIP_DOWNLOAD=1 \
LONGMEM_INPUT=benchmarks/longmemeval_s_full_normalized.jsonl \
EXTRACT_CACHE_ROOT=/workspace/cache \
bash scripts/run_reviewer_fixes_multigpu.sh' > /workspace/run.log 2>&1 &
```

---

## Next Steps (Priority Order)

1. **Add GPT-judged QA eval** to LME pipeline — replace logit L2 as primary metric for LME
2. **Spin new RunPod pod** with 100GB disk — run LME on 150 conversations
3. **Update evaluator model** — GPT-4o is outdated; use current best available (research needed)
4. **Frame geometry-behavior divergence** in paper using SAE citation
5. **Label Llama results as preliminary** — don't hide, don't overclaim
6. **Add second evaluator** for robustness (not just single-model judge)

---

---

## Second Research Round — Verified Findings (105 agents, 23 sources, 25 claims verified)

### (A) LLM Judge — GPT-4o Is Dead
**Confirmed (2-1):** Every single-judge setup is critiqueable due to egocentric (self-enhancement) bias and preference leakage. This is an EMNLP 2025 documented finding across 7 primary papers.

**Safe choice for ACL 2026:**
- Use **multi-judge ensemble**: Gemini 2.5 Flash + one open-weight judge (e.g., Llama-3-70B-Instruct)
- Report inter-judge agreement (Cohen's κ or Krippendorff's α)
- No single model (not GPT-5, not Claude Opus 4) is safe from reviewers citing judge bias
- Cheapest credible option: **Gemini 2.5 Flash** (cheapest frontier judge) + **Llama-3-70B-Instruct via HF/TogetherAI** (free/cheap open-weight)

**Key caveat:** The specific head-to-head accuracy comparisons between judges (the arXiv 2508.00454 paper) were fully refuted (0-3). No verified ranking survives — just use the ensemble approach.

### (B) SLM Novelty — KCD Is First
**Confirmed (3-0):** HyMem (arXiv 2602.13933), the closest competing work, was evaluated **exclusively on GPT-4o-mini and GPT-4.1-mini**. No experiments on open-source SLMs in the 1.5B-3B range.

**Exact claim to make:**
> "To our knowledge, KCD is the first geometric compression approach validated on open-source small language models (1.5B-3B parameters). Prior work (HyMem [cite], MemoryLLM [cite]) evaluates exclusively on proprietary frontier models or models ≥7B parameters."

**Refuted claim (don't make):** "SLMs benefit differently from compression than large models" — this was refuted 1-2. No verified source supports differential compression dynamics by model size. The novelty is by *absence* of prior work, not by proven differential benefit.

### (C) New Benchmarks for ACL 2026 Reviewers

| Benchmark | Venue | Type | SLM-feasible? |
|---|---|---|---|
| **MemBench** | ACL Findings 2025 | 2×2 factual/reflective × participation/observation | Risky (reflective tasks may be too hard for 1.5B) |
| **TReMu** | ACL Findings 2025 | 600 multi-choice QAs on LoCoMo; temporal reasoning | **Yes** — multi-choice, no generation |
| EvolMem | arXiv 2601.03543 | frontier-model-only | No |

**TReMu is the fastest add:**
- Multi-choice format — no generation needed, just answer selection
- Built on LoCoMo conversations (same data as some of our existing infrastructure)
- ACL Findings 2025 — reviewers will recognize it
- 600 QAs total — small enough to run quickly

**MemBench caveat:** ~100K token ceiling, reflective tasks may not be meaningful for 1.5B. Run MemBench only if you can spare the time.

### (D) Geometry Validation — What Holds Up
**Confirmed (2-1):** arXiv 2501.10573 ("The Geometry of Tokens in Internal Representations of Large Language Models", Jan 2025) validates:
- Intrinsic dimension (GRIDE), neighborhood overlap, cosine similarity as geometric probes
- Spearman rho **0.69-0.73** between intrinsic dimension and cross-entropy loss (p < 0.01)
- Validated on Llama 3 8B, Mistral 7B, Pythia 6.9B

**Critical caveat:** This is an unconfirmed arXiv preprint tested on 6B-8B models only. Direct applicability to 1.5B-3B is an *untested extrapolation*. Use as "principled motivation" not as proof.

**All curvature/trajectory claims refuted (0-3):** arXiv 2507.21107 (path curvature kappa in residual streams) claims were fully refuted. Do not cite it.

### (E) LongMemEval Post-Hoc Eval — Confirmed Shortcut
**Confirmed (3-0):** LongMemEval explicitly supports post-hoc evaluation.

From the official README:
> "If you only need to calculate the metrics on the outputs produced by your own system, you can run `src/evaluation/evaluate_qa.py`… save the outputs in a jsonl format with each line containing two fields: `question_id` and `hypothesis`."

**Published papers using this shortcut:** ENGRAM (arXiv 2511.12960), SGMem (arXiv 2509.21212), SimpleMem (arXiv 2601.02553) — all three use the post-hoc path. This is standard community practice, not a suspicious workaround.

**Practical implication:** After running LME conversations through KCD, save outputs as `{question_id, hypothesis}` jsonl and run the LME evaluator separately. The evaluator script handles the GPT judge call. This decouples compression runs from evaluation runs.

---

## Concrete Action Plan for Next RunPod Session

### Pod Configuration
```
Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
GPUs: L40S × 7 (or × 3 minimum)
Disk: 100GB+  ← CRITICAL, 20GB caused failure
Network volume: mount at /workspace/cache for EXTRACT_CACHE_ROOT
```

### Priority Order
1. **LME-150**: Run LongMemEval-S on 150 conversations (not all 500) with WORKERS_PER_GPU=3
2. **Post-hoc QA eval**: Save KCD outputs as `{question_id, hypothesis}` jsonl → run LME evaluator with multi-judge (Gemini 2.5 Flash + Llama-3-70B)
3. **TReMu**: Run TReMu (600 multi-choice QAs on LoCoMo) — cheapest new benchmark
4. **Llama-3B scale**: Get 50+ Llama-3.2-3B conversations (need 50 minimum for credibility)

### Run Command
```bash
nohup bash -c 'cd /workspace/RT && \
GITHUB_USER=SteveMama \
GITHUB_TOKEN=<new_token> \
HF_TOKEN=<hf_token> \
GPU_COUNT=7 \
JOB_MULTIPLIER=3 \
WORKERS_PER_GPU=3 \
RUN_LLAMA32_3B=0 \
SKIP_DOWNLOAD=1 \
LONGMEM_INPUT=benchmarks/longmemeval_s_full_normalized.jsonl \
EXTRACT_CACHE_ROOT=/workspace/cache \
bash scripts/run_reviewer_fixes_multigpu.sh' > /workspace/run.log 2>&1 &
```

### Paper Framing Changes Needed
1. Replace "logit L2" as primary metric with GPT-judged QA accuracy for LME section
2. Add multi-judge ensemble description (Gemini 2.5 Flash + Llama-3-70B) with inter-judge agreement
3. Add SLM novelty paragraph citing HyMem's GPT-4o-mini-only evaluation
4. Cite arXiv 2501.10573 for geometry-loss correlation (Spearman 0.69-0.73) as motivation
5. Label Llama-3.2-3B results as "preliminary (8 conversations)" — do not hide
6. Add TReMu results if time permits

---

## Open Research Questions (Unresolved)
- Is Gemini 2.5 Flash cheap enough to call for all 150 LME conversations × all policies? (Estimate cost before starting)
- Does MemBench reflective-level QA produce meaningful results for Qwen2.5-1.5B?
- Is there a pre-processed LoCoMo split we can use directly for TReMu without custom pipeline work?
- What is the minimum Llama-3.2-3B conversation count for 80% statistical power at effect size d=0.3?
