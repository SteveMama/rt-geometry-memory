#!/usr/bin/env bash
# run_criticism_fixes_multigpu.sh
#
# Addresses the three concrete reviewer criticisms identified in code audit:
#
#   1. KCD signal variants at full 1000-conv MSC scale
#      — The previous KCD study used only 50 conversations.  Reviewers will
#        ask why KCD variants were not compared at the same scale as external
#        baselines (1 000 conversations).  This run closes that gap directly.
#
#   2. sem_qcg ablation variants on MSC 1000
#      — Policies `sem_qcg_no_query` and `sem_qcg_no_support` already exist
#        in the registry.  Running them in the same study as sem_qcg gives a
#        component ablation: which part of the signal (query projection vs
#        support-awareness) is load-bearing?
#
#   3. LME with binding budget (--recent-window 0)
#      — Current fullLME results have token_fraction ~0.97 at B=0.20 because
#        the recent window exempts most of the prefix from compression.  This
#        run disables the recent window so the budget fraction applies to ALL
#        older turns, making the compression regime comparable to MSC.
#
# Results are written to:
#   results/criticism_fixes/msc_1000_kcd/
#   results/criticism_fixes/lme_binding_budget/
#
# Required env vars:
#   GITHUB_USER   — GitHub username (for git push)
#   GITHUB_TOKEN  — PAT with repo write scope
#
# Optional env vars:
#   HF_TOKEN           — HuggingFace token (not needed for Qwen models)
#   MODEL_KEY          — default: qwen25_15b
#   BUDGETS            — default: 0.20,0.35,0.50
#   GPU_COUNT          — override auto-detected count
#   WORKERS_PER_GPU    — default: 4 (tune for VRAM)
#   JOB_MULTIPLIER     — shards per GPU (default: 4)
#   SKIP_DOWNLOAD      — set to 1 to skip benchmark downloads
#   SKIP_PUSH          — set to 1 to skip git push
#   EXTRACT_CACHE_ROOT — large volume for .npz hidden-state cache
#
# Usage:
#   GITHUB_USER=SteveMama GITHUB_TOKEN=ghp_xxx \
#     bash scripts/run_criticism_fixes_multigpu.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── Python binary ──────────────────────────────────────────────────────────────
LOCAL_VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$LOCAL_VENV_PYTHON" ]]; then
    PYTHON_BIN="$LOCAL_VENV_PYTHON"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi
"$PYTHON_BIN" -c "import torch" >/dev/null 2>&1 || {
  echo "[criticism_fixes] ERROR: python at $PYTHON_BIN has no torch" >&2
  echo "[criticism_fixes] Run: bash scripts/install_reviewer_deps.sh" >&2
  exit 1
}
export PYTHON_BIN

# ── Config ─────────────────────────────────────────────────────────────────────
RUN_PREFIX="${RUN_PREFIX:-criticism_fixes}"
MODEL_KEY="${MODEL_KEY:-qwen25_15b}"
BUDGETS="${BUDGETS:-0.20,0.35,0.50}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
JOB_MULTIPLIER="${JOB_MULTIPLIER:-4}"
GPU_PREFLIGHT_RETRIES="${GPU_PREFLIGHT_RETRIES:-6}"
GPU_PREFLIGHT_SLEEP="${GPU_PREFLIGHT_SLEEP:-10}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

# Benchmark inputs
MSC_INPUT="benchmarks/msc_valid_normalized.jsonl"
LONGMEM_INPUT="benchmarks/longmemeval_s_full_normalized.jsonl"

[[ -n "${EXTRACT_CACHE_ROOT:-}" ]] && export EXTRACT_CACHE_ROOT

# ── Result directories ─────────────────────────────────────────────────────────
RESULTS_ROOT="results/criticism_fixes"
LOG_ROOT="$RESULTS_ROOT/logs"
QUEUE_ROOT="$RESULTS_ROOT/job_queue"
QUEUE_PENDING="$QUEUE_ROOT/pending"
QUEUE_RUNNING="$QUEUE_ROOT/running"
QUEUE_DONE="$QUEUE_ROOT/done"
QUEUE_FAILED="$QUEUE_ROOT/failed"
PLAN_ROOT="$RESULTS_ROOT/shard_plans"

mkdir -p "$LOG_ROOT" "$QUEUE_PENDING" "$QUEUE_RUNNING" "$QUEUE_DONE" \
         "$QUEUE_FAILED" "$PLAN_ROOT"

# ── Logging ────────────────────────────────────────────────────────────────────
MAIN_LOG="$LOG_ROOT/main_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [criticism_fixes] $*" >&2; }
die() { log "FATAL: $*"; exit 1; }

log "Run prefix : $RUN_PREFIX"
log "Model      : $MODEL_KEY   Budgets: $BUDGETS"
log "Log        : $MAIN_LOG"

# ── GPU detection ──────────────────────────────────────────────────────────────
mapfile -t GPU_INDICES < <(nvidia-smi --query-gpu=index --format=csv,noheader \
                             2>/dev/null | sed '/^$/d')
[[ ${#GPU_INDICES[@]} -gt 0 ]] || die "no GPUs detected via nvidia-smi"
GPU_COUNT="${GPU_COUNT:-${#GPU_INDICES[@]}}"
[[ "$GPU_COUNT" -le ${#GPU_INDICES[@]} ]] || GPU_COUNT=${#GPU_INDICES[@]}
JOB_SHARDS=$(( GPU_COUNT * JOB_MULTIPLIER ))
log "GPUs: ${GPU_INDICES[*]} | using $GPU_COUNT | shards=$JOB_SHARDS"

# Recover any interrupted jobs from a previous run of this script
for running_job in "$QUEUE_RUNNING"/*.job; do
  [[ -e "$running_job" ]] || continue
  base="$(basename "$running_job")"
  mv "$running_job" "$QUEUE_PENDING/${base%.gpu*.job}.job" 2>/dev/null || true
done
rm -f "$QUEUE_PENDING"/*.job "$QUEUE_FAILED"/*.job 2>/dev/null || true

# ── Job queue helpers ──────────────────────────────────────────────────────────
shard_complete() {
  local shard_dir="$1"
  [[ -f "$shard_dir/progress.json" ]] || return 1
  "$PYTHON_BIN" - "$shard_dir/progress.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
sys.exit(0 if p.get("status") == "complete" else 1)
EOF
}

enqueue_job() {
  local name="$1" shard_dir="$2" cmd="$3"
  if shard_complete "$shard_dir"; then
    log "skip (already complete): $name"
    return
  fi
  {
    printf 'SHARD_DIR=%q\n' "$shard_dir"
    printf 'LOG_PATH=%q\n'  "$LOG_ROOT/${name}.log"
    printf 'CMD=%q\n'       "$cmd"
  } > "$QUEUE_PENDING/${name}.job"
}

plan_shards() {
  local input="$1" tag="$2"
  local plan_dir="$PLAN_ROOT/$tag"
  mkdir -p "$plan_dir"
  "$PYTHON_BIN" -m paper3_codec.plan_conversation_shards \
    --input-path    "$input" \
    --shard-count   "$JOB_SHARDS" \
    --target-turn-stride 1 \
    --output-dir    "$plan_dir" >&2
  echo "$plan_dir"
}

# ── Step 1: Download benchmarks if missing ─────────────────────────────────────
log "=== STEP 1: Benchmarks ==="

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  log "SKIP_DOWNLOAD=1, skipping downloads"
else
  if [[ ! -f "$MSC_INPUT" ]]; then
    log "Downloading MSC valid..."
    "$PYTHON_BIN" scripts/download_public_benchmark.py \
      --benchmark msc_valid \
      --output benchmarks/msc_valid_raw.jsonl
    "$PYTHON_BIN" scripts/prepare_public_benchmark_jsonl.py \
      --format msc \
      --input  benchmarks/msc_valid_raw.jsonl \
      --output "$MSC_INPUT" \
      --family msc_valid
    log "MSC ready: $MSC_INPUT"
  else
    log "MSC already present: $MSC_INPUT"
  fi

  if [[ ! -f "$LONGMEM_INPUT" ]]; then
    log "Downloading LongMemEval-S (full, no truncation)..."
    "$PYTHON_BIN" scripts/download_public_benchmark.py \
      --benchmark longmemeval_s_cleaned \
      --output benchmarks/longmemeval_s_raw.json
    "$PYTHON_BIN" scripts/prepare_public_benchmark_jsonl.py \
      --format longmemeval \
      --input  benchmarks/longmemeval_s_raw.json \
      --output "$LONGMEM_INPUT" \
      --family longmemeval_s_full
    log "LME-S ready: $LONGMEM_INPUT"
  else
    log "LME-S already present: $LONGMEM_INPUT"
  fi
fi

[[ -f "$MSC_INPUT" ]]    || die "MSC benchmark missing: $MSC_INPUT"
[[ -f "$LONGMEM_INPUT" ]] || die "LME-S benchmark missing: $LONGMEM_INPUT"

# LME 100-conversation subset (same subset as reviewer_fixes run for comparability)
LME_SUBSET="benchmarks/longmemeval_s_subset_100.jsonl"
if [[ ! -f "$LME_SUBSET" ]]; then
  head -n 100 "$LONGMEM_INPUT" > "$LME_SUBSET"
  log "Created LME subset: $LME_SUBSET (100 conversations)"
else
  log "LME subset already present: $LME_SUBSET"
fi

# ── Step 2: Enqueue jobs ───────────────────────────────────────────────────────
log "=== STEP 2: Planning shards ==="

declare -a MERGE_SPECS=()

# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: KCD signal variants + ablations on full 1000-conv MSC
#
# Why:  Previous KCD study was n=50; external baselines were n=1000.
#       Reviewers noted the 78-unit gap in uniform L2 between the two runs as
#       evidence of sampling variance.  This run produces KCD variant results
#       at the same scale as the baseline comparison.
#
# Policies:
#   - uniform (reference)
#   - geometry_keep_compress_drop
#   - semantic_keep_compress_drop
#   - semantic_query_conditioned_geometry_keep_compress_drop   (sem_qcg full)
#   - sem_qcg_no_query:   removes query-conditioned projection  → tests whether
#       conditioning on the query hidden state is the load-bearing component
#   - sem_qcg_no_support: removes support-score weighting       → tests whether
#       the support-aware term contributes independently of query conditioning
#
# These last two are already registered policies; running them here gives a
# within-study component ablation with no code changes.
# ─────────────────────────────────────────────────────────────────────────────
log "Planning: KCD variants + ablations on MSC 1000 conversations"
plan_dir_msc_1k="$(plan_shards "$MSC_INPUT" "msc_1000_kcd")"
declare -a MSC_1K_SHARD_DIRS=()
for ((s=0; s<JOB_SHARDS; s++)); do
  ids_file="$plan_dir_msc_1k/shard_${s}_ids.txt"
  [[ -s "$ids_file" ]] || continue
  shard_dir="$RESULTS_ROOT/msc_1000_kcd/${RUN_PREFIX}_msc_1000_kcd_shard${s}of${JOB_SHARDS}"
  MSC_1K_SHARD_DIRS+=("$shard_dir")
  enqueue_job "msc_1000_kcd_s${s}" "$shard_dir" \
    "$PYTHON_BIN -m paper3_codec.study \
      --study-name ${RUN_PREFIX}_msc_1000_kcd_shard${s}of${JOB_SHARDS} \
      --model-keys $MODEL_KEY \
      --input-path $MSC_INPUT \
      --families msc_valid \
      --budgets $BUDGETS \
      --policies uniform,geometry_keep_compress_drop,semantic_keep_compress_drop,semantic_query_conditioned_geometry_keep_compress_drop,semantic_query_conditioned_geometry_keep_compress_drop_no_query,semantic_query_conditioned_geometry_keep_compress_drop_no_support \
      --recent-window 2 \
      --min-history 4 \
      --max-input-tokens 1024 \
      --segment-span 2 \
      --target-turn-stride 2 \
      --max-target-turns 16 \
      --output-root $RESULTS_ROOT/msc_1000_kcd \
      --conversation-ids-path $ids_file"
done
MERGE_SPECS+=("msc_1000_kcd|$(IFS=,; echo "${MSC_1K_SHARD_DIRS[*]}")")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: LME with binding budget (--recent-window 0)
#
# Why:  Current fullLME results show token_fraction ~0.97 at B=0.20 because
#       `recent_window=2` unconditionally retains the 2 most recent turns, and
#       the budget fraction is computed relative to the OLDER portion only.
#       On short LME evaluation prefixes (target-turn-stride 4, max 20 targets)
#       the older portion is very small, so the budget rarely binds.
#
#       Setting --recent-window 0 makes ALL turns eligible for compression.
#       This allows the budget fraction to apply to the full prefix, producing
#       token_fraction values comparable to the MSC runs (~0.20 at B=0.20).
#
#       Run on the same 100-conversation LME subset as the reviewer_fixes run
#       for direct comparability.  Results are saved separately so the original
#       reviewer_fixes/fullLME/ data is not overwritten.
# ─────────────────────────────────────────────────────────────────────────────
log "Planning: LME 100-conv with binding budget (recent-window 0)"
plan_dir_lme_bind="$(plan_shards "$LME_SUBSET" "lme_binding_budget")"
declare -a LME_BIND_SHARD_DIRS=()
for ((s=0; s<JOB_SHARDS; s++)); do
  ids_file="$plan_dir_lme_bind/shard_${s}_ids.txt"
  [[ -s "$ids_file" ]] || continue
  shard_dir="$RESULTS_ROOT/lme_binding_budget/${RUN_PREFIX}_lme_binding_shard${s}of${JOB_SHARDS}"
  LME_BIND_SHARD_DIRS+=("$shard_dir")
  enqueue_job "lme_binding_s${s}" "$shard_dir" \
    "$PYTHON_BIN -m paper3_codec.study \
      --study-name ${RUN_PREFIX}_lme_binding_shard${s}of${JOB_SHARDS} \
      --model-keys $MODEL_KEY \
      --input-path $LME_SUBSET \
      --families longmemeval_s_full \
      --budgets $BUDGETS \
      --policies uniform,geometry_keep_compress_drop,semantic_keep_compress_drop,semantic_query_conditioned_geometry_keep_compress_drop \
      --recent-window 0 \
      --min-history 4 \
      --max-input-tokens 1024 \
      --segment-span 2 \
      --target-turn-stride 4 \
      --max-target-turns 20 \
      --output-root $RESULTS_ROOT/lme_binding_budget \
      --conversation-ids-path $ids_file"
done
MERGE_SPECS+=("lme_binding_budget|$(IFS=,; echo "${LME_BIND_SHARD_DIRS[*]}")")

PENDING_COUNT=$(ls "$QUEUE_PENDING"/*.job 2>/dev/null | wc -l | tr -d ' ') || PENDING_COUNT=0
log "Enqueued $PENDING_COUNT total shards across $GPU_COUNT GPUs"

# ── Step 3: GPU worker pool ────────────────────────────────────────────────────
log "=== STEP 3: Running GPU workers ==="

gpu_preflight() {
  local gpu_id="$1" attempt
  for ((attempt=1; attempt<=GPU_PREFLIGHT_RETRIES; attempt++)); do
    if CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" - <<'EOF'
import torch
assert torch.cuda.is_available(), "cuda not available"
torch.cuda.set_device(0)
x = torch.zeros(8, device="cuda")
torch.cuda.synchronize()
print(f"gpu {torch.cuda.get_device_name(0)} ok")
EOF
    then return 0; fi
    log "gpu $gpu_id preflight attempt $attempt/$GPU_PREFLIGHT_RETRIES failed — retrying in ${GPU_PREFLIGHT_SLEEP}s"
    sleep "$GPU_PREFLIGHT_SLEEP"
  done
  return 1
}

worker_loop() {
  local gpu_id="$1"
  gpu_preflight "$gpu_id" || { log "gpu $gpu_id unusable, worker exiting"; return 1; }
  while true; do
    local claimed=""
    for pending in "$QUEUE_PENDING"/*.job; do
      [[ -e "$pending" ]] || break
      local base; base="$(basename "$pending" .job)"
      local running="$QUEUE_RUNNING/${base}.gpu${gpu_id}.job"
      if mv "$pending" "$running" 2>/dev/null; then claimed="$running"; break; fi
    done
    [[ -n "$claimed" ]] || break
    # shellcheck disable=SC1090
    source "$claimed"
    local job_name; job_name="$(basename "$claimed" .job)"
    mkdir -p "$(dirname "$LOG_PATH")"
    log "GPU=$gpu_id starting $job_name"
    local start_ts; start_ts=$(date +%s)
    if CUDA_VISIBLE_DEVICES="$gpu_id" \
       TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       bash -c "$CMD" >>"$LOG_PATH" 2>&1; then
      local elapsed=$(( $(date +%s) - start_ts ))
      log "GPU=$gpu_id DONE $job_name (${elapsed}s)"
      mv "$claimed" "$QUEUE_DONE/$(basename "$claimed")"
    else
      log "GPU=$gpu_id FAILED $job_name — see $LOG_PATH"
      mv "$claimed" "$QUEUE_FAILED/$(basename "$claimed")"
    fi
  done
}

declare -a WORKER_PIDS=()
for ((i=0; i<GPU_COUNT; i++)); do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    worker_loop "${GPU_INDICES[$i]}" &
    WORKER_PIDS+=("$!")
  done
done
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || true; done

FAILED_COUNT=$(ls "$QUEUE_FAILED"/*.job 2>/dev/null | wc -l | tr -d ' ') || FAILED_COUNT=0
if [[ "$FAILED_COUNT" -gt 0 ]]; then
  log "WARNING: $FAILED_COUNT shards failed. Re-run this script to retry (done shards are skipped)."
  ls "$QUEUE_FAILED"/*.job 2>/dev/null | while read -r f; do log "  $f"; done
fi

# ── Step 4: Merge shards ───────────────────────────────────────────────────────
log "=== STEP 4: Merging shards ==="

for spec in "${MERGE_SPECS[@]}"; do
  [[ -n "$spec" ]] || continue
  IFS='|' read -r tag dirs <<< "$spec"
  [[ -n "$dirs" ]] || continue
  log "Merging: $tag"
  case "$tag" in
    msc_1000_kcd)
      "$PYTHON_BIN" -m paper3_codec.merge_study_shards \
        --study-name "${RUN_PREFIX}_msc_1000_kcd_merged" \
        --output-root "$RESULTS_ROOT/msc_1000_kcd" \
        --shard-dirs  "$dirs" \
        || log "WARNING: merge failed for $tag"
      ;;
    lme_binding_budget)
      "$PYTHON_BIN" -m paper3_codec.merge_study_shards \
        --study-name "${RUN_PREFIX}_lme_binding_merged" \
        --output-root "$RESULTS_ROOT/lme_binding_budget" \
        --shard-dirs  "$dirs" \
        || log "WARNING: merge failed for $tag"
      ;;
  esac
done

# ── Step 5: Pairwise significance reports ──────────────────────────────────────
log "=== STEP 5: Pairwise reports ==="

for study_dir in \
    "$RESULTS_ROOT/msc_1000_kcd/${RUN_PREFIX}_msc_1000_kcd_merged" \
    "$RESULTS_ROOT/lme_binding_budget/${RUN_PREFIX}_lme_binding_merged"; do
  [[ -f "$study_dir/evaluation_rows.csv" ]] || continue
  log "Pairwise report: $study_dir"
  bash scripts/run_paper3_pairwise_report.sh "$study_dir" \
    || log "WARNING: pairwise report failed for $study_dir"
done

# ── Step 6: Summary ────────────────────────────────────────────────────────────
log "=== STEP 6: Writing summary ==="

SUMMARY_FILE="$RESULTS_ROOT/summary_${RUN_PREFIX}.md"
{
  echo "# Criticism-Fix Results — $RUN_PREFIX"
  echo "Generated: $(date)"
  echo "Model: $MODEL_KEY | Budgets: $BUDGETS | GPUs: $GPU_COUNT"
  echo ""
  echo "## What this run addresses"
  echo ""
  echo "### Criticism 1 + 2: KCD signal variants at full MSC scale + component ablation"
  echo "Previous KCD study: n=50. Baselines: n=1000. This run: n=1000 for all KCD variants."
  echo "Ablation policies included:"
  echo "  - sem_qcg_no_query   (removes query-conditioned projection)"
  echo "  - sem_qcg_no_support (removes support-score weighting)"
  echo "Output: $RESULTS_ROOT/msc_1000_kcd/"
  echo ""
  echo "### Criticism 3: LME with binding budget (recent-window 0)"
  echo "Previous LME: token_fraction ~0.97 at B=0.20 because budget applied only to"
  echo "older turns (recent_window=2). This run: --recent-window 0 so budget fraction"
  echo "applies to all turns. Expect token_fraction ~0.20 at B=0.20 if budget binds."
  echo "Output: $RESULTS_ROOT/lme_binding_budget/"
  echo ""
  echo "## Job outcomes"
  echo "- Done   : $(ls "$QUEUE_DONE"/*.job  2>/dev/null | wc -l | tr -d ' ' || echo 0)"
  echo "- Failed : $FAILED_COUNT"
  echo ""
  echo "## Output directories"
  for spec in "${MERGE_SPECS[@]}"; do
    IFS='|' read -r tag _ <<< "$spec"
    echo "- $tag"
  done
  echo ""
  echo "## How to read results"
  echo ""
  echo "### MSC 1000-conv KCD"
  echo "Compare pairwise_report.md in the merged directory."
  echo "Key comparisons:"
  echo "  sem_qcg vs semantic_KCD (primary claim)"
  echo "  sem_qcg vs sem_qcg_no_query (query projection ablation)"
  echo "  sem_qcg vs sem_qcg_no_support (support-score ablation)"
  echo ""
  echo "### LME binding budget"
  echo "Check token_fraction in study_summary.json."
  echo "  If token_fraction ~0.20 at B=0.20: budget is now binding."
  echo "  If token_fraction still ~0.97: conversation prefixes are too short"
  echo "  even without the recent window — need to increase target-turn-stride"
  echo "  or evaluate at later turns in the conversation."
  echo ""
  echo "## Logs"
  echo "Main log : $MAIN_LOG"
  echo "Per-job  : $LOG_ROOT/"
} > "$SUMMARY_FILE"
cat "$SUMMARY_FILE"

# ── Step 7: Git commit & push ──────────────────────────────────────────────────
log "=== STEP 7: Git commit & push ==="

if [[ "$SKIP_PUSH" == "1" ]]; then
  log "SKIP_PUSH=1, skipping git operations"
  log "All done. Results: $RESULTS_ROOT/"
  exit 0
fi

[[ -n "${GITHUB_USER:-}" ]]  || die "Set GITHUB_USER before running"
[[ -n "${GITHUB_TOKEN:-}" ]] || die "Set GITHUB_TOKEN before running"

git config user.name  "${GIT_NAME:-SteveMama}"
git config user.email "${GIT_EMAIL:-pranav@vizit.com}"

DONE_COUNT=$(ls "$QUEUE_DONE"/*.job 2>/dev/null | wc -l | tr -d ' ' || echo 0)

COMMIT_MSG="Add criticism-fix experiments: MSC 1000-conv KCD + LME binding budget

Closes three reviewer criticisms identified in code audit:

1. KCD signal variants at full 1000-conv MSC scale
   Previous KCD study was n=50; external baselines were n=1000.
   This run adds geometry_KCD, semantic_KCD, sem_qcg on all 1000
   MSC conversations using the same parameters as the baseline run.

2. sem_qcg component ablation (same 1000-conv MSC study)
   sem_qcg_no_query  — drops query-conditioned projection
   sem_qcg_no_support — drops support-score weighting
   Both policies were already registered; running them here gives
   a within-study ablation showing which component is load-bearing.

3. LME with binding budget (--recent-window 0)
   Previous fullLME had token_fraction ~0.97 at B=0.20 because
   recent_window=2 exempted most of the prefix from compression.
   This run sets recent_window=0 so budget applies to all turns.

Run: $RUN_PREFIX | Model: $MODEL_KEY | GPUs: $GPU_COUNT
Jobs done: $DONE_COUNT | Jobs failed: $FAILED_COUNT | $(date +%Y-%m-%d)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git add \
  "$RESULTS_ROOT/" \
  "scripts/run_criticism_fixes_multigpu.sh" \
  2>/dev/null || true

if git diff --cached --quiet; then
  log "No staged changes to commit"
else
  git commit -m "$COMMIT_MSG"
  git remote set-url origin \
    "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/SteveMama/rt-geometry-memory.git"
  git push origin main
  log "Pushed to GitHub ✓"
fi

log "=== ALL DONE ==="
log "Results: $RESULTS_ROOT/"
log "Summary: $SUMMARY_FILE"
log "Main log: $MAIN_LOG"

if [[ "$FAILED_COUNT" -gt 0 ]]; then
  log "Re-run this script to retry $FAILED_COUNT failed shards (done shards are skipped)."
  exit 1
fi
