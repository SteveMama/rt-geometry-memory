#!/usr/bin/env bash
# run_reviewer_fixes_multigpu.sh
#
# Runs all experiments needed to close the reviewer gaps identified in
# the ACL Findings submission. Designed for a multi-GPU cloud instance.
#
# What this runs (in order):
#   1. Downloads LongMemEval-S (full, no turn truncation) and MSC if missing
#   2. Hard stress set: longllmlingua vs geometry_KCD vs semantic_KCD (Exp 1 extension)
#   3. Full LongMemEval-S: geometry_KCD + baselines (removes 40-turn cap)
#   4. Llama-3.2-3B scale validation on hard stress set (second 3B model)
#   5. Merges all shards
#   6. Commits results and pushes to GitHub
#
# Required env vars:
#   GITHUB_USER        your GitHub username
#   GITHUB_TOKEN       personal access token with repo write scope
#
# Optional env vars:
#   HF_TOKEN           HuggingFace token (needed for llama32_3b)
#   MODEL_KEY          primary model (default: qwen25_15b)
#   BUDGETS            comma-separated (default: 0.20,0.35,0.50)
#   GPU_COUNT          override auto-detected GPU count
#   JOB_MULTIPLIER     shards per GPU (default: 2)
#   SKIP_DOWNLOAD      set to 1 to skip benchmark downloads
#   SKIP_PUSH          set to 1 to skip git push
#
# Usage:
#   GITHUB_USER=SteveMama GITHUB_TOKEN=ghp_xxx bash scripts/run_reviewer_fixes_multigpu.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── Python binary ─────────────────────────────────────────────────────────────
LOCAL_VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$LOCAL_VENV_PYTHON" ]]; then
    PYTHON_BIN="$LOCAL_VENV_PYTHON"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi
"$PYTHON_BIN" -c "import torch" >/dev/null 2>&1 || {
  echo "[reviewer_fixes] ERROR: python at $PYTHON_BIN has no torch" >&2
  echo "[reviewer_fixes] Run: bash scripts/install_reviewer_deps.sh" >&2
  exit 1
}
export PYTHON_BIN

# ── Config ────────────────────────────────────────────────────────────────────
RUN_PREFIX="${RUN_PREFIX:-reviewer_fixes}"
MODEL_KEY="${MODEL_KEY:-qwen25_15b}"
BUDGETS="${BUDGETS:-0.20,0.35,0.50}"
JOB_MULTIPLIER="${JOB_MULTIPLIER:-2}"
GPU_PREFLIGHT_RETRIES="${GPU_PREFLIGHT_RETRIES:-6}"
GPU_PREFLIGHT_SLEEP="${GPU_PREFLIGHT_SLEEP:-10}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

HARDSET_INPUT="paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl"
MSC_INPUT="benchmarks/msc_valid_normalized.jsonl"
LONGMEM_INPUT="benchmarks/longmemeval_s_full_normalized.jsonl"   # full, no turn cap

LOG_ROOT="results/reviewer_fixes/logs/${RUN_PREFIX}"
QUEUE_ROOT="results/reviewer_fixes/job_queue/${RUN_PREFIX}"
QUEUE_PENDING="$QUEUE_ROOT/pending"
QUEUE_RUNNING="$QUEUE_ROOT/running"
QUEUE_DONE="$QUEUE_ROOT/done"
QUEUE_FAILED="$QUEUE_ROOT/failed"
PLAN_ROOT="results/reviewer_fixes/shard_plans/${RUN_PREFIX}"

mkdir -p "$LOG_ROOT" "$QUEUE_PENDING" "$QUEUE_RUNNING" "$QUEUE_DONE" "$QUEUE_FAILED" "$PLAN_ROOT"

# ── Logging helpers ───────────────────────────────────────────────────────────
MAIN_LOG="$LOG_ROOT/main_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [reviewer_fixes] $*" >&2; }
die() { log "FATAL: $*"; exit 1; }

log "Run prefix: $RUN_PREFIX"
log "Model: $MODEL_KEY  Budgets: $BUDGETS"
log "Log: $MAIN_LOG"

# ── GPU detection ─────────────────────────────────────────────────────────────
mapfile -t GPU_INDICES < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d')
[[ ${#GPU_INDICES[@]} -gt 0 ]] || die "no GPUs detected via nvidia-smi"
GPU_COUNT="${GPU_COUNT:-${#GPU_INDICES[@]}}"
[[ "$GPU_COUNT" -le ${#GPU_INDICES[@]} ]] || GPU_COUNT=${#GPU_INDICES[@]}
JOB_SHARDS=$(( GPU_COUNT * JOB_MULTIPLIER ))
log "GPUs: ${GPU_INDICES[*]} | using $GPU_COUNT | shards/workload=$JOB_SHARDS"

# restore any interrupted jobs from previous run
for running_job in "$QUEUE_RUNNING"/*.job; do
  [[ -e "$running_job" ]] || continue
  base="$(basename "$running_job")"
  mv "$running_job" "$QUEUE_PENDING/${base%.gpu*.job}.job"
done
rm -f "$QUEUE_PENDING"/*.job "$QUEUE_FAILED"/*.job 2>/dev/null || true

# ── Job queue helpers ─────────────────────────────────────────────────────────
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
    log "skip (complete): $name"
    return
  fi
  {
    printf 'SHARD_DIR=%q\n' "$shard_dir"
    printf 'LOG_PATH=%q\n' "$LOG_ROOT/${name}.log"
    printf 'CMD=%q\n' "$cmd"
  } > "$QUEUE_PENDING/${name}.job"
}

plan_shards() {
  local input="$1" tag="$2"
  local plan_dir="$PLAN_ROOT/$tag"
  mkdir -p "$plan_dir"
  "$PYTHON_BIN" -m paper3_codec.plan_conversation_shards \
    --input-path "$input" \
    --shard-count "$JOB_SHARDS" \
    --target-turn-stride 1 \
    --output-dir "$plan_dir" >&2
  echo "$plan_dir"
}

# ── Step 1: Download benchmarks ───────────────────────────────────────────────
log "=== STEP 1: Download benchmarks ==="
mkdir -p benchmarks

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  log "SKIP_DOWNLOAD=1, skipping downloads"
else
  # MSC
  if [[ ! -f "$MSC_INPUT" ]]; then
    log "Downloading MSC valid..."
    "$PYTHON_BIN" scripts/download_public_benchmark.py \
      --benchmark msc_valid \
      --output benchmarks/msc_valid_raw.jsonl
    "$PYTHON_BIN" scripts/prepare_public_benchmark_jsonl.py \
      --format msc \
      --input benchmarks/msc_valid_raw.jsonl \
      --output "$MSC_INPUT" \
      --family msc_valid
    log "MSC valid ready: $MSC_INPUT"
  else
    log "MSC valid already present: $MSC_INPUT"
  fi

  # LongMemEval-S FULL (no turn cap — key fix from reviewer)
  if [[ ! -f "$LONGMEM_INPUT" ]]; then
    log "Downloading LongMemEval-S (full conversations, no truncation)..."
    "$PYTHON_BIN" scripts/download_public_benchmark.py \
      --benchmark longmemeval_s_cleaned \
      --output benchmarks/longmemeval_s_raw.json
    # prepare WITHOUT --max-turns-per-conversation (removes the 40-turn cap)
    "$PYTHON_BIN" scripts/prepare_public_benchmark_jsonl.py \
      --format longmemeval \
      --input benchmarks/longmemeval_s_raw.json \
      --output "$LONGMEM_INPUT" \
      --family longmemeval_s_full
    CONV_COUNT=$("$PYTHON_BIN" -c "
import json
count = sum(1 for line in open('$LONGMEM_INPUT') if line.strip())
print(count)
")
    log "LongMemEval-S full ready: $LONGMEM_INPUT ($CONV_COUNT conversations)"
  else
    log "LongMemEval-S full already present: $LONGMEM_INPUT"
  fi
fi

[[ -f "$HARDSET_INPUT" ]] || die "hard stress set missing: $HARDSET_INPUT"

# ── Step 2: Enqueue jobs ──────────────────────────────────────────────────────
log "=== STEP 2: Planning jobs ==="

declare -a MERGE_SPECS=()

# ── 2a. Hard stress set: longllmlingua baseline (Experiment 1 extension) ──────
log "Planning: longllmlingua baseline on hard stress set"
plan_dir_hs="$(plan_shards "$HARDSET_INPUT" "baselines_hardset")"
declare -a HS_SHARD_DIRS=()
for ((s=0; s<JOB_SHARDS; s++)); do
  ids_file="$plan_dir_hs/shard_${s}_ids.txt"
  [[ -s "$ids_file" ]] || continue
  shard_dir="results/reviewer_fixes/baselines/${RUN_PREFIX}_hardset_shard${s}of${JOB_SHARDS}"
  HS_SHARD_DIRS+=("$shard_dir")
  enqueue_job "baselines_hardset_s${s}" "$shard_dir" \
    "$PYTHON_BIN -m june_fixes.baselines.baseline_study \
      --study-name ${RUN_PREFIX}_hardset_shard${s}of${JOB_SHARDS} \
      --model-keys $MODEL_KEY \
      --input-path $HARDSET_INPUT \
      --families long_dependency,retrieval_heavy,code_conversation \
      --budgets $BUDGETS \
      --policies uniform,longllmlingua,recency,lexical_tfidf,recency_keep_compress_drop \
      --recent-window 2 \
      --min-history 4 \
      --max-input-tokens 768 \
      --segment-span 2 \
      --output-root results/reviewer_fixes/baselines \
      --conversation-ids-path $ids_file"
done
MERGE_SPECS+=("baselines_hardset|$(IFS=,; echo "${HS_SHARD_DIRS[*]}")")

# ── 2b. Full LongMemEval-S: geometry_KCD + baselines (no 40-turn cap) ─────────
if [[ -f "$LONGMEM_INPUT" ]]; then
  log "Planning: full LongMemEval-S study"
  plan_dir_lme="$(plan_shards "$LONGMEM_INPUT" "fullLME")"
  declare -a LME_SHARD_DIRS=()
  for ((s=0; s<JOB_SHARDS; s++)); do
    ids_file="$plan_dir_lme/shard_${s}_ids.txt"
    [[ -s "$ids_file" ]] || continue
    shard_dir="results/reviewer_fixes/fullLME/${RUN_PREFIX}_lme_shard${s}of${JOB_SHARDS}"
    LME_SHARD_DIRS+=("$shard_dir")
    enqueue_job "fullLME_s${s}" "$shard_dir" \
      "$PYTHON_BIN -m paper3_codec.study \
        --study-name ${RUN_PREFIX}_lme_shard${s}of${JOB_SHARDS} \
        --model-keys $MODEL_KEY \
        --input-path $LONGMEM_INPUT \
        --budgets $BUDGETS \
        --policies uniform,semantic,geometry,geometry_keep_compress_drop,semantic_query_conditioned_geometry_keep_compress_drop \
        --recent-window 2 \
        --min-history 4 \
        --max-input-tokens 1024 \
        --segment-span 2 \
        --target-turn-stride 4 \
        --max-target-turns 20 \
        --output-root results/reviewer_fixes/fullLME \
        --conversation-ids-path $ids_file"
  done
  MERGE_SPECS+=("fullLME|$(IFS=,; echo "${LME_SHARD_DIRS[*]}")")
else
  log "WARNING: $LONGMEM_INPUT not found, skipping full LME study"
fi

# ── 2c. Llama-3.2-3B signal comparison (second 3B model) ─────────────────────
if [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  log "Planning: Llama-3.2-3B scale validation (second 3B model)"
  shard_dir_llama="results/reviewer_fixes/scale_llama32_3b/${RUN_PREFIX}_signal_swap_llama32_3b"
  enqueue_job "scale_llama32_3b" "$shard_dir_llama" \
    "$PYTHON_BIN -m paper3_codec.study \
      --study-name ${RUN_PREFIX}_signal_swap_llama32_3b \
      --model-keys llama32_3b \
      --input-path $HARDSET_INPUT \
      --families long_dependency,retrieval_heavy,code_conversation \
      --budgets $BUDGETS \
      --policies uniform,semantic,geometry,geometry_keep_compress_drop,semantic_keep_compress_drop \
      --recent-window 2 \
      --min-history 4 \
      --max-input-tokens 768 \
      --segment-span 2 \
      --output-root results/reviewer_fixes/scale_llama32_3b"
  MERGE_SPECS+=("scale_llama32_3b|$shard_dir_llama")
else
  log "WARNING: no HF_TOKEN set, skipping Llama-3.2-3B (gated model)"
  log "  Set HF_TOKEN=hf_xxx to enable the second 3B model"
fi

# ── 2d. MSC baselines with longllmlingua ─────────────────────────────────────
if [[ -f "$MSC_INPUT" ]]; then
  log "Planning: longllmlingua baseline on MSC"
  plan_dir_msc="$(plan_shards "$MSC_INPUT" "baselines_msc")"
  declare -a MSC_SHARD_DIRS=()
  for ((s=0; s<JOB_SHARDS; s++)); do
    ids_file="$plan_dir_msc/shard_${s}_ids.txt"
    [[ -s "$ids_file" ]] || continue
    shard_dir="results/reviewer_fixes/baselines/${RUN_PREFIX}_msc_shard${s}of${JOB_SHARDS}"
    MSC_SHARD_DIRS+=("$shard_dir")
    enqueue_job "baselines_msc_s${s}" "$shard_dir" \
      "$PYTHON_BIN -m june_fixes.baselines.baseline_study \
        --study-name ${RUN_PREFIX}_msc_shard${s}of${JOB_SHARDS} \
        --model-keys $MODEL_KEY \
        --input-path $MSC_INPUT \
        --budgets $BUDGETS \
        --policies uniform,longllmlingua,recency,lexical_tfidf,recency_keep_compress_drop \
        --recent-window 2 \
        --min-history 4 \
        --max-input-tokens 1024 \
        --segment-span 2 \
        --target-turn-stride 2 \
        --max-target-turns 16 \
        --output-root results/reviewer_fixes/baselines \
        --conversation-ids-path $ids_file"
  done
  MERGE_SPECS+=("baselines_msc|$(IFS=,; echo "${MSC_SHARD_DIRS[*]}")")
fi

PENDING_COUNT=$(ls "$QUEUE_PENDING"/*.job 2>/dev/null | wc -l | tr -d ' ')
log "Enqueued $PENDING_COUNT jobs across $GPU_COUNT GPUs"

# ── Step 3: GPU worker pool ───────────────────────────────────────────────────
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
    log "gpu $gpu_id preflight attempt $attempt/$GPU_PREFLIGHT_RETRIES failed, retrying..."
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
  worker_loop "${GPU_INDICES[$i]}" &
  WORKER_PIDS+=("$!")
done
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || true; done

FAILED_COUNT=$(ls "$QUEUE_FAILED"/*.job 2>/dev/null | wc -l | tr -d ' ')
if [[ "$FAILED_COUNT" -gt 0 ]]; then
  log "WARNING: $FAILED_COUNT jobs failed. Re-run this script to retry (completed shards are skipped)."
  log "Failed jobs:"
  ls "$QUEUE_FAILED"/*.job 2>/dev/null | while read -r f; do log "  $f"; done
fi

# ── Step 4: Merge shards ──────────────────────────────────────────────────────
log "=== STEP 4: Merging shards ==="

for spec in "${MERGE_SPECS[@]}"; do
  [[ -n "$spec" ]] || continue
  IFS='|' read -r tag dirs <<< "$spec"
  [[ -n "$dirs" ]] || continue
  log "Merging: $tag"
  case "$tag" in
    baselines_*)
      "$PYTHON_BIN" -m paper3_codec.merge_study_shards \
        --study-name "${RUN_PREFIX}_${tag}_merged" \
        --output-root results/reviewer_fixes/baselines \
        --shard-dirs "$dirs" || log "WARNING: merge failed for $tag"
      ;;
    fullLME)
      "$PYTHON_BIN" -m paper3_codec.merge_study_shards \
        --study-name "${RUN_PREFIX}_fullLME_merged" \
        --output-root results/reviewer_fixes/fullLME \
        --shard-dirs "$dirs" || log "WARNING: merge failed for fullLME"
      ;;
    scale_llama32_3b)
      log "(llama32_3b is a single-shard job, no merge needed)"
      ;;
  esac
done

# ── Step 5: Run pairwise reports ──────────────────────────────────────────────
log "=== STEP 5: Pairwise reports ==="

for study_dir in \
    "results/reviewer_fixes/baselines/${RUN_PREFIX}_baselines_hardset_merged" \
    "results/reviewer_fixes/baselines/${RUN_PREFIX}_baselines_msc_merged" \
    "results/reviewer_fixes/fullLME/${RUN_PREFIX}_fullLME_merged" \
    "results/reviewer_fixes/scale_llama32_3b/${RUN_PREFIX}_signal_swap_llama32_3b"; do
  [[ -f "$study_dir/evaluation_rows.csv" ]] || continue
  log "Pairwise report: $study_dir"
  bash scripts/run_paper3_pairwise_report.sh "$study_dir" || log "WARNING: pairwise report failed for $study_dir"
done

# ── Step 6: Summarize results ─────────────────────────────────────────────────
log "=== STEP 6: Writing summary ==="

SUMMARY_FILE="results/reviewer_fixes/summary_${RUN_PREFIX}.md"
{
  echo "# Reviewer Fix Results — $RUN_PREFIX"
  echo "Generated: $(date)"
  echo "Model: $MODEL_KEY | Budgets: $BUDGETS | GPUs: $GPU_COUNT"
  echo ""
  echo "## Jobs"
  echo "- Done: $(ls "$QUEUE_DONE"/*.job 2>/dev/null | wc -l | tr -d ' ')"
  echo "- Failed: $FAILED_COUNT"
  echo ""
  echo "## Output dirs"
  for spec in "${MERGE_SPECS[@]}"; do
    [[ -n "$spec" ]] || continue
    IFS='|' read -r tag _ <<< "$spec"
    echo "- $tag"
  done
  echo ""
  echo "## Logs"
  echo "Main log: $MAIN_LOG"
  echo "Per-job logs: $LOG_ROOT/"
} > "$SUMMARY_FILE"

log "Summary written to $SUMMARY_FILE"

# ── Step 7: Git commit and push ───────────────────────────────────────────────
log "=== STEP 7: Git commit & push ==="

if [[ "$SKIP_PUSH" == "1" ]]; then
  log "SKIP_PUSH=1, skipping git operations"
  log "All done. Results under results/reviewer_fixes/"
  exit 0
fi

[[ -n "${GITHUB_USER:-}" ]] || die "Set GITHUB_USER before running (needed for git push)"
[[ -n "${GITHUB_TOKEN:-}" ]] || die "Set GITHUB_TOKEN before running (needed for git push)"

COMMIT_MSG="Add reviewer-fix results: longllmlingua baseline, full LME, llama32_3b scale

- LongLLMLingua (question-aware) baseline on hard stress set and MSC
- Full LongMemEval-S study (no 40-turn truncation cap removed)
- Llama-3.2-3B second 3B model signal comparison
- Run prefix: $RUN_PREFIX | GPUs: $GPU_COUNT | $(date +%Y-%m-%d)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git config user.name "${GIT_NAME:-SteveMama}"
git config user.email "${GIT_EMAIL:-pranav@vizit.com}"

# stage only results — not raw benchmark downloads
git add \
  results/reviewer_fixes/ \
  "$SUMMARY_FILE" \
  june_fixes/baselines/baseline_study.py \
  paper1_geometry/modeling.py \
  scripts/install_reviewer_deps.sh \
  scripts/run_reviewer_fixes_multigpu.sh \
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
log "Results: results/reviewer_fixes/"
log "Summary: $SUMMARY_FILE"
log "Main log: $MAIN_LOG"

if [[ "$FAILED_COUNT" -gt 0 ]]; then
  log "Re-run this script to retry $FAILED_COUNT failed jobs (completed shards are cached)"
  exit 1
fi
