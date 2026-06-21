#!/usr/bin/env bash
# run_msc_1000_kcd_multigpu.sh
#
# Runs KCD signal variants + component ablations on full 1000-conv MSC.
#
# Addresses two reviewer criticisms:
#   1. KCD signal variants were only evaluated on 50 conversations.
#      External baselines used 1000.  This run closes that gap.
#   2. No component ablation of sem_qcg was provided.
#      sem_qcg_no_query and sem_qcg_no_support are included here.
#
# Results saved to:
#   results/msc_1000_kcd/
#
# Required env vars:
#   GITHUB_USER   — GitHub username
#   GITHUB_TOKEN  — PAT with repo write scope
#
# Optional env vars:
#   MODEL_KEY          — default: qwen25_15b
#   BUDGETS            — default: 0.20,0.35,0.50
#   GPU_COUNT          — override auto-detected count
#   WORKERS_PER_GPU    — default: 4
#   JOB_MULTIPLIER     — shards per GPU (default: 4)
#   SKIP_DOWNLOAD      — set to 1 to skip benchmark download
#   SKIP_PUSH          — set to 1 to skip git push
#   EXTRACT_CACHE_ROOT — large volume for .npz hidden-state cache
#
# Usage:
#   GITHUB_USER=SteveMama GITHUB_TOKEN=ghp_xxx \
#     bash scripts/run_msc_1000_kcd_multigpu.sh

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
  echo "[msc_1000_kcd] ERROR: python at $PYTHON_BIN has no torch" >&2
  exit 1
}
export PYTHON_BIN

# ── Config ─────────────────────────────────────────────────────────────────────
RUN_PREFIX="${RUN_PREFIX:-msc_1000_kcd}"
MODEL_KEY="${MODEL_KEY:-qwen25_15b}"
BUDGETS="${BUDGETS:-0.20,0.35,0.50}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
JOB_MULTIPLIER="${JOB_MULTIPLIER:-4}"
GPU_PREFLIGHT_RETRIES="${GPU_PREFLIGHT_RETRIES:-6}"
GPU_PREFLIGHT_SLEEP="${GPU_PREFLIGHT_SLEEP:-10}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

MSC_INPUT="benchmarks/msc_valid_normalized.jsonl"

[[ -n "${EXTRACT_CACHE_ROOT:-}" ]] && export EXTRACT_CACHE_ROOT

# ── Result directories ─────────────────────────────────────────────────────────
RESULTS_ROOT="results/msc_1000_kcd"
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
log() { echo "[$(ts)] [msc_1000_kcd] $*" >&2; }
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

# Recover any jobs interrupted mid-run
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

# ── Step 1: Download MSC if missing ───────────────────────────────────────────
log "=== STEP 1: Benchmark ==="

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  log "SKIP_DOWNLOAD=1, skipping"
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
fi
[[ -f "$MSC_INPUT" ]] || die "MSC benchmark missing: $MSC_INPUT"

# ── Step 2: Plan + enqueue shards ─────────────────────────────────────────────
log "=== STEP 2: Planning shards ==="

log "Planning: MSC 1000-conv KCD variants + ablations"
plan_dir="$(plan_shards "$MSC_INPUT" "msc_1000_kcd")"

declare -a SHARD_DIRS=()
for ((s=0; s<JOB_SHARDS; s++)); do
  ids_file="$plan_dir/shard_${s}_ids.txt"
  [[ -s "$ids_file" ]] || continue
  shard_dir="$RESULTS_ROOT/${RUN_PREFIX}_shard${s}of${JOB_SHARDS}"
  SHARD_DIRS+=("$shard_dir")
  enqueue_job "msc_s${s}" "$shard_dir" \
    "$PYTHON_BIN -m paper3_codec.study \
      --study-name ${RUN_PREFIX}_shard${s}of${JOB_SHARDS} \
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
      --output-root $RESULTS_ROOT \
      --conversation-ids-path $ids_file"
done

PENDING_COUNT=$(ls "$QUEUE_PENDING"/*.job 2>/dev/null | wc -l | tr -d ' ') || PENDING_COUNT=0
log "Enqueued $PENDING_COUNT shards across $GPU_COUNT GPUs (shards already complete are skipped)"

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
DONE_COUNT=$(ls "$QUEUE_DONE"/*.job 2>/dev/null | wc -l | tr -d ' ') || DONE_COUNT=0
if [[ "$FAILED_COUNT" -gt 0 ]]; then
  log "WARNING: $FAILED_COUNT shards failed. Re-run to retry (done shards are skipped)."
  ls "$QUEUE_FAILED"/*.job 2>/dev/null | while read -r f; do log "  $f"; done
fi

# ── Step 4: Merge shards ───────────────────────────────────────────────────────
log "=== STEP 4: Merging shards ==="

MERGED_DIR="$RESULTS_ROOT/${RUN_PREFIX}_merged"
SHARD_DIRS_CSV="$(IFS=,; echo "${SHARD_DIRS[*]}")"

"$PYTHON_BIN" -m paper3_codec.merge_study_shards \
  --study-name "${RUN_PREFIX}_merged" \
  --output-root "$RESULTS_ROOT" \
  --shard-dirs  "$SHARD_DIRS_CSV" \
  || die "merge failed"

log "Merged → $MERGED_DIR"

# ── Step 5: Pairwise significance reports ──────────────────────────────────────
log "=== STEP 5: Pairwise reports ==="

if [[ -f "$MERGED_DIR/evaluation_rows.csv" ]]; then
  bash scripts/run_paper3_pairwise_report.sh "$MERGED_DIR" \
    || log "WARNING: pairwise report failed"
  log "Pairwise report → $MERGED_DIR/pairwise_report.md"
else
  log "WARNING: no evaluation_rows.csv — skipping pairwise report"
fi

# ── Step 6: Summary ────────────────────────────────────────────────────────────
log "=== STEP 6: Writing summary ==="

SUMMARY_FILE="$RESULTS_ROOT/summary_${RUN_PREFIX}.md"
{
  echo "# MSC 1000-conv KCD Results — $RUN_PREFIX"
  echo "Generated: $(date)"
  echo "Model: $MODEL_KEY | Budgets: $BUDGETS | GPUs: $GPU_COUNT"
  echo ""
  echo "## What this run addresses"
  echo ""
  echo "### Criticism 1: KCD signal variants at full MSC scale"
  echo "Previous KCD study: n=50. External baselines: n=1000."
  echo "This run: n=1000 for all KCD variants + ablations."
  echo ""
  echo "### Criticism 2: sem_qcg component ablation"
  echo "Policies included:"
  echo "  uniform"
  echo "  geometry_keep_compress_drop"
  echo "  semantic_keep_compress_drop"
  echo "  semantic_query_conditioned_geometry_keep_compress_drop  (sem_qcg full)"
  echo "  sem_qcg_no_query   — removes query-conditioned projection"
  echo "  sem_qcg_no_support — removes support-score weighting"
  echo ""
  echo "## Job outcomes"
  echo "- Done   : $DONE_COUNT"
  echo "- Failed : $FAILED_COUNT"
  echo ""
  echo "## Key comparisons (see pairwise_report.md)"
  echo "  sem_qcg vs semantic_KCD      (primary signal claim)"
  echo "  sem_qcg vs sem_qcg_no_query  (query projection ablation)"
  echo "  sem_qcg vs sem_qcg_no_support (support-score ablation)"
  echo ""
  echo "## Logs"
  echo "Main log : $MAIN_LOG"
  echo "Per-shard: $LOG_ROOT/"
} > "$SUMMARY_FILE"
cat "$SUMMARY_FILE"

# ── Step 7: Git commit & push ──────────────────────────────────────────────────
log "=== STEP 7: Git commit & push ==="

if [[ "$SKIP_PUSH" == "1" ]]; then
  log "SKIP_PUSH=1, skipping git operations"
  log "=== ALL DONE === Results: $RESULTS_ROOT/"
  exit 0
fi

[[ -n "${GITHUB_USER:-}" ]]  || die "Set GITHUB_USER before running"
[[ -n "${GITHUB_TOKEN:-}" ]] || die "Set GITHUB_TOKEN before running"

git config user.name  "${GIT_NAME:-SteveMama}"
git config user.email "${GIT_EMAIL:-pranav@vizit.com}"

git add "$RESULTS_ROOT/" "scripts/run_msc_1000_kcd_multigpu.sh" 2>/dev/null || true

COMMIT_MSG="Add MSC 1000-conv KCD results: signal variants + component ablation

Addresses two reviewer criticisms identified in code audit:

1. KCD signal variants at full 1000-conv MSC scale
   Previous KCD study was n=50; external baselines were n=1000.
   This run adds geometry_KCD, semantic_KCD, sem_qcg on all 1000
   MSC conversations at the same parameters as the baseline run.

2. sem_qcg component ablation
   sem_qcg_no_query  — drops query-conditioned projection
   sem_qcg_no_support — drops support-score weighting
   Within-study ablation showing which signal component is load-bearing.

Run: $RUN_PREFIX | Model: $MODEL_KEY | GPUs: $GPU_COUNT
Done: $DONE_COUNT | Failed: $FAILED_COUNT | $(date +%Y-%m-%d)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

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
log "Results  : $RESULTS_ROOT/"
log "Merged   : $MERGED_DIR/"
log "Pairwise : $MERGED_DIR/pairwise_report.md"
log "Summary  : $SUMMARY_FILE"

if [[ "$FAILED_COUNT" -gt 0 ]]; then
  log "Re-run to retry $FAILED_COUNT failed shards."
  exit 1
fi
