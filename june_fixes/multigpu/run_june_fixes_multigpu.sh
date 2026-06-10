#!/usr/bin/env bash
# June-fixes multi-GPU orchestrator.
#
# Runs all GPU work from the review fixes through one filesystem job queue,
# in the same style as scripts/run_paper3_gate1_scaleup_multigpu.sh:
#   - balanced per-conversation shards (paper3_codec.plan_conversation_shards)
#   - one worker per GPU, jobs claimed by atomic mv, CUDA_VISIBLE_DEVICES pinning
#   - GPU preflight with retries
#   - resume: completed shards detected via progress.json, interrupted jobs
#     restored from running/ to pending/ on restart
#   - merge phase at the end
#
# Workloads (toggle via env, all default on where inputs exist):
#   RUN_BASELINES=1     fix 3: baseline policies on hard set + MSC (+LME)
#   RUN_QA=1            fix 1: QA accuracy over existing study dirs
#   RUN_CROSSFAMILY=1   fix 5: signal swap on smollm2_17b / llama32_3b
#
# Inputs (env):
#   HARDSET_INPUT   default paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl
#   MSC_INPUT       e.g. benchmarks/msc_valid_normalized.jsonl            (optional)
#   LONGMEM_INPUT   e.g. benchmarks/longmemeval_s_cleaned_normalized.jsonl (optional)
#   QA_SPECS        semicolon list of name:study_dir:input_jsonl for QA accuracy
#                   default derives from MSC/LME refinement study dirs if present
#   MODEL_KEY       default qwen25_15b
#   BUDGETS         default 0.20,0.35,0.50
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOCAL_VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$LOCAL_VENV_PYTHON" ]]; then PYTHON_BIN="$LOCAL_VENV_PYTHON"; else PYTHON_BIN="$(command -v python3 || command -v python)"; fi
fi
"$PYTHON_BIN" -c "import torch" >/dev/null 2>&1 || { echo "[june_fixes] python lacks torch: $PYTHON_BIN" >&2; exit 1; }
export PYTHON_BIN

RUN_PREFIX="${RUN_PREFIX:-june_fixes}"
MODEL_KEY="${MODEL_KEY:-qwen25_15b}"
BUDGETS="${BUDGETS:-0.20,0.35,0.50}"
HARDSET_INPUT="${HARDSET_INPUT:-paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl}"
MSC_INPUT="${MSC_INPUT:-benchmarks/msc_valid_normalized.jsonl}"
LONGMEM_INPUT="${LONGMEM_INPUT:-benchmarks/longmemeval_s_cleaned_normalized.jsonl}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_QA="${RUN_QA:-1}"
RUN_CROSSFAMILY="${RUN_CROSSFAMILY:-1}"
JOB_MULTIPLIER="${JOB_MULTIPLIER:-2}"
GPU_PREFLIGHT_RETRIES="${GPU_PREFLIGHT_RETRIES:-6}"
GPU_PREFLIGHT_SLEEP_SECONDS="${GPU_PREFLIGHT_SLEEP_SECONDS:-10}"
QA_STUDY_DIR_MSC="${QA_STUDY_DIR_MSC:-results/paper3/studies/paper3_gate1_scaleup_multigpu_refinement_msc_valid_32conv}"
QA_STUDY_DIR_LME="${QA_STUDY_DIR_LME:-results/paper3/studies/paper3_gate1_scaleup_multigpu_refinement_longmemeval_s_cleaned_12conv}"

mapfile -t GPU_INDICES < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d')
[[ ${#GPU_INDICES[@]} -gt 0 ]] || { echo "[june_fixes] no GPUs via nvidia-smi" >&2; exit 1; }
GPU_COUNT="${GPU_COUNT:-${#GPU_INDICES[@]}}"
[[ "$GPU_COUNT" -le ${#GPU_INDICES[@]} ]] || GPU_COUNT=${#GPU_INDICES[@]}
JOB_SHARDS=$(( GPU_COUNT * JOB_MULTIPLIER ))
echo "[june_fixes] GPUs=${GPU_INDICES[*]} using=$GPU_COUNT shards-per-workload=$JOB_SHARDS" >&2

QUEUE_ROOT="results/june_fixes/job_queue/${RUN_PREFIX}"
QUEUE_PENDING="$QUEUE_ROOT/pending"; QUEUE_RUNNING="$QUEUE_ROOT/running"
QUEUE_DONE="$QUEUE_ROOT/done"; QUEUE_FAILED="$QUEUE_ROOT/failed"
PLAN_ROOT="results/june_fixes/shard_plans/${RUN_PREFIX}"
LOG_ROOT="results/june_fixes/logs/${RUN_PREFIX}"
mkdir -p "$QUEUE_PENDING" "$QUEUE_RUNNING" "$QUEUE_DONE" "$QUEUE_FAILED" "$PLAN_ROOT" "$LOG_ROOT"
rm -f "$QUEUE_PENDING"/*.job "$QUEUE_FAILED"/*.job 2>/dev/null || true
for running_job in "$QUEUE_RUNNING"/*.job; do
  [[ -e "$running_job" ]] || continue
  base="$(basename "$running_job")"
  mv "$running_job" "$QUEUE_PENDING/${base%.gpu*.job}.job"
done

shard_complete() {
  local shard_dir="$1"
  [[ -f "$shard_dir/progress.json" ]] && \
    "$PYTHON_BIN" - "$shard_dir/progress.json" <<'EOF'
import json, sys
payload = json.load(open(sys.argv[1]))
sys.exit(0 if payload.get("status") == "complete" else 1)
EOF
}

enqueue_job() {  # name, shard_dir, cmd
  local name="$1" shard_dir="$2" cmd="$3"
  if shard_complete "$shard_dir"; then
    echo "[june_fixes] skip complete: $name" >&2
    return
  fi
  {
    printf 'SHARD_DIR=%q\n' "$shard_dir"
    printf 'LOG_PATH=%q\n' "$LOG_ROOT/${name}.log"
    printf 'CMD=%q\n' "$cmd"
  } > "$QUEUE_PENDING/${name}.job"
}

plan_shards() {  # input_jsonl, tag, max_turns(optional) -> writes shard id files, echoes plan dir
  local input="$1" tag="$2" max_turns="${3:-}"
  local plan_dir="$PLAN_ROOT/$tag"
  mkdir -p "$plan_dir"
  local extra=()
  [[ -n "$max_turns" ]] && extra+=(--max-turns-per-conversation "$max_turns")
  "$PYTHON_BIN" -m paper3_codec.plan_conversation_shards \
    --input-path "$input" \
    --shard-count "$JOB_SHARDS" \
    --target-turn-stride 1 \
    --output-dir "$plan_dir" \
    "${extra[@]}" >&2
  echo "$plan_dir"
}

# ---------------------------------------------------------------- baselines
declare -a BASELINE_MERGE_SPECS=()
if [[ "$RUN_BASELINES" == "1" ]]; then
  declare -a BASELINE_BENCHES=("hardset|$HARDSET_INPUT|")
  [[ -f "$MSC_INPUT" ]] && BASELINE_BENCHES+=("msc|$MSC_INPUT|")
  [[ -f "$LONGMEM_INPUT" ]] && BASELINE_BENCHES+=("lme|$LONGMEM_INPUT|40")
  for bench_spec in "${BASELINE_BENCHES[@]}"; do
    IFS='|' read -r tag input max_turns <<< "$bench_spec"
    [[ -f "$input" ]] || { echo "[june_fixes] missing $input, skipping baselines:$tag" >&2; continue; }
    plan_dir="$(plan_shards "$input" "baselines_$tag" "$max_turns")"
    shard_dirs=()
    for ((s=0; s<JOB_SHARDS; s++)); do
      ids_file="$plan_dir/shard_${s}_ids.txt"
      [[ -s "$ids_file" ]] || continue
      shard_dir="results/june_fixes/baselines/${RUN_PREFIX}_${tag}_shard${s}of${JOB_SHARDS}"
      shard_dirs+=("$shard_dir")
      extra=""
      [[ -n "$max_turns" ]] && extra="--max-turns-per-conversation $max_turns --target-turn-stride 4 --max-target-turns 16"
      enqueue_job "baseline_${tag}_s${s}" "$shard_dir" \
        "$PYTHON_BIN -m june_fixes.baselines.baseline_study \
          --study-name ${RUN_PREFIX}_${tag}_shard${s}of${JOB_SHARDS} \
          --model-keys $MODEL_KEY --input-path $input --budgets $BUDGETS \
          --output-root results/june_fixes/baselines \
          --conversation-ids-path $ids_file $extra"
    done
    BASELINE_MERGE_SPECS+=("${tag}|$(IFS=,; echo "${shard_dirs[*]}")")
  done
fi

# --------------------------------------------------------------------- QA
declare -a QA_MERGE_SPECS=()
if [[ "$RUN_QA" == "1" ]]; then
  QA_SPECS_DEFAULT=""
  [[ -d "$QA_STUDY_DIR_MSC" && -f "$MSC_INPUT" ]] && QA_SPECS_DEFAULT="msc:$QA_STUDY_DIR_MSC:$MSC_INPUT"
  if [[ -d "$QA_STUDY_DIR_LME" && -f "$LONGMEM_INPUT" ]]; then
    QA_SPECS_DEFAULT="${QA_SPECS_DEFAULT:+$QA_SPECS_DEFAULT;}lme:$QA_STUDY_DIR_LME:$LONGMEM_INPUT"
  fi
  QA_SPECS="${QA_SPECS:-$QA_SPECS_DEFAULT}"
  if [[ -z "$QA_SPECS" ]]; then
    echo "[june_fixes] no QA study dirs found; set QA_SPECS=name:study_dir:input_jsonl" >&2
  fi
  IFS=';' read -ra QA_ITEMS <<< "$QA_SPECS"
  for item in "${QA_ITEMS[@]}"; do
    [[ -n "$item" ]] || continue
    IFS=':' read -r tag study_dir input <<< "$item"
    [[ -f "$study_dir/evaluation_rows.csv" ]] || { echo "[june_fixes] no evaluation_rows.csv in $study_dir, skipping qa:$tag" >&2; continue; }
    plan_dir="$(plan_shards "$input" "qa_$tag")"
    shard_dirs=()
    for ((s=0; s<JOB_SHARDS; s++)); do
      ids_file="$plan_dir/shard_${s}_ids.txt"
      [[ -s "$ids_file" ]] || continue
      shard_dir="results/june_fixes/qa_accuracy/${RUN_PREFIX}_${tag}_shard${s}of${JOB_SHARDS}"
      shard_dirs+=("$shard_dir")
      enqueue_job "qa_${tag}_s${s}" "$shard_dir" \
        "$PYTHON_BIN -m june_fixes.qa_accuracy.qa_accuracy_study \
          --study-dir $study_dir --input-path $input --model-key $MODEL_KEY \
          --output-dir $shard_dir --conversation-ids-path $ids_file"
    done
    QA_MERGE_SPECS+=("${tag}|$(IFS=,; echo "${shard_dirs[*]}")")
  done
fi

# ------------------------------------------------------------- crossfamily
if [[ "$RUN_CROSSFAMILY" == "1" ]]; then
  CF_MODELS=("smollm2_17b")
  if [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then CF_MODELS+=("llama32_3b"); else
    echo "[june_fixes] no HF token; skipping gated llama32_3b" >&2; fi
  for cf_model in "${CF_MODELS[@]}"; do
    shard_dir="results/june_fixes/crossfamily/${RUN_PREFIX}_signal_swap_${cf_model}"
    enqueue_job "crossfamily_${cf_model}" "$shard_dir" \
      "$PYTHON_BIN -m paper3_codec.study \
        --study-name ${RUN_PREFIX}_signal_swap_${cf_model} \
        --model-keys $cf_model --input-path $HARDSET_INPUT \
        --families long_dependency,retrieval_heavy,code_conversation \
        --budgets $BUDGETS \
        --policies uniform,semantic,geometry,semantic_keep_compress_drop,geometry_keep_compress_drop \
        --recent-window 2 --min-history 4 --max-input-tokens 768 \
        --output-root results/june_fixes/crossfamily"
  done
fi

PENDING_COUNT=$(ls "$QUEUE_PENDING"/*.job 2>/dev/null | wc -l | tr -d ' ')
echo "[june_fixes] enqueued $PENDING_COUNT jobs" >&2

# ------------------------------------------------------------ worker pool
gpu_preflight() {
  local gpu_id="$1" attempt
  for ((attempt=1; attempt<=GPU_PREFLIGHT_RETRIES; attempt++)); do
    if CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" - <<'EOF'
import torch
assert torch.cuda.is_available()
torch.cuda.set_device(0)
x = torch.zeros(8, device="cuda"); torch.cuda.synchronize()
EOF
    then return 0; fi
    echo "[june_fixes] gpu $gpu_id preflight failed ($attempt), retrying" >&2
    sleep "$GPU_PREFLIGHT_SLEEP_SECONDS"
  done
  return 1
}

worker_loop() {
  local gpu_id="$1"
  gpu_preflight "$gpu_id" || { echo "[june_fixes] gpu $gpu_id unusable, worker exiting" >&2; return 1; }
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
    mkdir -p "$(dirname "$LOG_PATH")"
    echo "[june_fixes] gpu=$gpu_id job=$(basename "$claimed")" >&2
    if CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "$CMD" >>"$LOG_PATH" 2>&1; then
      mv "$claimed" "$QUEUE_DONE/$(basename "$claimed")"
    else
      echo "[june_fixes] FAILED job=$(basename "$claimed") log=$LOG_PATH" >&2
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
  echo "[june_fixes] $FAILED_COUNT jobs failed — rerun this script to retry (completed shards are skipped)" >&2
fi

# ------------------------------------------------------------------ merge
for spec in "${BASELINE_MERGE_SPECS[@]:-}"; do
  [[ -n "$spec" ]] || continue
  IFS='|' read -r tag dirs <<< "$spec"
  [[ -n "$dirs" ]] || continue
  "$PYTHON_BIN" -m paper3_codec.merge_study_shards \
    --study-name "${RUN_PREFIX}_baselines_${tag}_merged" \
    --output-root results/june_fixes/baselines \
    --shard-dirs "$dirs" || echo "[june_fixes] baseline merge failed for $tag" >&2
done
for spec in "${QA_MERGE_SPECS[@]:-}"; do
  [[ -n "$spec" ]] || continue
  IFS='|' read -r tag dirs <<< "$spec"
  [[ -n "$dirs" ]] || continue
  "$PYTHON_BIN" -m june_fixes.qa_accuracy.merge_qa_shards \
    --study-name "${RUN_PREFIX}_qa_${tag}_merged" \
    --output-root results/june_fixes/qa_accuracy \
    --shard-dirs "$dirs" || echo "[june_fixes] qa merge failed for $tag" >&2
done

echo "[june_fixes] all done. Outputs under results/june_fixes/" >&2
