#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$LOCAL_VENV_PYTHON" ]]; then
    PYTHON_BIN="$LOCAL_VENV_PYTHON"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[run_paper3_gate1_scaleup_multigpu] could not find .venv/bin/python, python3, or python on PATH" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "[run_paper3_gate1_scaleup_multigpu] selected python does not have torch: $PYTHON_BIN" >&2
  exit 1
fi
export PYTHON_BIN

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_gate1_scaleup_multigpu.sh <msc-input-jsonl> <longmemeval-input-jsonl> [model-key] [budgets] [target-turn-stride] [max-target-turns] [longmemeval-max-turns-per-conversation] [run-prefix]" >&2
  exit 1
fi

MSC_INPUT="$1"
LONGMEM_INPUT="$2"
MODEL_KEY="${3:-qwen25_15b}"
BUDGETS="${4:-0.20,0.35,0.50}"
TARGET_TURN_STRIDE="${5:-4}"
MAX_TARGET_TURNS="${6:-16}"
LONGMEM_MAX_TURNS="${7:-40}"
RUN_PREFIX="${8:-paper3_gate1_scaleup_multigpu}"
ENABLE_ORACLE_ATTENTION_SUMMARY="${ENABLE_ORACLE_ATTENTION_SUMMARY:-0}"
GPU_COUNT="${GPU_COUNT:-0}"
JOB_MULTIPLIER="${JOB_MULTIPLIER:-2}"

MSC_LIMIT=32
LONGMEM_LIMIT=12

if [[ ! -f "$MSC_INPUT" || ! -f "$LONGMEM_INPUT" ]]; then
  echo "[run_paper3_gate1_scaleup_multigpu] missing input benchmark file" >&2
  exit 1
fi

mapfile -t GPU_INDICES < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d')
if [[ ${#GPU_INDICES[@]} -eq 0 ]]; then
  echo "[run_paper3_gate1_scaleup_multigpu] no visible GPUs found via nvidia-smi" >&2
  exit 1
fi
if [[ "$GPU_COUNT" -le 0 || "$GPU_COUNT" -gt ${#GPU_INDICES[@]} ]]; then
  GPU_COUNT="${#GPU_INDICES[@]}"
fi

echo "[run_paper3_gate1_scaleup_multigpu] GPUs=${GPU_INDICES[*]} using=$GPU_COUNT model=${MODEL_KEY}" >&2

QUEUE_ROOT="results/paper3/job_queue/${RUN_PREFIX}"
QUEUE_PENDING="$QUEUE_ROOT/pending"
QUEUE_RUNNING="$QUEUE_ROOT/running"
QUEUE_DONE="$QUEUE_ROOT/done"
QUEUE_FAILED="$QUEUE_ROOT/failed"
mkdir -p "$QUEUE_PENDING" "$QUEUE_RUNNING" "$QUEUE_DONE" "$QUEUE_FAILED"
rm -f "$QUEUE_PENDING"/* "$QUEUE_RUNNING"/* "$QUEUE_DONE"/* "$QUEUE_FAILED"/* 2>/dev/null || true

JOB_SHARDS=$(( GPU_COUNT * JOB_MULTIPLIER ))
if [[ "$JOB_SHARDS" -lt "$GPU_COUNT" ]]; then
  JOB_SHARDS="$GPU_COUNT"
fi

join_by_comma() {
  local first=1
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      printf '%s' "$item"
      first=0
    else
      printf ',%s' "$item"
    fi
  done
}

declare -a ORACLE_MSC_DIRS=()
declare -a STUDY_MSC_DIRS=()
declare -a ORACLE_LONGMEM_DIRS=()
declare -a STUDY_LONGMEM_DIRS=()

write_job_file() {
  local job_path="$1"
  local kind="$2"
  local study_name="$3"
  local input_path="$4"
  local benchmark_name="$5"
  local max_turns="$6"
  local shard_ids_path="$7"
  local shard_dir="$8"
  local log_path="$9"
  cat > "$job_path" <<EOF
KIND="$kind"
STUDY_NAME="$study_name"
INPUT_PATH="$input_path"
BENCHMARK_NAME="$benchmark_name"
MAX_TURNS="$max_turns"
SHARD_IDS_PATH="$shard_ids_path"
SHARD_DIR="$shard_dir"
LOG_PATH="$log_path"
EOF
}

prepare_oracle_jobs() {
  local benchmark_name="$1"
  local input_path="$2"
  local total_limit="$3"
  local max_turns="$4"
  local study_prefix="$5"
  local target_array_name="$6"
  local plan_dir="results/paper3/shard_plans/${study_prefix}"
  mkdir -p "$plan_dir"
  local max_turns_arg=()
  if [[ -n "$max_turns" ]]; then
    max_turns_arg=(--max-turns-per-conversation "$max_turns")
  fi
  "$PYTHON_BIN" -m paper3_codec.plan_conversation_shards \
    --input-path "$input_path" \
    --limit-conversations "$total_limit" \
    --target-turn-stride "$TARGET_TURN_STRIDE" \
    --max-target-turns "$MAX_TARGET_TURNS" \
    --shard-count "$JOB_SHARDS" \
    --output-dir "$plan_dir" \
    "${max_turns_arg[@]}"
  for ((i=0; i<JOB_SHARDS; i++)); do
    local shard_ids_path="${plan_dir}/shard_${i}_ids.txt"
    if [[ ! -s "$shard_ids_path" ]]; then
      continue
    fi
    local shard_name="${study_prefix}_shard${i}of${JOB_SHARDS}"
    local shard_dir="results/paper3/harm_oracle/${shard_name}"
    local log_path="${shard_dir}/worker.log"
    mkdir -p "$shard_dir"
    eval "$target_array_name+=(\"$shard_dir\")"
    local job_path="${QUEUE_PENDING}/${shard_name}.job"
    write_job_file "$job_path" "oracle" "$shard_name" "$input_path" "$benchmark_name" "$max_turns" "$shard_ids_path" "$shard_dir" "$log_path"
    echo "[queue][oracle] staged job=${job_path} ids=${shard_ids_path} study=${shard_name}" >&2
  done
}

prepare_study_jobs() {
  local input_path="$1"
  local total_limit="$2"
  local max_turns="$3"
  local study_prefix="$4"
  local target_array_name="$5"
  local plan_dir="results/paper3/shard_plans/${study_prefix}"
  mkdir -p "$plan_dir"
  local max_turns_arg=()
  if [[ -n "$max_turns" ]]; then
    max_turns_arg=(--max-turns-per-conversation "$max_turns")
  fi
  "$PYTHON_BIN" -m paper3_codec.plan_conversation_shards \
    --input-path "$input_path" \
    --limit-conversations "$total_limit" \
    --target-turn-stride "$TARGET_TURN_STRIDE" \
    --max-target-turns "$MAX_TARGET_TURNS" \
    --shard-count "$JOB_SHARDS" \
    --output-dir "$plan_dir" \
    "${max_turns_arg[@]}"
  for ((i=0; i<JOB_SHARDS; i++)); do
    local shard_ids_path="${plan_dir}/shard_${i}_ids.txt"
    if [[ ! -s "$shard_ids_path" ]]; then
      continue
    fi
    local shard_name="${study_prefix}_shard${i}of${JOB_SHARDS}"
    local shard_dir="results/paper3/studies/${shard_name}"
    local log_path="${shard_dir}/worker.log"
    mkdir -p "$shard_dir"
    eval "$target_array_name+=(\"$shard_dir\")"
    local job_path="${QUEUE_PENDING}/${shard_name}.job"
    write_job_file "$job_path" "study" "$shard_name" "$input_path" "" "$max_turns" "$shard_ids_path" "$shard_dir" "$log_path"
    echo "[queue][study] staged job=${job_path} ids=${shard_ids_path} study=${shard_name}" >&2
  done
}

worker_loop() {
  local gpu_id="$1"
  local worker_log="$QUEUE_ROOT/worker_gpu${gpu_id}.log"
  touch "$worker_log"
  while true; do
    local claimed=""
    local pending_path
    for pending_path in "$QUEUE_PENDING"/*.job; do
      [[ -e "$pending_path" ]] || break
      local basename
      basename="$(basename "$pending_path")"
      local target="$QUEUE_RUNNING/${basename%.job}.gpu${gpu_id}.job"
      if mv "$pending_path" "$target" 2>/dev/null; then
        claimed="$target"
        break
      fi
    done
    if [[ -z "$claimed" ]]; then
      break
    fi

    # shellcheck source=/dev/null
    source "$claimed"
    echo "[worker][gpu=${gpu_id}] starting kind=${KIND} study=${STUDY_NAME} ids=${SHARD_IDS_PATH}" | tee -a "$worker_log"
    local exit_code=0
    if [[ "$KIND" == "oracle" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu_id" bash scripts/run_paper3_harm_oracle_probe.sh \
        "$STUDY_NAME" \
        "$BENCHMARK_NAME" \
        "$INPUT_PATH" \
        "$MODEL_KEY" \
        "$BUDGETS" \
        "999999" \
        "$TARGET_TURN_STRIDE" \
        "$MAX_TARGET_TURNS" \
        "$MAX_TURNS" \
        "$ENABLE_ORACLE_ATTENTION_SUMMARY" \
        "0" \
        "$SHARD_IDS_PATH" \
        2>&1 | tee "$LOG_PATH"
      exit_code=${PIPESTATUS[0]}
    else
      CUDA_VISIBLE_DEVICES="$gpu_id" bash scripts/run_paper3_gate1_refinement_probe.sh \
        "$STUDY_NAME" \
        "$INPUT_PATH" \
        "$MODEL_KEY" \
        "$BUDGETS" \
        "999999" \
        "$TARGET_TURN_STRIDE" \
        "$MAX_TARGET_TURNS" \
        "$MAX_TURNS" \
        "0" \
        "$SHARD_IDS_PATH" \
        2>&1 | tee "$LOG_PATH"
      exit_code=${PIPESTATUS[0]}
    fi

    if [[ "$exit_code" -eq 0 ]]; then
      mv "$claimed" "$QUEUE_DONE/$(basename "$claimed")"
      echo "[worker][gpu=${gpu_id}] completed kind=${KIND} study=${STUDY_NAME}" | tee -a "$worker_log"
    else
      mv "$claimed" "$QUEUE_FAILED/$(basename "$claimed")"
      echo "[worker][gpu=${gpu_id}] failed kind=${KIND} study=${STUDY_NAME} exit=${exit_code}" | tee -a "$worker_log"
      return "$exit_code"
    fi
  done
}

prepare_oracle_jobs "msc_valid" "$MSC_INPUT" "$MSC_LIMIT" "" "${RUN_PREFIX}_oracle_msc_valid_32conv" ORACLE_MSC_DIRS
prepare_study_jobs "$MSC_INPUT" "$MSC_LIMIT" "" "${RUN_PREFIX}_refinement_msc_valid_32conv" STUDY_MSC_DIRS
prepare_oracle_jobs "longmemeval_s_cleaned" "$LONGMEM_INPUT" "$LONGMEM_LIMIT" "$LONGMEM_MAX_TURNS" "${RUN_PREFIX}_oracle_longmemeval_s_cleaned_12conv" ORACLE_LONGMEM_DIRS
prepare_study_jobs "$LONGMEM_INPUT" "$LONGMEM_LIMIT" "$LONGMEM_MAX_TURNS" "${RUN_PREFIX}_refinement_longmemeval_s_cleaned_12conv" STUDY_LONGMEM_DIRS

echo "[run_paper3_gate1_scaleup_multigpu] queued jobs=$(find "$QUEUE_PENDING" -type f -name '*.job' | wc -l | tr -d ' ')" >&2

declare -a WORKER_PIDS=()
for ((i=0; i<GPU_COUNT; i++)); do
  worker_loop "${GPU_INDICES[$i]}" &
  WORKER_PIDS+=("$!")
done
for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid"
done

if compgen -G "$QUEUE_FAILED/*.job" > /dev/null; then
  echo "[run_paper3_gate1_scaleup_multigpu] failed jobs remain in $QUEUE_FAILED" >&2
  exit 1
fi

"$PYTHON_BIN" -m paper3_codec.merge_oracle_shards \
  --study-name "${RUN_PREFIX}_oracle_msc_valid_32conv" \
  --benchmark-name "msc_valid" \
  --output-root "results/paper3/harm_oracle" \
  --shard-dirs "$(join_by_comma "${ORACLE_MSC_DIRS[@]}")"

"$PYTHON_BIN" -m paper3_codec.merge_study_shards \
  --study-name "${RUN_PREFIX}_refinement_msc_valid_32conv" \
  --output-root "results/paper3/studies" \
  --shard-dirs "$(join_by_comma "${STUDY_MSC_DIRS[@]}")"

"$PYTHON_BIN" -m paper3_codec.merge_oracle_shards \
  --study-name "${RUN_PREFIX}_oracle_longmemeval_s_cleaned_12conv" \
  --benchmark-name "longmemeval_s_cleaned" \
  --output-root "results/paper3/harm_oracle" \
  --shard-dirs "$(join_by_comma "${ORACLE_LONGMEM_DIRS[@]}")"

"$PYTHON_BIN" -m paper3_codec.merge_study_shards \
  --study-name "${RUN_PREFIX}_refinement_longmemeval_s_cleaned_12conv" \
  --output-root "results/paper3/studies" \
  --shard-dirs "$(join_by_comma "${STUDY_LONGMEM_DIRS[@]}")"

echo "[run_paper3_gate1_scaleup_multigpu] complete" >&2
