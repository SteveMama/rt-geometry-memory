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

compute_shards() {
  local total="$1"
  local shard_count="$2"
  local base=$(( total / shard_count ))
  local rem=$(( total % shard_count ))
  local offset=0
  SHARD_LIMITS=()
  SHARD_SKIPS=()
  for ((i=0; i<shard_count; i++)); do
    local size="$base"
    if (( i < rem )); then
      size=$(( size + 1 ))
    fi
    SHARD_LIMITS+=("$size")
    SHARD_SKIPS+=("$offset")
    offset=$(( offset + size ))
  done
}

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

launch_oracle_shards() {
  local benchmark_name="$1"
  local input_path="$2"
  local total_limit="$3"
  local max_turns="$4"
  local study_prefix="$5"
  compute_shards "$total_limit" "$GPU_COUNT"
  local pids=()
  local shard_dirs=()
  for ((i=0; i<GPU_COUNT; i++)); do
    local shard_limit="${SHARD_LIMITS[$i]}"
    local shard_skip="${SHARD_SKIPS[$i]}"
    if [[ "$shard_limit" -le 0 ]]; then
      continue
    fi
    local shard_name="${study_prefix}_shard${i}of${GPU_COUNT}"
    shard_dirs+=("results/paper3/harm_oracle/${shard_name}")
    echo "[multigpu][oracle] gpu=${GPU_INDICES[$i]} shard=${i} skip=${shard_skip} limit=${shard_limit} study=${shard_name}" >&2
    CUDA_VISIBLE_DEVICES="${GPU_INDICES[$i]}" bash scripts/run_paper3_harm_oracle_probe.sh \
      "$shard_name" \
      "$benchmark_name" \
      "$input_path" \
      "$MODEL_KEY" \
      "$BUDGETS" \
      "$shard_limit" \
      "$TARGET_TURN_STRIDE" \
      "$MAX_TARGET_TURNS" \
      "$max_turns" \
      "$ENABLE_ORACLE_ATTENTION_SUMMARY" \
      "$shard_skip" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$PYTHON_BIN" -m paper3_codec.merge_oracle_shards \
    --study-name "$study_prefix" \
    --benchmark-name "$benchmark_name" \
    --output-root "results/paper3/harm_oracle" \
    --shard-dirs "$(join_by_comma "${shard_dirs[@]}")"
}

launch_study_shards() {
  local input_path="$1"
  local total_limit="$2"
  local max_turns="$3"
  local study_prefix="$4"
  compute_shards "$total_limit" "$GPU_COUNT"
  local pids=()
  local shard_dirs=()
  for ((i=0; i<GPU_COUNT; i++)); do
    local shard_limit="${SHARD_LIMITS[$i]}"
    local shard_skip="${SHARD_SKIPS[$i]}"
    if [[ "$shard_limit" -le 0 ]]; then
      continue
    fi
    local shard_name="${study_prefix}_shard${i}of${GPU_COUNT}"
    shard_dirs+=("results/paper3/studies/${shard_name}")
    echo "[multigpu][study] gpu=${GPU_INDICES[$i]} shard=${i} skip=${shard_skip} limit=${shard_limit} study=${shard_name}" >&2
    CUDA_VISIBLE_DEVICES="${GPU_INDICES[$i]}" bash scripts/run_paper3_gate1_refinement_probe.sh \
      "$shard_name" \
      "$input_path" \
      "$MODEL_KEY" \
      "$BUDGETS" \
      "$shard_limit" \
      "$TARGET_TURN_STRIDE" \
      "$MAX_TARGET_TURNS" \
      "$max_turns" \
      "$shard_skip" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$PYTHON_BIN" -m paper3_codec.merge_study_shards \
    --study-name "$study_prefix" \
    --output-root "results/paper3/studies" \
    --shard-dirs "$(join_by_comma "${shard_dirs[@]}")"
}

launch_oracle_shards "msc_valid" "$MSC_INPUT" "$MSC_LIMIT" "" "${RUN_PREFIX}_oracle_msc_valid_32conv"
launch_study_shards "$MSC_INPUT" "$MSC_LIMIT" "" "${RUN_PREFIX}_refinement_msc_valid_32conv"
launch_oracle_shards "longmemeval_s_cleaned" "$LONGMEM_INPUT" "$LONGMEM_LIMIT" "$LONGMEM_MAX_TURNS" "${RUN_PREFIX}_oracle_longmemeval_s_cleaned_12conv"
launch_study_shards "$LONGMEM_INPUT" "$LONGMEM_LIMIT" "$LONGMEM_MAX_TURNS" "${RUN_PREFIX}_refinement_longmemeval_s_cleaned_12conv"

echo "[run_paper3_gate1_scaleup_multigpu] complete" >&2
