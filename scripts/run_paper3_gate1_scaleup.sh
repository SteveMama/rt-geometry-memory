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
  echo "[run_paper3_gate1_scaleup] could not find .venv/bin/python, python3, or python on PATH" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "[run_paper3_gate1_scaleup] selected python does not have torch: $PYTHON_BIN" >&2
  echo "[run_paper3_gate1_scaleup] create .venv and install dependencies first." >&2
  exit 1
fi
export PYTHON_BIN

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_gate1_scaleup.sh <msc-input-jsonl> <longmemeval-input-jsonl> [model-key] [budgets] [target-turn-stride] [max-target-turns] [longmemeval-max-turns-per-conversation] [run-prefix]" >&2
  exit 1
fi

MSC_INPUT="$1"
LONGMEM_INPUT="$2"
MODEL_KEY="${3:-qwen25_15b}"
BUDGETS="${4:-0.20,0.35,0.50}"
TARGET_TURN_STRIDE="${5:-4}"
MAX_TARGET_TURNS="${6:-16}"
LONGMEM_MAX_TURNS="${7:-40}"
RUN_PREFIX="${8:-paper3_gate1_scaleup}"
ENABLE_ORACLE_ATTENTION_SUMMARY="${ENABLE_ORACLE_ATTENTION_SUMMARY:-0}"

MSC_LIMIT=32
LONGMEM_LIMIT=12

if [[ ! -f "$MSC_INPUT" ]]; then
  echo "[run_paper3_gate1_scaleup] missing MSC input: $MSC_INPUT" >&2
  exit 1
fi
if [[ ! -f "$LONGMEM_INPUT" ]]; then
  echo "[run_paper3_gate1_scaleup] missing LongMemEval input: $LONGMEM_INPUT" >&2
  exit 1
fi

echo "[run_paper3_gate1_scaleup] model=${MODEL_KEY} budgets=${BUDGETS}" >&2
echo "[run_paper3_gate1_scaleup] MSC limit=${MSC_LIMIT} input=${MSC_INPUT}" >&2
echo "[run_paper3_gate1_scaleup] LongMemEval limit=${LONGMEM_LIMIT} input=${LONGMEM_INPUT} max_turns_per_conv=${LONGMEM_MAX_TURNS}" >&2
echo "[run_paper3_gate1_scaleup] oracle attention summaries=${ENABLE_ORACLE_ATTENTION_SUMMARY}" >&2

bash scripts/run_paper3_harm_oracle_probe.sh \
  "${RUN_PREFIX}_oracle_msc_valid_32conv" \
  msc_valid \
  "$MSC_INPUT" \
  "$MODEL_KEY" \
  "$BUDGETS" \
  "$MSC_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS" \
  "" \
  "$ENABLE_ORACLE_ATTENTION_SUMMARY"

bash scripts/run_paper3_gate1_refinement_probe.sh \
  "${RUN_PREFIX}_refinement_msc_valid_32conv" \
  "$MSC_INPUT" \
  "$MODEL_KEY" \
  "$BUDGETS" \
  "$MSC_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS"

bash scripts/run_paper3_harm_oracle_probe.sh \
  "${RUN_PREFIX}_oracle_longmemeval_s_cleaned_12conv" \
  longmemeval_s_cleaned \
  "$LONGMEM_INPUT" \
  "$MODEL_KEY" \
  "$BUDGETS" \
  "$LONGMEM_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS" \
  "$LONGMEM_MAX_TURNS" \
  "$ENABLE_ORACLE_ATTENTION_SUMMARY"

bash scripts/run_paper3_gate1_refinement_probe.sh \
  "${RUN_PREFIX}_refinement_longmemeval_s_cleaned_12conv" \
  "$LONGMEM_INPUT" \
  "$MODEL_KEY" \
  "$BUDGETS" \
  "$LONGMEM_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS" \
  "$LONGMEM_MAX_TURNS"

echo "[run_paper3_gate1_scaleup] complete" >&2
echo "  results/paper3/harm_oracle/${RUN_PREFIX}_oracle_msc_valid_32conv" >&2
echo "  results/paper3/studies/${RUN_PREFIX}_refinement_msc_valid_32conv" >&2
echo "  results/paper3/harm_oracle/${RUN_PREFIX}_oracle_longmemeval_s_cleaned_12conv" >&2
echo "  results/paper3/studies/${RUN_PREFIX}_refinement_longmemeval_s_cleaned_12conv" >&2
