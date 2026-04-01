#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_gate1_real.sh <msc-input-jsonl> <longmemeval-input-jsonl> [models] [budgets] [msc-limit-conversations] [longmemeval-limit-conversations] [target-turn-stride] [max-target-turns]" >&2
  exit 1
fi

MSC_INPUT="$1"
LONGMEM_INPUT="$2"
MODEL_KEYS="${3:-qwen25_15b}"
BUDGETS="${4:-0.20,0.35,0.50}"
MSC_LIMIT="${5:-24}"
LONGMEM_LIMIT="${6:-24}"
TARGET_TURN_STRIDE="${7:-4}"
MAX_TARGET_TURNS="${8:-16}"

if [[ ! -f "$MSC_INPUT" ]]; then
  echo "[run_paper3_gate1_real] missing MSC input: $MSC_INPUT" >&2
  exit 1
fi
if [[ ! -f "$LONGMEM_INPUT" ]]; then
  echo "[run_paper3_gate1_real] missing LongMemEval input: $LONGMEM_INPUT" >&2
  exit 1
fi

echo "[run_paper3_gate1_real] models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_gate1_real] MSC input=${MSC_INPUT} limit=${MSC_LIMIT}" >&2
echo "[run_paper3_gate1_real] LongMemEval input=${LONGMEM_INPUT} limit=${LONGMEM_LIMIT}" >&2

echo "[run_paper3_gate1_real] Running MSC oracle headroom..." >&2
bash scripts/run_paper3_harm_oracle_probe.sh \
  paper3_gate1_oracle_msc_valid \
  msc_valid \
  "$MSC_INPUT" \
  "$MODEL_KEYS" \
  "$BUDGETS" \
  "$MSC_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS"

echo "[run_paper3_gate1_real] Running MSC refinement study..." >&2
bash scripts/run_paper3_gate1_refinement_probe.sh \
  paper3_gate1_refinement_msc_valid \
  "$MSC_INPUT" \
  "$MODEL_KEYS" \
  "$BUDGETS" \
  "$MSC_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS"

echo "[run_paper3_gate1_real] Running LongMemEval oracle headroom..." >&2
bash scripts/run_paper3_harm_oracle_probe.sh \
  paper3_gate1_oracle_longmemeval_s_cleaned \
  longmemeval_s_cleaned \
  "$LONGMEM_INPUT" \
  "$MODEL_KEYS" \
  "$BUDGETS" \
  "$LONGMEM_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS"

echo "[run_paper3_gate1_real] Running LongMemEval refinement study..." >&2
bash scripts/run_paper3_gate1_refinement_probe.sh \
  paper3_gate1_refinement_longmemeval_s_cleaned \
  "$LONGMEM_INPUT" \
  "$MODEL_KEYS" \
  "$BUDGETS" \
  "$LONGMEM_LIMIT" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS"

echo "[run_paper3_gate1_real] Gate 1 decision surface complete." >&2
echo "[run_paper3_gate1_real] Oracle outputs:" >&2
echo "  results/paper3/harm_oracle/paper3_gate1_oracle_msc_valid" >&2
echo "  results/paper3/harm_oracle/paper3_gate1_oracle_longmemeval_s_cleaned" >&2
echo "[run_paper3_gate1_real] Refinement study outputs:" >&2
echo "  results/paper3/studies/paper3_gate1_refinement_msc_valid" >&2
echo "  results/paper3/studies/paper3_gate1_refinement_longmemeval_s_cleaned" >&2
