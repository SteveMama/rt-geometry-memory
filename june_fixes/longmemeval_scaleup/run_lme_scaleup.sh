#!/usr/bin/env bash
# Review fix #6: the LongMemEval slice in the paper is 12 conversations
# truncated to 40 turns, which guts a long-memory benchmark. This wrapper
# re-runs the existing (already multi-GPU, queued, resumable) Gate 1 scale-up
# with a substantially less truncated LongMemEval slice:
#   - LONGMEM_LIMIT: 12 -> 40 conversations
#   - longmemeval-max-turns-per-conversation: 40 -> 80
# MSC settings are left at the published 32-conversation configuration so the
# MSC numbers stay comparable.
#
# Usage:
#   bash june_fixes/longmemeval_scaleup/run_lme_scaleup.sh \
#     <msc_valid_normalized.jsonl> <longmemeval_s_cleaned_normalized.jsonl> [model-key]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_lme_scaleup.sh <msc-jsonl> <longmemeval-jsonl> [model-key]" >&2
  exit 1
fi

MSC_INPUT="$1"
LONGMEM_INPUT="$2"
MODEL_KEY="${3:-qwen25_15b}"
BUDGETS="${BUDGETS:-0.20,0.35,0.50}"
TARGET_TURN_STRIDE="${TARGET_TURN_STRIDE:-4}"
MAX_TARGET_TURNS="${MAX_TARGET_TURNS:-16}"
LONGMEM_MAX_TURNS="${LONGMEM_MAX_TURNS:-80}"
RUN_PREFIX="${RUN_PREFIX:-june_lme_scaleup}"

export MSC_LIMIT="${MSC_LIMIT:-32}"
export LONGMEM_LIMIT="${LONGMEM_LIMIT:-40}"

echo "[lme_scaleup] longmem: limit=$LONGMEM_LIMIT max_turns=$LONGMEM_MAX_TURNS model=$MODEL_KEY" >&2

bash scripts/run_paper3_gate1_scaleup_multigpu.sh \
  "$MSC_INPUT" \
  "$LONGMEM_INPUT" \
  "$MODEL_KEY" \
  "$BUDGETS" \
  "$TARGET_TURN_STRIDE" \
  "$MAX_TARGET_TURNS" \
  "$LONGMEM_MAX_TURNS" \
  "$RUN_PREFIX" \
  "$MSC_LIMIT" \
  "$LONGMEM_LIMIT"

echo "[lme_scaleup] done. Merged outputs under results/paper3 with prefix $RUN_PREFIX" >&2
echo "[lme_scaleup] re-run june_fixes.answer_harm_oracle.answer_harm_gate1 on the new candidate_rows.csv" >&2
