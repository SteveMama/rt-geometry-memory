#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_semantic_object_probe.sh <study-name> <input-jsonl> [oracle-study-dir] [models] [budgets] [limit-conversations] [target-turn-stride] [max-target-turns]" >&2
  exit 1
fi

RUN_NAME="$1"
INPUT_PATH="$2"
ORACLE_STUDY_DIR="${3:-}"
MODEL_KEYS="${4:-qwen25_15b}"
BUDGETS="${5:-0.20,0.35,0.50}"
LIMIT_CONVERSATIONS="${6:-24}"
TARGET_TURN_STRIDE="${7:-4}"
MAX_TARGET_TURNS="${8:-16}"
PREDICTOR_DIR="results/paper3/harm_predictor_models/${RUN_NAME}"
BASE_POLICIES="semantic,budget_aware_semantic_keep_compress_drop,semantic_object_keep_compress_drop"
POLICIES="$BASE_POLICIES"
EXTRA_ARGS=()

if [[ -n "$ORACLE_STUDY_DIR" ]]; then
  echo "[run_paper3_semantic_object_probe] training object-aware harm predictor from ${ORACLE_STUDY_DIR}/candidate_rows.csv" >&2
  python -m paper3_codec.harm_predictor \
    --candidate-csv "${ORACLE_STUDY_DIR}/candidate_rows.csv" \
    --output-dir "${PREDICTOR_DIR}"
  POLICIES="${POLICIES},semantic_object_harm_keep_compress_drop"
  EXTRA_ARGS+=(--harm-predictor-path "${PREDICTOR_DIR}/harm_predictor.pt")
fi

echo "[run_paper3_semantic_object_probe] study=${RUN_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_semantic_object_probe] input=${INPUT_PATH}" >&2
echo "[run_paper3_semantic_object_probe] policies=${POLICIES}" >&2

PYTHONUNBUFFERED=1 python -u -m paper3_codec.study \
  --study-name "$RUN_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path "$INPUT_PATH" \
  --families "" \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768 \
  --segment-span 3 \
  --limit-conversations "$LIMIT_CONVERSATIONS" \
  --target-turn-stride "$TARGET_TURN_STRIDE" \
  --max-target-turns "$MAX_TARGET_TURNS" \
  "${EXTRA_ARGS[@]}"

echo "[run_paper3_semantic_object_probe] Building pairwise report..." >&2
bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"
echo "[run_paper3_semantic_object_probe] Done." >&2
