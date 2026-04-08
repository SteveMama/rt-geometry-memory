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
  echo "[run_paper3_gate1_refinement_probe] could not find .venv/bin/python, python3, or python on PATH" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "[run_paper3_gate1_refinement_probe] selected python does not have torch: $PYTHON_BIN" >&2
  echo "[run_paper3_gate1_refinement_probe] create .venv and install dependencies first." >&2
  exit 1
fi
export PYTHON_BIN

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_gate1_refinement_probe.sh <study-name> <input-jsonl> [models] [budgets] [limit-conversations] [target-turn-stride] [max-target-turns] [max-turns-per-conversation] [skip-conversations]" >&2
  exit 1
fi

RUN_NAME="$1"
INPUT_PATH="$2"
MODEL_KEYS="${3:-qwen25_15b}"
BUDGETS="${4:-0.20,0.35,0.50}"
LIMIT_CONVERSATIONS="${5:-24}"
TARGET_TURN_STRIDE="${6:-4}"
MAX_TARGET_TURNS="${7:-16}"
MAX_TURNS_PER_CONVERSATION="${8:-}"
SKIP_CONVERSATIONS="${9:-0}"
POLICIES="semantic,budget_aware_semantic_keep_compress_drop,semantic_ambient_geometry_keep_compress_drop,semantic_query_conditioned_geometry_keep_compress_drop"

echo "[run_paper3_gate1_refinement_probe] study=${RUN_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_gate1_refinement_probe] input=${INPUT_PATH}" >&2
echo "[run_paper3_gate1_refinement_probe] policies=${POLICIES}" >&2
echo "[run_paper3_gate1_refinement_probe] max_turns_per_conversation=${MAX_TURNS_PER_CONVERSATION:-none} skip_conversations=${SKIP_CONVERSATIONS}" >&2

MAX_TURNS_ARG=""
if [[ -n "$MAX_TURNS_PER_CONVERSATION" ]]; then
  MAX_TURNS_ARG="--max-turns-per-conversation $MAX_TURNS_PER_CONVERSATION"
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m paper3_codec.study \
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
  --skip-conversations "$SKIP_CONVERSATIONS" \
  --target-turn-stride "$TARGET_TURN_STRIDE" \
  --max-target-turns "$MAX_TARGET_TURNS" \
  $MAX_TURNS_ARG

echo "[run_paper3_gate1_refinement_probe] Building pairwise report..." >&2
bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"
echo "[run_paper3_gate1_refinement_probe] Done." >&2
