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
  echo "[run_paper3_harm_oracle_probe] could not find .venv/bin/python, python3, or python on PATH" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "[run_paper3_harm_oracle_probe] selected python does not have torch: $PYTHON_BIN" >&2
  echo "[run_paper3_harm_oracle_probe] create .venv and install dependencies first." >&2
  exit 1
fi
export PYTHON_BIN

if [[ $# -lt 3 ]]; then
  echo "Usage: run_paper3_harm_oracle_probe.sh <study-name> <benchmark-name> <input-jsonl> [models] [budgets] [limit-conversations] [target-turn-stride] [max-target-turns] [max-turns-per-conversation] [enable-attention-summary]" >&2
  exit 1
fi

RUN_NAME="$1"
BENCHMARK_NAME="$2"
INPUT_PATH="$3"
MODEL_KEYS="${4:-qwen25_15b}"
BUDGETS="${5:-0.20,0.35,0.50}"
LIMIT_CONVERSATIONS="${6:-24}"
TARGET_TURN_STRIDE="${7:-4}"
MAX_TARGET_TURNS="${8:-16}"
# Cap turns extracted per conversation. Critical for long-conversation benchmarks
# like LongMemEval — without this each conversation processes 100s of turns.
# Default: no cap (None). Pass an integer to truncate (e.g. 40).
MAX_TURNS_PER_CONVERSATION="${9:-}"
ENABLE_ATTENTION_SUMMARY="${10:-0}"

echo "[run_paper3_harm_oracle_probe] study=${RUN_NAME} benchmark=${BENCHMARK_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_harm_oracle_probe] input=${INPUT_PATH}" >&2
echo "[run_paper3_harm_oracle_probe] limit_conversations=${LIMIT_CONVERSATIONS} target_turn_stride=${TARGET_TURN_STRIDE} max_target_turns=${MAX_TARGET_TURNS} max_turns_per_conversation=${MAX_TURNS_PER_CONVERSATION:-none} enable_attention_summary=${ENABLE_ATTENTION_SUMMARY}" >&2

MAX_TURNS_ARG=""
if [[ -n "$MAX_TURNS_PER_CONVERSATION" ]]; then
  MAX_TURNS_ARG="--max-turns-per-conversation $MAX_TURNS_PER_CONVERSATION"
fi
ATTENTION_ARG=""
if [[ "$ENABLE_ATTENTION_SUMMARY" == "1" ]]; then
  ATTENTION_ARG="--enable-attention-summary"
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m paper3_codec.harm_oracle_study \
  --study-name "$RUN_NAME" \
  --benchmark-name "$BENCHMARK_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path "$INPUT_PATH" \
  --budgets "$BUDGETS" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768 \
  --limit-conversations "$LIMIT_CONVERSATIONS" \
  --target-turn-stride "$TARGET_TURN_STRIDE" \
  --max-target-turns "$MAX_TARGET_TURNS" \
  $MAX_TURNS_ARG \
  $ATTENTION_ARG

echo "[run_paper3_harm_oracle_probe] Done." >&2
