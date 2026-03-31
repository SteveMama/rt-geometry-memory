#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 3 ]]; then
  echo "Usage: run_paper3_harm_oracle_probe.sh <study-name> <benchmark-name> <input-jsonl> [models] [budgets] [limit-conversations] [target-turn-stride] [max-target-turns]" >&2
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

echo "[run_paper3_harm_oracle_probe] study=${RUN_NAME} benchmark=${BENCHMARK_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_harm_oracle_probe] input=${INPUT_PATH}" >&2
echo "[run_paper3_harm_oracle_probe] limit_conversations=${LIMIT_CONVERSATIONS} target_turn_stride=${TARGET_TURN_STRIDE} max_target_turns=${MAX_TARGET_TURNS}" >&2

PYTHONUNBUFFERED=1 python -u -m paper3_codec.harm_oracle_study \
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
  --max-target-turns "$MAX_TARGET_TURNS"

echo "[run_paper3_harm_oracle_probe] Done." >&2
