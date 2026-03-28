#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper2_custom_benchmark.sh <study-name> <input-jsonl> [models] [families] [budgets] [policies]" >&2
  exit 1
fi

STUDY_NAME="$1"
INPUT_PATH="$2"
MODEL_KEYS="${3:-qwen25_05b,qwen25_15b,smollm2_17b}"
FAMILIES="${4:-long_dependency,retrieval_heavy,code_conversation}"
BUDGETS="${5:-0.20,0.35,0.50}"
POLICIES="${6:-uniform,semantic,lexical,geometry,geometry_segment_actions}"

python -m paper2_memory.study \
  --study-name "$STUDY_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path "$INPUT_PATH" \
  --families "$FAMILIES" \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768
