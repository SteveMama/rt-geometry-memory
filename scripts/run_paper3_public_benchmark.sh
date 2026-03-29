#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_public_benchmark.sh <study-name> <input-jsonl> [models] [budgets] [policies]" >&2
  exit 1
fi

RUN_NAME="$1"
INPUT_PATH="$2"
MODEL_KEYS="${3:-qwen25_15b}"
BUDGETS="${4:-0.20,0.35,0.50}"
POLICIES="${5:-uniform,semantic,geometry,geometry_segment_actions,geometry_keep_compress_drop}"

echo "[run_paper3_public_benchmark] study=${RUN_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_public_benchmark] input=${INPUT_PATH}" >&2

python -m paper3_codec.study \
  --study-name "$RUN_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path "$INPUT_PATH" \
  --families "" \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 1024 \
  --segment-span 2

echo "[run_paper3_public_benchmark] Building pairwise report..." >&2
bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"
