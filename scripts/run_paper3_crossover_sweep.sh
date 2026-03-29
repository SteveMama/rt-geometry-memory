#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_crossover_sweep.sh <study-name> <benchmark-jsonl> [models] [budgets] [policies]" >&2
  exit 1
fi

RUN_NAME="$1"
BENCHMARK_PATH="$2"
MODEL_KEYS="${3:-qwen25_15b,llama32_3b}"
BUDGETS="${4:-0.20,0.24,0.28,0.32,0.35,0.38,0.42,0.46,0.50}"
POLICIES="${5:-uniform,semantic,geometry,geometry_keep_compress_drop}"

echo "[run_paper3_crossover_sweep] study=${RUN_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_crossover_sweep] hard-set + benchmark=${BENCHMARK_PATH}" >&2

python -m paper3_codec.study \
  --study-name "$RUN_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --extra-input-paths "$BENCHMARK_PATH" \
  --families "" \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 1024 \
  --segment-span 2

echo "[run_paper3_crossover_sweep] Building pairwise report..." >&2
bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"
