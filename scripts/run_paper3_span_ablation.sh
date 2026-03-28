#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREFIX="${1:-paper3_span_ablation_v1}"
MODEL_KEYS="${2:-qwen25_15b}"
SPANS="${3:-1,2,3,4}"
BUDGETS="${4:-0.20,0.35,0.50}"

IFS=',' read -r -a SPAN_ARRAY <<< "$SPANS"
for SPAN in "${SPAN_ARRAY[@]}"; do
  RUN_NAME="${PREFIX}_span${SPAN}"
  python -m paper3_codec.study \
    --study-name "$RUN_NAME" \
    --model-keys "$MODEL_KEYS" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --families long_dependency,retrieval_heavy,code_conversation \
    --budgets "$BUDGETS" \
    --recent-window 2 \
    --min-history 4 \
    --max-input-tokens 768 \
    --segment-span "$SPAN"
  bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"
done
