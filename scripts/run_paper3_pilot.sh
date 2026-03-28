#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${1:-paper3_pilot_v1}"
MODEL_KEY="${2:-qwen25_05b}"

python -m paper3_codec.run_paper3 \
  --run-name "$RUN_NAME" \
  --model-key "$MODEL_KEY" \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets 0.20,0.35,0.50 \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768
