#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STUDY_NAME="${1:-paper2_fairness_sweep_v1}"
MODEL_KEYS="${2:-qwen25_15b}"
BUDGETS="${3:-0.24,0.28,0.32,0.35,0.38,0.42,0.46,0.50}"
POLICIES="${4:-uniform,semantic,geometry,geometry_segment_actions}"

python -m paper2_memory.study \
  --study-name "$STUDY_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768
