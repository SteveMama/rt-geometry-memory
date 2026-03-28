#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STUDY_NAME="${1:-behavior_stress_v1_rerun}"
MODELS="${2:-qwen25_05b,qwen25_15b,smollm2_17b}"

python -m paper2_memory.study \
  --study-name "$STUDY_NAME" \
  --model-keys "$MODELS" \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets 0.20,0.35,0.50 \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768
