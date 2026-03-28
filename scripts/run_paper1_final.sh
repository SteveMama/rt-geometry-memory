#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STUDY_NAME="${1:-expanded_v8_final_rerun}"
MODELS="${2:-qwen25_05b,qwen25_15b,smollm2_17b}"

python -m paper1_geometry.study \
  --study-name "$STUDY_NAME" \
  --model-keys "$MODELS" \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
