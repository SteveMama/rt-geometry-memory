#!/bin/zsh
set -euo pipefail

cd /Users/pranav/Documents/RT
source .venv/bin/activate
python -m paper1_geometry.study \
  --study-name expanded_v8_final \
  --model-keys qwen25_05b,qwen25_15b,smollm2_17b \
  --input-path paper1_geometry/assets/paper1_study_conversations.jsonl \
  --extra-input-paths paper1_geometry/assets/paper1_h2_stress_conversations.jsonl \
  --max-turns 7 \
  --max-input-tokens 768
