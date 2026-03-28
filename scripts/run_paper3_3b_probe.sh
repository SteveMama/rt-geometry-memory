#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${1:-paper3_3b_probe_v1}"
MODEL_KEYS="${2:-qwen25_3b}"
BUDGETS="${3:-0.20,0.35,0.50}"
SEGMENT_SPAN="${4:-2}"

python -m paper3_codec.study \
  --study-name "$RUN_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets "$BUDGETS" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 1024 \
  --segment-span "$SEGMENT_SPAN"

bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"

IFS=',' read -r -a MODEL_ARRAY <<< "$MODEL_KEYS"
for MODEL_KEY in "${MODEL_ARRAY[@]}"; do
  python -m paper3_codec.memory_critical_analysis \
    --evaluation-csv "results/paper3/studies/${RUN_NAME}/evaluation_rows.csv" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --model-key "$MODEL_KEY" \
    --budget-fraction 0.35 \
    --policy-name geometry_keep_compress_drop \
    --output-path "results/paper3/studies/${RUN_NAME}/memory_critical_${MODEL_KEY}_keep_compress_drop_b035.md"

  python -m paper3_codec.memory_critical_analysis \
    --evaluation-csv "results/paper3/studies/${RUN_NAME}/evaluation_rows.csv" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --model-key "$MODEL_KEY" \
    --budget-fraction 0.35 \
    --policy-name geometry_segment_actions \
    --output-path "results/paper3/studies/${RUN_NAME}/memory_critical_${MODEL_KEY}_segment_actions_b035.md"
done
