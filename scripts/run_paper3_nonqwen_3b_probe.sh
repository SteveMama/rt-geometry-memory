#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${1:-paper3_nonqwen_3b_probe_v1}"
MODEL_KEYS="${2:-llama32_3b}"
BUDGETS="${3:-0.20,0.35,0.50}"
INPUT_PATH="${4:-paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl}"
POLICIES="${5:-uniform,geometry,geometry_keep_compress_drop}"
LIMIT_CONVERSATIONS="${6:-9}"
TARGET_TURN_STRIDE="${7:-1}"
MAX_TARGET_TURNS="${8:-64}"

echo "[run_paper3_nonqwen_3b_probe] study=${RUN_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_nonqwen_3b_probe] input=${INPUT_PATH}" >&2
echo "[run_paper3_nonqwen_3b_probe] limit_conversations=${LIMIT_CONVERSATIONS} target_turn_stride=${TARGET_TURN_STRIDE} max_target_turns=${MAX_TARGET_TURNS}" >&2

python -m paper3_codec.study \
  --study-name "$RUN_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path "$INPUT_PATH" \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 1024 \
  --segment-span 2 \
  --limit-conversations "$LIMIT_CONVERSATIONS" \
  --target-turn-stride "$TARGET_TURN_STRIDE" \
  --max-target-turns "$MAX_TARGET_TURNS"

echo "[run_paper3_nonqwen_3b_probe] Building pairwise report..." >&2
bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"

IFS=',' read -r -a MODEL_ARRAY <<< "$MODEL_KEYS"
for MODEL_KEY in "${MODEL_ARRAY[@]}"; do
  echo "[run_paper3_nonqwen_3b_probe] Memory-critical analysis for ${MODEL_KEY}..." >&2
  python -m paper3_codec.memory_critical_analysis \
    --evaluation-csv "results/paper3/studies/${RUN_NAME}/evaluation_rows.csv" \
    --input-path "$INPUT_PATH" \
    --model-key "$MODEL_KEY" \
    --budget-fraction 0.35 \
    --policy-name geometry_keep_compress_drop \
    --output-path "results/paper3/studies/${RUN_NAME}/memory_critical_${MODEL_KEY}_keep_compress_drop_b035.md"
done
