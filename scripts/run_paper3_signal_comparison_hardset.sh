#!/usr/bin/env bash
set -euo pipefail

# Experiment 1: geometry_KCD vs semantic_KCD on the hard stress set.
#
# This is the critical head-to-head that answers whether the geometry signal
# is specifically necessary for constraint-critical memory tasks, or whether
# the KCD codec structure alone does the work regardless of signal.
#
# Prediction: semantic_KCD will be weaker because semantic scoring cannot
# distinguish a user constraint turn from its assistant echo — they are
# topically identical. Geometry CAN because the user constraint creates a
# larger state displacement.
#
# Usage:
#   bash scripts/run_paper3_signal_comparison_hardset.sh [run-name] [models] [budgets]
#
# Defaults:
#   run-name: paper3_signal_comparison_hardset_v1
#   models:   qwen25_15b
#   budgets:  0.20,0.35,0.50

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${1:-paper3_signal_comparison_hardset_v1}"
MODEL_KEYS="${2:-qwen25_15b}"
BUDGETS="${3:-0.20,0.35,0.50}"

POLICIES="uniform,semantic,geometry,geometry_keep_compress_drop,semantic_keep_compress_drop"

echo "[run_paper3_signal_comparison_hardset] study=${RUN_NAME}" >&2
echo "[run_paper3_signal_comparison_hardset] models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_signal_comparison_hardset] policies=${POLICIES}" >&2

PYTHONUNBUFFERED=1 python -u -m paper3_codec.study \
  --study-name "$RUN_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --families long_dependency,retrieval_heavy,code_conversation \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768 \
  --segment-span 2

echo "[run_paper3_signal_comparison_hardset] Building pairwise report..." >&2
bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"

IFS=',' read -r -a MODEL_ARRAY <<< "$MODEL_KEYS"
for MODEL_KEY in "${MODEL_ARRAY[@]}"; do
  python -m paper3_codec.memory_critical_analysis \
    --evaluation-csv "results/paper3/studies/${RUN_NAME}/evaluation_rows.csv" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --model-key "$MODEL_KEY" \
    --budget-fraction 0.35 \
    --policy-name geometry_keep_compress_drop \
    --output-path "results/paper3/studies/${RUN_NAME}/memory_critical_${MODEL_KEY}_geometry_kcd_b035.md"

  python -m paper3_codec.memory_critical_analysis \
    --evaluation-csv "results/paper3/studies/${RUN_NAME}/evaluation_rows.csv" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --model-key "$MODEL_KEY" \
    --budget-fraction 0.35 \
    --policy-name semantic_keep_compress_drop \
    --output-path "results/paper3/studies/${RUN_NAME}/memory_critical_${MODEL_KEY}_semantic_kcd_b035.md"
done

echo "[run_paper3_signal_comparison_hardset] Done." >&2
echo "[run_paper3_signal_comparison_hardset] Output: results/paper3/studies/${RUN_NAME}" >&2
echo "" >&2
echo "Reading the results:" >&2
echo "  Pairwise report: results/paper3/studies/${RUN_NAME}/pairwise_report.md" >&2
echo "  Memory critical (geometry-KCD): results/paper3/studies/${RUN_NAME}/memory_critical_*_geometry_kcd_b035.md" >&2
echo "  Memory critical (semantic-KCD): results/paper3/studies/${RUN_NAME}/memory_critical_*_semantic_kcd_b035.md" >&2
echo "" >&2
echo "What to look for:" >&2
echo "  If geometry_KCD beats semantic_KCD on the hard set -> signal is doing real work" >&2
echo "  If semantic_KCD ties geometry_KCD -> codec structure was doing the work, not signal" >&2
