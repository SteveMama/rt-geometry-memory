#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODELS="${1:-qwen25_05b,qwen25_15b,smollm2_17b}"

IFS=',' read -r -a MODEL_ARRAY <<< "$MODELS"
for MODEL_KEY in "${MODEL_ARRAY[@]}"; do
  STUDY_NAME="behavior_stress_${MODEL_KEY}_cases"
  python -m paper2_memory.study \
    --study-name "$STUDY_NAME" \
    --model-keys "$MODEL_KEY" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --families long_dependency,retrieval_heavy,code_conversation \
    --budgets 0.20,0.35,0.50 \
    --recent-window 2 \
    --min-history 4 \
    --max-input-tokens 768

  python -m paper2_memory.case_analysis \
    --evaluation-csv "results/paper2/studies/${STUDY_NAME}/evaluation_rows.csv" \
    --behavior-csv "results/paper2/studies/${STUDY_NAME}/behavior_rows.csv" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --model-key "$MODEL_KEY" \
    --budget-fraction 0.35 \
    --top-n 5 \
    --output-path "results/paper2/studies/${STUDY_NAME}/case_analysis_${MODEL_KEY}_b035.md"

  python -m paper2_memory.memory_critical_analysis \
    --evaluation-csv "results/paper2/studies/${STUDY_NAME}/evaluation_rows.csv" \
    --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
    --model-key "$MODEL_KEY" \
    --budget-fraction 0.35 \
    --output-path "results/paper2/studies/${STUDY_NAME}/memory_critical_${MODEL_KEY}_b035.md"
done

python -m paper2_memory.cross_model_memory_summary \
  --evaluation-csvs "results/paper2/studies/behavior_stress_qwen25_05b_cases/evaluation_rows.csv,results/paper2/studies/behavior_stress_qwen25_15b_cases/evaluation_rows.csv,results/paper2/studies/behavior_stress_smollm2_17b_cases/evaluation_rows.csv" \
  --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
  --budget-fraction 0.35 \
  --output-md results/paper2/studies/cross_model_memory_summary_b035.md \
  --output-csv results/paper2/studies/cross_model_memory_summary_b035.csv
