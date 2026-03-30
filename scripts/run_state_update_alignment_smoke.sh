#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/run_state_update_alignment_smoke.sh <study_name> <model_key> [max_input_tokens]"
  exit 1
fi

study_name="$1"
model_key="$2"
max_input_tokens="${3:-768}"

output_dir="results/state_update_alignment/${study_name}"

echo "[run_state_update_alignment_smoke] study=${study_name} model=${model_key} max_input_tokens=${max_input_tokens}"
echo "[run_state_update_alignment_smoke] input=benchmarks/state_update_synthetic_conversations.jsonl"
PYTHONUNBUFFERED=1 python -u scripts/run_state_update_alignment_check.py \
  --input-path benchmarks/state_update_synthetic_conversations.jsonl \
  --labels-path benchmarks/state_update_synthetic_labels.json \
  --model-key "${model_key}" \
  --max-input-tokens "${max_input_tokens}" \
  --output-dir "${output_dir}"
echo "[run_state_update_alignment_smoke] Done. Outputs in ${output_dir}"
