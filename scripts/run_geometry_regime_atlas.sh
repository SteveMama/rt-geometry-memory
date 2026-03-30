#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: bash scripts/run_geometry_regime_atlas.sh <study_name> <input_path> <model_key> [extra_input_paths_csv] [cluster_count] [limit_conversations]"
  exit 1
fi

STUDY_NAME="$1"
INPUT_PATH="$2"
MODEL_KEY="$3"
EXTRA_INPUT_PATHS="${4:-}"
CLUSTER_COUNT="${5:-4}"
LIMIT_CONVERSATIONS="${6:-16}"

echo "[run_geometry_regime_atlas] study=${STUDY_NAME} model=${MODEL_KEY}"
echo "[run_geometry_regime_atlas] input=${INPUT_PATH}"
if [[ -n "${EXTRA_INPUT_PATHS}" ]]; then
  echo "[run_geometry_regime_atlas] extra_input_paths=${EXTRA_INPUT_PATHS}"
fi
echo "[run_geometry_regime_atlas] cluster_count=${CLUSTER_COUNT} limit_conversations=${LIMIT_CONVERSATIONS}"

PYTHONUNBUFFERED=1 python -u -m paper1_geometry.run_regime_atlas \
  --study-name "${STUDY_NAME}" \
  --model-key "${MODEL_KEY}" \
  --input-path "${INPUT_PATH}" \
  --extra-input-paths "${EXTRA_INPUT_PATHS}" \
  --cluster-count "${CLUSTER_COUNT}" \
  --limit-conversations "${LIMIT_CONVERSATIONS}" \
  --max-turns 96 \
  --max-input-tokens 1024 \
  --max-segment-len 8 \
  --min-segment-len 3

echo "[run_geometry_regime_atlas] Done."
