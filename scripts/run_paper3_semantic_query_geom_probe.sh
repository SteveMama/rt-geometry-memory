#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_semantic_query_geom_probe.sh <study-name> <input-jsonl> [models] [budgets] [limit-conversations] [target-turn-stride] [max-target-turns]" >&2
  exit 1
fi

RUN_NAME="$1"
INPUT_PATH="$2"
MODEL_KEYS="${3:-qwen25_15b}"
BUDGETS="${4:-0.20,0.35,0.50}"
LIMIT_CONVERSATIONS="${5:-24}"
TARGET_TURN_STRIDE="${6:-4}"
MAX_TARGET_TURNS="${7:-16}"
POLICIES="uniform,semantic,geometry,support_aware_geometry_keep_compress_drop,semantic_keep_compress_drop,budget_aware_semantic_keep_compress_drop,query_conditioned_geometry_v2,query_conditioned_geometry_keep_compress_drop_v2,semantic_query_conditioned_geometry_keep_compress_drop,semantic_ambient_geometry_keep_compress_drop,semantic_query_conditioned_geometry_keep_compress_drop_no_query,semantic_query_conditioned_geometry_keep_compress_drop_no_support"

echo "[run_paper3_semantic_query_geom_probe] study=${RUN_NAME} models=${MODEL_KEYS} budgets=${BUDGETS}" >&2
echo "[run_paper3_semantic_query_geom_probe] input=${INPUT_PATH}" >&2
echo "[run_paper3_semantic_query_geom_probe] policies=${POLICIES}" >&2

PYTHONUNBUFFERED=1 python -u -m paper3_codec.study \
  --study-name "$RUN_NAME" \
  --model-keys "$MODEL_KEYS" \
  --input-path "$INPUT_PATH" \
  --families "" \
  --budgets "$BUDGETS" \
  --policies "$POLICIES" \
  --recent-window 2 \
  --min-history 4 \
  --max-input-tokens 768 \
  --segment-span 3 \
  --limit-conversations "$LIMIT_CONVERSATIONS" \
  --target-turn-stride "$TARGET_TURN_STRIDE" \
  --max-target-turns "$MAX_TARGET_TURNS"

echo "[run_paper3_semantic_query_geom_probe] Building pairwise report..." >&2
bash scripts/run_paper3_pairwise_report.sh "results/paper3/studies/${RUN_NAME}"
echo "[run_paper3_semantic_query_geom_probe] Done." >&2
