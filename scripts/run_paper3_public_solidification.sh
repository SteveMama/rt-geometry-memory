#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: run_paper3_public_solidification.sh <prefix> <benchmark-jsonl> [qwen-model] [nonqwen-model]" >&2
  exit 1
fi

PREFIX="$1"
BENCHMARK_PATH="$2"
QWEN_MODEL="${3:-qwen25_15b}"
NONQWEN_MODEL="${4:-llama32_3b}"

echo "[run_paper3_public_solidification] Stage 1/3: public benchmark on ${QWEN_MODEL}" >&2
bash scripts/run_paper3_public_benchmark.sh "${PREFIX}_public_benchmark" "$BENCHMARK_PATH" "$QWEN_MODEL"
echo "[run_paper3_public_solidification] Stage 2/3: non-Qwen 3B probe on ${NONQWEN_MODEL}" >&2
bash scripts/run_paper3_nonqwen_3b_probe.sh "${PREFIX}_nonqwen_3b" "$NONQWEN_MODEL"
echo "[run_paper3_public_solidification] Stage 3/3: dense crossover sweep on ${QWEN_MODEL},${NONQWEN_MODEL}" >&2
bash scripts/run_paper3_crossover_sweep.sh "${PREFIX}_crossover" "$BENCHMARK_PATH" "${QWEN_MODEL},${NONQWEN_MODEL}"

echo "Paper 3 public solidification batch finished."
echo "Suggested publish commands:"
echo "  bash scripts/publish_artifact.sh --paper paper3 --source results/paper3/studies/${PREFIX}_public_benchmark --name ${PREFIX}_public_benchmark"
echo "  bash scripts/publish_artifact.sh --paper paper3 --source results/paper3/studies/${PREFIX}_nonqwen_3b --name ${PREFIX}_nonqwen_3b"
echo "  bash scripts/publish_artifact.sh --paper paper3 --source results/paper3/studies/${PREFIX}_crossover --name ${PREFIX}_crossover"
