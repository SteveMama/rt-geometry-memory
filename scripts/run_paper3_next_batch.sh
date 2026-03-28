#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREFIX="${1:-paper3_batch_v1}"
FAIR_MODEL="${2:-qwen25_15b}"
PROBE_MODEL="${3:-qwen25_3b}"

bash scripts/run_paper3_fairness_sweep.sh "${PREFIX}_fairness" "$FAIR_MODEL"
bash scripts/run_paper3_3b_probe.sh "${PREFIX}_3b" "$PROBE_MODEL"

echo "Paper 3 next batch finished."
echo "Suggested publish commands:"
echo "  bash scripts/publish_artifact.sh --paper paper3 --source results/paper3/studies/${PREFIX}_fairness --name ${PREFIX}_fairness"
echo "  bash scripts/publish_artifact.sh --paper paper3 --source results/paper3/studies/${PREFIX}_3b --name ${PREFIX}_3b"
