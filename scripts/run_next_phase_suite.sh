#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREFIX="${1:-next_phase}"
COMP_MODELS="${2:-qwen25_05b,qwen25_15b,smollm2_17b}"
PROBE_MODEL="${3:-qwen25_3b}"

bash scripts/run_paper2_competitor_matrix.sh "${PREFIX}_paper2_competitor_matrix" "$COMP_MODELS"
bash scripts/run_paper2_3b_probe.sh "${PREFIX}_paper2_3b_probe" "$PROBE_MODEL"
bash scripts/run_paper2_fairness_sweep.sh "${PREFIX}_paper2_fairness_sweep"
bash scripts/run_paper3_head_to_head.sh "${PREFIX}_paper3_head_to_head" "$COMP_MODELS"

echo "Next-phase suite finished."
echo "Suggested publish commands:"
echo "  bash scripts/publish_artifact.sh --paper paper2 --source results/paper2/studies/${PREFIX}_paper2_competitor_matrix --name ${PREFIX}_paper2_competitor_matrix"
echo "  bash scripts/publish_artifact.sh --paper paper2 --source results/paper2/studies/${PREFIX}_paper2_3b_probe --name ${PREFIX}_paper2_3b_probe"
echo "  bash scripts/publish_artifact.sh --paper paper2 --source results/paper2/studies/${PREFIX}_paper2_fairness_sweep --name ${PREFIX}_paper2_fairness_sweep"
echo "  bash scripts/publish_artifact.sh --paper paper3 --source results/paper3/studies/${PREFIX}_paper3_head_to_head --name ${PREFIX}_paper3_head_to_head"
