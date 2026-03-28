#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: run_paper3_pairwise_report.sh <study-dir>" >&2
  exit 1
fi

STUDY_DIR="$1"

python -m paper3_codec.pairwise_analysis \
  --evaluation-csv "${STUDY_DIR}/evaluation_rows.csv" \
  --behavior-csv "${STUDY_DIR}/behavior_rows.csv" \
  --output-json "${STUDY_DIR}/pairwise_summary.json" \
  --output-md "${STUDY_DIR}/pairwise_report.md"
