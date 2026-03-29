#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/build_manuscript_assets.py
tectonic --outdir manuscript/build manuscript/paper_checkpoint.tex

echo "Built manuscript/build/paper_checkpoint.pdf"
