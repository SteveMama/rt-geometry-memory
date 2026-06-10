#!/usr/bin/env bash
# Build the anonymized supplementary bundle for the ARR submission
# (review fix #8, item 13): code + hard stress set, no results/artifacts,
# no git history, and a hard failure if any identifying string survives.
#
# Usage: bash june_fixes/manuscript/make_anonymous_bundle.sh [output.zip]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ZIP="${1:-anonymous_supplementary.zip}"
STAGE_DIR="$(mktemp -d)/rt_supplementary"
mkdir -p "$STAGE_DIR"

INCLUDE_DIRS=(
  paper1_geometry
  paper2_memory
  paper3_codec
  june_fixes
  scripts
)
INCLUDE_FILES=(
  pyproject.toml
)

for dir in "${INCLUDE_DIRS[@]}"; do
  rsync -a \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
    --exclude 'results' --exclude 'artifacts' --exclude '*.zip' \
    --exclude '*.npz' --exclude '*.log' \
    "$dir" "$STAGE_DIR/"
done
for file in "${INCLUDE_FILES[@]}"; do
  cp "$file" "$STAGE_DIR/"
done

cat > "$STAGE_DIR/README_SUPPLEMENTARY.md" <<'EOF'
# Anonymous Supplementary Material

Code and data for "Signal-Conditioned Memory Compression for Multi-Turn
Conversations" (ARR submission).

- paper1_geometry/   geometry extraction, characterization, conversation assets
                     (includes the 36-conversation hard stress set under
                     paper1_geometry/assets/)
- paper2_memory/     budgeted retention policies and studies
- paper3_codec/      KCD codec, policies, oracle studies, statistics
- june_fixes/        QA-accuracy evaluation, baselines, answer-harm oracle,
                     multiple-comparison correction, regime detector
- scripts/           single-run and multi-GPU entry points

Setup: `pip install -e .` then see each package's module docstrings.
Public benchmarks (MSC, LoCoMo, LongMemEval) are fetched by
scripts/download_public_benchmark.py under their original licenses.
EOF

# Anonymity gate: fail loudly if identifying strings survive.
IDENTIFYING_PATTERN='SteveMama|pkompally|kompally|[Pp]ranav|northeastern'
if MATCHES=$(grep -rIlE "$IDENTIFYING_PATTERN" "$STAGE_DIR" 2>/dev/null); then
  echo "[make_anonymous_bundle] FATAL: identifying strings found in:" >&2
  echo "$MATCHES" >&2
  exit 1
fi

(cd "$(dirname "$STAGE_DIR")" && zip -qr "$ROOT_DIR/$OUTPUT_ZIP" "$(basename "$STAGE_DIR")")
echo "[make_anonymous_bundle] wrote $ROOT_DIR/$OUTPUT_ZIP"
unzip -l "$ROOT_DIR/$OUTPUT_ZIP" | tail -3
