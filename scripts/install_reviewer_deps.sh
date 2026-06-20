#!/usr/bin/env bash
# install_reviewer_deps.sh
#
# Installs all dependencies needed for the reviewer-fix experiments on a
# multi-GPU cloud instance (CUDA 11.8+ / 12.x).
#
# Run once before launching run_reviewer_fixes_multigpu.sh.
#
# Usage:
#   bash scripts/install_reviewer_deps.sh          # auto-detect CUDA
#   CUDA_VERSION=12.1 bash scripts/install_reviewer_deps.sh
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-}"
PYTHON="${PYTHON_BIN:-python3}"

echo "[deps] Python: $($PYTHON --version)" >&2
echo "[deps] Detecting CUDA..." >&2

if [[ -z "$CUDA_VERSION" ]]; then
  if command -v nvcc &>/dev/null; then
    CUDA_VERSION="$(nvcc --version | grep 'release' | sed 's/.*release //' | sed 's/,.*//')"
    echo "[deps] nvcc reports CUDA $CUDA_VERSION" >&2
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION="$(nvidia-smi | grep 'CUDA Version' | sed 's/.*CUDA Version: //' | awk '{print $1}')"
    echo "[deps] nvidia-smi reports CUDA $CUDA_VERSION" >&2
  else
    echo "[deps] WARNING: no CUDA detected, installing CPU-only torch" >&2
    CUDA_VERSION="cpu"
  fi
fi

CUDA_MAJOR="${CUDA_VERSION%%.*}"

# ── core project deps ────────────────────────────────────────────────────────
echo "[deps] Installing core project..." >&2
$PYTHON -m pip install -e ".[dev]" --quiet 2>/dev/null || \
  $PYTHON -m pip install -e . --quiet

# ── torch with correct CUDA index ───────────────────────────────────────────
echo "[deps] Installing torch (CUDA $CUDA_VERSION)..." >&2
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  $PYTHON -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu -q
elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
  $PYTHON -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121 -q
else
  $PYTHON -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu118 -q
fi

# ── transformers + accelerate ────────────────────────────────────────────────
echo "[deps] Installing transformers stack..." >&2
$PYTHON -m pip install \
  "transformers>=4.45.0" \
  "accelerate>=0.30.0" \
  "huggingface_hub>=0.22.0" \
  "tokenizers>=0.19" \
  "datasets>=2.18" \
  "sentencepiece" \
  "protobuf" \
  -q

# ── flash attention 2 (major speedup, skip gracefully if build fails) ────────
echo "[deps] Trying flash-attn-2..." >&2
if [[ "$CUDA_VERSION" != "cpu" ]]; then
  $PYTHON -m pip install "flash-attn>=2.5.0" --no-build-isolation -q 2>/dev/null && \
    echo "[deps] flash-attn-2 installed ✓" >&2 || \
    echo "[deps] flash-attn-2 build failed, continuing without it" >&2
fi

# ── xformers (optional, helps on some GPU configs) ───────────────────────────
if [[ "$CUDA_VERSION" != "cpu" ]]; then
  $PYTHON -m pip install xformers --quiet 2>/dev/null || true
fi

# ── llmlingua (needed for longllmlingua baseline) ────────────────────────────
echo "[deps] Installing llmlingua..." >&2
$PYTHON -m pip install llmlingua -q

# ── eval / science deps ──────────────────────────────────────────────────────
echo "[deps] Installing eval stack..." >&2
$PYTHON -m pip install \
  "numpy>=2.0" \
  "scipy>=1.12" \
  "scikit-learn>=1.4" \
  "tqdm>=4.66" \
  "rouge_score" \
  "nltk" \
  -q

# ── git + github push deps ───────────────────────────────────────────────────
echo "[deps] Checking git..." >&2
git --version >&2

# verify key imports work
echo "[deps] Verifying imports..." >&2
$PYTHON -c "import torch; print(f'[deps] torch={torch.__version__} cuda={torch.cuda.is_available()}')" >&2
$PYTHON -c "import transformers; print(f'[deps] transformers={transformers.__version__}')" >&2
$PYTHON -c "import llmlingua; print('[deps] llmlingua ✓')" 2>/dev/null && true || \
  echo "[deps] WARNING: llmlingua import failed" >&2

echo "[deps] All dependencies installed." >&2
