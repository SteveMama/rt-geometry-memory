#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

python - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
