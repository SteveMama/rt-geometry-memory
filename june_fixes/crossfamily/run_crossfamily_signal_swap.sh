#!/usr/bin/env bash
# Review fix #5: the signal-swap ablation (geometry_KCD vs semantic_KCD) is
# reported only on Qwen models. This runs the identical study on two non-Qwen
# families: smollm2_17b (SmolLM2) and llama32_3b (Llama 3.2, HF-gated).
#
# With >=2 GPUs the two models run in parallel, one model per GPU. With one
# GPU they run sequentially. Resume is inherited from study.py progress files.
#
# Usage: bash june_fixes/crossfamily/run_crossfamily_signal_swap.sh [budgets] [run-prefix]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOCAL_VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$LOCAL_VENV_PYTHON" ]]; then PYTHON_BIN="$LOCAL_VENV_PYTHON"; else PYTHON_BIN="$(command -v python3 || command -v python)"; fi
fi

BUDGETS="${1:-0.20,0.35,0.50}"
RUN_PREFIX="${2:-june_crossfamily}"
INPUT_PATH="${HARDSET_INPUT:-paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl}"
POLICIES="uniform,semantic,geometry,semantic_keep_compress_drop,geometry_keep_compress_drop"
MODELS=("smollm2_17b" "llama32_3b")

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "[crossfamily] missing hard stress set: $INPUT_PATH" >&2
  exit 1
fi

# llama32_3b is gated on Hugging Face.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[crossfamily] WARNING: no HF_TOKEN set; skipping gated llama32_3b" >&2
  MODELS=("smollm2_17b")
fi

mapfile -t GPU_INDICES < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)
GPU_AVAILABLE=${#GPU_INDICES[@]}

run_model() {
  local model_key="$1"
  local gpu_id="$2"
  local study_name="${RUN_PREFIX}_signal_swap_${model_key}"
  local env_prefix=""
  if [[ -n "$gpu_id" ]]; then env_prefix="CUDA_VISIBLE_DEVICES=$gpu_id"; fi
  echo "[crossfamily] model=$model_key gpu=${gpu_id:-cpu/auto} study=$study_name" >&2
  env ${env_prefix} "$PYTHON_BIN" -m paper3_codec.study \
    --study-name "$study_name" \
    --model-keys "$model_key" \
    --input-path "$INPUT_PATH" \
    --families long_dependency,retrieval_heavy,code_conversation \
    --budgets "$BUDGETS" \
    --policies "$POLICIES" \
    --recent-window 2 \
    --min-history 4 \
    --max-input-tokens 768 \
    --output-root results/june_fixes/crossfamily \
    2>&1 | tee "results/june_fixes/crossfamily_${study_name}.log"
}

mkdir -p results/june_fixes/crossfamily

if [[ "$GPU_AVAILABLE" -ge 2 && ${#MODELS[@]} -ge 2 ]]; then
  PIDS=()
  for i in "${!MODELS[@]}"; do
    run_model "${MODELS[$i]}" "${GPU_INDICES[$i]}" &
    PIDS+=("$!")
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
  [[ "$FAIL" -eq 0 ]] || { echo "[crossfamily] one model run failed" >&2; exit 1; }
else
  for i in "${!MODELS[@]}"; do
    gpu=""
    [[ "$GPU_AVAILABLE" -ge 1 ]] && gpu="${GPU_INDICES[0]}"
    run_model "${MODELS[$i]}" "$gpu"
  done
fi

echo "[crossfamily] done. Studies under results/june_fixes/crossfamily/" >&2
echo "[crossfamily] compare geometry_keep_compress_drop vs semantic_keep_compress_drop in each study's significance_summary.json" >&2
