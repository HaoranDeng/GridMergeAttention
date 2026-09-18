#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
GPU="${GPU:?Set GPU to an exclusively clean H200 index}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_svg2_ear_turbo_20videos_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
COUNT="${COUNT:-20}"
START="${START:-0}"

SVG_ROOT="$SVG_ROOT" PYTHON="$PYTHON" \
  "$REPO_DIR/scripts/build_svg_fast_kernels_monarch_rt.sh"

IFS=',' read -r utilization memory_used <<< "$(
  nvidia-smi -i "$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits
)"
utilization="${utilization// /}"
memory_used="${memory_used// /}"
if [[ "$utilization" != "0" || "$memory_used" -gt 16 ]]; then
  echo "GPU $GPU is not clean: utilization=${utilization}%, memory=${memory_used} MiB" >&2
  exit 2
fi
if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo "GPU $GPU has an active compute process" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:$SVG_ROOT/svg/kernels/build:${PYTHONPATH:-}"

"$PYTHON" - <<'PY'
import torch
import triton
import _kernels
from analysis.benchmark_wan14b_720p_methods import install_rocm_import_stubs

install_rocm_import_stubs()
from svg.models.wan.attention import ENABLE_FAST_KERNEL

if not ENABLE_FAST_KERNEL:
    raise RuntimeError("SVG fast kernels were not enabled")
print(f"Torch {torch.__version__}; Triton {triton.__version__}")
print(f"Fast kernels enabled: {_kernels.__file__}")
PY

for backend in svg2 ear; do
  if [[ "$backend" == "svg2" ]]; then
    method="svg2_turbo"
  else
    method="svg_ear_turbo"
  fi
  mkdir -p "$OUTPUT_ROOT/$method" "$OUTPUT_ROOT/triton_cache_$method"
  export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache_$method"

  "$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
    --backend "$backend" \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$method" \
    --count "$COUNT" \
    --start "$START" \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_steps 50 \
    --seed 0 \
    --wan_forward_backend svg \
    --qk_norm_backend triton \
    --rope_backend svg \
    2>&1 | tee "$OUTPUT_ROOT/logs/$method.log"
done

touch "$OUTPUT_ROOT/completed"
