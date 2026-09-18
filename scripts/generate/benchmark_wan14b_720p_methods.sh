#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON_SVG="${PYTHON_SVG:-$HOME/miniconda3/envs/SVG/bin/python}"
PYTHON_GMA="${PYTHON_GMA:-$PYTHON_SVG}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_four_methods_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"

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

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

for backend in dense svg svg2 gma; do
  python="$PYTHON_SVG"
  if [[ "$backend" == "dense" || "$backend" == "gma" ]]; then
    python="$PYTHON_GMA"
  fi
  "$python" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
    --backend "$backend" \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$backend" \
    --count 20 \
    --seed 0 \
    2>&1 | tee "$OUTPUT_ROOT/logs/$backend.log"
done
