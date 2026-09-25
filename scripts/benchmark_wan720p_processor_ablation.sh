#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
GPU="${GPU:?Set GPU to an exclusively clean H200 index}"
RUN_ID="${RUN_ID:-wan14b_720p_processor_ablation_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
COUNT="${COUNT:-2}"
STEPS="${STEPS:-5}"

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

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:$SVG_ROOT/svg/kernels/build:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

run_variant() {
  local name="$1"
  local wan_forward="$2"
  local qk_norm="$3"
  local rope="$4"
  mkdir -p "$OUTPUT_ROOT/$name"
  "$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
    --backend gma_c4_mix \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$name" \
    --count "$COUNT" \
    --num_steps "$STEPS" \
    --seed 0 \
    --wan_forward_backend "$wan_forward" \
    --qk_norm_backend "$qk_norm" \
    --rope_backend "$rope" \
    2>&1 | tee "$OUTPUT_ROOT/logs/$name.log"
}

run_variant legacy diffusers pytorch legacy
run_variant fast_qk_rope diffusers triton svg
run_variant svg_forward svg triton svg
