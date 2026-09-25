#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
GPU="${GPU:?Set GPU to an exclusively clean H200 index}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_gma_c4_mix_no_entropy_bias_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"

IFS=',' read -r utilization memory_used <<<"$(
  nvidia-smi -i "$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits
)"
processes="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
utilization="${utilization// /}"
memory_used="${memory_used// /}"
if [[ "$utilization" != "0" || "$memory_used" -gt 16 || -n "$processes" ]]; then
  echo "GPU $GPU is not clean: utilization=${utilization}%, memory=${memory_used} MiB, processes=${processes:-none}" >&2
  exit 2
fi

OUTPUT_DIR="$OUTPUT_ROOT/gma_c4_mix_no_entropy_bias"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SVG_ROOT/svg/kernels/build:$SVG_ROOT:$REPO_DIR:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

"$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
  --backend gma_c4_mix_no_entropy_bias \
  --model "$MODEL" \
  --sparse_video_gen_root "$SVG_ROOT" \
  --prompt_path "$PROMPTS" \
  --output_dir "$OUTPUT_DIR" \
  --count 20 \
  --seed 0 \
  >"$OUTPUT_ROOT/logs/gma_c4_mix_no_entropy_bias.log" 2>&1
