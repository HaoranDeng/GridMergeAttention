#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
FASTVIDEO_ROOT="${FASTVIDEO_ROOT:-/local2/denghaoran/GridMergeAttention/external/FastVideo}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
GPU="${GPU:?Set GPU to an exclusively clean H200 index}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_vsa_training_free_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
COUNT="${COUNT:-20}"
START="${START:-0}"

gpu_processes="$(
  nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null
)"
IFS=',' read -r utilization memory_used <<< "$(
  nvidia-smi -i "$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits
)"
utilization="${utilization// /}"
memory_used="${memory_used// /}"
if [[ -n "$gpu_processes" || "$utilization" != "0" || "$memory_used" -gt 16 ]]; then
  echo "GPU $GPU is not clean: utilization=${utilization}%, memory=${memory_used} MiB" >&2
  exit 2
fi

backend=vsa_training_free_all
mkdir -p "$OUTPUT_ROOT/$backend" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export CUDA_VISIBLE_DEVICES="$GPU"
export FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1
export PYTHONPATH="$FASTVIDEO_ROOT/fastvideo-kernel/python:$SVG_ROOT/svg/kernels/build:$SVG_ROOT:$REPO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

"$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
  --backend "$backend" \
  --model "$MODEL" \
  --sparse_video_gen_root "$SVG_ROOT" \
  --prompt_path "$PROMPTS" \
  --output_dir "$OUTPUT_ROOT/$backend" \
  --count "$COUNT" \
  --start "$START" \
  --height 720 \
  --width 1280 \
  --num_frames 81 \
  --num_steps 50 \
  --seed 0 \
  --vsa_sparsity 0.9 \
  --attention_error_calls 1 \
  --attention_error_inline \
  2>&1 | tee "$OUTPUT_ROOT/logs/$backend.log"
