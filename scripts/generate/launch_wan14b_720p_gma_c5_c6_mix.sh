#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_gma_c5_c6_mix_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
GPU_C5="${GPU_C5:-1}"
GPU_C6="${GPU_C6:-2}"
COUNT="${COUNT:-8}"

check_clean_gpu() {
  local gpu="$1"
  local utilization memory_used processes
  IFS=',' read -r utilization memory_used <<<"$(
    nvidia-smi -i "$gpu" \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits
  )"
  utilization="${utilization// /}"
  memory_used="${memory_used// /}"
  processes="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ "$utilization" != "0" || "$memory_used" -gt 16 || -n "$processes" ]]; then
    echo "GPU $gpu is not clean: utilization=${utilization}%, memory=${memory_used} MiB, processes=${processes:-none}" >&2
    return 1
  fi
}

run_backend() {
  local backend="$1"
  local gpu="$2"
  local output="$OUTPUT_ROOT/$backend"
  check_clean_gpu "$gpu"
  mkdir -p "$output" "$OUTPUT_ROOT/triton_cache/$backend"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$SVG_ROOT/svg/kernels/build:$SVG_ROOT:$REPO_DIR:${PYTHONPATH:-}" \
  TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache/$backend" \
    "$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
      --backend "$backend" \
      --model "$MODEL" \
      --sparse_video_gen_root "$SVG_ROOT" \
      --prompt_path "$PROMPTS" \
      --output_dir "$output" \
      --count "$COUNT" \
      --seed 0 \
      >"$OUTPUT_ROOT/logs/$backend.log" 2>&1
}

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
  --format=csv >"$OUTPUT_ROOT/gpu_snapshot_start.csv"

run_backend gma_c5_mix "$GPU_C5" &
pid_c5=$!
run_backend gma_c6_mix "$GPU_C6" &
pid_c6=$!
printf '%s\n' "$pid_c5" >"$OUTPUT_ROOT/gma_c5_mix.pid"
printf '%s\n' "$pid_c6" >"$OUTPUT_ROOT/gma_c6_mix.pid"

status=0
wait "$pid_c5" || status=1
wait "$pid_c6" || status=1
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
  --format=csv >"$OUTPUT_ROOT/gpu_snapshot_end.csv"
exit "$status"
