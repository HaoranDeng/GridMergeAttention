#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
GPU="${GPU:-4}"
COUNT="${COUNT:-20}"
BACKEND="${BACKEND:?Set BACKEND to svg_all, svg2_all, or ear_all}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_${BACKEND}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"

case "$BACKEND" in
  svg_all|svg2_all|ear_all) ;;
  *) echo "Unsupported all-layer backend: $BACKEND" >&2; exit 2 ;;
esac

IFS=',' read -r utilization memory_used <<< "$(
  nvidia-smi -i "$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits
)"
utilization="${utilization// /}"
memory_used="${memory_used// /}"
compute_processes="$(
  nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d' \
    | wc -l
)"
if [[ "$utilization" != "0" || "$memory_used" -gt 16 || "$compute_processes" -ne 0 ]]; then
  echo "GPU $GPU is not clean: utilization=${utilization}%, memory=${memory_used} MiB, processes=${compute_processes}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:$SVG_ROOT/svg/kernels/build:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

"$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --sparse_video_gen_root "$SVG_ROOT" \
  --prompt_path "$PROMPTS" \
  --output_dir "$OUTPUT_ROOT/$BACKEND" \
  --count "$COUNT" \
  --seed 0 \
  2>&1 | tee "$OUTPUT_ROOT/logs/$BACKEND.log"
