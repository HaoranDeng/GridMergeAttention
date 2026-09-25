#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
GMA_RUNTIME="${GMA_RUNTIME:-fast}"
case "$GMA_RUNTIME" in
  fast)
    PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
    ;;
  exact)
    PYTHON="${PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
    ;;
  *)
    echo "Unknown GMA_RUNTIME=$GMA_RUNTIME (expected fast or exact)" >&2
    exit 2
    ;;
esac
GPU="${GPU:?Set GPU to an exclusively clean H200 index}"
BACKEND="${BACKEND:-gma}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_gma_all_fast_$(date +%Y%m%d_%H%M%S)}"
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
  [[ -z "$gpu_processes" ]] || echo "$gpu_processes" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/$BACKEND" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SVG_ROOT/svg/kernels/build:$SVG_ROOT:$REPO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

echo "GMA backend: $BACKEND"
echo "GMA runtime: $GMA_RUNTIME ($PYTHON)"
if [[ "$GMA_RUNTIME" == "fast" ]]; then
  echo "Fast mode preserves the algorithm but is not bitwise-identical to the exact SVG runtime."
fi

"$PYTHON" - <<'PY'
import _kernels

print(f"SVG fast kernels: {_kernels.__file__}")
PY

"$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --sparse_video_gen_root "$SVG_ROOT" \
  --prompt_path "$PROMPTS" \
  --output_dir "$OUTPUT_ROOT/$BACKEND" \
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
  --gma_preprocess_backend fused \
  --gma_output_layout token \
  --gma_head_chunk 10 \
  2>&1 | tee "$OUTPUT_ROOT/logs/$BACKEND.log"
