#!/usr/bin/env bash
#SBATCH -J GMA-C4-NoH
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
PYTHON="${PYTHON:-$WORK_ROOT/miniconda3/envs/gma_amd/bin/python}"
SVG_ROOT="${SVG_ROOT:-$WORK_ROOT/Sparse-VideoGen}"
MODEL="${MODEL:-$WORK_ROOT/models/Wan2.1-T2V-14B-Diffusers}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_gma_c4_mix_no_entropy_bias_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/GridMergeAttention/experiments/$RUN_ID}"
OUTPUT_DIR="$OUTPUT_ROOT/gma_c4_mix_no_entropy_bias"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/cache" "$OUTPUT_ROOT/triton_cache"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
export TORCH_HOME="$OUTPUT_ROOT/cache/torch"
export MPLCONFIGDIR="$OUTPUT_ROOT/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
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

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
