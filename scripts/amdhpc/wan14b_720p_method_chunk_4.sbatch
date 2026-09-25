#!/usr/bin/env bash
#SBATCH -J GMA720Chunk
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
PYTHON="${PYTHON:-$WORK_ROOT/miniconda3/envs/gma_amd/bin/python}"
SVG_ROOT="${SVG_ROOT:-$WORK_ROOT/Sparse-VideoGen}"
MODEL="${MODEL:-$WORK_ROOT/models/Wan2.1-T2V-14B-Diffusers}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT}"
BACKEND="${BACKEND:?Set BACKEND}"
START="${START:?Set START to 0 or 4}"
COUNT="${COUNT:-4}"
ATTENTION_ERROR_INLINE="${ATTENTION_ERROR_INLINE:-0}"
OUTPUT_DIR="$EXPERIMENT_ROOT/chunks/$BACKEND/start_$START"

mkdir -p "$OUTPUT_DIR" "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/triton_cache/$BACKEND-$START"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_ROOT/cache"
export TORCH_HOME="$EXPERIMENT_ROOT/cache/torch"
export MPLCONFIGDIR="$EXPERIMENT_ROOT/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="$EXPERIMENT_ROOT/triton_cache/$BACKEND-$START"

error_args=(--attention_error_calls -1)
if [[ "$ATTENTION_ERROR_INLINE" == 1 ]]; then
  error_args+=(--attention_error_inline)
fi

"$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --sparse_video_gen_root "$SVG_ROOT" \
  --prompt_path "$PROMPTS" \
  --output_dir "$OUTPUT_DIR" \
  --start "$START" \
  --count "$COUNT" \
  --seed 0 \
  "${error_args[@]}" \
  >"$EXPERIMENT_ROOT/logs/${BACKEND}_start_${START}.log" 2>&1
