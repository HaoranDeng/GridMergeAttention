#!/usr/bin/env bash
#SBATCH -J GMA-C4-Alg
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
MODEL="${MODEL:-$WORK_ROOT/models/Wan2.1-T2V-14B-Diffusers}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
RUN_ID="${RUN_ID:-wan14b_720p_hard_c4_algorithm_sweep}"
OUTPUT_ROOT="$WORK_ROOT/GridMergeAttention/analysis/$RUN_ID"

mkdir -p "$OUTPUT_ROOT" "$REPO_DIR/out"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
export TORCH_HOME="$OUTPUT_ROOT/cache/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

extra_args=()
if [[ "${CHECK_TRITON_WIDE:-0}" == 1 ]]; then
  extra_args+=(--check_triton_wide)
fi
if [[ -n "${TRITON_BASELINE_MODULE:-}" ]]; then
  extra_args+=(--triton_baseline_module "$TRITON_BASELINE_MODULE")
fi
if [[ "${TIME_TRITON:-0}" == 1 ]]; then
  extra_args+=(--time_triton)
fi
if [[ "${TIME_DENSE_ALL_HEADS:-0}" == 1 ]]; then
  extra_args+=(--time_dense_all_heads)
fi
if [[ "${TIME_TRITON_ALL_HEADS_CHUNKED:-0}" == 1 ]]; then
  extra_args+=(--time_triton_all_heads_chunked)
fi
if [[ -n "${TRITON_HEAD_CHUNK:-}" ]]; then
  extra_args+=(--triton_head_chunk "$TRITON_HEAD_CHUNK")
fi
if [[ -n "${TIMING_HEAD_MULTIPLIER:-}" ]]; then
  extra_args+=(--timing_head_multiplier "$TIMING_HEAD_MULTIPLIER")
fi

"$PYTHON" -u "$REPO_DIR/analysis/sweep_wan14b_720p_hard_c4_algorithm.py" \
  --model "$MODEL" \
  --prompt_path "$PROMPTS" \
  --steps "${STEPS:-4}" \
  --layers "${LAYERS:-0,13,26,39}" \
  --heads "${HEADS:-0,20}" \
  --variants "${VARIANTS:-all}" \
  --prompt_index "${PROMPT_INDEX:-0}" \
  "${extra_args[@]}" \
  --output "$OUTPUT_ROOT/metrics.json"
