#!/usr/bin/env bash
#SBATCH -J GMA-C4-Acc
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
RUN_ID="${RUN_ID:-wan14b_720p_hard_c4_accuracy_validation}"
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

for prompt_index in ${PROMPT_INDICES:-0 1 2}; do
  "$PYTHON" -u "$REPO_DIR/analysis/sweep_wan14b_720p_hard_c4_algorithm.py" \
    --model "$MODEL" \
    --prompt_path "$PROMPTS" \
    --steps 4 \
    --layers all \
    --heads "${HEADS:-0,10,20,30}" \
    --variants "${VARIANTS:-baseline:seed_wide:seed_wider_left:seed_wider_balanced:seed_wider_symmetric}" \
    --prompt_index "$prompt_index" \
    --output "$OUTPUT_ROOT/metrics_prompt_${prompt_index}.json"
done
