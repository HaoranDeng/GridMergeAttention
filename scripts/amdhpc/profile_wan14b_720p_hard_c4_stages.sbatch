#!/usr/bin/env bash
#SBATCH -J GMA720Prof
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
PYTHON="${PYTHON:-$WORK_ROOT/miniconda3/envs/gma_amd/bin/python}"
RUN_ID="${RUN_ID:-wan14b_720p_hard_c4_stage_profile}"
OUTPUT_ROOT="$WORK_ROOT/GridMergeAttention/analysis/$RUN_ID"

mkdir -p "$OUTPUT_ROOT" "$REPO_DIR/out"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

"$PYTHON" -u "$REPO_DIR/analysis/profile_wan14b_720p_hard_c4_stages.py" \
  --heads "${HEADS:-8}" \
  --query_outer "${QUERY_OUTER:-21}" \
  --key_outer "${KEY_OUTER:-21}" \
  --warmup "${WARMUP:-2}" \
  --repeats "${REPEATS:-10}" \
  --output "$OUTPUT_ROOT/profile.json"
