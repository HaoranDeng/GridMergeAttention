#!/usr/bin/env bash
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --exclusive
set -euo pipefail
ROOT=/work1/jasoncong/denghaoran/GridMergeAttention/experiments/mi300_hybrid_20x50_20260914
cd "$ROOT/repo"
export ROCM_PATH=/opt/rocm-7.2.0
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:$ROOT/svg:$ROOT/eval_deps"
export HF_HOME=/work1/jasoncong/denghaoran/huggingface
export TORCH_HOME=/work1/jasoncong/denghaoran/cache/torch
export GMA_HIP_CACHE="$ROOT/hip_cache"
export TRITON_CACHE_DIR="$ROOT/cache/$1/triton"
export XDG_CACHE_HOME="$ROOT/cache/$1"
export MIOPEN_USER_DB_PATH="$ROOT/cache/$1/miopen-db"
export MIOPEN_CUSTOM_CACHE_DIR="$ROOT/cache/$1/miopen-kernels"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
# Preserve Slurm's ROCm binding; do not replace physical IDs with zero.
env | sort | grep -E 'SLURM|VISIBLE_DEVICES' > "$ROOT/logs/binding.$SLURM_JOB_ID.txt"
set +e
/work1/jasoncong/denghaoran/miniconda3/envs/gma_amd/bin/python -u \
  analysis/run_mi300_hybrid_comparison.py --root "$ROOT" --method "$1"
status=$?
set -e
if [[ "$status" == 75 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
