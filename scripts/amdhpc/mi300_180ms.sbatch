#!/usr/bin/env bash
#SBATCH -J GMA-HIP180
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH --exclusive
#SBATCH -o /work1/jasoncong/denghaoran/GridMergeAttention/hip_180ms_20260914/job.%j.out
#SBATCH -e /work1/jasoncong/denghaoran/GridMergeAttention/hip_180ms_20260914/job.%j.err
set -euo pipefail
ROOT=/work1/jasoncong/denghaoran/GridMergeAttention/hip_180ms_20260914
CAPTURE_ROOT=/work1/jasoncong/denghaoran/GridMergeAttention/hip_20260914
cd "$CAPTURE_ROOT/repo"
export ROCM_PATH=/opt/rocm-7.2.0
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
export GMA_HIP_CACHE="$CAPTURE_ROOT/hip_cache"
export TRITON_CACHE_DIR="$ROOT/triton_cache"
export OMP_NUM_THREADS=4
env | sort | grep -E 'SLURM|VISIBLE_DEVICES' > "$ROOT/binding.$SLURM_JOB_ID.txt"
rocm-smi --showpids --showuse --showmeminfo vram > "$ROOT/gpu_before.$SLURM_JOB_ID.txt"
/work1/jasoncong/denghaoran/miniconda3/envs/gma_amd/bin/python -u \
  "${BENCHMARK:-analysis/mi300_180ms_worker.py}" \
  --capture "$CAPTURE_ROOT/${CAPTURE:-real_full40_step0_layer00.pt}" \
  --output "$ROOT/result.$SLURM_JOB_ID.json" "$@"
