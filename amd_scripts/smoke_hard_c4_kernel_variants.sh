#!/bin/bash

#SBATCH -J HardC4Smoke
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -p mi3001x

set -euo pipefail

if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -n "${WORK:-}" ] && [ -f "${WORK}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${WORK}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "/work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh"
fi

CONDA_ENV="${CONDA_ENV:-monarch_rt}"
conda activate "${CONDA_ENV}"

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work1/jasoncong/denghaoran/training-free-videogen}"
RUN_ID="${RUN_ID:-hard_c4_kernel_variant_smoke_20260808}"
RESULT_DIR="${RESULT_DIR:-${SCRATCH_ROOT}/analysis/${RUN_ID}}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULT_DIR}/smoke.json}"
DTYPE="${DTYPE:-bf16}"
GPU_ID="${GPU_ID:-0}"

cd "${REPO_DIR}"
mkdir -p out "${RESULT_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_ROOT}/triton_cache/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "Smoke-testing hard C4 kernel variants"
echo "  output: ${OUTPUT_JSON}"
echo "  dtype: ${DTYPE}"
echo "  triton cache: ${TRITON_CACHE_DIR}"

HIP_VISIBLE_DEVICES="${GPU_ID}" ROCR_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONUNBUFFERED=1 \
python analysis/smoke_hard_c4_kernel_variants.py \
  --output_json "${OUTPUT_JSON}" \
  --dtype "${DTYPE}"
