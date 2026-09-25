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
SMOKE_VARIANTS="${SMOKE_VARIANTS:-}"
SMOKE_BATCH="${SMOKE_BATCH:-1}"
SMOKE_QUERY_OUTER="${SMOKE_QUERY_OUTER:-2}"
SMOKE_KEY_OUTER="${SMOKE_KEY_OUTER:-2}"
SMOKE_ROWS="${SMOKE_ROWS:-30}"
SMOKE_COLUMNS="${SMOKE_COLUMNS:-52}"
SMOKE_HEADS="${SMOKE_HEADS:-2}"
SMOKE_DIM="${SMOKE_DIM:-128}"
SMOKE_WARMUP="${SMOKE_WARMUP:-2}"
SMOKE_REPEATS="${SMOKE_REPEATS:-5}"

cd "${REPO_DIR}"
mkdir -p out "${RESULT_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_ROOT}/triton_cache/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "Smoke-testing hard C4 kernel variants"
echo "  output: ${OUTPUT_JSON}"
echo "  dtype: ${DTYPE}"
echo "  variants: ${SMOKE_VARIANTS:-<default>}"
echo "  shape: B=${SMOKE_BATCH} A=${SMOKE_QUERY_OUTER} F=${SMOKE_KEY_OUTER} rows=${SMOKE_ROWS} cols=${SMOKE_COLUMNS} heads=${SMOKE_HEADS} dim=${SMOKE_DIM}"
echo "  triton cache: ${TRITON_CACHE_DIR}"

variant_args=()
if [ -n "${SMOKE_VARIANTS}" ]; then
  variant_args+=(--variants "${SMOKE_VARIANTS}")
fi

HIP_VISIBLE_DEVICES="${GPU_ID}" ROCR_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONUNBUFFERED=1 \
python analysis/smoke_hard_c4_kernel_variants.py \
  --output_json "${OUTPUT_JSON}" \
  --dtype "${DTYPE}" \
  --batch "${SMOKE_BATCH}" \
  --query_outer "${SMOKE_QUERY_OUTER}" \
  --key_outer "${SMOKE_KEY_OUTER}" \
  --rows "${SMOKE_ROWS}" \
  --columns "${SMOKE_COLUMNS}" \
  --heads "${SMOKE_HEADS}" \
  --dim "${SMOKE_DIM}" \
  --warmup "${SMOKE_WARMUP}" \
  --repeats "${SMOKE_REPEATS}" \
  "${variant_args[@]}"
