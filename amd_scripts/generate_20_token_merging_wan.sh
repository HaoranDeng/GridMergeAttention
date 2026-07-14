#!/bin/bash

#SBATCH -J TFVGen
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:30:00
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
fi

CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV to the environment with torch/Wan dependencies.}"
conda activate "${CONDA_ENV}"

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
CONFIG="${CONFIG:-configs/token_merging_wan_fewstep.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/token_merging_wan}"

cd "${REPO_DIR}"
mkdir -p out "${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -mindepth 1 -delete

MIOPEN_CACHE_ROOT="${TMPDIR:-${PWD}/assets}/miopen-${SLURM_JOB_ID:-manual}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE_ROOT}/user-db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE_ROOT}/kernel-cache"
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python generate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt_path prompts/MovieGenVideoBench_extended.txt \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos 20 \
  --use_ema
