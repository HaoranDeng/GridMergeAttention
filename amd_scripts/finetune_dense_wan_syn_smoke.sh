#!/bin/bash

#SBATCH -J TFVFitDense
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
fi

CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV to the environment with torch/Wan dependencies.}"
conda activate "${CONDA_ENV}"

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/wan_syn_smoke}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/checkpoints/dense_wan_syn_smoke}"
MAX_STEPS="${MAX_STEPS:-5}"

cd "${REPO_DIR}"
mkdir -p out "${DATA_DIR}" "${OUTPUT_DIR}"

if ! find "${DATA_DIR}" -type f -name '*.parquet' | grep -q .; then
  hf download \
    FastVideo/Wan-Syn_77x448x832_600k \
    Part_100/latents_chunk_0000.parquet \
    --repo-type dataset \
    --local-dir "${DATA_DIR}"
fi

MIOPEN_CACHE_ROOT="${TMPDIR:-${PWD}/assets}/miopen-${SLURM_JOB_ID:-manual}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE_ROOT}/user-db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE_ROOT}/kernel-cache"
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python finetune.py \
  --config configs/dense_wan_448p_train.yaml \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --use_ema \
  --max_steps "${MAX_STEPS}" \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --weight_decay 1e-4 \
  --gradient_checkpointing
