#!/bin/bash

#SBATCH -J TFVFitDense
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -p mi3001x

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV to the environment with torch/Wan dependencies.}"
WANDB_PROJECT="${WANDB_PROJECT:?Set WANDB_PROJECT to the Weights & Biases project name.}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/wan_syn_smoke}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/checkpoints/dense_wan_syn_smoke}"

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
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
mkdir -p out "${OUTPUT_DIR}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python finetune.py \
  --config configs/dense_wan_448p_train.yaml \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --use_ema \
  --max_steps "${MAX_STEPS:-5}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --save_steps "${SAVE_STEPS:-0}" \
  --gradient_checkpointing \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_run_name "${WANDB_RUN_NAME:-dense-wan-448p-${SLURM_JOB_ID:-manual}}" \
  --wandb_group "${WANDB_GROUP:-}" \
  --wandb_mode "${WANDB_MODE:-online}"
