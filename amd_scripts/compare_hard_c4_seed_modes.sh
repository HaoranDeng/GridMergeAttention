#!/bin/bash

#SBATCH -J SeedModes
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
RUN_ID="${RUN_ID:-hard_c4_seed_modes_$(date +%Y%m%d_%H%M%S)}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
CONFIG="${CONFIG:-configs/dense_wan_fewstep.yaml}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRATCH_ROOT}/analysis/${RUN_ID}/seed_mode_metrics.json}"
USE_EMA="${USE_EMA:-1}"

cd "${REPO_DIR}"
mkdir -p out "$(dirname "${OUTPUT_JSON}")"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKEN_MERGING_FORCE_TORCH_RMSNORM="${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-1}"

EXTRA_ARGS=()
if [ "${USE_EMA}" != "0" ]; then
  EXTRA_ARGS+=(--use_ema)
fi

python analysis/compare_hard_c4_seed_modes.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt_path "${PROMPT_PATH}" \
  --prompt_index "${PROMPT_INDEX:-0}" \
  --num_prompts "${NUM_PROMPTS:-1}" \
  --seed "${SEED:-0}" \
  --seed_mode_seed "${SEED_MODE_SEED:-0}" \
  --seed_modes "${SEED_MODES:-local_with_ith,random_with_ith,fully_random}" \
  --dtype "${DTYPE:-bf16}" \
  --max_calls "${MAX_CALLS:-4}" \
  --query_outer_chunk "${QUERY_OUTER_CHUNK:-1}" \
  --head_chunk "${HEAD_CHUNK:-2}" \
  --output_json "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"
