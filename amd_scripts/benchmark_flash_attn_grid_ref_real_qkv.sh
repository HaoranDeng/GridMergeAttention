#!/bin/bash

#SBATCH -J FlashGridRef
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -p mi3001x

set -euo pipefail

if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -n "${WORK:-}" ] && [ -f "${WORK}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${WORK}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh"
fi

conda activate "${CONDA_ENV:-monarch_rt}"

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work1/jasoncong/denghaoran/training-free-videogen}"
RUN_ID="${RUN_ID:-flash_attn_grid_ref_real_qkv_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRATCH_ROOT}/analysis/${RUN_ID}/metrics.json}"

cd "${REPO_DIR}"
mkdir -p out "$(dirname "${OUTPUT_JSON}")"

export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKEN_MERGING_FORCE_TORCH_RMSNORM="${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_ROOT}/triton_cache/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TRITON_CACHE_DIR}"

extra_args=()
if [ "${USE_EMA:-1}" != "0" ]; then
  extra_args+=(--use_ema)
fi

python analysis/benchmark_flash_attn_grid_ref_real_qkv.py \
  --config "${CONFIG:-configs/dense_wan_fewstep.yaml}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt_path "${PROMPT_PATH}" \
  --prompt_index "${PROMPT_INDEX:-0}" \
  --num_prompts "${NUM_PROMPTS:-1}" \
  --seed "${SEED:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --max_calls "${MAX_CALLS:-10}" \
  --output_json "${OUTPUT_JSON}" \
  "${extra_args[@]}"
