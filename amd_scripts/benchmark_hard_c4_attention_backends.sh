#!/bin/bash

#SBATCH -J HardC4Attn
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
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
RUN_ID="${RUN_ID:-hard_c4_attention_backends_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-${SCRATCH_ROOT}/analysis/${RUN_ID}}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"

CONFIG="${CONFIG:-configs/dense_wan_fewstep.yaml}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULT_DIR}/hard_c4_attention_backends.json}"
MAX_CALLS="${MAX_CALLS:-30}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
VARIANTS="${VARIANTS:-hard_c4_bestlogit_tokenmajor}"
QUERY_OUTER_CHUNK="${QUERY_OUTER_CHUNK:-7}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"
GPU_ID="${GPU_ID:-0}"

cd "${REPO_DIR}"
mkdir -p out "${RESULT_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKEN_MERGING_FORCE_TORCH_RMSNORM="${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_ROOT}/triton_cache/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "Benchmarking non-shared hard C4 attention backends"
echo "  max calls: ${MAX_CALLS}"
echo "  prompts:   ${NUM_PROMPTS}"
echo "  variants:  ${VARIANTS}"
echo "  skip baseline: ${SKIP_BASELINE}"
echo "  output:    ${OUTPUT_JSON}"
echo "  triton cache: ${TRITON_CACHE_DIR}"

baseline_args=()
case "${SKIP_BASELINE,,}" in
  1|true|yes|y) baseline_args+=(--skip_baseline) ;;
esac

HIP_VISIBLE_DEVICES="${GPU_ID}" ROCR_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONUNBUFFERED=1 \
python analysis/hard_route_nonshared_backend_real_qkv.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt_path "${PROMPT_PATH}" \
  --dtype "${DTYPE}" \
  --use_ema \
  --max_calls "${MAX_CALLS}" \
  --num_prompts "${NUM_PROMPTS}" \
  --query_outer_chunk "${QUERY_OUTER_CHUNK}" \
  --variants "${VARIANTS}" \
  --seed "${SEED}" \
  "${baseline_args[@]}" \
  --output_json "${OUTPUT_JSON}"
