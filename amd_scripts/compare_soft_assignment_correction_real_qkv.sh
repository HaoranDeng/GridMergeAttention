#!/bin/bash

#SBATCH -J GMA-SoftCorr
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -p mi2101x

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

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${WORK_ROOT}/GridMergeAttention}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/wan_models}"
MODEL_DIR="${MODEL_ROOT}/Wan2.1-T2V-14B-StepDistill-CfgDistill"
CHECKPOINT="${CHECKPOINT:-${MODEL_DIR}/distill_model.pt}"
RUN_ID="${RUN_ID:-wan14b_720p_4step_soft_assignment_correction_real_qkv_$(date +%Y%m%d_%H%M%S)}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
OUTPUT="${OUTPUT:-${SCRATCH_ROOT}/experiments/${RUN_ID}/real_qkv_metrics.json}"

cd "${REPO_DIR}"
mkdir -p out "$(dirname "${OUTPUT}")"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WAN_MODEL_ROOT="${MODEL_ROOT}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

python analysis/compare_soft_assignment_correction_real_qkv.py \
  --config "${CONFIG:-configs/dense_wan_14b_720p_fewstep.yaml}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt_path "${PROMPT_PATH}" \
  --output_json "${OUTPUT}" \
  --heads "${HEADS:-0,20}" \
  --temperatures "${TEMPERATURES:-0.1,0.25,0.5,1.0}" \
  --record_calls "${RECORD_CALLS:-1,14,27,40,41,54,67,80,81,94,107,120,121,134,147,160}" \
  --prompt_index "${PROMPT_INDEX:-0}" \
  --num_prompts "${NUM_PROMPTS:-1}" \
  --seed "${SEED:-0}"

echo "RESULT_JSON=${OUTPUT}"
