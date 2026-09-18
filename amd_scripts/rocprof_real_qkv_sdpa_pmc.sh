#!/bin/bash

#SBATCH -J SdpaPmc
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

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work1/jasoncong/denghaoran/training-free-videogen}"
RUN_ID="${RUN_ID:-real_qkv_sdpa_rocprof_pmc_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${SCRATCH_ROOT}/analysis/${RUN_ID}"

cd "${REPO_DIR}"
mkdir -p out "${OUT_DIR}"

export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKEN_MERGING_FORCE_TORCH_RMSNORM="${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-1}"

cat > "${OUT_DIR}/pmc.txt" <<'EOF'
pmc : FetchSize WriteSize
EOF

extra_args=()
if [ "${USE_EMA:-1}" != "0" ]; then
  extra_args+=(--use_ema)
fi

rocprof --stats \
  -i "${OUT_DIR}/pmc.txt" \
  -o "${OUT_DIR}/rocprof.csv" \
  python analysis/profile_real_qkv_flash_load_time.py \
    --config "${CONFIG:-configs/dense_wan_fewstep.yaml}" \
    --checkpoint "${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}" \
    --prompt_path "${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}" \
    --prompt_index "${PROMPT_INDEX:-0}" \
    --attention_call_index "${ATTENTION_CALL_INDEX:-0}" \
    --seed "${SEED:-0}" \
    --dtype "${DTYPE:-bf16}" \
    --block_ms "${BLOCK_MS:-}" \
    --warmup "${WARMUP:-1}" \
    --repeats "${REPEATS:-2}" \
    --output_json "${OUT_DIR}/metrics.json" \
    "${extra_args[@]}"
