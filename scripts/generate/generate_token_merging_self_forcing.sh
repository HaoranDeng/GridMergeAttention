#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WAN_MODEL_DIR="${WAN_MODEL_DIR:-${REPO_DIR}/wan_models/Wan2.1-T2V-1.3B}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
CONFIG="${CONFIG:-${REPO_DIR}/configs/token_merging_wan_hard_c4_triton_fewstep.yaml}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/vbench_all_dimension.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/videos/hard_c4_triton_self_forcing}"
# generate.py interprets a negative limit as "use every prompt in the file".
NUM_VIDEOS="${NUM_VIDEOS:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"

if [[ ! -d "${WAN_MODEL_DIR}" ]]; then
  echo "Wan model directory not found: ${WAN_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Self-forcing checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_PATH}" ]]; then
  echo "Prompt file not found: ${PROMPT_PATH}" >&2
  exit 1
fi

# The loader appends model_name from the config, so WAN_MODEL_ROOT must be the
# directory containing Wan2.1-T2V-1.3B rather than the model directory itself.
export WAN_MODEL_ROOT="$(dirname "${WAN_MODEL_DIR}")"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

EXTRA_ARGS=()
if [[ -n "${QUERY_OUTER_CHUNK:-}" ]]; then
  EXTRA_ARGS+=(--token_merging_query_outer_chunk "${QUERY_OUTER_CHUNK}")
fi

python generate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt_path "${PROMPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --fps "${FPS}" \
  --dtype "${DTYPE}" \
  --use_ema \
  "${EXTRA_ARGS[@]}" \
  "$@"
