#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONFIG="${CONFIG:-${REPO_DIR}/configs/token_merging_hunyuan_480p.yaml}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/videos/token_merging_hunyuan}"
MODEL_ID="${MODEL_ID:-tencent/HunyuanVideo}"
NUM_VIDEOS="${NUM_VIDEOS:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-0}"
FPS="${FPS:-24}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Generation config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_PATH}" ]]; then
  echo "Prompt file not found: ${PROMPT_PATH}" >&2
  exit 1
fi

# Keep model downloads and temporary artifacts on the shared project volume.
export HF_HOME="${HF_HOME:-/xuanwu-tank/east/zrl/huggingface_cache}"
export TMPDIR="${TMPDIR:-/xuanwu-tank/east/zrl/tmp}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${HF_HOME}" "${TMPDIR}" "${OUTPUT_DIR}"
cd "${REPO_DIR}"

EXTRA_ARGS=()
if [[ -n "${W_REDUCE:-}" ]]; then
  EXTRA_ARGS+=(--token_merging_w_reduce "${W_REDUCE}")
fi
if [[ -n "${QUERY_OUTER_CHUNK:-}" ]]; then
  EXTRA_ARGS+=(--token_merging_query_outer_chunk "${QUERY_OUTER_CHUNK}")
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--local_files_only)
fi

python generate_hunyuan.py \
  --config "${CONFIG}" \
  --model_id "${MODEL_ID}" \
  --prompt_path "${PROMPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --fps "${FPS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
