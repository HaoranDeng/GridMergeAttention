#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_token_merging_configs.sh CONFIG [CONFIG ...]

Runs one generation config per GPU. By default, N configs use GPUs 0 through
N-1. Set GPU_IDS to a space-separated list to select different GPUs.

Examples:
  bash scripts/generate/launch_token_merging_configs.sh \
    configs/config_a.yaml configs/config_b.yaml

  GPU_IDS="2 5 7" bash scripts/generate/launch_token_merging_configs.sh \
    configs/config_a.yaml configs/config_b.yaml configs/config_c.yaml
EOF
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WAN_MODEL_DIR="${WAN_MODEL_DIR:-${REPO_DIR}/wan_models/Wan2.1-T2V-1.3B}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/vbench_all_dimension.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/videos/token_merging_configs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
NUM_VIDEOS="${NUM_VIDEOS:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-2}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_ARGS=("$@")
CONFIG_PATHS=()
RUN_NAMES=()
declare -A SEEN_RUN_NAMES=()

for config_arg in "${CONFIG_ARGS[@]}"; do
  if [[ "${config_arg}" = /* ]]; then
    config_path="${config_arg}"
  else
    config_path="${REPO_DIR}/${config_arg}"
  fi

  config_file="${config_path##*/}"
  run_name="${config_file%.*}"
  if [[ -n "${SEEN_RUN_NAMES[${run_name}]:-}" ]]; then
    echo "Config filenames must have unique stems; duplicate output name: ${run_name}" >&2
    exit 1
  fi
  SEEN_RUN_NAMES["${run_name}"]=1
  CONFIG_PATHS+=("${config_path}")
  RUN_NAMES+=("${run_name}")
done

GPU_ID_LIST=()
if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
else
  for ((i = 0; i < ${#CONFIG_PATHS[@]}; i++)); do
    GPU_ID_LIST+=("${i}")
  done
fi

if (( ${#GPU_ID_LIST[@]} != ${#CONFIG_PATHS[@]} )); then
  echo "GPU/config count mismatch: got ${#GPU_ID_LIST[@]} GPUs and ${#CONFIG_PATHS[@]} configs." >&2
  echo "GPU_IDS: ${GPU_ID_LIST[*]:-(none)}" >&2
  exit 1
fi

required_files=("${REPO_DIR}/generate.py" "${CHECKPOINT}" "${PROMPT_PATH}")
required_files+=("${CONFIG_PATHS[@]}")

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    missing=1
  fi
done
if (( missing )); then
  exit 1
fi
if [[ ! -d "${WAN_MODEL_DIR}" ]]; then
  echo "Wan model directory not found: ${WAN_MODEL_DIR}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export WAN_MODEL_ROOT="$(dirname "${WAN_MODEL_DIR}")"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

pids=()
for i in "${!CONFIG_PATHS[@]}"; do
  gpu_id="${GPU_ID_LIST[$i]}"
  config="${CONFIG_PATHS[$i]}"
  run_name="${RUN_NAMES[$i]}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  log_path="${LOG_DIR}/${run_name}.log"

  mkdir -p "${output_dir}"
  echo "Launching ${run_name} on GPU ${gpu_id}"
  echo "  config: ${config}"
  echo "  output: ${output_dir}"
  echo "  log:    ${log_path}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" generate.py \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --prompt_path "${PROMPT_PATH}" \
      --output_dir "${output_dir}" \
      --num_videos "${NUM_VIDEOS}" \
      --num_samples "${NUM_SAMPLES}" \
      --sample_batch_size "${SAMPLE_BATCH_SIZE}" \
      --skip_existing \
      --seed "${SEED}" \
      --fps "${FPS}" \
      --dtype "${DTYPE}" \
      --use_ema \
      --vbench_filenames \
      >"${log_path}" 2>&1 &
  pids+=("$!")
done

stop_jobs() {
  echo "Stopping generation jobs..." >&2
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}
trap stop_jobs INT TERM

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Completed ${RUN_NAMES[$i]} (GPU ${GPU_ID_LIST[$i]})."
  else
    job_status=$?
    echo "Failed ${RUN_NAMES[$i]} (GPU ${GPU_ID_LIST[$i]}) with status ${job_status}. See ${LOG_DIR}/${RUN_NAMES[$i]}.log" >&2
    status=1
  fi
done

if (( status == 0 )); then
  echo "All ${#CONFIG_PATHS[@]} generation jobs completed successfully."
fi
exit "${status}"
