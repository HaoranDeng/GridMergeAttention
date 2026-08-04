#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WAN_MODEL_DIR="${WAN_MODEL_DIR:-${REPO_DIR}/wan_models/Wan2.1-T2V-1.3B}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/vbench_all_dimension.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/videos/token_merging_six_configs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
NUM_VIDEOS="${NUM_VIDEOS:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"

# Override with a space-separated list, for example GPU_IDS="2 3 4 5 6 7".
read -r -a GPU_ID_LIST <<< "${GPU_IDS:-0 1 2 3 4 5}"

CONFIGS=(
  "configs/token_merging_wan_two_seed_fewstep.yaml"
  "configs/token_merging_wan_four_seed_fewstep.yaml"
  "configs/token_merging_wan_six_seed_fewstep.yaml"
  "configs/token_merging_wan_select_c2_fewstep.yaml"
  "configs/token_merging_wan_select_c4_fewstep.yaml"
  "configs/token_merging_wan_select_c6_fewstep.yaml"
)

RUN_NAMES=(
  "hard_c2"
  "hard_c4"
  "hard_c6"
  "select_c2"
  "select_c4"
  "select_c6"
)

if (( ${#GPU_ID_LIST[@]} != ${#CONFIGS[@]} )); then
  echo "Expected ${#CONFIGS[@]} GPU IDs, got ${#GPU_ID_LIST[@]}: ${GPU_ID_LIST[*]}" >&2
  exit 1
fi

required_files=("${REPO_DIR}/generate.py" "${CHECKPOINT}" "${PROMPT_PATH}")
for config in "${CONFIGS[@]}"; do
  required_files+=("${REPO_DIR}/${config}")
done

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

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export WAN_MODEL_ROOT="$(dirname "${WAN_MODEL_DIR}")"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

pids=()
for i in "${!CONFIGS[@]}"; do
  gpu_id="${GPU_ID_LIST[$i]}"
  config="${REPO_DIR}/${CONFIGS[$i]}"
  run_name="${RUN_NAMES[$i]}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  log_path="${LOG_DIR}/${run_name}.log"

  mkdir -p "${output_dir}"
  echo "Launching ${run_name} on GPU ${gpu_id}"
  echo "  config: ${config}"
  echo "  output: ${output_dir}"
  echo "  log:    ${log_path}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 \
    python generate.py \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --prompt_path "${PROMPT_PATH}" \
      --output_dir "${output_dir}" \
      --num_videos "${NUM_VIDEOS}" \
      --num_samples "${NUM_SAMPLES}" \
      --seed "${SEED}" \
      --fps "${FPS}" \
      --dtype "${DTYPE}" \
      --use_ema \
      --vbench_filenames \
      "$@" >"${log_path}" 2>&1 &
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
  echo "All six generation jobs completed successfully."
fi
exit "${status}"
