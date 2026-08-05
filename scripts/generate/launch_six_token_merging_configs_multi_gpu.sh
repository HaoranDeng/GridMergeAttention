#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WAN_MODEL_DIR="${WAN_MODEL_DIR:-${REPO_DIR}/wan_models/Wan2.1-T2V-1.3B}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/vbench_all_dimension.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/videos/token_merging_six_configs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs_multi_gpu}"
NUM_VIDEOS="${NUM_VIDEOS:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-2}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${GPU_IDS:-}" ]]; then
  echo 'Set GPU_IDS to the GPUs reserved for this run, for example GPU_IDS="0 1 2 5 6 7".' >&2
  exit 2
fi
normalized_gpu_ids="${GPU_IDS//,/ }"
read -r -a GPU_ID_LIST <<< "${normalized_gpu_ids}"
if (( ${#GPU_ID_LIST[@]} == 0 )); then
  echo "GPU_IDS did not contain any GPU IDs." >&2
  exit 2
fi

declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_ID_LIST[@]}"; do
  if [[ -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]]; then
    echo "GPU_IDS contains a duplicate GPU: ${gpu_id}" >&2
    exit 2
  fi
  SEEN_GPU_IDS["${gpu_id}"]=1
done

CONFIGS=(
  "configs/token_merging_wan_two_seed_fewstep.yaml"
  "configs/token_merging_wan_select_c2_fewstep.yaml"
  "configs/token_merging_wan_four_seed_fewstep.yaml"
  "configs/token_merging_wan_select_c4_fewstep.yaml"
  "configs/token_merging_wan_six_seed_fewstep.yaml"
  "configs/token_merging_wan_select_c6_fewstep.yaml"
)

RUN_NAMES=(
  "hard_c2"
  "select_c2"
  "hard_c4"
  "select_c4"
  "hard_c6"
  "select_c6"
)

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
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

active_generators="$(pgrep -af 'python(3)? .*generate\.py' || true)"
if [[ "${active_generators}" == *"${OUTPUT_ROOT}/"* ]]; then
  echo "Another generation process is already writing under ${OUTPUT_ROOT}:" >&2
  echo "${active_generators}" >&2
  echo "Stop the existing launcher before starting this one." >&2
  exit 1
fi

export WAN_MODEL_ROOT="$(dirname "${WAN_MODEL_DIR}")"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

pids=()
stop_jobs() {
  echo "Stopping workers for the current config..." >&2
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}
trap stop_jobs INT TERM

num_shards="${#GPU_ID_LIST[@]}"
for config_idx in "${!CONFIGS[@]}"; do
  config="${REPO_DIR}/${CONFIGS[$config_idx]}"
  run_name="${RUN_NAMES[$config_idx]}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${output_dir}"

  echo "Starting ${run_name} with ${num_shards} prompt shards across GPUs: ${GPU_ID_LIST[*]}"
  echo "  config: ${config}"
  echo "  output: ${output_dir}"

  pids=()
  for shard_id in "${!GPU_ID_LIST[@]}"; do
    gpu_id="${GPU_ID_LIST[$shard_id]}"
    log_path="${LOG_DIR}/${run_name}.gpu_${gpu_id}.log"
    echo "  shard ${shard_id}/${num_shards} -> GPU ${gpu_id}, log ${log_path}"

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
        --prompt_shard_id "${shard_id}" \
        --num_prompt_shards "${num_shards}" \
        --seed "${SEED}" \
        --fps "${FPS}" \
        --dtype "${DTYPE}" \
        --use_ema \
        --vbench_filenames \
        >"${log_path}" 2>&1 &
    pids+=("$!")
  done

  config_status=0
  for shard_id in "${!pids[@]}"; do
    if wait "${pids[$shard_id]}"; then
      echo "Completed ${run_name} shard ${shard_id} on GPU ${GPU_ID_LIST[$shard_id]}."
    else
      worker_status=$?
      echo "Failed ${run_name} shard ${shard_id} on GPU ${GPU_ID_LIST[$shard_id]} with status ${worker_status}." >&2
      echo "See ${LOG_DIR}/${run_name}.gpu_${GPU_ID_LIST[$shard_id]}.log" >&2
      config_status=1
    fi
  done
  pids=()

  if (( config_status != 0 )); then
    echo "Not starting the next config because ${run_name} had a failed worker." >&2
    exit 1
  fi
  echo "Finished ${run_name}; moving to the next config."
done

echo "All six configs completed successfully."
