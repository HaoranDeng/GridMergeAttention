#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/vbench_all_dimension.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/videos/token_merging_hunyuan_hard_configs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs_multi_gpu}"
MODEL_ID="${MODEL_ID:-tencent/HunyuanVideo}"
NUM_VIDEOS="${NUM_VIDEOS:--1}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SEED="${SEED:-0}"
FPS="${FPS:-24}"
# The parity test cached only transformer weights. Keep downloads enabled by
# default so the first generation run can fetch text encoders, tokenizers, and
# the VAE. Set LOCAL_FILES_ONLY=1 only after the complete pipeline is cached.
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/zrlu/miniforge3/envs/SVG/bin/python}"

if [[ -z "${GPU_IDS:-}" ]]; then
  echo 'Set GPU_IDS to the GPUs reserved for this run, for example GPU_IDS="0 3 5 6".' >&2
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

# Only the requested hard-route variants are included. Select-route remains
# unsupported by the Hunyuan joint video/text attention implementation.
CONFIGS=(
  "configs/token_merging_hunyuan_hard_route_c2_480p.yaml"
  "configs/token_merging_hunyuan_hard_route_c4_480p.yaml"
  "configs/token_merging_hunyuan_hard_route_c6_480p.yaml"
)

RUN_NAMES=(
  "hard_c2"
  "hard_c4"
  "hard_c6"
)

required_files=(
  "${REPO_DIR}/generate_hunyuan.py"
  "${REPO_DIR}/scripts/generate/summarize_hunyuan_timings.py"
  "${PROMPT_PATH}"
)
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
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is missing or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export HF_HOME="${HF_HOME:-/xuanwu-tank/east/zrl/huggingface_cache}"
export TMPDIR="${TMPDIR:-/xuanwu-tank/east/zrl/tmp}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${TMPDIR}"

active_generators="$(pgrep -af 'python(3)? .*generate_hunyuan\.py' || true)"
if [[ "${active_generators}" == *"${OUTPUT_ROOT}/"* ]]; then
  echo "Another Hunyuan generation process is already writing under ${OUTPUT_ROOT}:" >&2
  echo "${active_generators}" >&2
  echo "Stop the existing launcher before starting this one." >&2
  exit 1
fi

cd "${REPO_DIR}"

local_file_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_file_args+=(--local_files_only)
fi

pids=()
stop_jobs() {
  echo "Stopping Hunyuan workers for the current config..." >&2
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

  echo "Starting Hunyuan ${run_name} with ${num_shards} prompt shards across GPUs: ${GPU_ID_LIST[*]}"
  echo "  config: ${config}"
  echo "  output: ${output_dir}"

  pids=()
  timing_paths=()
  for shard_id in "${!GPU_ID_LIST[@]}"; do
    gpu_id="${GPU_ID_LIST[$shard_id]}"
    log_path="${LOG_DIR}/${run_name}.gpu_${gpu_id}.log"
    timing_path="${LOG_DIR}/${run_name}.gpu_${gpu_id}.timing.json"
    timing_paths+=("${timing_path}")
    echo "  shard ${shard_id}/${num_shards} -> GPU ${gpu_id}, log ${log_path}, timing ${timing_path}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" generate_hunyuan.py \
        --config "${config}" \
        --model_id "${MODEL_ID}" \
        --prompt_path "${PROMPT_PATH}" \
        --output_dir "${output_dir}" \
        --num_videos "${NUM_VIDEOS}" \
        --num_samples "${NUM_SAMPLES}" \
        --skip_existing \
        --prompt_shard_id "${shard_id}" \
        --num_prompt_shards "${num_shards}" \
        --seed "${SEED}" \
        --fps "${FPS}" \
        --vbench_filenames \
        --timing_output "${timing_path}" \
        "${local_file_args[@]}" \
        "$@" >"${log_path}" 2>&1 &
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

  timing_summary_path="${OUTPUT_ROOT}/${run_name}.timing_summary.json"
  "${PYTHON_BIN}" scripts/generate/summarize_hunyuan_timings.py \
    --label "${run_name}" \
    --output "${timing_summary_path}" \
    "${timing_paths[@]}"
  echo "Finished ${run_name}; moving to the next hard-route config."
done

echo "All three Hunyuan hard-route configs completed successfully."
