#!/usr/bin/env bash
# Preparation only; select a clean H200 only after the active run ends.
# GPU=5 RESULT_DIR=/local2/.../new_run bash analysis/run_rf0_layer33_capture.sh \
#   --checkpoint /xuanwu-tank/.../distill_model.pt
set -euo pipefail

gpu_index="${GPU:?Set GPU to exactly one physical H200 index}"
result_dir="${RESULT_DIR:?Set an isolated outer result directory}"
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/denghaoran/miniconda3/envs/monarch_rt/bin/python}"
model_root="${WAN_MODEL_ROOT:-/xuanwu-tank/east/zrl/MonarchRT/wan_models}"
checkpoint=""
config="${source_root}/configs/dense_wan_14b_720p_fewstep.yaml"
prompts="${source_root}/prompts/MovieGenVideoBench_extended.txt"
[[ "${gpu_index}" =~ ^[0-9]+$ ]] || { echo "GPU must be one physical numeric index" >&2; exit 2; }
while [ "$#" -gt 0 ]; do
  [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
  case "$1" in
    --checkpoint) checkpoint="$2" ;;
    --config) config="$2" ;;
    --prompt_path) prompts="$2" ;;
    *) echo "Unsupported argument $1; --output-root is managed by this wrapper" >&2; exit 2 ;;
  esac
  shift 2
done
[ -n "${checkpoint}" ] || { echo "Explicit --checkpoint is required" >&2; exit 2; }
[ "${WAN_SDPA_BACKEND:-flash}" = flash ] || { echo "WAN_SDPA_BACKEND must be flash" >&2; exit 2; }
case "${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-}" in
  ""|0) ;;
  *) echo "Native RMSNorm required; remove TOKEN_MERGING_FORCE_TORCH_RMSNORM" >&2; exit 2 ;;
esac
if [ -n "${TOKEN_MERGING_SAVE_FIRST_QKV_PATH:-}" ] || [ -n "${TOKEN_MERGING_EXIT_AFTER_QKV_SAVE:-}" ]; then
  echo "Remove legacy capture/early-exit environment overrides" >&2; exit 2
fi
for required in "${python_bin}" "${checkpoint}" "${config}" "${prompts}" "${source_root}/analysis/capture_rf0_layer33_diagnostic.py"; do
  [ -f "${required}" ] || { echo "Missing file: ${required}" >&2; exit 2; }
done
[[ "${result_dir}" = /* && "${result_dir}" != / ]] || { echo "RESULT_DIR must be a named absolute directory" >&2; exit 2; }
mkdir -p -- "${result_dir}"
result_dir="$(cd -- "${result_dir}" && pwd -P)"
capture_root="${result_dir}/captures"
for artifact in captures capture.log pid purity.log exit_code worker_exit_code startup_gpu_status.txt; do
  if [ -e "${result_dir}/${artifact}" ] || [ -L "${result_dir}/${artifact}" ]; then
    echo "Refusing existing artifact: ${result_dir}/${artifact}" >&2; exit 4
  fi
done
# Do not create capture_root: the Python tool requires a nonexistent child.
export CUDA_VISIBLE_DEVICES="${gpu_index}" WAN_MODEL_ROOT="${model_root}" WAN_SDPA_BACKEND=flash
unset TOKEN_MERGING_FORCE_TORCH_RMSNORM
export OMP_NUM_THREADS=4
export PYTHONPATH="${source_root}:${source_root}/analysis:${PYTHONPATH:-}"
export PATH="$(dirname "${python_bin}"):/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$(dirname "${python_bin}")/../lib:${LD_LIBRARY_PATH:-}"
cd -- "${source_root}"
process=""
finish() {
  local status=$?
  trap - EXIT INT TERM
  if [ -n "${process}" ] && kill -0 "${process}" 2>/dev/null; then
    kill -TERM "${process}" 2>/dev/null || true  # Only our own worker, never a foreign PID.
    wait "${process}" 2>/dev/null || true
  fi
  printf '%s\n' "${status}" > "${result_dir}/exit_code"
  exit "${status}"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
compute_processes() {
  nvidia-smi -i "${gpu_uuid:-${gpu_index}}" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits
}
gpu_state="$(nvidia-smi -i "${gpu_index}" --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
[[ "${gpu_state}" != *$'\n'* ]] || { echo "Expected exactly one GPU" >&2; exit 3; }
IFS=, read -r gpu_name gpu_uuid used util <<< "${gpu_state}"
gpu_uuid="${gpu_uuid// /}"; used="${used// /}"; util="${util// /}"
[[ "${gpu_name}" = *H200* && "${gpu_uuid}" = GPU-* && "${used}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || {
  echo "Require an H200 with valid memory/utilization readings" >&2; exit 3;
}
process_list="$(compute_processes)"
if [ -n "${process_list}" ] || [ "${used}" -gt 100 ] || [ "${util}" -ne 0 ]; then
  echo "GPU ${gpu_index} not exclusively clean; no capture launched" >&2; exit 3
fi
printf '%s physical_gpu=%s name_uuid_memory_util=%s compute_processes=none\n' "$(date -Iseconds)" "${gpu_index}" "${gpu_state}" > "${result_dir}/startup_gpu_status.txt"
: > "${result_dir}/purity.log"
"${python_bin}" -u analysis/capture_rf0_layer33_diagnostic.py \
  --checkpoint "${checkpoint}" --config "${config}" --prompt_path "${prompts}" \
  --output-root "${capture_root}" > "${result_dir}/capture.log" 2>&1 &
process=$!
printf '%s\n' "${process}" > "${result_dir}/pid"
purity_status=0
while kill -0 "${process}" 2>/dev/null; do
  if ! process_list="$(compute_processes)"; then
    printf '%s monitor_query_failed\n' "$(date -Iseconds)" >> "${result_dir}/purity.log"
    purity_status=6
    break
  fi
  while IFS=, read -r pid memory name; do
    pid="${pid// /}"
    if [ -n "${pid}" ] && [ "${pid}" != "${process}" ]; then
      printf '%s foreign=%s,%s,%s\n' "$(date -Iseconds)" "${pid}" "${memory}" "${name}" >> "${result_dir}/purity.log"
      purity_status=5
    fi
  done <<< "${process_list}"
  [ "${purity_status}" -eq 0 ] || break
  sleep 15
done
if [ "${purity_status}" -ne 0 ]; then
  echo "GPU purity failed; stopping only this capture worker" >&2
  kill -TERM "${process}" 2>/dev/null || true
fi
set +e
wait "${process}"
worker_status=$?
set -e
process=""
printf '%s\n' "${worker_status}" > "${result_dir}/worker_exit_code"
tail -30 "${result_dir}/capture.log"
[ "${purity_status}" -eq 0 ] || exit "${purity_status}"
exit "${worker_status}"
