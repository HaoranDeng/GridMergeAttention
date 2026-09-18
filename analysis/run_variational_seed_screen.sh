#!/usr/bin/env bash
set -euo pipefail

gpu_index="${GPU:?physical H200 index required}"
result_dir="${RESULT_DIR:?result directory required}"
result_json="${RESULT_JSON:-${result_dir}/screen.json}"
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/denghaoran/miniconda3/envs/monarch_rt/bin/python}"
experiment_script="${EXPERIMENT_SCRIPT:-analysis/screen_partition_population_seeds.py}"
compute_processes() {
  nvidia-smi -i "${gpu_index}" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits
}
process_list="$(compute_processes)"
IFS=, read -r used util <<< "$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
if [ -n "${process_list}" ] || [ "${used}" -gt 100 ] || [ "${util}" -ne 0 ]; then
  echo "GPU ${gpu_index} not clean; no experiment launched" >&2
  exit 3
fi
mkdir -p "${result_dir}"
if [ -e "${result_json}" ]; then
  echo "Result already exists; use a new experiment directory" >&2
  exit 4
fi
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export WAN_SDPA_BACKEND=flash
export OMP_NUM_THREADS=4
export PYTHONPATH="${source_root}:${source_root}/analysis:${PYTHONPATH:-}"
export PATH="$(dirname "${python_bin}"):/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$(dirname "${python_bin}")/../lib:${LD_LIBRARY_PATH:-}"
cd "${source_root}"
"${python_bin}" -u "${experiment_script}" --output "${result_json}" "$@" > "${result_dir}/screen.log" 2>&1 &
process=$!
printf '%s\n' "${process}" > "${result_dir}/pid"
: > "${result_dir}/purity.log"
while kill -0 "${process}" 2>/dev/null; do
  while IFS=, read -r pid memory name; do
    pid="${pid// /}"
    if [ -n "${pid}" ] && [ "${pid}" != "${process}" ]; then
      printf '%s foreign=%s,%s,%s\n' "$(date -Iseconds)" "${pid}" "${memory}" "${name}" >> "${result_dir}/purity.log"
    fi
  done < <(compute_processes)
  sleep 15
done
set +e
wait "${process}"
rc=$?
set -e
printf '%s\n' "${rc}" > "${result_dir}/exit_code"
tail -30 "${result_dir}/screen.log"
if [ -s "${result_dir}/purity.log" ]; then
  echo "GPU contention observed; timing invalid" >&2
  exit 5
fi
exit "${rc}"
