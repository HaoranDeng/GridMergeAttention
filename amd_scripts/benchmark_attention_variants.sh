#!/bin/bash

#SBATCH -J AttnBench
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 01:30:00
#SBATCH -p mi3001x

set -euo pipefail

if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -n "${WORK:-}" ] && [ -f "${WORK}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${WORK}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV to the environment with torch/Wan dependencies.}"
conda activate "${CONDA_ENV}"

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work1/jasoncong/denghaoran/training-free-videogen}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/videos/attention_variant_runtime}"
RESULT_DIR="${RESULT_DIR:-${SCRATCH_ROOT}/analysis/attention_variant_runtime}"
NUM_VIDEOS="${NUM_VIDEOS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"

read -r -a GPU_ID_LIST <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

NAMES=(
  "dense"
  "gridmerge_one"
  "gridmerge_one_transpose"
  "hard_route_c6"
  "hard_route_c6_transpose"
  "select_route_c6"
  "select_route_c6_transpose"
  "global10"
)

CONFIGS=(
  "configs/dense_wan_fewstep.yaml"
  "configs/token_merging_wan_fewstep.yaml"
  "configs/token_merging_wan_gridmerge_one_transpose_fewstep.yaml"
  "configs/token_merging_wan_hard_route_c6_fewstep.yaml"
  "configs/token_merging_wan_hard_route_c6_transpose_fewstep.yaml"
  "configs/token_merging_wan_select_route_c6_fewstep.yaml"
  "configs/token_merging_wan_select_route_c6_transpose_fewstep.yaml"
  "configs/token_merging_wan_global10_fewstep.yaml"
)

if (( ${#GPU_ID_LIST[@]} != ${#NAMES[@]} )); then
  echo "Expected ${#NAMES[@]} GPU ids, got ${#GPU_ID_LIST[@]}: ${GPU_ID_LIST[*]}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p out "${OUTPUT_ROOT}" "${RESULT_DIR}"
rm -rf "${OUTPUT_ROOT:?}/"*
rm -f "${RESULT_DIR}/"*.json "${RESULT_DIR}/runtime_summary.tsv"

required_files=("${CHECKPOINT}" "${PROMPT_PATH}")
for config in "${CONFIGS[@]}"; do
  required_files+=("${REPO_DIR}/${config}")
done
for path in "${required_files[@]}"; do
  if [ ! -f "${path}" ]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
done

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKEN_MERGING_FORCE_TORCH_RMSNORM="${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-1}"

run_one() {
  local index="$1"
  local gpu_id="${GPU_ID_LIST[$index]}"
  local name="${NAMES[$index]}"
  local config="${CONFIGS[$index]}"
  local output_dir="${OUTPUT_ROOT}/${name}"
  local log_path="${RESULT_DIR}/${name}.log"
  local result_path="${RESULT_DIR}/${name}.json"
  local miopen_cache_root="${TMPDIR:-${REPO_DIR}/assets}/miopen-${SLURM_JOB_ID:-manual}-${name}"
  local start_ns end_ns elapsed_ms mp4_count status

  mkdir -p "${output_dir}"
  mkdir -p "${miopen_cache_root}/user-db" "${miopen_cache_root}/kernel-cache"
  echo "Launching ${name} on GPU ${gpu_id}"
  echo "  config: ${config}"
  echo "  output: ${output_dir}"
  echo "  log:    ${log_path}"

  start_ns="$(date +%s%N)"
  set +e
  HIP_VISIBLE_DEVICES="${gpu_id}" ROCR_VISIBLE_DEVICES="${gpu_id}" CUDA_VISIBLE_DEVICES="${gpu_id}" \
  MIOPEN_USER_DB_PATH="${miopen_cache_root}/user-db" \
  MIOPEN_CUSTOM_CACHE_DIR="${miopen_cache_root}/kernel-cache" \
  PYTHONUNBUFFERED=1 \
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
      >"${log_path}" 2>&1
  status="$?"
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms="$(((end_ns - start_ns) / 1000000))"
  mp4_count="$(find "${output_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"

  python - "$result_path" "$name" "$config" "$output_dir" "$gpu_id" "$status" "$elapsed_ms" "$mp4_count" "$NUM_VIDEOS" "$NUM_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

path, name, config, output_dir, gpu_id, status, elapsed_ms, mp4_count, num_videos, num_samples = sys.argv[1:]
elapsed_ms = int(elapsed_ms)
mp4_count = int(mp4_count)
payload = {
    "name": name,
    "config": config,
    "output_dir": output_dir,
    "gpu_id": int(gpu_id),
    "status": int(status),
    "elapsed_seconds": elapsed_ms / 1000.0,
    "mp4_count": mp4_count,
    "requested_videos": int(num_videos),
    "num_samples": int(num_samples),
    "seconds_per_mp4": (elapsed_ms / 1000.0 / mp4_count) if mp4_count else None,
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY

  return "${status}"
}

pids=()
for index in "${!NAMES[@]}"; do
  run_one "${index}" &
  pids+=("$!")
done

stop_jobs() {
  echo "Stopping benchmark jobs..." >&2
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}
trap stop_jobs INT TERM

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${NAMES[$index]}."
  else
    job_status=$?
    echo "Failed ${NAMES[$index]} with status ${job_status}. See ${RESULT_DIR}/${NAMES[$index]}.log" >&2
    status=1
  fi
done

python - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
rows = []
for path in sorted(result_dir.glob("*.json")):
    rows.append(json.loads(path.read_text()))

summary_path = result_dir / "runtime_summary.tsv"
with summary_path.open("w", encoding="utf-8") as f:
    f.write("name\tstatus\tmp4_count\telapsed_seconds\tseconds_per_mp4\tconfig\toutput_dir\n")
    for row in rows:
        per_mp4 = row["seconds_per_mp4"]
        f.write(
            f"{row['name']}\t{row['status']}\t{row['mp4_count']}\t"
            f"{row['elapsed_seconds']:.3f}\t"
            f"{'' if per_mp4 is None else f'{per_mp4:.3f}'}\t"
            f"{row['config']}\t{row['output_dir']}\n"
        )

print(summary_path)
PY

exit "${status}"
