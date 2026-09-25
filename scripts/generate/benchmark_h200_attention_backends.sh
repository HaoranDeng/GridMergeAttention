#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

RUN_ID="${RUN_ID:-h200_attention_backends_$(date +%Y%m%d_%H%M%S)}"
LOCAL2_ROOT="${LOCAL2_ROOT:-/local2/denghaoran/GridMergeAttention}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOCAL2_ROOT}/videos/${RUN_ID}}"
RESULT_ROOT="${RESULT_ROOT:-${LOCAL2_ROOT}/analysis/${RUN_ID}}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL2_ROOT}/wan_models}"
CHECKPOINT="${CHECKPOINT:-${LOCAL2_ROOT}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
NUM_VIDEOS="${NUM_VIDEOS:-20}"
GPU_ID="${GPU_ID:-2}"

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${LOCAL2_ROOT}/triton_cache/${RUN_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WAN_MODEL_ROOT="${MODEL_ROOT}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${LOCAL2_ROOT}/triton_cache/${RUN_ID}"
export TOKEN_MERGING_FORCE_TORCH_RMSNORM="${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-1}"

names=(sdpa flash_attn_triton monarch_rt hard_c4)
configs=(
  configs/dense_wan_fewstep.yaml
  configs/dense_wan_flash_attn_triton_fewstep.yaml
  configs/token_merging_wan_monarchrt_triton_fewstep.yaml
  configs/token_merging_wan_hard_c4_triton_fewstep.yaml
)

for idx in "${!names[@]}"; do
  name="${names[$idx]}"
  config="${configs[$idx]}"
  output_dir="${OUTPUT_ROOT}/${name}"
  result_json="${RESULT_ROOT}/${name}.json"
  log_file="${RESULT_ROOT}/${name}.log"
  mkdir -p "${output_dir}"

  echo "[$(date -Is)] starting ${name}"
  start_ms="$(date +%s%3N)"
  status="success"
  if ! python generate.py \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --prompt_path "${PROMPT_PATH}" \
      --output_dir "${output_dir}" \
      --num_videos "${NUM_VIDEOS}" \
      --num_samples 1 \
      --batch_size 1 \
      --sample_batch_size 1 \
      --seed 0 \
      --use_ema \
      --dtype bf16 \
      --video_preset ultrafast \
      --video_crf 23 \
      >"${log_file}" 2>&1; then
    status="failed"
  fi
  end_ms="$(date +%s%3N)"
  elapsed_ms="$((end_ms - start_ms))"
  mp4_count="$(find "${output_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  python - "$result_json" "$name" "$config" "$output_dir" "$status" "$elapsed_ms" "$mp4_count" "$NUM_VIDEOS" <<'PY'
import json
import sys

path, name, config, output_dir, status, elapsed_ms, mp4_count, num_videos = sys.argv[1:]
elapsed_seconds = int(elapsed_ms) / 1000.0
mp4_count_i = int(mp4_count)
payload = {
    "name": name,
    "config": config,
    "output_dir": output_dir,
    "status": status,
    "mp4_count": mp4_count_i,
    "num_videos": int(num_videos),
    "elapsed_seconds": elapsed_seconds,
    "seconds_per_mp4": elapsed_seconds / mp4_count_i if mp4_count_i else None,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
  echo "[$(date -Is)] ${name}: status=${status}, mp4=${mp4_count}, seconds=$((elapsed_ms / 1000))"
  if [ "${status}" != "success" ]; then
    echo "${name} failed; see ${log_file}" >&2
    exit 1
  fi
done

python - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [
    json.loads((root / f"{name}.json").read_text())
    for name in ("sdpa", "flash_attn_triton", "monarch_rt", "hard_c4")
]
baseline = rows[0]["elapsed_seconds"]
with (root / "runtime_summary.tsv").open("w", encoding="utf-8") as f:
    f.write("name\tstatus\tmp4_count\telapsed_seconds\tseconds_per_mp4\tspeedup_vs_sdpa\tconfig\toutput_dir\n")
    for row in rows:
        speedup = baseline / row["elapsed_seconds"] if row["elapsed_seconds"] else None
        f.write(
            f"{row['name']}\t{row['status']}\t{row['mp4_count']}\t"
            f"{row['elapsed_seconds']:.3f}\t{row['seconds_per_mp4']:.3f}\t"
            f"{speedup:.3f}\t{row['config']}\t{row['output_dir']}\n"
        )
PY

cat "${RESULT_ROOT}/runtime_summary.tsv"
