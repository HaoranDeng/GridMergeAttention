#!/bin/bash

#SBATCH -J WanProfile
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
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
elif [ -f "/work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "/work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh"
fi

CONDA_ENV="${CONDA_ENV:-monarch_rt}"
conda activate "${CONDA_ENV}"

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work1/jasoncong/denghaoran/training-free-videogen}"
RUN_ID="${RUN_ID:-video_pipeline_profile_$(date +%Y%m%d_%H%M%S)}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${REPO_DIR}/wan_models}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/self_forcing_dmd.pt}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/prompts/MovieGenVideoBench_extended.txt}"
RESULT_DIR="${RESULT_DIR:-${SCRATCH_ROOT}/analysis/${RUN_ID}}"
VIDEO_DIR="${VIDEO_DIR:-${SCRATCH_ROOT}/videos/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-1}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
BASELINE_LABEL="${BASELINE_LABEL:-dense}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/dense_wan_fewstep.yaml}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-hard_c4_triton}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-configs/token_merging_wan_hard_c4_triton_fewstep.yaml}"

cd "${REPO_DIR}"
mkdir -p out "${RESULT_DIR}" "${VIDEO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKEN_MERGING_FORCE_TORCH_RMSNORM="${TOKEN_MERGING_FORCE_TORCH_RMSNORM:-1}"

run_profile() {
  local label="$1"
  local config="$2"
  local miopen_cache_root="${TMPDIR:-${REPO_DIR}/assets}/miopen-${SLURM_JOB_ID:-manual}-${label}"

  mkdir -p "${miopen_cache_root}/user-db" "${miopen_cache_root}/kernel-cache"
  echo "Profiling ${label}"
  echo "  config: ${config}"
  HIP_VISIBLE_DEVICES="${GPU_ID}" ROCR_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MIOPEN_USER_DB_PATH="${miopen_cache_root}/user-db" \
  MIOPEN_CUSTOM_CACHE_DIR="${miopen_cache_root}/kernel-cache" \
  PYTHONUNBUFFERED=1 \
    python analysis/profile_video_pipeline.py \
      --label "${label}" \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}" \
      --prompt_path "${PROMPT_PATH}" \
      --output_json "${RESULT_DIR}/${label}.json" \
      --output_video "${VIDEO_DIR}/${label}.mp4" \
      --seed "${SEED}" \
      --warmup "${WARMUP}" \
      --fps "${FPS}" \
      --dtype "${DTYPE}" \
      --use_ema
}

run_profile "${BASELINE_LABEL}" "${BASELINE_CONFIG}"
run_profile "${CANDIDATE_LABEL}" "${CANDIDATE_CONFIG}"

python - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
rows = []
for path in sorted(result_dir.glob("*.json")):
    payload = json.loads(path.read_text())
    total = payload["total_wall_seconds"]
    groups = payload["groups"]
    attention = groups["self_attention_core"]["seconds"] + groups["cross_attention_core"]["seconds"]
    rows.append({
        "label": payload["label"],
        "total": total,
        "self_attention_core": groups["self_attention_core"]["seconds"],
        "cross_attention_core": groups["cross_attention_core"]["seconds"],
        "all_attention_core": attention,
        "all_attention_pct": 100.0 * attention / total,
        "ffn": groups["ffn"]["seconds"],
        "ffn_pct": groups["ffn"]["percent_of_total"],
        "vae_decode": groups["vae_decode"]["seconds"],
        "vae_decode_pct": groups["vae_decode"]["percent_of_total"],
        "save_video": groups["save_video"]["seconds"],
        "save_video_pct": groups["save_video"]["percent_of_total"],
    })

summary = result_dir / "profile_summary.tsv"
with summary.open("w", encoding="utf-8") as f:
    f.write("label\ttotal_s\tself_attn_core_s\tcross_attn_core_s\tall_attn_core_s\tall_attn_pct\tffn_s\tffn_pct\tvae_decode_s\tvae_decode_pct\tsave_video_s\tsave_video_pct\n")
    for row in rows:
        f.write(
            f"{row['label']}\t{row['total']:.3f}\t{row['self_attention_core']:.3f}\t"
            f"{row['cross_attention_core']:.3f}\t{row['all_attention_core']:.3f}\t"
            f"{row['all_attention_pct']:.2f}\t{row['ffn']:.3f}\t{row['ffn_pct']:.2f}\t"
            f"{row['vae_decode']:.3f}\t{row['vae_decode_pct']:.2f}\t"
            f"{row['save_video']:.3f}\t{row['save_video_pct']:.2f}\n"
        )
print(summary)
PY
