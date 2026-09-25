#!/usr/bin/env bash

#SBATCH -J GMA-C468
#SBATCH -o out/job.%j.hellinger_common_c.out
#SBATCH -e out/job.%j.hellinger_common_c.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 04:00:00
#SBATCH -p mi2104x

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
RUN_ID="${RUN_ID:-wan14b_720p_4step_hellinger_common_c_20260830}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work1/jasoncong/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
MODEL_ROOT="${WAN_MODEL_ROOT:-/work1/jasoncong/denghaoran/wan_models}"
CHECKPOINT="$MODEL_ROOT/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt"
PROMPTS="$REPO_DIR/prompts/MovieGenVideoBench_extended.txt"
PYTHON="${PYTHON:-/work1/jasoncong/denghaoran/miniconda3/envs/gma_amd/bin/python}"
NUM_VIDEOS="${NUM_VIDEOS:-20}"

cd "$REPO_DIR"
mkdir -p out "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/cache"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WAN_MODEL_ROOT="$MODEL_ROOT"
export TOKENIZERS_PARALLELISM=false

for required in "$PYTHON" "$CHECKPOINT" "$PROMPTS"; do
  [ -e "$required" ] || { echo "FATAL: missing $required" >&2; exit 1; }
done

"$PYTHON" analysis/verify_prompt_contract.py \
  --prompt_path "$PROMPTS" \
  --num_videos "$NUM_VIDEOS" \
  --seed 0 \
  --manifest_out "$OUTPUT_ROOT/prompt_preflight.json"

labels=(
  dense_flash
  hellinger_common_c4
  hellinger_common_c6
  hellinger_common_c8
)
configs=(
  configs/dense_wan_14b_720p_fewstep.yaml
  configs/token_merging_wan_720p_hellinger_span_common_c4_c32g45_a0p2_p16_j4_t1_rf0_fewstep.yaml
  configs/token_merging_wan_720p_hellinger_span_common_c6_c32g45_a0p2_p16_j4_t1_rf0_fewstep.yaml
  configs/token_merging_wan_720p_hellinger_span_common_c8_c32g45_a0p2_p16_j4_t1_rf0_fewstep.yaml
)

pids=()
for index in 0 1 2 3; do
  label="${labels[$index]}"
  config="${configs[$index]}"
  stage="$OUTPUT_ROOT/$label"
  videos="$stage/videos"
  mkdir -p "$videos" "$OUTPUT_ROOT/cache/$label"
  echo "[$label] gpu=$index config=$config start=$(date -Iseconds)"
  ROCR_VISIBLE_DEVICES="$index" \
  WAN_SDPA_BACKEND=flash \
  TRITON_CACHE_DIR="$OUTPUT_ROOT/cache/$label" \
  "$PYTHON" -u generate.py \
    --config "$config" \
    --checkpoint "$CHECKPOINT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$videos" \
    --runtime_json "$stage/runtime.json" \
    --num_videos "$NUM_VIDEOS" \
    --num_samples 1 \
    --seed 0 \
    --skip_existing \
    --use_ema \
    > "$stage/generate.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in 0 1 2 3; do
  if wait "${pids[$index]}"; then
    echo "[${labels[$index]}] generation complete"
  else
    echo "[${labels[$index]}] generation failed" >&2
    failed=1
  fi
done
[ "$failed" -eq 0 ] || exit 1

OUTPUT_ROOT="$OUTPUT_ROOT" NUM_VIDEOS="$NUM_VIDEOS" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from utils.prompt_contract import CANONICAL_FIRST_PROMPT_SHA256, load_output_prompt_manifest

root = Path(os.environ["OUTPUT_ROOT"])
expected = int(os.environ["NUM_VIDEOS"])
for label in (
    "dense_flash",
    "hellinger_common_c4",
    "hellinger_common_c6",
    "hellinger_common_c8",
):
    stage = root / label
    videos = stage / "videos"
    files = sorted(videos.glob("*.mp4"))
    records = json.loads((stage / "runtime.json").read_text())["records"]
    manifest = load_output_prompt_manifest(videos)
    first_hash = manifest["prompts"][0]["sha256"]
    if len(files) != expected or len(records) != expected:
        raise SystemExit(f"{label}: videos={len(files)} runtime_records={len(records)}")
    if first_hash != CANONICAL_FIRST_PROMPT_SHA256:
        raise SystemExit(f"{label}: video 000 is not the Tokyo-woman prompt")
    print(f"{label}: verified {expected} videos and runtime records")
PY

"$PYTHON" analysis/evaluate_psnr_ssim.py \
  --reference_dir "$OUTPUT_ROOT/dense_flash/videos" \
  --method_name hellinger_common_c4 \
  --method_dir "$OUTPUT_ROOT/hellinger_common_c4/videos" \
  --method_name hellinger_common_c6 \
  --method_dir "$OUTPUT_ROOT/hellinger_common_c6/videos" \
  --method_name hellinger_common_c8 \
  --method_dir "$OUTPUT_ROOT/hellinger_common_c8/videos" \
  --max_videos "$NUM_VIDEOS" \
  --output "$OUTPUT_ROOT/results/paired_psnr_ssim.json" \
  > "$OUTPUT_ROOT/results/paired_psnr_ssim.log" 2>&1

printf 'complete=%s\ncommit=%s\nnode=%s\n' \
  "$(date -Iseconds)" "$(git rev-parse HEAD)" "$(hostname)" \
  > "$OUTPUT_ROOT/COMPLETE"
