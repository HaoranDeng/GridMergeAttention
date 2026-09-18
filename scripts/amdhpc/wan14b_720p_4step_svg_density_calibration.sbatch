#!/usr/bin/env bash
#SBATCH -J SVG-DENS-CAL
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 01:00:00
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
PYTHON="${PYTHON:-$WORK_ROOT/miniconda3/envs/gma_amd/bin/python}"
SVG_ROOT="${SVG_ROOT:-$WORK_ROOT/Sparse-VideoGen}"
MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/wan_models}"
MODEL_DIR="$MODEL_ROOT/Wan2.1-T2V-14B-StepDistill-CfgDistill"
CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/distill_model.pt}"
RUN_ID="${RUN_ID:-wan14b_720p_4step_svg_density_calibration_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
CASES="${CASES:-svg2_all:0.72 ear_all:0.72 svg2_all:0.48 ear_all:0.48}"
VIDEO_COUNT="${VIDEO_COUNT:-1}"
EXPECTED_CHECKPOINT_BYTES=28577333801

mkdir -p "$REPO_DIR/out" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/cache" "$OUTPUT_ROOT/results"
while [[ ! -f "$CHECKPOINT" ]] || (( $(stat -c %s "$CHECKPOINT" 2>/dev/null || echo 0) < EXPECTED_CHECKPOINT_BYTES )); do
  echo "Waiting for complete StepDistill checkpoint: $CHECKPOINT"
  sleep 60
done

export WAN_MODEL_ROOT="$MODEL_ROOT"
export SPARSE_VIDEO_GEN_ROOT="$SVG_ROOT"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

config_for() {
  case "$1" in
    svg2_all) echo "$REPO_DIR/configs/svg2_wan_14b_720p_4step_all.yaml" ;;
    ear_all) echo "$REPO_DIR/configs/svg_ear_wan_14b_720p_4step_all.yaml" ;;
    *) echo "Unsupported backend: $1" >&2; return 2 ;;
  esac
}

run_case() {
  local spec="$1"
  local gpu="$2"
  local backend="${spec%%:*}"
  local top_p="${spec#*:}"
  local label="${backend}_p${top_p/./}"
  local output="$OUTPUT_ROOT/$label"
  mkdir -p "$output" "$OUTPUT_ROOT/triton_cache/$label"

  env \
    -u HIP_VISIBLE_DEVICES \
    -u CUDA_VISIBLE_DEVICES \
    ROCR_VISIBLE_DEVICES="$gpu" \
    SVG_TOP_P="$top_p" \
    SVG_DENSITY_LOG="$output/density.jsonl" \
    TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache/$label" \
    "$PYTHON" -u "$REPO_DIR/generate.py" \
      --config "$(config_for "$backend")" \
      --checkpoint "$CHECKPOINT" \
      --prompt_path "$PROMPTS" \
      --output_dir "$output" \
      --num_videos "$VIDEO_COUNT" \
      --num_samples 1 \
      --batch_size 1 \
      --sample_batch_size 1 \
      --seed 0 \
      --dtype bf16 \
      --skip_existing \
      --runtime_json "$output/runtime.json" \
      >"$OUTPUT_ROOT/logs/$label.log" 2>&1

  "$PYTHON" "$REPO_DIR/analysis/summarize_svg_density.py" \
    --input "$output/density.jsonl" \
    --output "$OUTPUT_ROOT/results/${label}_density.json" \
    --backend "$backend" \
    --calls_per_video 160 \
    >"$OUTPUT_ROOT/logs/${label}_density.log" 2>&1
}

read -r -a cases <<< "$CASES"
if (( ${#cases[@]} > 4 )); then
  echo "mi2104x calibration supports at most four concurrent cases" >&2
  exit 2
fi

pids=()
for index in "${!cases[@]}"; do
  run_case "${cases[$index]}" "$index" &
  pids+=("$!")
  sleep 30
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit 1

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
