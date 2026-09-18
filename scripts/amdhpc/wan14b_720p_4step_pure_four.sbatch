#!/usr/bin/env bash
#SBATCH -J GMA4STEP
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -p mi3008x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 12:00:00
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
PYTHON="${PYTHON:-$WORK_ROOT/miniconda3/envs/gma_amd/bin/python}"
SVG_ROOT="${SVG_ROOT:-$WORK_ROOT/Sparse-VideoGen}"
MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/wan_models}"
MODEL_DIR="$MODEL_ROOT/Wan2.1-T2V-14B-StepDistill-CfgDistill"
CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/distill_model.pt}"
RUN_ID="${RUN_ID:-wan14b_720p_4step_pure_four_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
VIDEO_COUNT="${VIDEO_COUNT:-20}"
EXPECTED_CHECKPOINT_BYTES=28577333801

mkdir -p "$REPO_DIR/out" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/cache"
while [[ ! -f "$CHECKPOINT" ]] || (( $(stat -c %s "$CHECKPOINT" 2>/dev/null || echo 0) < EXPECTED_CHECKPOINT_BYTES )); do
  echo "Waiting for complete StepDistill checkpoint: $CHECKPOINT"
  sleep 60
done

export WAN_MODEL_ROOT="$MODEL_ROOT"
export SPARSE_VIDEO_GEN_ROOT="$SVG_ROOT"
FASTVIDEO_ROOT="${FASTVIDEO_ROOT:-/home1/denghaoran/workspace/FastVideo}"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:$FASTVIDEO_ROOT/fastvideo-kernel/python:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

config_for() {
  case "$1" in
    gma) echo "$REPO_DIR/configs/token_merging_wan_14b_hard_c4_triton_720p_fewstep.yaml" ;;
    gma_h2) echo "$REPO_DIR/configs/token_merging_wan_14b_hard_c4_h2_720p_4step.yaml" ;;
    gma_h4) echo "$REPO_DIR/configs/token_merging_wan_14b_hard_c4_h4_720p_4step.yaml" ;;
    gma_c4_first_frame_full) echo "$REPO_DIR/configs/token_merging_wan_14b_hard_c4_first_frame_full_720p_4step.yaml" ;;
    monarch_rt_all) echo "$REPO_DIR/configs/token_merging_wan_14b_monarchrt_720p_4step_all.yaml" ;;
    monarch_rt_h2_all) echo "$REPO_DIR/configs/token_merging_wan_14b_monarchrt_h2_720p_4step_all.yaml" ;;
    monarch_rt_h2w2_all) echo "$REPO_DIR/configs/token_merging_wan_14b_monarchrt_h2w2_720p_4step_all.yaml" ;;
    monarch_rt_h4_all) echo "$REPO_DIR/configs/token_merging_wan_14b_monarchrt_h4_720p_4step_all.yaml" ;;
    monarch_rt_w2_all) echo "$REPO_DIR/configs/token_merging_wan_14b_monarchrt_w2_720p_4step_all.yaml" ;;
    monarch_rt_w4_all) echo "$REPO_DIR/configs/token_merging_wan_14b_monarchrt_w4_720p_4step_all.yaml" ;;
    vmonarch_all) echo "$REPO_DIR/configs/vmonarch_wan_14b_720p_4step_all.yaml" ;;
    vsa_training_free_all) echo "$REPO_DIR/configs/vsa_training_free_wan_14b_720p_4step_all.yaml" ;;
    ear_all) echo "$REPO_DIR/configs/svg_ear_wan_14b_720p_4step_all.yaml" ;;
    svg2_all) echo "$REPO_DIR/configs/svg2_wan_14b_720p_4step_all.yaml" ;;
    svg_all) echo "$REPO_DIR/configs/svg_wan_14b_720p_4step_all.yaml" ;;
  esac
}

run_method() {
  local method="$1"
  local gpu="$2"
  local output="$OUTPUT_ROOT/$method"
  local extra_args=()
  local density_env=()
  if [[ "$method" =~ ^(gma|monarch_rt) && -n "${GMA_HEAD_CHUNK:-}" ]]; then
    extra_args+=(--token_merging_head_chunk "$GMA_HEAD_CHUNK")
  fi
  if [[ "$method" =~ ^gma && -n "${GMA_NUM_MERGE_TOKENS:-}" ]]; then
    extra_args+=(--token_merging_num_merge_tokens "$GMA_NUM_MERGE_TOKENS")
  fi
  if [[ "${RECORD_SVG_DENSITY:-0}" == "1" && "$method" =~ ^(svg2_all|ear_all)$ ]]; then
    density_env+=(SVG_DENSITY_LOG="$output/density.jsonl")
  fi
  mkdir -p "$output" "$OUTPUT_ROOT/triton_cache/$method"
  env \
    -u HIP_VISIBLE_DEVICES \
    -u CUDA_VISIBLE_DEVICES \
    ROCR_VISIBLE_DEVICES="$gpu" \
    TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache/$method" \
    "${density_env[@]}" \
    "$PYTHON" -u "$REPO_DIR/generate.py" \
      --config "$(config_for "$method")" \
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
      "${extra_args[@]}" \
      >"$OUTPUT_ROOT/logs/$method.log" 2>&1
}

read -r -a methods <<< "${METHODS:-gma ear_all svg2_all svg_all}"
if [[ "${SLURM_JOB_PARTITION:-}" =~ ^(mi3008x|mi2508x|mi2104x)$ ]]; then
  pids=()
  for index in "${!methods[@]}"; do
    run_method "${methods[$index]}" "$index" &
    pids+=("$!")
    sleep 30
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || exit 1
else
  for method in "${methods[@]}"; do
    run_method "$method" 0
  done
fi

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
