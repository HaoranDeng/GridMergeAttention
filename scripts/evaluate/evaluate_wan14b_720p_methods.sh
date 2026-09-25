#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_four_methods_20260816}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
SVG_PYTHON="${SVG_PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/local2/denghaoran/GridMergeAttention/envs/vbench/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/workspace/VBench}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
GPU="${GPU:-5}"
SECONDARY_GPU="${SECONDARY_GPU:-4}"
RESULTS="$EXPERIMENT_ROOT/results"
VBENCH_RESULTS="$RESULTS/vbench"
VBENCH_HOME="/local2/denghaoran/GridMergeAttention/vbench_home"
VBENCH_CACHE_DIR="$VBENCH_HOME/.cache/vbench"

gpu_is_clean() {
  local gpu="$1"
  local utilization memory_used
  IFS=',' read -r utilization memory_used < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits
  )
  utilization="${utilization// /}"
  memory_used="${memory_used// /}"
  [[ "$utilization" == "0" && "$memory_used" -le 16 ]]
}

for backend in dense svg svg2 gma; do
  count=$(find "$EXPERIMENT_ROOT/$backend" -maxdepth 1 -name '*.mp4' | wc -l)
  [[ "$count" -eq 20 ]] || { echo "$backend has $count/20 videos" >&2; exit 2; }
done

gpu_is_clean "$GPU" || {
  echo "Primary GPU $GPU is not clean" >&2
  exit 3
}

mkdir -p "$RESULTS" "$VBENCH_RESULTS" "$VBENCH_HOME"
export TOKENIZERS_PARALLELISM=false
export VBENCH_CACHE_DIR

if [[ ! -f "$RESULTS/paired_quality.json" ]]; then
  env CUDA_VISIBLE_DEVICES="$GPU" \
    "$SVG_PYTHON" -u "$REPO_DIR/analysis/evaluate_video_quality.py" \
    --reference_dir "$EXPERIMENT_ROOT/dense" \
    --method_dir "$EXPERIMENT_ROOT/svg" --method_name svg \
    --method_dir "$EXPERIMENT_ROOT/svg2" --method_name svg2 \
    --method_dir "$EXPERIMENT_ROOT/gma" --method_name gma \
    --output "$RESULTS/paired_quality.json" \
    2>&1 | tee "$RESULTS/paired_quality.log"
fi

dimensions=(
  subject_consistency background_consistency motion_smoothness
  dynamic_degree aesthetic_quality imaging_quality
)

run_vbench() {
  local backend="$1"
  local gpu="$2"
  local output_dir="$VBENCH_RESULTS/$backend"
  mkdir -p "$output_dir"
  "$SVG_PYTHON" "$REPO_DIR/analysis/prepare_vbench_prompt_map.py" \
    --prompt_path "$PROMPTS" \
    --video_dir "$EXPERIMENT_ROOT/$backend" \
    --output "$output_dir/prompts.json"
  if ! find "$output_dir" -name '*_eval_results.json' -print -quit | grep -q .; then
    echo "Running VBench $backend on physical GPU $gpu"
    env CUDA_VISIBLE_DEVICES="$gpu" \
      HOME="$VBENCH_HOME" XDG_CACHE_HOME="$VBENCH_HOME/.cache" \
      TORCH_HOME="$VBENCH_HOME/.cache/torch" HF_HOME="$VBENCH_HOME/.cache/huggingface" \
      "$VBENCH_PYTHON" -u "$VBENCH_ROOT/evaluate.py" \
        --videos_path "$EXPERIMENT_ROOT/$backend" \
        --dimension "${dimensions[@]}" \
        --mode custom_input \
        --load_ckpt_from_local True \
        --prompt_file "$output_dir/prompts.json" \
        --output_path "$output_dir" \
        2>&1 | tee "$output_dir/evaluate.log"
  fi
}

run_vbench_pair() {
  local first="$1"
  local second="$2"
  local first_pid second_pid status=0
  run_vbench "$first" "$GPU" &
  first_pid=$!
  run_vbench "$second" "$SECONDARY_GPU" &
  second_pid=$!
  wait "$first_pid" || status=1
  wait "$second_pid" || status=1
  return "$status"
}

precompute_marker="$VBENCH_RESULTS/.precompute_running"
if [[ -f "$precompute_marker" ]]; then
  precompute_pid=$(cat "$precompute_marker")
  while kill -0 "$precompute_pid" 2>/dev/null; do
    echo "Waiting for VBench precompute process $precompute_pid"
    sleep 30
  done
  rm -f "$precompute_marker"
fi

if [[ "$SECONDARY_GPU" != "$GPU" ]] && gpu_is_clean "$SECONDARY_GPU"; then
  echo "Using clean GPUs $GPU and $SECONDARY_GPU for parallel VBench evaluation"
  run_vbench_pair dense svg
  run_vbench_pair svg2 gma
else
  echo "Secondary GPU $SECONDARY_GPU is unavailable; running VBench serially on GPU $GPU"
  for backend in dense svg svg2 gma; do
    run_vbench "$backend" "$GPU"
  done
fi

"$SVG_PYTHON" "$REPO_DIR/analysis/summarize_wan14b_720p_experiment.py" \
  --experiment_root "$EXPERIMENT_ROOT" \
  --quality_json "$RESULTS/paired_quality.json" \
  --vbench_root "$VBENCH_RESULTS" \
  --output_json "$RESULTS/summary.json" \
  --output_tsv "$RESULTS/summary.tsv"
