#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
QUALITY_PYTHON="${QUALITY_PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
MODEL="${MODEL:-/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers}"
SVG_ROOT="${SVG_ROOT:-/home/denghaoran/workspace/Sparse-VideoGen}"
PROMPT_PATH="${PROMPT_PATH:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
GPU="${GPU:?Set GPU to one exclusively clean physical H200 index}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_svg_density_curve_20260902}"
RUN_ROOT="${RUN_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
DENSE_REFERENCE="${DENSE_REFERENCE:-/local2/denghaoran/GridMergeAttention/experiments/wan14b_720p_50step_hellinger_p16_c32g45_a0p2_20260829/dense_reference_exact_prompt_20260816}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/local2/denghaoran/GridMergeAttention/gpu_locks}"

METHODS=(
  svg_density05
  svg_density10
  svg_density15
  svg_density20
  svg_density25
  svg_density30
  svg_density35
)
REQUESTED_DENSITIES=(0.05 0.10 0.15 0.20 0.25 0.30 0.35)
# The upstream parameter excludes the mandatory first-frame sink and is rounded
# to a 128-token half-band. These values target the requested realized density.
PATTERN_DENSITIES=(
  0.0016924050
  0.0517980572
  0.1038222222
  0.1543786568
  0.2064866717
  0.2598538675
  0.3113634669
)
VIDEOS=20

check_clean_gpu() {
  local utilization memory_used compute_processes
  IFS=',' read -r utilization memory_used < <(
    nvidia-smi -i "$GPU" \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits | tr -d ' '
  )
  compute_processes="$(
    nvidia-smi -i "$GPU" --query-compute-apps=pid \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )"
  if [[ "$utilization" != "0" || "$memory_used" -gt 16 || -n "$compute_processes" ]]; then
    echo "GPU $GPU is not exclusively clean: utilization=${utilization}%, memory=${memory_used} MiB, processes=${compute_processes:-none}" >&2
    return 2
  fi
}

run_with_gpu_audit() {
  local method="$1"
  local log="$2"
  shift 2
  local audit_log="${log%.log}.gpu_audit.log"
  local overlap_log="${log%.log}.gpu_overlap.log"
  local method_pid status=0 pids pid
  : >"$audit_log"
  : >"$overlap_log"

  "$@" >>"$log" 2>&1 &
  method_pid=$!
  while kill -0 "$method_pid" 2>/dev/null; do
    pids="$(
      nvidia-smi -i "$GPU" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d' || true
    )"
    printf '%s method_pid=%s gpu_pids=%s\n' \
      "$(date --iso-8601=seconds)" "$method_pid" "${pids//$'\n'/,}" \
      >>"$audit_log"
    for pid in $pids; do
      if [[ "$pid" != "$method_pid" ]]; then
        printf '%s foreign_pid=%s method_pid=%s\n' \
          "$(date --iso-8601=seconds)" "$pid" "$method_pid" \
          >>"$overlap_log"
      fi
    done
    sleep 30
  done
  wait "$method_pid" || status=$?
  if [[ -s "$overlap_log" ]]; then
    echo "Foreign GPU process overlapped $method; timing is invalid." >&2
    return 3
  fi
  return "$status"
}

for path in "$PYTHON" "$QUALITY_PYTHON" "$MODEL" "$SVG_ROOT" "$PROMPT_PATH" "$DENSE_REFERENCE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results" "$RUN_ROOT/triton_cache"
mkdir -p "$GPU_LOCK_DIR"
exec 9>"$GPU_LOCK_DIR/h200_gpu${GPU}.lock"
if ! flock -n 9; then
  echo "GPU $GPU is already reserved by another GMA runner." >&2
  exit 2
fi
check_clean_gpu

export PYTHONPATH="$REPO_DIR:$SVG_ROOT:$SVG_ROOT/svg/kernels/build:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"
"$PYTHON" analysis/verify_prompt_contract.py \
  --prompt_path "$PROMPT_PATH" \
  --num_videos "$VIDEOS" \
  --seed 0 \
  --num_samples 1 \
  --manifest_out "$RUN_ROOT/prompt_contract.json" \
  >"$RUN_ROOT/logs/prompt_contract.log" 2>&1

{
  echo -e "method\trequested_density\tsvg_pattern_density"
  for i in "${!METHODS[@]}"; do
    echo -e "${METHODS[$i]}\t${REQUESTED_DENSITIES[$i]}\t${PATTERN_DENSITIES[$i]}"
  done
} >"$RUN_ROOT/sweep_manifest.tsv"

{
  echo "run_id=$RUN_ID"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "physical_gpu=$GPU"
  echo "model=$MODEL"
  echo "prompt_path=$PROMPT_PATH"
  echo "dense_reference=$DENSE_REFERENCE"
  echo "backend=svg_all"
  nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader
} >"$RUN_ROOT/launch_manifest.txt"

for i in "${!METHODS[@]}"; do
  method="${METHODS[$i]}"
  output_dir="$RUN_ROOT/$method"
  runtime_json="$output_dir/runtime.json"
  density_summary="$RUN_ROOT/results/${method}_density.json"
  log="$RUN_ROOT/logs/$method.log"
  completed=0
  if [[ -d "$output_dir" ]]; then
    completed="$(find "$output_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  fi

  if [[ "$completed" -ge "$VIDEOS" && -s "$runtime_json" ]]; then
    "$QUALITY_PYTHON" analysis/summarize_svg_static_density.py \
      --svg_pattern_density "${PATTERN_DENSITIES[$i]}" \
      --requested_density "${REQUESTED_DENSITIES[$i]}" \
      --output "$density_summary" \
      >>"$log" 2>&1
    echo "Skipping complete method $method ($completed videos)." | tee -a "$log"
    continue
  fi
  if [[ "$completed" -gt 0 && ! -s "$runtime_json" ]]; then
    echo "$method has videos but no runtime JSON; refusing an unverifiable resume." >&2
    exit 3
  fi

  check_clean_gpu
  export TRITON_CACHE_DIR="$RUN_ROOT/triton_cache/$method"
  mkdir -p "$output_dir" "$TRITON_CACHE_DIR"

  "$PYTHON" analysis/verify_prompt_contract.py \
    --prompt_path "$PROMPT_PATH" \
    --num_videos "$VIDEOS" \
    --seed 0 \
    --num_samples 1 \
    --manifest_out "$output_dir/prompt_manifest.json" \
    >>"$log" 2>&1

  run_with_gpu_audit "$method" "$log" \
    "$PYTHON" -u analysis/benchmark_wan14b_720p_methods.py \
    --backend svg_all \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPT_PATH" \
    --output_dir "$output_dir" \
    --start 0 \
    --count "$VIDEOS" \
    --seed 0 \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_steps 50 \
    --svg_pattern_density "${PATTERN_DENSITIES[$i]}" \
    --wan_forward_backend svg \
    --qk_norm_backend triton \
    --rope_backend svg

  "$QUALITY_PYTHON" analysis/summarize_svg_static_density.py \
    --svg_pattern_density "${PATTERN_DENSITIES[$i]}" \
    --requested_density "${REQUESTED_DENSITIES[$i]}" \
    --output "$density_summary" \
    >>"$log" 2>&1
done

quality_args=()
for method in "${METHODS[@]}"; do
  quality_args+=(--method_dir "$RUN_ROOT/$method" --method_name "$method")
done

check_clean_gpu
"$QUALITY_PYTHON" analysis/evaluate_video_quality.py \
  --reference_dir "$DENSE_REFERENCE" \
  "${quality_args[@]}" \
  --output "$RUN_ROOT/results/paired_quality_20videos.json" \
  --lpips_batch_size 8 \
  >"$RUN_ROOT/logs/quality.log" 2>&1

echo "SVG density sweep and quality evaluation complete: $RUN_ROOT"
