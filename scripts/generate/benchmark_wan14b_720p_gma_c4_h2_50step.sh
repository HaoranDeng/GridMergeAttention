#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
MODEL="${MODEL:-/home/denghaoran/wan_models/Wan2.1-T2V-14B-Diffusers}"
SVG_ROOT="${SVG_ROOT:-/home/denghaoran/workspace/Sparse-VideoGen}"
PROMPT_PATH="${PROMPT_PATH:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
GPU="${GPU:?Set GPU to one exclusively clean physical H200 index}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_gma_c4_h2_$(date +%Y%m%d)}"
RUN_ROOT="${RUN_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
DENSE_REFERENCE="${DENSE_REFERENCE:-/local2/denghaoran/GridMergeAttention/experiments/wan14b_720p_50step_hellinger_p16_c32g45_a0p2_20260829/dense_reference_exact_prompt_20260816}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/local2/denghaoran/GridMergeAttention/gpu_locks}"
METHOD="gma_c4_h2_all"
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
  local log="$1"
  shift
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
    echo "Foreign GPU process overlapped $METHOD; timing is invalid." >&2
    return 3
  fi
  return "$status"
}

for path in "$PYTHON" "$MODEL" "$SVG_ROOT" "$PROMPT_PATH" "$DENSE_REFERENCE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 2
  fi
done

OUTPUT_DIR="$RUN_ROOT/$METHOD"
RUNTIME_JSON="$OUTPUT_DIR/runtime.json"
LOG="$RUN_ROOT/logs/$METHOD.log"
mkdir -p "$OUTPUT_DIR" "$RUN_ROOT/logs" "$RUN_ROOT/results" "$RUN_ROOT/triton_cache/$METHOD"
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
export TRITON_CACHE_DIR="$RUN_ROOT/triton_cache/$METHOD"

cd "$REPO_DIR"
"$PYTHON" analysis/verify_prompt_contract.py \
  --prompt_path "$PROMPT_PATH" \
  --num_videos "$VIDEOS" \
  --seed 0 \
  --num_samples 1 \
  --manifest_out "$RUN_ROOT/prompt_contract.json" \
  >"$RUN_ROOT/logs/prompt_contract.log" 2>&1
cp "$RUN_ROOT/prompt_contract.json" "$OUTPUT_DIR/prompt_manifest.json"

completed=0
if [[ -d "$OUTPUT_DIR" ]]; then
  completed="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
fi

manifest="$RUN_ROOT/launch_manifest.txt"
if [[ ! -s "$manifest" || "$completed" -lt "$VIDEOS" ]]; then
  {
    echo "run_id=$RUN_ID"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "physical_gpu=$GPU"
    echo "model=$MODEL"
    echo "prompt_path=$PROMPT_PATH"
    echo "dense_reference=$DENSE_REFERENCE"
    echo "backend=gma_grid24x80"
    echo "method=pure all-layer GMA C4"
    echo "h_reduce=2"
    echo "w_reduce=1"
    nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.used,utilization.gpu \
      --format=csv,noheader
  } >"$manifest"
else
  {
    echo "quality_gpu=$GPU"
    echo "quality_git_commit=$(git rev-parse HEAD)"
    nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.used,utilization.gpu \
      --format=csv,noheader
  } >"$RUN_ROOT/quality_launch_manifest.txt"
fi

if [[ "$completed" -gt 0 && ! -s "$RUNTIME_JSON" ]]; then
  echo "$METHOD has videos but no runtime JSON; refusing an unverifiable resume." >&2
  exit 3
fi

if [[ "$completed" -lt "$VIDEOS" ]]; then
  run_with_gpu_audit "$LOG" \
    "$PYTHON" -u analysis/benchmark_wan14b_720p_methods.py \
    --backend gma_grid24x80 \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start 0 \
    --count "$VIDEOS" \
    --seed 0 \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_steps 50 \
    --wan_forward_backend svg \
    --qk_norm_backend triton \
    --rope_backend svg \
    --gma_head_chunk 10
else
  echo "Skipping complete generation ($completed videos)." | tee -a "$LOG"
fi

check_clean_gpu
run_with_gpu_audit "$RUN_ROOT/logs/quality.log" \
  "$PYTHON" analysis/evaluate_video_quality.py \
    --reference_dir "$DENSE_REFERENCE" \
    --method_dir "$OUTPUT_DIR" \
    --method_name "$METHOD" \
    --output "$RUN_ROOT/results/paired_quality_20videos.json" \
    --lpips_batch_size 8

echo "GMA C4 h_reduce=2 experiment complete: $RUN_ROOT"
