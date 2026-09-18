#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/wan_models}"
MODEL_DIR="$MODEL_ROOT/Wan2.1-T2V-14B-StepDistill-CfgDistill"
CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/distill_model.pt}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
PROMPT_PATH="${PROMPT_PATH:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
CONFIG="${CONFIG:-$REPO_DIR/configs/svg_ear_wan_14b_720p_4step_all.yaml}"
GPU="${GPU:?Set GPU to one exclusively clean physical H200 index}"
TOP_P="${TOP_P:?Set the SVG-EAR top-p value}"
TARGET_DENSITY="${TARGET_DENSITY:?Set the requested realized density}"
VIDEO_COUNT="${VIDEO_COUNT:-20}"
RUN_ID="${RUN_ID:-wan14b_720p_4step_svg_ear_density_$(date +%Y%m%d_%H%M%S)}"
METHOD_LABEL="${METHOD_LABEL:-svg_ear_density${TARGET_DENSITY/./}}"
RUN_ROOT="${RUN_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
OUTPUT_DIR="$RUN_ROOT/$METHOD_LABEL"
LOG="$RUN_ROOT/logs/$METHOD_LABEL.log"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/local2/denghaoran/GridMergeAttention/gpu_locks}"
EXPECTED_CHECKPOINT_BYTES=28577333801
CALLS_PER_VIDEO=160

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
  local method_pid status=0 pids pid
  local audit_log="$RUN_ROOT/logs/$METHOD_LABEL.gpu_audit.log"
  local overlap_log="$RUN_ROOT/logs/$METHOD_LABEL.gpu_overlap.log"
  : >"$audit_log"
  : >"$overlap_log"

  "$@" >>"$LOG" 2>&1 &
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
    echo "Foreign GPU process overlapped $METHOD_LABEL; timing is invalid." >&2
    return 3
  fi
  return "$status"
}

for path in "$PYTHON" "$CHECKPOINT" "$SVG_ROOT" "$PROMPT_PATH" "$CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 2
  fi
done
if (( $(stat -c %s "$CHECKPOINT") < EXPECTED_CHECKPOINT_BYTES )); then
  echo "StepDistill checkpoint is incomplete: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$RUN_ROOT/logs" "$RUN_ROOT/results" \
  "$RUN_ROOT/triton_cache/$METHOD_LABEL" "$GPU_LOCK_DIR"
exec 9>"$GPU_LOCK_DIR/h200_gpu${GPU}.lock"
if ! flock -n 9; then
  echo "GPU $GPU is already reserved by another GMA runner." >&2
  exit 2
fi
check_clean_gpu

completed="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
if [[ "$completed" -gt 0 && ! -s "$OUTPUT_DIR/prompt_manifest.json" ]]; then
  echo "$OUTPUT_DIR contains videos without a prompt manifest; refusing resume." >&2
  exit 3
fi
if [[ "$completed" -eq 0 ]]; then
  : >"$OUTPUT_DIR/density.jsonl"
fi

export WAN_MODEL_ROOT="$MODEL_ROOT"
export SPARSE_VIDEO_GEN_ROOT="$SVG_ROOT"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export SVG_TOP_P="$TOP_P"
export SVG_DENSITY_LOG="$OUTPUT_DIR/density.jsonl"
export TRITON_CACHE_DIR="$RUN_ROOT/triton_cache/$METHOD_LABEL"
export HF_HOME="/local2/denghaoran/huggingface"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
"$PYTHON" analysis/verify_prompt_contract.py \
  --prompt_path "$PROMPT_PATH" \
  --num_videos "$VIDEO_COUNT" \
  --seed 0 \
  --num_samples 1 \
  --manifest_out "$OUTPUT_DIR/prompt_manifest.json" \
  >"$RUN_ROOT/logs/$METHOD_LABEL.prompt_contract.log" 2>&1

{
  echo "run_id=$RUN_ID"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "physical_gpu=$GPU"
  echo "backend=ear_all"
  echo "top_p=$TOP_P"
  echo "target_density=$TARGET_DENSITY"
  echo "video_count=$VIDEO_COUNT"
  echo "checkpoint=$CHECKPOINT"
  echo "prompt_path=$PROMPT_PATH"
  nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader
} >"$RUN_ROOT/launch_manifest.txt"

run_with_gpu_audit \
  "$PYTHON" -u generate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --prompt_path "$PROMPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_videos "$VIDEO_COUNT" \
    --num_samples 1 \
    --batch_size 1 \
    --sample_batch_size 1 \
    --seed 0 \
    --dtype bf16 \
    --skip_existing \
    --runtime_json "$OUTPUT_DIR/runtime.json"

"$PYTHON" analysis/summarize_svg_density.py \
  --input "$OUTPUT_DIR/density.jsonl" \
  --output "$RUN_ROOT/results/${METHOD_LABEL}_density.json" \
  --backend ear_all \
  --calls_per_video "$CALLS_PER_VIDEO" \
  >>"$RUN_ROOT/logs/$METHOD_LABEL.density.log" 2>&1

video_count="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
density_records="$(wc -l <"$OUTPUT_DIR/density.jsonl")"
expected_records=$((video_count * CALLS_PER_VIDEO))
if [[ "$video_count" -ne "$VIDEO_COUNT" || "$density_records" -ne "$expected_records" ]]; then
  echo "Incomplete output: videos=$video_count/$VIDEO_COUNT density_records=$density_records/$expected_records" >&2
  exit 4
fi

echo "SVG-EAR run complete: $OUTPUT_DIR"
