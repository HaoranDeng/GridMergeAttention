#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-$REPO_DIR/wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill}"
CHECKPOINT="${CHECKPOINT:-$WAN_MODEL_DIR/distill_model.pt}"
MIN_CHECKPOINT_BYTES="${MIN_CHECKPOINT_BYTES:-28000000000}"
PROMPT_PATH="${PROMPT_PATH:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
CONFIG="${CONFIG:-$REPO_DIR/configs/token_merging_wan_14b_monarchrt_h2_720p_50step_all.yaml}"
GPU="${GPU:-3}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_monarchrt_h2_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
OUTPUT_DIR="$RUN_ROOT/monarch_rt_h2"
LOG_DIR="$RUN_ROOT/logs"
RUNTIME_JSON="$OUTPUT_DIR/runtime.json"

IFS=',' read -r utilization memory_used < <(
  nvidia-smi -i "$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ' '
)
compute_processes="$(
  nvidia-smi -i "$GPU" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
)"
if [[ "$utilization" != "0" || "$memory_used" -gt 16 || -n "$compute_processes" ]]; then
  echo "GPU $GPU is not exclusively clean: utilization=${utilization}%, memory=${memory_used} MiB, processes=${compute_processes:-none}" >&2
  exit 2
fi

for path in "$PYTHON" "$WAN_MODEL_DIR" "$CHECKPOINT" "$PROMPT_PATH" "$CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 2
  fi
done

checkpoint_bytes="$(stat -c '%s' "$CHECKPOINT")"
if [[ "$checkpoint_bytes" -lt "$MIN_CHECKPOINT_BYTES" ]]; then
  echo "Checkpoint is too small for Wan2.1-14B: $CHECKPOINT ($checkpoint_bytes bytes)" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export WAN_MODEL_ROOT="$(dirname "$WAN_MODEL_DIR")"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/local2/denghaoran/GridMergeAttention/triton_cache/monarchrt_h2_720p}"
mkdir -p "$TRITON_CACHE_DIR"

cd "$REPO_DIR"
{
  echo "run_id=$RUN_ID"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "gpu=$GPU"
  echo "config=$CONFIG"
  echo "output_dir=$OUTPUT_DIR"
  nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
} >"$RUN_ROOT/launch_manifest.txt"

exec "$PYTHON" generate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt_path "$PROMPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --runtime_json "$RUNTIME_JSON" \
  --num_videos 20 \
  --num_samples 1 \
  --batch_size 1 \
  --sample_batch_size 1 \
  --skip_existing \
  --seed 0 \
  --fps 16 \
  --dtype bf16 \
  --use_ema \
  >"$LOG_DIR/monarch_rt_h2.log" 2>&1
