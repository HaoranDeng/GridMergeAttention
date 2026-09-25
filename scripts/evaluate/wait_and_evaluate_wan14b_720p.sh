#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_four_methods_20260816}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
GPU="${GPU:-5}"
POLL_SECONDS="${POLL_SECONDS:-300}"

while true; do
  ready=true
  for backend in dense svg svg2 gma; do
    directory="$EXPERIMENT_ROOT/$backend"
    if [[ ! -d "$directory" ]]; then
      ready=false
      break
    fi
    count=$(find "$directory" -maxdepth 1 -name '*.mp4' | wc -l)
    if [[ "$count" -ne 20 ]]; then
      ready=false
      break
    fi
  done
  $ready && break
  sleep "$POLL_SECONDS"
done

while true; do
  IFS=',' read -r utilization memory_used <<< "$(
    nvidia-smi -i "$GPU" \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits
  )"
  utilization="${utilization// /}"
  memory_used="${memory_used// /}"
  if [[ "$utilization" == "0" && "$memory_used" -le 16 ]]; then
    break
  fi
  sleep 30
done

cd "$REPO_DIR"
exec env RUN_ID="$RUN_ID" GPU="$GPU" \
  bash scripts/evaluate/evaluate_wan14b_720p_methods.sh
