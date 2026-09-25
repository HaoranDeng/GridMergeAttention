#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ID="${RUN_ID:-wan14b_720p_4step_dense_sdpa_backends_$(date +%Y%m%d_%H%M%S)}"

GPU="${GPU:?Set GPU to the physical H200 index}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$SOURCE_ROOT/prompts/MovieGenVideoBench_extended.txt}"
MODEL_ROOT="${WAN_MODEL_ROOT:-/xuanwu-tank/east/zrl/MonarchRT/wan_models}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt}"
CONFIG="$SOURCE_ROOT/configs/dense_wan_14b_720p_fewstep.yaml"
ENV_ROOT="${ENV_ROOT:-/xuanwu-tank/east/zrl/exp-gridmergeattn/environments/etm}"

mkdir -p "$OUTPUT_ROOT"

for required in "$SOURCE_ROOT/generate.py" "$CONFIG" "$CHECKPOINT" "$PROMPTS" "$ENV_ROOT/activate.sh"; do
  [ -e "$required" ] || { echo "FATAL: missing $required" >&2; exit 1; }
done

gpu_processes() {
  nvidia-smi -i "$GPU" \
    --query-compute-apps=pid,used_memory,process_name \
    --format=csv,noheader,nounits 2>/dev/null
}

require_clean_gpu() {
  local used util processes
  IFS=, read -r used util <<< "$(nvidia-smi -i "$GPU" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  processes="$(gpu_processes)"
  if [ -n "$processes" ] || [ "$used" -gt 100 ] || [ "$util" -ne 0 ]; then
    echo "FATAL: GPU $GPU is not clean: memory=${used}MiB util=${util}%" >&2
    [ -n "$processes" ] && echo "$processes" >&2
    exit 1
  fi
}

is_ours() {
  local pid="$1" root="$2" depth=0
  while [ -n "$pid" ] && [ "$pid" != "1" ] && [ "$depth" -lt 8 ]; do
    [ "$pid" = "$root" ] && return 0
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    depth=$((depth + 1))
  done
  return 1
}

watch_purity() {
  local root="$1" output="$2" pid mem name
  while :; do
    sleep 15
    while IFS=, read -r pid mem name; do
      pid="$(echo "$pid" | tr -d ' ')"
      [ -n "$pid" ] || continue
      is_ours "$pid" "$root" && continue
      echo "$(date -Iseconds) pid=$pid mem=$(echo "$mem" | tr -d ' ') name=$name" \
        >> "$output"
    done < <(gpu_processes)
  done
}

write_manifest() {
  local backend="$1" output="$2"
  BACKEND="$backend" OUTPUT="$output" GPU_INDEX="$GPU" python - <<'PY'
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

def command(*args):
    return subprocess.check_output(args, text=True).strip()

payload = {
    "backend": os.environ["BACKEND"],
    "gpu_index": int(os.environ["GPU_INDEX"]),
    "git_commit": command("git", "rev-parse", "HEAD"),
    "git_status_tracked": command(
        "git", "status", "--short", "--untracked-files=no"
    ),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "triton": __import__("triton").__version__,
    "checkpoint": os.environ["CHECKPOINT"],
    "config": os.environ["CONFIG"],
    "prompts": os.environ["PROMPTS"],
    "seed": 0,
    "resolution": [1280, 720],
    "frames": 81,
    "denoise_steps": 4,
    "attention_calls_per_video": 160,
}
Path(os.environ["OUTPUT"]).write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_backend() {
  local backend="$1"
  local stage="$OUTPUT_ROOT/$backend"
  local videos="$stage/videos" runtime="$stage/runtime.json"
  local log="$stage/generate.log" purity="$stage/purity.log"
  local count records
  mkdir -p "$videos"

  count="$(find "$videos" -maxdepth 1 -name '*.mp4' | wc -l)"
  records=0
  if [ -s "$runtime" ]; then
    records="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("records", [])))' "$runtime")"
  fi
  if [ "$count" -ge 20 ] && [ "$records" -ge 20 ]; then
    echo "[$backend] already complete: videos=$count records=$records"
    return 0
  fi

  require_clean_gpu
  export WAN_SDPA_BACKEND="$backend"
  write_manifest "$backend" "$stage/manifest.json"
  : > "$purity"
  echo "=== backend=$backend start=$(date -Iseconds) physical_gpu=$GPU ===" | tee -a "$log"

  CUDA_VISIBLE_DEVICES="$GPU" python -u generate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$videos" \
    --runtime_json "$runtime" \
    --num_videos -1 \
    --num_samples 1 \
    --seed 0 \
    --skip_existing \
    >> "$log" 2>&1 &
  local process=$!
  watch_purity "$process" "$purity" &
  local watcher=$!
  set +e
  wait "$process"
  local rc=$?
  set -e
  kill "$watcher" 2>/dev/null || true
  wait "$watcher" 2>/dev/null || true

  count="$(find "$videos" -maxdepth 1 -name '*.mp4' | wc -l)"
  records=0
  [ -s "$runtime" ] && records="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("records", [])))' "$runtime")"
  echo "=== backend=$backend rc=$rc end=$(date -Iseconds) videos=$count records=$records ===" \
    | tee -a "$log"
  if [ "$rc" -ne 0 ] || [ "$count" -ne 20 ] || [ "$records" -ne 20 ]; then
    echo "FATAL: incomplete $backend run" >&2
    exit 1
  fi
  if [ -s "$purity" ]; then
    echo "FATAL: foreign GPU process observed during $backend run" >&2
    cat "$purity" >&2
    exit 1
  fi

  python - "$runtime" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))["summary"]
print(json.dumps(summary, indent=2))
PY
}

require_clean_gpu
export HOME=/home/zrlu
source "$ENV_ROOT/activate.sh"
export HOME=/home/denghaoran
export WAN_MODEL_ROOT="$MODEL_ROOT"
export CHECKPOINT CONFIG PROMPTS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SOURCE_ROOT"

read -r -a backends <<< "${BACKENDS:-cudnn flash}"
for index in "${!backends[@]}"; do
  run_backend "${backends[$index]}"
  if [ "$index" -lt "$((${#backends[@]} - 1))" ]; then
    sleep 20
  fi
done
