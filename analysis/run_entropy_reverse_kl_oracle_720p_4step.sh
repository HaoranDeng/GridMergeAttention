#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GPU="${GPU:?Set GPU to the physical H200 index}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new experiment directory}"
NUM_VIDEOS="${NUM_VIDEOS:-20}"
PROMPTS="${PROMPTS:-$SOURCE_ROOT/prompts/MovieGenVideoBench_extended.txt}"
CONFIG="${CONFIG:-$SOURCE_ROOT/configs/token_merging_wan_720p_entropy_reverse_kl_mass_c32_t1_rf10_oracle_fewstep.yaml}"
MODEL_ROOT="${WAN_MODEL_ROOT:-/xuanwu-tank/east/zrl/MonarchRT/wan_models}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt}"
SVG_ROOT="${SVG_ROOT:-/xuanwu-tank/east/zrl/exp-gridmergeattn/third_party/Sparse-VideoGen}"
PYTHON_BIN="${PYTHON_BIN:-/home/denghaoran/miniconda3/envs/monarch_rt/bin/python}"

VIDEOS="$OUTPUT_ROOT/videos"
RUNTIME="$OUTPUT_ROOT/runtime.json"
LOG="$OUTPUT_ROOT/generate.log"
PURITY="$OUTPUT_ROOT/purity.log"
MANIFEST="$OUTPUT_ROOT/manifest.json"

mkdir -p "$VIDEOS"
for required in "$SOURCE_ROOT/generate.py" "$CHECKPOINT" "$PROMPTS" "$CONFIG" "$PYTHON_BIN"; do
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
  local root="$1" pid mem name
  while :; do
    sleep 15
    while IFS=, read -r pid mem name; do
      pid="$(echo "$pid" | tr -d ' ')"
      [ -n "$pid" ] || continue
      is_ours "$pid" "$root" && continue
      echo "$(date -Iseconds) pid=$pid mem=$(echo "$mem" | tr -d ' ') name=$name" \
        >> "$PURITY"
    done < <(gpu_processes)
  done
}

require_clean_gpu
export HOME=/home/denghaoran
export WAN_MODEL_ROOT="$MODEL_ROOT"
export SPARSE_VIDEO_GEN_ROOT="$SVG_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/home/denghaoran/miniconda3/envs/monarch_rt/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/local2/denghaoran/flashinfer_scai6_cuda129}"
export PYTHONPATH="$SOURCE_ROOT:$SVG_ROOT:$SVG_ROOT/svg/kernels/build:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CONFIG CHECKPOINT PROMPTS GPU NUM_VIDEOS
cd "$SOURCE_ROOT"

"$PYTHON_BIN" - "$MANIFEST" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import triton

def command(*args):
    return subprocess.check_output(args, text=True).strip()

config = Path(os.environ["CONFIG"])
payload = {
    "method": "entropy_reverse_kl_mass_c32_t1_rf10_oracle",
    "method_classification": "dense-router oracle diagnostic",
    "deployable_pure_gca": False,
    "dense_router_reason": "Seed selection evaluates exact query-key block masses over all keys.",
    "gpu_index": int(os.environ["GPU"]),
    "git_commit": command("git", "rev-parse", "HEAD"),
    "git_status": command("git", "status", "--short"),
    "config": str(config),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "triton": triton.__version__,
    "checkpoint": os.environ["CHECKPOINT"],
    "checkpoint_weights": "generator_ema",
    "prompts": os.environ["PROMPTS"],
    "seed": 0,
    "resolution": [1280, 720],
    "frames": 81,
    "denoise_steps": 4,
    "attention_calls_per_video": 160,
    "num_videos": int(os.environ["NUM_VIDEOS"]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

: > "$PURITY"
echo "=== start=$(date -Iseconds) physical_gpu=$GPU videos=$NUM_VIDEOS ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u generate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt_path "$PROMPTS" \
  --output_dir "$VIDEOS" \
  --runtime_json "$RUNTIME" \
  --num_videos "$NUM_VIDEOS" \
  --num_samples 1 \
  --seed 0 \
  --use_ema \
  --skip_existing \
  >> "$LOG" 2>&1 &
process=$!
watch_purity "$process" &
watcher=$!
set +e
wait "$process"
rc=$?
set -e
kill "$watcher" 2>/dev/null || true
wait "$watcher" 2>/dev/null || true

count="$(find "$VIDEOS" -maxdepth 1 -name '*.mp4' | wc -l)"
records=0
[ -s "$RUNTIME" ] && records="$("$PYTHON_BIN" -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("records", [])))' "$RUNTIME")"
echo "=== rc=$rc end=$(date -Iseconds) videos=$count records=$records ===" | tee -a "$LOG"
if [ "$rc" -ne 0 ] || [ "$count" -ne "$NUM_VIDEOS" ] || [ "$records" -ne "$NUM_VIDEOS" ]; then
  echo "FATAL: incomplete oracle run" >&2
  exit 1
fi
if [ -s "$PURITY" ]; then
  echo "FATAL: foreign GPU process observed during oracle run" >&2
  cat "$PURITY" >&2
  exit 1
fi

"$PYTHON_BIN" - "$RUNTIME" <<'PY'
import json
import sys
print(json.dumps(json.load(open(sys.argv[1]))["summary"], indent=2))
PY
