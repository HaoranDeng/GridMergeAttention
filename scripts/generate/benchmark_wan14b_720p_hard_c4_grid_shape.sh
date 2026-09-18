#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
MODEL="${MODEL:-$HOME/wan_models/Wan2.1-T2V-14B-Diffusers}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
GPU="${GPU:?Set GPU to an exclusively clean H200 index}"
BACKEND="${BACKEND:?Set BACKEND to gma_grid24x80 or gma_grid48x40}"
RUN_ID="${RUN_ID:-wan14b_720p_${BACKEND}_4step_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
STEPS="${STEPS:-4}"

case "$BACKEND" in
  gma_grid24x80|gma_grid48x40) ;;
  *) echo "Unsupported BACKEND=$BACKEND" >&2; exit 2 ;;
esac

IFS=',' read -r utilization memory_used <<< "$(
  nvidia-smi -i "$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits
)"
utilization="${utilization// /}"
memory_used="${memory_used// /}"
if [[ "$utilization" != "0" || "$memory_used" -gt 16 ]]; then
  echo "GPU $GPU is not clean: utilization=${utilization}%, memory=${memory_used} MiB" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for method in dense gma "$BACKEND"; do
  mkdir -p "$OUTPUT_ROOT/$method" "$OUTPUT_ROOT/triton_cache/$method"
  export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache/$method"
  error_args=()
  if [[ "$method" != dense ]]; then
    error_args=(
      --attention_error_calls -1
      --attention_error_inline
      --attention_error_sparse_region_only
    )
  fi
  "$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
    --backend "$method" \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$method" \
    --start 0 \
    --count 1 \
    --seed 0 \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_steps "$STEPS" \
    "${error_args[@]}" \
    >"$OUTPUT_ROOT/logs/$method.log" 2>&1
done

"$PYTHON" - "$OUTPUT_ROOT" "$BACKEND" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
variant = sys.argv[2]
results = {}
for method in ("dense", "gma", variant):
    payload = json.loads((root / method / "runtime.json").read_text())
    results[method] = {
        "summary": payload["summary"],
        "attention_error_vs_dense": payload.get("attention_error_vs_dense"),
        "attention_layout": payload["protocol"].get("attention_layout"),
    }
(root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
PY
