#!/usr/bin/env bash
#SBATCH -J EARvsC4
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
PYTHON="${PYTHON:-$WORK_ROOT/miniconda3/envs/gma_amd/bin/python}"
SVG_ROOT="${SVG_ROOT:-$WORK_ROOT/Sparse-VideoGen}"
MODEL="${MODEL:-$WORK_ROOT/models/Wan2.1-T2V-14B-Diffusers}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
RUN_ID="${RUN_ID:-wan14b_720p_ear_vs_hard_c4_error_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="$WORK_ROOT/GridMergeAttention/experiments/$RUN_ID"
STEPS="${STEPS:-50}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/triton_cache"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
export TORCH_HOME="$OUTPUT_ROOT/cache/torch"
export MPLCONFIGDIR="$OUTPUT_ROOT/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

for backend in dense ear gma; do
  output_dir="$OUTPUT_ROOT/$backend"
  mkdir -p "$output_dir"
  export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache/$backend"
  mkdir -p "$TRITON_CACHE_DIR"

  error_args=()
  if [[ "$backend" != dense ]]; then
    error_args=(
      --attention_error_calls -1
      --attention_error_inline
      --attention_error_sparse_region_only
    )
  fi

  "$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
    --backend "$backend" \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$output_dir" \
    --start 0 \
    --count 1 \
    --seed 0 \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_steps "$STEPS" \
    "${error_args[@]}" \
    >"$OUTPUT_ROOT/logs/$backend.log" 2>&1
done

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = {}
for backend in ("dense", "ear", "gma"):
    payload = json.loads((root / backend / "runtime.json").read_text())
    results[backend] = {
        "summary": payload["summary"],
        "attention_error_vs_dense": payload.get("attention_error_vs_dense"),
    }
(root / "comparison.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
PY
