#!/usr/bin/env bash
# Usage: bash scripts/run_svg_ear25.sh MODEL_DIR SPARSE_VIDEOGEN_DIR [GPU_ID=0] [OUTPUT_DIR]
# Requires an installed Wan/SVG environment. Recorded versions: torch 2.8.0+cu128,
# diffusers 0.34.0, transformers 4.52.4, triton 3.7.1, flashinfer-python 0.6.6.
# Recorded Sparse-VideoGen commit: f89aedaf169ac2ae5b186bda674e53c3dc08c476.
set -euo pipefail
if [[ $# -lt 2 || ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: bash scripts/run_svg_ear25.sh MODEL_DIR SPARSE_VIDEOGEN_DIR [GPU_ID=0] [OUTPUT_DIR]'
  exit 0
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$(realpath "$1")"
SVG_ROOT="$(realpath "$2")"
OUT="${4:-$ROOT/experiments/svg_ear25}"
PYTHON="${PYTHON:-python}"
[[ -f "$MODEL/model_index.json" ]] || { echo 'MODEL_DIR must be a local Wan2.1-T2V-14B-Diffusers checkpoint.' >&2; exit 1; }
[[ -f "$SVG_ROOT/svg/models/wan/attention.py" ]] || { echo 'SPARSE_VIDEOGEN_DIR must contain svg/models/wan/attention.py.' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Use a new OUTPUT_DIR: $OUT already exists." >&2; exit 1; }
mkdir -p "$OUT/cache/"{tmp,triton,torch_extensions,flashinfer}
OUT="$(realpath "$OUT")"
export CUDA_VISIBLE_DEVICES="${3:-0}" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT:$SVG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$OUT/cache/tmp" TMP="$OUT/cache/tmp" TEMP="$OUT/cache/tmp"
export TRITON_CACHE_DIR="$OUT/cache/triton" TORCH_EXTENSIONS_DIR="$OUT/cache/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$OUT/cache/flashinfer" SVG_DENSITY_LOG="$OUT/density.jsonl"
export MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPYCACHEPREFIX
"$PYTHON" "$ROOT/analysis/verify_prompt_contract.py" \
  --prompt_path "$ROOT/prompts/MovieGenVideoBench_extended.txt" \
  --num_videos 20 --seed 0 --num_samples 1 --manifest_out "$OUT/prompt_manifest.json" > "$OUT/prompt_check.log"
# One process, prompts 0..19, seed 0. Do not add --original_script_seeding.
# 25% is the target density label; top_p=0.8749, not 0.25.
exec "$PYTHON" -u "$ROOT/analysis/benchmark_wan14b_720p_methods.py" \
  --backend ear_all --model "$MODEL" --sparse_video_gen_root "$SVG_ROOT" \
  --prompt_path "$ROOT/prompts/MovieGenVideoBench_extended.txt" --output_dir "$OUT" \
  --canonical_names --start 0 --count 20 --seed 0 \
  --height 720 --width 1280 --num_frames 81 --num_steps 50 \
  --svg_top_p 0.8749 --svg_min_kc_ratio 0.0 \
  --wan_forward_backend svg --qk_norm_backend triton --rope_backend svg \
  --gma_preprocess_backend fused --gma_output_layout token --gma_head_chunk 10
