#!/usr/bin/env bash
# Wan2.1-14B 720p video generation with the native Q90K80 even-capacity GCA
# kernels (C4/C6/C8/C10) replacing every self-attention call.
#
#   CAP=8 GPU=0 OUT=/path/to/out scripts/generate/run_q90k80_even_video.sh [extra generate.py args]
#
# CAP=0 runs the same pipeline with dense SDPA (Flash backend) as the paired
# reference. Per-call attention timings go to $OUT/runtime_c$CAP.json and the
# videos to $OUT/c$CAP/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAP="${CAP:-8}"                 # 4, 6, 8, 10, or 0 for dense
GPU="${GPU:-0}"
HC="${HC:-8}"                   # heads per native call
CONFIG="${CONFIG:-configs/dense_wan_14b_720p_fewstep.yaml}"   # 4-step StepDistill
CHECKPOINT="${CHECKPOINT:-$HOME/wan_models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt}"
NUM_VIDEOS="${NUM_VIDEOS:-3}"
SEED="${SEED:-0}"
OUT="${OUT:-$REPO/videos/q90k80_even}"
PYTHON="${PYTHON:-python}"
ROUTE_FLAGS="${ROUTE_FLAGS:-}"  # e.g. -DROUTE_LAZY=0 for the previous route softmax

export CUDA_VISIBLE_DEVICES="$GPU"
export WAN_SDPA_BACKEND="${WAN_SDPA_BACKEND:-flash}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-$(dirname "$(dirname "$CHECKPOINT")")}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO/.torch_ext_$(hostname)}"

mkdir -p "$OUT"
cd "$REPO"
exec "$PYTHON" analysis/q90k80_even_generate.py \
  --gma-cap "$CAP" --gma-hc "$HC" --gma-route-flags="$ROUTE_FLAGS" \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --num_videos "$NUM_VIDEOS" --seed "$SEED" \
  --output_dir "$OUT/c$CAP" --runtime_json "$OUT/runtime_c$CAP.json" \
  "$@"
