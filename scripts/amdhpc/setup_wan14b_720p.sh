#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
CONDA_ROOT="$WORK_ROOT/miniconda3"
SOURCE_ENV="$CONDA_ROOT/envs/monarch_rt"
TARGET_ENV="$CONDA_ROOT/envs/gma_amd"
SVG_ROOT="$WORK_ROOT/Sparse-VideoGen"
MODEL_ROOT="$WORK_ROOT/models/Wan2.1-T2V-14B-Diffusers"
HF_HOME="$WORK_ROOT/huggingface"

export HF_HOME
export XDG_CACHE_HOME="$WORK_ROOT/cache"
export PIP_CACHE_DIR="$WORK_ROOT/cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$WORK_ROOT/models"

if [[ ! -x "$TARGET_ENV/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create -y --prefix "$TARGET_ENV" --clone "$SOURCE_ENV"
fi

"$TARGET_ENV/bin/python" -m pip install --no-cache-dir --upgrade \
  diffusers==0.34.0 transformers==4.52.4 termcolor \
  loguru pandas scikit-learn

# The source environment contains a CUDA torchao build that Diffusers probes
# at import time. It is unused here and incompatible with the ROCm PyTorch.
"$TARGET_ENV/bin/python" -m pip uninstall -y torchao || true

if [[ ! -d "$SVG_ROOT/.git" ]]; then
  git clone https://github.com/svg-project/Sparse-VideoGen.git "$SVG_ROOT"
fi
git -C "$SVG_ROOT" fetch origin main
git -C "$SVG_ROOT" checkout f89aedaf169ac2ae5b186bda674e53c3dc08c476

"$TARGET_ENV/bin/python" - "$MODEL_ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    local_dir=sys.argv[1],
    max_workers=8,
)
PY

"$TARGET_ENV/bin/python" - <<'PY'
from diffusers import AutoencoderKLWan, WanPipeline
import torch

print("torch", torch.__version__, "hip", torch.version.hip)
print("WanPipeline import OK")
PY
