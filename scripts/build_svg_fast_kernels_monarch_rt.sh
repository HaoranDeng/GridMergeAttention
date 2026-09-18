#!/usr/bin/env bash
set -euo pipefail

SVG_ROOT="${SVG_ROOT:-$HOME/workspace/Sparse-VideoGen}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
BUILD_DIR="${BUILD_DIR:-/local2/denghaoran/GridMergeAttention/build/svg_kernels_monarch_rt}"
KERNEL_DIR="$SVG_ROOT/svg/kernels"
EXT_SUFFIX="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
TARGET="$KERNEL_DIR/build/_kernels$EXT_SUFFIX"

if [[ -f "$TARGET" ]]; then
  echo "Fast RoPE extension already exists: $TARGET"
  exit 0
fi

mkdir -p "$BUILD_DIR" "$KERNEL_DIR/build"
TORCH_PREFIX="$($PYTHON -c 'import torch; print(torch.utils.cmake_prefix_path)')"

CUDACXX="$CUDA_HOME/bin/nvcc" cmake \
  -S "$KERNEL_DIR" \
  -B "$BUILD_DIR" \
  -DPython_EXECUTABLE="$PYTHON" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build "$BUILD_DIR" --parallel
cp -f "$BUILD_DIR/_kernels$EXT_SUFFIX" "$TARGET"
echo "Built fast RoPE extension: $TARGET"
