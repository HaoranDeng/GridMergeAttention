#!/usr/bin/env bash
set -euo pipefail

gpu_index=1
worktree=/local2/denghaoran/GridMergeAttention_worktrees/temporal_entropy_barycenter_20260904a
output_dir=/local2/denghaoran/GridMergeAttention/experiments/temporal_entropy_barycenter_screen_20260904a
python_bin=/home/denghaoran/miniconda3/envs/monarch_rt/bin/python

gpu_memory=$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
gpu_utilization=$(nvidia-smi -i "${gpu_index}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
gpu_processes=$(nvidia-smi -i "${gpu_index}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -c '[0-9]' || true)
if (( gpu_memory > 10 || gpu_utilization > 0 || gpu_processes > 0 )); then
  echo "GPU ${gpu_index} is not clean: memory=${gpu_memory}MiB util=${gpu_utilization}% processes=${gpu_processes}" >&2
  exit 3
fi

mkdir -p "${output_dir}"
cd "${worktree}"
export PATH=/home/denghaoran/miniconda3/envs/monarch_rt/bin:/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/denghaoran/miniconda3/envs/monarch_rt/lib
export FLASHINFER_WORKSPACE_BASE=/local2/denghaoran/flashinfer_scai6_cuda129
export PYTHONPATH="${worktree}:${worktree}/analysis"
export CUDA_VISIBLE_DEVICES="${gpu_index}"

"${python_bin}" analysis/screen_temporal_entropy_barycenter_seeds.py \
  --device cuda:0 \
  --qkv \
    /local2/denghaoran/GridMergeAttention/experiments/learned_seed_capture_20260822/call020_train/qkv.pt.npz \
    /local2/denghaoran/GridMergeAttention/experiments/learned_seed_capture_20260822/call020_holdout/qkv.pt.npz \
    /local2/denghaoran/GridMergeAttention/experiments/learned_seed_capture_20260822/call100_train/qkv.pt.npz \
    /local2/denghaoran/GridMergeAttention/experiments/learned_seed_capture_20260822/call100_holdout/qkv.pt.npz \
  --heads 0,10,20,30 \
  --frames 0,10,20 \
  --output "${output_dir}/full48.json" \
  2>&1 | tee "${output_dir}/run.log"

nvidia-smi -i "${gpu_index}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
