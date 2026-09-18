#!/usr/bin/env bash
#SBATCH -J GMA-SAP-C4
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
MODEL="${MODEL:-$WORK_ROOT/models/Wan2.1-T2V-14B-Diffusers}"
SVG_ROOT="${SVG_ROOT:-$WORK_ROOT/Sparse-VideoGen}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
RUN_ID="${RUN_ID:-wan14b_720p_sap_hard_c4_screen}"
OUTPUT_ROOT="$WORK_ROOT/GridMergeAttention/analysis/$RUN_ID"

mkdir -p "$OUTPUT_ROOT" "$REPO_DIR/out"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

"$PYTHON" -u "$REPO_DIR/analysis/screen_wan14b_720p_sap_hard_c4.py" \
  --model "$MODEL" \
  --sparse_video_gen_root "$SVG_ROOT" \
  --prompt_path "$PROMPTS" \
  --output "$OUTPUT_ROOT/metrics.json" \
  --steps "${STEPS:-1}" \
  --layers "${LAYERS:-0}" \
  --heads "${HEADS:-0}" \
  --q_clusters "${Q_CLUSTERS:-300}" \
  --k_clusters "${K_CLUSTERS:-1000}" \
  --kmeans_iters "${KMEANS_ITERS:-10}" \
  --prompt_index "${PROMPT_INDEX:-0}" \
  --seed "${SEED:-0}"
