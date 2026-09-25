#!/usr/bin/env bash
#SBATCH -J GMA720AMD8
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -p mi3008x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 192
#SBATCH -t 12:00:00
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/GridMergeAttention}"
WORK_ROOT="${WORK_ROOT:-/work1/jasoncong/denghaoran}"
PYTHON="${PYTHON:-$WORK_ROOT/miniconda3/envs/gma_amd/bin/python}"
SVG_ROOT="${SVG_ROOT:-$WORK_ROOT/Sparse-VideoGen}"
MODEL="${MODEL:-$WORK_ROOT/models/Wan2.1-T2V-14B-Diffusers}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_amdhpc_8_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/training-free-videogen/experiments/$RUN_ID}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
VIDEO_COUNT="${VIDEO_COUNT:-8}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/cache" "$OUTPUT_ROOT/results"
export PYTHONPATH="$REPO_DIR:$SVG_ROOT:${PYTHONPATH:-}"
export HF_HOME="$WORK_ROOT/huggingface"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
export TORCH_HOME="$OUTPUT_ROOT/cache/torch"
export MPLCONFIGDIR="$OUTPUT_ROOT/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

backend_args() {
  local backend="$1"
  printf '%s\n' \
    "$PYTHON" -u "$REPO_DIR/analysis/benchmark_wan14b_720p_methods.py" \
    --backend "$backend" \
    --model "$MODEL" \
    --sparse_video_gen_root "$SVG_ROOT" \
    --prompt_path "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$backend" \
    --count "$VIDEO_COUNT" \
    --seed 0 \
    --attention_error_calls -1
}

run_backend() {
  local backend="$1"
  local gpu="$2"
  local -a command
  mapfile -t command < <(backend_args "$backend")
  mkdir -p "$OUTPUT_ROOT/triton_cache/$backend"
  env \
    ROCR_VISIBLE_DEVICES="$gpu" \
    HIP_VISIBLE_DEVICES="$gpu" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache/$backend" \
    "${command[@]}" >"$OUTPUT_ROOT/logs/$backend.log" 2>&1
}

run_backend_mi3001() {
  local backend="$1"
  local -a command
  mapfile -t command < <(backend_args "$backend")
  mkdir -p "$OUTPUT_ROOT/triton_cache/$backend"
  srun --exclusive -N 1 -n 1 -c 16 env \
    ROCR_VISIBLE_DEVICES=0 \
    HIP_VISIBLE_DEVICES=0 \
    CUDA_VISIBLE_DEVICES=0 \
    TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache/$backend" \
    "${command[@]}" >"$OUTPUT_ROOT/logs/$backend.log" 2>&1
}

wait_for_backends() {
  local status=0
  local pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  [[ "$status" -eq 0 ]] || { echo "At least one backend failed" >&2; exit 1; }
}

backends=(dense svg svg2 gma)
if [[ "${SLURM_JOB_PARTITION:-}" == "mi3001x" ]]; then
  concurrency="${SLURM_JOB_NUM_NODES:-1}"
  pids=()
  for backend in "${backends[@]}"; do
    run_backend_mi3001 "$backend" &
    pids+=("$!")
    if [[ "${#pids[@]}" -eq "$concurrency" ]]; then
      wait_for_backends "${pids[@]}"
      pids=()
    fi
  done
  [[ "${#pids[@]}" -eq 0 ]] || wait_for_backends "${pids[@]}"
else
  pids=()
  for index in "${!backends[@]}"; do
    run_backend "${backends[$index]}" "$index" &
    pids+=("$!")
  done
  wait_for_backends "${pids[@]}"
fi

"$PYTHON" -u "$REPO_DIR/analysis/evaluate_psnr_ssim.py" \
  --reference_dir "$OUTPUT_ROOT/dense" \
  --method_dir "$OUTPUT_ROOT/svg" --method_name svg \
  --method_dir "$OUTPUT_ROOT/svg2" --method_name svg2 \
  --method_dir "$OUTPUT_ROOT/gma" --method_name gma \
  --output "$OUTPUT_ROOT/results/paired_psnr_ssim.json" \
  >"$OUTPUT_ROOT/logs/paired_psnr_ssim.log" 2>&1

"$PYTHON" "$REPO_DIR/analysis/summarize_amdhpc_wan14b_720p.py" \
  --experiment_root "$OUTPUT_ROOT" \
  --quality_json "$OUTPUT_ROOT/results/paired_psnr_ssim.json" \
  --output_json "$OUTPUT_ROOT/results/summary.json" \
  --output_tsv "$OUTPUT_ROOT/results/summary.tsv" \
  | tee "$OUTPUT_ROOT/logs/summary.log"

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
