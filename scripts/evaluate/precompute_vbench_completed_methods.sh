#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_four_methods_20260816}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/local2/denghaoran/GridMergeAttention/envs/vbench/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/workspace/VBench}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
GPU="${GPU:-4}"
METHODS=(${METHODS:-dense svg gma})
RESULTS="$EXPERIMENT_ROOT/results"
VBENCH_RESULTS="$RESULTS/vbench"
VBENCH_HOME="/local2/denghaoran/GridMergeAttention/vbench_home"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/local2/denghaoran/GridMergeAttention/gpu_locks}"
MARKER="$VBENCH_RESULTS/.precompute_running"

dimensions=(
  subject_consistency background_consistency motion_smoothness
  dynamic_degree aesthetic_quality imaging_quality
)

for backend in "${METHODS[@]}"; do
  count=$(find "$EXPERIMENT_ROOT/$backend" -maxdepth 1 -name '*.mp4' | wc -l)
  [[ "$count" -eq 20 ]] || { echo "$backend has $count/20 videos" >&2; exit 2; }
done

IFS=',' read -r utilization memory_used < <(
  nvidia-smi -i "$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits
)
utilization="${utilization// /}"
memory_used="${memory_used// /}"
compute_processes="$(
  nvidia-smi -i "$GPU" --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
)"
[[ "$utilization" == "0" && "$memory_used" -le 16 && -z "$compute_processes" ]] || {
  echo "GPU $GPU is not clean: utilization=${utilization}%, memory=${memory_used} MiB, processes=${compute_processes:-none}" >&2
  exit 3
}

mkdir -p "$VBENCH_RESULTS" "$VBENCH_HOME" "$GPU_LOCK_DIR"
exec 9>"$GPU_LOCK_DIR/h200_gpu${GPU}.lock"
flock -n 9 || { echo "GPU $GPU is reserved by another GMA runner" >&2; exit 3; }
printf '%s\n' "$$" > "$MARKER"
trap 'rm -f "$MARKER"' EXIT
export TOKENIZERS_PARALLELISM=false
export VBENCH_CACHE_DIR="$VBENCH_HOME/.cache/vbench"

for backend in "${METHODS[@]}"; do
  output_dir="$VBENCH_RESULTS/$backend"
  mkdir -p "$output_dir"
  "$HOME/miniconda3/envs/SVG/bin/python" \
    "$REPO_DIR/analysis/prepare_vbench_prompt_map.py" \
    --prompt_path "$PROMPTS" \
    --video_dir "$EXPERIMENT_ROOT/$backend" \
    --output "$output_dir/prompts.json"
  if find "$output_dir" -name '*_eval_results.json' -print -quit | grep -q .; then
    continue
  fi
  audit_log="$output_dir/gpu_audit.log"
  overlap_log="$output_dir/gpu_overlap.log"
  : >"$audit_log"
  : >"$overlap_log"
  env CUDA_VISIBLE_DEVICES="$GPU" \
    HOME="$VBENCH_HOME" XDG_CACHE_HOME="$VBENCH_HOME/.cache" \
    TORCH_HOME="$VBENCH_HOME/.cache/torch" HF_HOME="$VBENCH_HOME/.cache/huggingface" \
    "$VBENCH_PYTHON" -u "$VBENCH_ROOT/evaluate.py" \
      --videos_path "$EXPERIMENT_ROOT/$backend" \
      --dimension "${dimensions[@]}" \
      --mode custom_input \
      --load_ckpt_from_local True \
      --prompt_file "$output_dir/prompts.json" \
      --output_path "$output_dir" \
      >"$output_dir/evaluate.log" 2>&1 &
  vbench_pid=$!
  while kill -0 "$vbench_pid" 2>/dev/null; do
    gpu_pids="$(
      nvidia-smi -i "$GPU" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d' || true
    )"
    printf '%s method_pid=%s gpu_pids=%s\n' \
      "$(date --iso-8601=seconds)" "$vbench_pid" "${gpu_pids//$'\n'/,}" \
      >>"$audit_log"
    for pid in $gpu_pids; do
      if [[ "$pid" != "$vbench_pid" ]]; then
        printf '%s foreign_pid=%s method_pid=%s\n' \
          "$(date --iso-8601=seconds)" "$pid" "$vbench_pid" \
          >>"$overlap_log"
      fi
    done
    sleep 30
  done
  wait "$vbench_pid"
  if [[ -s "$overlap_log" ]]; then
    echo "Foreign GPU process overlapped VBench $backend" >&2
    exit 4
  fi
done

touch "$VBENCH_RESULTS/.precompute_done"
