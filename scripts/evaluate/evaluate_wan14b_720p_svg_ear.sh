#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
RUN_ID="${RUN_ID:-wan14b_720p_50step_four_methods_20260816}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"
SVG_PYTHON="${SVG_PYTHON:-$HOME/miniconda3/envs/SVG/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/local2/denghaoran/GridMergeAttention/envs/vbench/bin/python}"
VBENCH_ROOT="${VBENCH_ROOT:-$HOME/workspace/VBench}"
PROMPTS="${PROMPTS:-$REPO_DIR/prompts/MovieGenVideoBench_extended.txt}"
GPU="${GPU:-0}"
COUNT="${COUNT:-8}"
RUN_QUALITY="${RUN_QUALITY:-1}"
RESULTS="$EXPERIMENT_ROOT/results"
EAR_VBENCH="$RESULTS/vbench/ear"
VBENCH_HOME="/local2/denghaoran/GridMergeAttention/vbench_home"
QUALITY_JSON="${QUALITY_JSON:-$RESULTS/paired_ear_quality.json}"

ear_count=$(find "$EXPERIMENT_ROOT/ear" -maxdepth 1 -name '*.mp4' | wc -l)
dense_count=$(find "$EXPERIMENT_ROOT/dense" -maxdepth 1 -name '*.mp4' | wc -l)
[[ "$ear_count" -eq "$COUNT" ]] || { echo "EAR has $ear_count/$COUNT videos" >&2; exit 2; }
[[ "$dense_count" -ge "$COUNT" ]] || { echo "Dense has $dense_count/$COUNT videos" >&2; exit 2; }

mkdir -p "$RESULTS" "$EAR_VBENCH" "$VBENCH_HOME"
export TOKENIZERS_PARALLELISM=false
export VBENCH_CACHE_DIR="$VBENCH_HOME/.cache/vbench"

if [[ "$RUN_QUALITY" == "1" ]]; then
  "$SVG_PYTHON" -u "$REPO_DIR/analysis/evaluate_video_quality.py" \
    --reference_dir "$EXPERIMENT_ROOT/dense" \
    --method_dir "$EXPERIMENT_ROOT/ear" --method_name ear \
    --output "$QUALITY_JSON" \
    --max_videos "$COUNT" \
    2>&1 | tee "$RESULTS/paired_ear_quality.log"
else
  [[ -f "$QUALITY_JSON" ]] || { echo "Missing $QUALITY_JSON" >&2; exit 2; }
fi

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
if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo "GPU $GPU has an active compute process" >&2
  exit 2
fi

"$SVG_PYTHON" "$REPO_DIR/analysis/prepare_vbench_prompt_map.py" \
  --prompt_path "$PROMPTS" \
  --video_dir "$EXPERIMENT_ROOT/ear" \
  --output "$EAR_VBENCH/prompts.json"

dimensions=(
  subject_consistency background_consistency motion_smoothness
  dynamic_degree aesthetic_quality imaging_quality
)
env CUDA_VISIBLE_DEVICES="$GPU" \
  HOME="$VBENCH_HOME" XDG_CACHE_HOME="$VBENCH_HOME/.cache" \
  TORCH_HOME="$VBENCH_HOME/.cache/torch" HF_HOME="$VBENCH_HOME/.cache/huggingface" \
  "$VBENCH_PYTHON" -u "$VBENCH_ROOT/evaluate.py" \
    --videos_path "$EXPERIMENT_ROOT/ear" \
    --dimension "${dimensions[@]}" \
    --mode custom_input \
    --load_ckpt_from_local True \
    --prompt_file "$EAR_VBENCH/prompts.json" \
    --output_path "$EAR_VBENCH" \
    2>&1 | tee "$EAR_VBENCH/evaluate.log"

"$SVG_PYTHON" "$REPO_DIR/analysis/summarize_wan14b_720p_experiment.py" \
  --experiment_root "$EXPERIMENT_ROOT" \
  --quality_json "$QUALITY_JSON" \
  --vbench_root "$RESULTS/vbench" \
  --methods dense,ear \
  --output_json "$RESULTS/ear_summary.json" \
  --output_tsv "$RESULTS/ear_summary.tsv"
