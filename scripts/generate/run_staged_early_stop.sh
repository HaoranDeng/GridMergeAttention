#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/workspace/GridMergeAttention}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/monarch_rt/bin/python}"
GENERATOR="${GENERATOR:?Set GENERATOR to a generation script}"
GPU="${GPU:?Set GPU to an exclusively clean H200 index}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
BACKEND="${BACKEND:?Set BACKEND to the candidate output label}"
REFERENCE_DIR="${REFERENCE_DIR:?Set REFERENCE_DIR to matched dense videos}"
BASELINE_DIR="${BASELINE_DIR:?Set BASELINE_DIR to all-layer SVG-EAR videos}"
THEORETICAL_RATIO_VS_SVG_EAR="${THEORETICAL_RATIO_VS_SVG_EAR:?Set the FLOP ratio}"
STAGES="${STAGES:-1 3 5 20}"

mkdir -p "$OUTPUT_ROOT/results"
for stage in $STAGES; do
  echo "[early-stop] generating through $stage videos"
  COUNT="$stage" START=0 GPU="$GPU" OUTPUT_ROOT="$OUTPUT_ROOT" \
    BACKEND="$BACKEND" bash "$GENERATOR"

  echo "[early-stop] evaluating $stage paired videos"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    "$REPO_DIR/analysis/evaluate_staged_early_stop.py" \
    --reference_dir "$REFERENCE_DIR" \
    --candidate_dir "$OUTPUT_ROOT/$BACKEND" \
    --baseline_dir "$BASELINE_DIR" \
    --candidate_name "$BACKEND" \
    --expected_videos "$stage" \
    --theoretical_ratio_vs_svg_ear "$THEORETICAL_RATIO_VS_SVG_EAR" \
    --output "$OUTPUT_ROOT/results/${BACKEND}_stage${stage}.json"
  status=$?
  set -e

  if [[ "$status" -eq 3 ]]; then
    echo "[early-stop] rejected at $stage videos; no larger stage will run"
    exit 3
  fi
  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
done
