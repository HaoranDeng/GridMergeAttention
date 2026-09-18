#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="${RUN_ID:-wan14b_720p_4step_h200_comparison_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/local2/denghaoran/GridMergeAttention/experiments/$RUN_ID}"

DENSE_GPU="${DENSE_GPU:?Set DENSE_GPU to an idle physical H200 index}"
METHOD_GPU="${METHOD_GPU:?Set METHOD_GPU to a different idle physical H200 index}"
BACKENDS="${BACKENDS:-cudnn flash}"
METHODS="${METHODS:-hard_c4 hellinger_top1 svg svg2 svg_ear}"

if [ "$DENSE_GPU" = "$METHOD_GPU" ]; then
  echo "FATAL: DENSE_GPU and METHOD_GPU must be different" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "run_id=$RUN_ID"
echo "output_root=$OUTPUT_ROOT"
echo "dense_gpu=$DENSE_GPU backends=$BACKENDS"
echo "method_gpu=$METHOD_GPU methods=$METHODS"

GPU="$DENSE_GPU" \
SOURCE_ROOT="$REPO_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT/dense" \
BACKENDS="$BACKENDS" \
bash "$REPO_ROOT/analysis/run_dense_sdpa_backends_720p_4step.sh" &
dense_pid=$!

GPU="$METHOD_GPU" \
SOURCE_ROOT="$REPO_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT/methods" \
METHODS="$METHODS" \
bash "$REPO_ROOT/analysis/run_wan14b_720p_4step_method_queue.sh" &
method_pid=$!

cleanup() {
  kill "$dense_pid" "$method_pid" 2>/dev/null || true
}
trap cleanup INT TERM

set +e
wait "$dense_pid"
dense_rc=$?
wait "$method_pid"
method_rc=$?
set -e
trap - INT TERM

if [ "$dense_rc" -ne 0 ] || [ "$method_rc" -ne 0 ]; then
  echo "FATAL: dense_rc=$dense_rc method_rc=$method_rc" >&2
  exit 1
fi

python "$REPO_ROOT/analysis/summarize_dense_sdpa_backends.py" \
  --root "$OUTPUT_ROOT/dense"
echo "All requested 4-step experiments completed: $OUTPUT_ROOT"
