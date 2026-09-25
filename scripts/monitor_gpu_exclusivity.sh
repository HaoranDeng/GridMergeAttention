#!/usr/bin/env bash
set -u

GPU="${GPU:?Set GPU index}"
RUNNER_ROOT="${RUNNER_ROOT:?Set the runner process ID}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:?Set the experiment directory}"
DONE_FILE="${DONE_FILE:-$EXPERIMENT_ROOT/completed}"
LOG="${LOG:-$EXPERIMENT_ROOT/gpu${GPU}_exclusivity.log}"
VIOLATION="${VIOLATION:-$EXPERIMENT_ROOT/gpu${GPU}_contamination_detected}"

is_descendant() {
  local pid="$1"
  local root="$2"
  local parent
  while [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 )); do
    [[ "$pid" == "$root" ]] && return 0
    parent=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -n "$parent" ]] || break
    pid="$parent"
  done
  return 1
}

mkdir -p "$EXPERIMENT_ROOT"
printf '%s watchdog_started runner=%s\n' \
  "$(date -Is)" "$RUNNER_ROOT" >> "$LOG"

checks=0
while [[ ! -f "$DONE_FILE" ]]; do
  mapfile -t pids < <(
    nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | tr -d ' ' | sed '/^$/d'
  )

  for pid in "${pids[@]}"; do
    if ! is_descendant "$pid" "$RUNNER_ROOT"; then
      printf '%s contamination pid=%s process=%s\n' \
        "$(date -Is)" "$pid" "$(ps -o user=,cmd= -p "$pid")" >> "$LOG"
      touch "$VIOLATION"
    fi
  done

  if (( checks % 60 == 0 )); then
    printf '%s healthy gpu_pids=%s\n' \
      "$(date -Is)" "${pids[*]:-none}" >> "$LOG"
  fi
  checks=$((checks + 1))

  if ! kill -0 "$RUNNER_ROOT" 2>/dev/null; then
    printf '%s runner_stopped_without_completion\n' "$(date -Is)" >> "$LOG"
    exit 1
  fi
  sleep 5
done

printf '%s run_complete contamination=%s\n' \
  "$(date -Is)" "$([[ -e "$VIOLATION" ]] && echo yes || echo no)" >> "$LOG"
