#!/usr/bin/env bash
set -euo pipefail

# GPU Utilization Instability One-Click Diagnosis
# - Runs a set of controlled experiments
# - Captures logs with built-in profiler enabled
# - Extracts the last "Time Distribution" block from each run
# - Prints a concise summary of likely main cause
#
# Requirements:
# - Linux + bash
# - accelerate installed and working on the node
# - This repo checked out; run from anywhere (script will cd to repo root)
#
# Notes:
# - Uses the trainer's internal profiler via RECTIFIED_FLOW_PROFILE=1
# - Keeps runs short (default 300-600 steps)
# - Oxford Flowers dataset is pulled via HF on first run (needs network/cache)

########################
# Configurable defaults #
########################
STEPS=${STEPS:-300}
BATCH=${BATCH:-128}
IMSIZE=${IMSIZE:-128}
WORKERS=${WORKERS:-8}
GPU_COUNT_ENV=${GPU_COUNT:-}
SAVE_MANY=${SAVE_MANY:-1000000} # effectively disable sampling/ckpt
SAVE_SPARSE=${SAVE_SPARSE:-100}
CKPT_SPARSE=${CKPT_SPARSE:-200}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_oxford.py}
# Whether to include sampling/checkpoint periodic test (0=skip, 1=run)
INCLUDE_SAMPLING=${INCLUDE_SAMPLING:-0}

########################
# Resolve repo root     #
########################
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

########################
# GPU count detection   #
########################
if [[ -n "${GPU_COUNT_ENV}" ]]; then
  GPUS="$GPU_COUNT_ENV"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _arr <<< "$CUDA_VISIBLE_DEVICES"
  GPUS="${#_arr[@]}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L | wc -l)"
  else
    GPUS=1
  fi
fi

########################
# Output directories    #
########################
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="diagnostics_logs/$TS"
mkdir -p "$OUT_DIR"

########################
# Helpers               #
########################
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Extract the last "Time Distribution" block from a log file and print lines as "name: pct%"
extract_last_time_dist() {
  local logfile="$1"
  local last_line
  last_line="$(grep -n "^Time Distribution:" "$logfile" | tail -n1 | cut -d: -f1 || true)"
  if [[ -z "$last_line" ]]; then
    return 1
  fi
  awk -v start="$last_line" 'NR>start && /^[A-Za-z_]+: [0-9.]+%$/ {print; next} NR>start {exit}' "$logfile"
}

# Return "TOP_NAME TOP_PCT%" from a time distribution block
top_from_time_dist() {
  awk -F': ' '{name=$1; gsub(/%/, "", $2); val=$2+0; if(val>max){max=val; top=name}} END{if(max>0) printf "%s %.1f%%\n", top, max}'
}

run_exp() {
  local label="$1"; shift
  local np="$1"; shift
  local extra_args=("$@")
  local log_file="$OUT_DIR/${label}.log"

  log "Running: $label (np=$np) -> $log_file"
  RECTIFIED_FLOW_PROFILE=1 \
  accelerate launch --num_processes "$np" \
    "$TRAIN_SCRIPT" \
    --batch_size "$BATCH" --image_size "$IMSIZE" \
    --num_train_steps "$STEPS" --num_workers "$WORKERS" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"

  if ! grep -q "^Time Distribution:" "$log_file"; then
    log "WARN: No profiling Time Distribution found in $label log."
  fi
}

summarize_exp() {
  local label="$1"
  local log_file="$OUT_DIR/${label}.log"
  local block
  block="$(extract_last_time_dist "$log_file" || true)"
  if [[ -z "$block" ]]; then
    echo "$label: (no profile block found)"
    return
  fi
  local top
  top="$(printf "%s\n" "$block" | top_from_time_dist)"
  echo "$label: $top"
}

########################
# Experiments          #
########################
log "Detected GPUs: $GPUS"
log "Logs: $OUT_DIR"

# A1) Single-GPU baseline (no sampling/ckpt)
run_exp "A1_single_gpu_baseline" 1 --save_results_every "$SAVE_MANY" --checkpoint_every "$SAVE_MANY"

# A2) Multi-GPU baseline (no sampling/ckpt)
if (( GPUS > 1 )); then
  run_exp "A2_multi_gpu_baseline" "$GPUS" --save_results_every "$SAVE_MANY" --checkpoint_every "$SAVE_MANY"
else
  log "Skipping A2 (only one GPU detected)."
fi

# B) Sampling/Checkpoint periodic work (observe periodic dips) — disabled by default
if (( GPUS > 1 )) && (( INCLUDE_SAMPLING == 1 )); then
  run_exp "B_sampling_checkpoint" "$GPUS" --save_results_every "$SAVE_SPARSE" --checkpoint_every "$CKPT_SPARSE"
fi

# C) Data loader sweep (no sampling/ckpt)
if (( GPUS > 1 )); then
  for W in 4 8 12 16; do
    WORKERS="$W" run_exp "C_workers_${W}" "$GPUS" --save_results_every "$SAVE_MANY" --checkpoint_every "$SAVE_MANY"
  done
fi

########################
# Summary              #
########################
log "\n==== Diagnosis Summary (top time share in last profile block) ===="
summarize_exp "A1_single_gpu_baseline"
if (( GPUS > 1 )); then
  summarize_exp "A2_multi_gpu_baseline"
  if (( INCLUDE_SAMPLING == 1 )); then
    summarize_exp "B_sampling_checkpoint"
  fi
  for W in 4 8 12 16; do
    summarize_exp "C_workers_${W}"
  done
fi

cat << 'EOF'

Interpretation:
- If A1 is smooth and A2 shows a different top (e.g., synchronization/optimizer/metrics), multi-GPU sync or main-rank side work is likely the main cause.
- If B shows sampling or checkpointing as top (and GPU dips align with those intervals), periodic side work is the main cause.
- If C shows data_loading dominant and improves as workers increase, the input pipeline is the main cause.
- Otherwise, compare A1 vs A2 vs B to decide which component consistently dominates.

Tips:
- Increase --num_workers when data_loading dominates.
- Increase --save_results_every/--checkpoint_every (or separate evaluation) when sampling/ckpt dominates.
- Reduce per-step CPU metrics/logging when metrics/synchronization dominates.

All logs kept under diagnostics_logs/<timestamp>.
EOF
