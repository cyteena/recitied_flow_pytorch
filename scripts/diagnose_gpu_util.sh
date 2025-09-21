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
# New: Multi-experiment mode (0=original diagnosis, 1=run multiple single-GPU experiments in parallel)
MULTI_EXP_MODE=${MULTI_EXP_MODE:-0}
# List of experiment configs (e.g., "lr:3e-4,lr:1e-4,batch:64")
EXP_CONFIGS=${EXP_CONFIGS:-"default"}

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

# Parse config string like "lr:3e-4,batch:64" into CLI args
parse_config() {
  local config="$1"
  local args=()
  IFS=',' read -r -a parts <<< "$config"
  for part in "${parts[@]}"; do
    IFS=':' read -r key val <<< "$part"
    case "$key" in
      lr) args+=(--learning_rate "$val") ;;
      batch) args+=(--batch_size "$val") ;;
      steps) args+=(--num_train_steps "$val") ;;
      workers) args+=(--num_workers "$val") ;;
      *) log "WARN: Unknown config key '$key', ignoring" ;;
    esac
  done
  printf '%s\n' "${args[@]}"
}

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
  local temp_log="$OUT_DIR/${label}_temp.log"

  log "Running: $label (np=$np) -> $log_file"
  RECTIFIED_FLOW_PROFILE=1 \
  accelerate launch --num_processes "$np" \
    "$TRAIN_SCRIPT" \
    --batch_size "$BATCH" --image_size "$IMSIZE" \
    --num_train_steps "$STEPS" --num_workers "$WORKERS" \
    "${extra_args[@]}" \
    > "$temp_log" 2>&1

  # Extract only profiling reports from temp log
  awk '
    /^=== Profiling Report/ { in_report=1; print; next }
    /^=== Final Profiling Summary/ { in_report=1; print; next }
    in_report && /^===/ && /Summary/ { print; next }
    in_report && /^[A-Za-z_]+: [0-9.]+s/ { print; next }
    in_report && /^Time Distribution:/ { print; next }
    in_report && /^[A-Za-z_]+: [0-9.]+%$/ { print; next }
    in_report && /^$/ { print; next }
    /^===/ && /Summary/ { in_report=0; print; next }
    /^$/ { if(in_report) print }
  ' "$temp_log" > "$log_file"

  rm -f "$temp_log"

  if ! grep -q "^Time Distribution:" "$log_file"; then
    log "WARN: No profiling Time Distribution found in $label log."
  fi
}

# Run a single-GPU experiment with specific config and GPU
run_single_gpu_exp() {
  local exp_id="$1"
  local gpu_id="$2"
  local config="$3"
  local label="exp_${exp_id}_gpu${gpu_id}_${config//:/_}"
  local log_file="$OUT_DIR/${label}.log"
  local temp_log="$OUT_DIR/${label}_temp.log"

  local extra_args
  extra_args="$(parse_config "$config")"

  log "Starting exp $exp_id on GPU $gpu_id with config '$config' -> $log_file"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  RECTIFIED_FLOW_PROFILE=1 \
  accelerate launch --num_processes 1 \
    "$TRAIN_SCRIPT" \
    --batch_size "$BATCH" --image_size "$IMSIZE" \
    --num_train_steps "$STEPS" --num_workers "$WORKERS" \
    $extra_args \
    --save_results_every "$SAVE_MANY" --checkpoint_every "$SAVE_MANY" \
    > "$temp_log" 2>&1 &

  # Extract profiling in background
  (
    wait $!
    awk '
      /^=== Profiling Report/ { in_report=1; print; next }
      /^=== Final Profiling Summary/ { in_report=1; print; next }
      in_report && /^===/ && /Summary/ { print; next }
      in_report && /^[A-Za-z_]+: [0-9.]+s/ { print; next }
      in_report && /^Time Distribution:/ { print; next }
      in_report && /^[A-Za-z_]+: [0-9.]+%$/ { print; next }
      in_report && /^$/ { print; next }
      /^===/ && /Summary/ { in_report=0; print; next }
      /^$/ { if(in_report) print }
    ' "$temp_log" > "$log_file"
    rm -f "$temp_log"
    log "Finished exp $exp_id on GPU $gpu_id"
  ) &
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

if (( MULTI_EXP_MODE == 1 )); then
  # Multi-experiment mode: run multiple single-GPU experiments in parallel
  IFS=',' read -r -a configs <<< "$EXP_CONFIGS"
  local exp_count="${#configs[@]}"
  local gpu_idx=0
  local pids=()

  log "Starting $exp_count experiments across $GPUS GPUs"

  for ((exp_id=0; exp_id<exp_count; exp_id++)); do
    local config="${configs[$exp_id]}"
    local gpu_id="$gpu_idx"
    run_single_gpu_exp "$exp_id" "$gpu_id" "$config"
    pids+=($!)
    gpu_idx=$(( (gpu_idx + 1) % GPUS ))
  done

  # Wait for all experiments to finish
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  log "All experiments completed"

  # Summarize all experiments
  log "\n==== Multi-Experiment Summary ===="
  for ((exp_id=0; exp_id<exp_count; exp_id++)); do
    local config="${configs[$exp_id]}"
    local gpu_id=$(( exp_id % GPUS ))
    local label="exp_${exp_id}_gpu${gpu_id}_${config//:/_}"
    summarize_exp "$label"
  done

else
  # Original diagnosis mode
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
fi
