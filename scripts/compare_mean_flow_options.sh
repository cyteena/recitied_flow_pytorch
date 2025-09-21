#!/usr/bin/env bash
set -euo pipefail

# MeanFlow Training Options Comparison Script
# Runs 12 experiments comparing:
# - random_fourier_features vs learned_sinusoidal_cond vs none
# - use_logit_normal_sampler vs uniform sampling
# - warmup + lr_decay vs constant lr
#
# Requirements:
# - Linux + bash
# - accelerate installed and working
# - This repo checked out; run from repo root
#
# Usage:
# ./scripts/compare_mean_flow_options.sh
########################
# Configurable defaults #
########################
STEPS=${STEPS:-70000}
BATCH=${BATCH:-64}  # Increased for better GPU utilization
IMSIZE=${IMSIZE:-128}  # Increased for more computation
WORKERS=${WORKERS:-8}  # Increased for better data loading
TRAIN_SCRIPT=${TRAIN_SCRIPT:-compare_mean_flow_options.py}

########################
# Resolve repo root     #
########################
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

########################
# GPU count detection   #
########################
if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L | wc -l)"
else
    GPUS=1
fi

########################
# Helpers               #
########################
log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

run_experiment() {
  local exp_id="$1"
  local gpu_id="$2"
  local time_cond="$3"
  local sampler="$4"
  local sched="$5"

  local exp_name="meanflow_${exp_id}_${time_cond}_${sampler}_${sched}"

  log "Starting MeanFlow exp $exp_id on GPU $gpu_id: $exp_name"

  # Convert options to flags
  local rff_flag=""
  local lsc_flag=""
  local logit_flag=""
  local warmup_flag=""
  local lr_decay_flag=""
  
  # Time conditioning
  if [[ "$time_cond" == "rff" ]]; then
    rff_flag="--random_fourier_features"
  elif [[ "$time_cond" == "lsc" ]]; then
    lsc_flag="--learned_sinusoidal_cond"
  fi
  
  # Sampling method
  if [[ "$sampler" == "logit" ]]; then
    logit_flag="--use_logit_normal_sampler"
  fi
  
  # Learning rate scheduling
  if [[ "$sched" == "sched" ]]; then
    warmup_flag="--use_warmup"
    lr_decay_flag="--use_lr_decay"
  fi

  CUDA_VISIBLE_DEVICES="$gpu_id" python "$TRAIN_SCRIPT" \
    --num_train_steps "$STEPS" \
    --batch_size "$BATCH" \
    --image_size "$IMSIZE" \
    --num_workers "$WORKERS" \
    $rff_flag \
    $lsc_flag \
    $logit_flag \
    $warmup_flag \
    $lr_decay_flag \
    --results_folder "results/${exp_name}" \
    --checkpoints_folder "checkpoints/${exp_name}" \
    --use_wandb \
    --wandb_run_name "$exp_name" \
    --prefetch_factor 8 \
    --persistent_workers &
}

########################
# Main experiment loop #
########################
log "Starting MeanFlow training options comparison with $GPUS GPUs"
log "Using wandb for logging - no local log files"

# Define experiments: (time_cond, sampler, sched)
# time_cond: "none", "rff" (random_fourier_features), "lsc" (learned_sinusoidal_cond)
# sampler: "uniform", "logit" (use_logit_normal_sampler)
# sched: "const" (constant lr), "sched" (warmup + lr_decay)
experiments=(
  "none uniform const"    # 1: baseline
  "none uniform sched"    # 2: baseline + scheduling
  "none logit const"      # 3: logit sampling only
  "none logit sched"      # 4: logit sampling + scheduling
  "rff uniform const"     # 5: random fourier features only
  "rff uniform sched"     # 6: rff + scheduling
  "rff logit const"       # 7: rff + logit sampling
  "rff logit sched"       # 8: rff + logit + scheduling
  "lsc uniform const"     # 9: learned sinusoidal cond only
  "lsc uniform sched"     # 10: lsc + scheduling
  "lsc logit const"       # 11: lsc + logit sampling
  "lsc logit sched"       # 12: all features enabled
)

pids=()
gpu_idx=0

for i in "${!experiments[@]}"; do
  exp_params="${experiments[$i]}"
  # Parse the parameters properly
  read -r time_cond sampler sched <<< "$exp_params"
  run_experiment "$((i+1))" "$gpu_idx" "$time_cond" "$sampler" "$sched" &
  pids+=($!)

  gpu_idx=$(( (gpu_idx + 1) % GPUS ))
done

log "All MeanFlow experiments started. Waiting for completion..."

# Wait for all experiments to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done

log "All MeanFlow experiments completed!"

########################
# Summary              #
########################
log "\n==== MeanFlow Comparison Summary ===="
log "All experiments have been launched with wandb logging enabled."
log "Monitor progress at: https://wandb.ai"
log "Project: mean-flow-comparison"

cat << 'EOF'

MeanFlow Experiment configurations:
1. baseline: no time_cond, uniform sampling, constant lr
2. baseline + sched: no time_cond, uniform sampling, warmup + lr_decay
3. logit sampling: no time_cond, logit_normal sampling, constant lr
4. logit + sched: no time_cond, logit_normal sampling, warmup + lr_decay
5. rff: random_fourier_features, uniform sampling, constant lr
6. rff + sched: random_fourier_features, uniform sampling, warmup + lr_decay
7. rff + logit: random_fourier_features, logit_normal sampling, constant lr
8. rff + logit + sched: random_fourier_features, logit_normal sampling, warmup + lr_decay
9. lsc: learned_sinusoidal_cond, uniform sampling, constant lr
10. lsc + sched: learned_sinusoidal_cond, uniform sampling, warmup + lr_decay
11. lsc + logit: learned_sinusoidal_cond, logit_normal sampling, constant lr
12. all features: learned_sinusoidal_cond, logit_normal sampling, warmup + lr_decay

Results and logs are available in wandb: mean-flow-comparison project.
EOF