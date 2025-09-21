#!/usr/bin/env bash
set -euo pipefail

# Training Options Comparison Script
# Runs 8 experiments comparing mean-variance-net, consistency_flow_matching, and warm_up + lr_decay
#
# Requirements:
# - Linux + bash
# - accelerate installed and working
# - This repo checked out; run from repo root
#
# Usage:
# ./scripts/compare_training_options.sh

########################
# Configurable defaults #
########################
STEPS=${STEPS:-10000}
BATCH=${BATCH:-64}  # Increased from 16 to 64 for better GPU utilization
IMSIZE=${IMSIZE:-128}  # Increased from 64 to 128 for more computation
WORKERS=${WORKERS:-8}  # Increased from 4 to 8 for better data loading
TRAIN_SCRIPT=${TRAIN_SCRIPT:-compare_training_options.py}

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
  local mean_var="$3"
  local consistency="$4"
  local sched="$5"

  local exp_name="exp_${exp_id}_mv${mean_var}_cons${consistency}_sched${sched}"

  log "Starting exp $exp_id on GPU $gpu_id: $exp_name"

  # Convert boolean values to lowercase for Python
  local mean_var_flag=""
  local consistency_flag=""
  local warmup_flag=""
  local lr_decay_flag=""
  
  if [[ "$mean_var" == "True" ]]; then
    mean_var_flag="--mean_variance_net"
  fi
  
  if [[ "$consistency" == "True" ]]; then
    consistency_flag="--use_consistency"
  fi
  
  if [[ "$sched" == "True" ]]; then
    warmup_flag="--use_warmup"
    lr_decay_flag="--use_lr_decay"
  fi

  CUDA_VISIBLE_DEVICES="$gpu_id" python "$TRAIN_SCRIPT" \
    --num_train_steps "$STEPS" \
    --batch_size "$BATCH" \
    --image_size "$IMSIZE" \
    --num_workers "$WORKERS" \
    $mean_var_flag \
    $consistency_flag \
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
log "Starting training options comparison with $GPUS GPUs"
log "Using wandb for logging - no local log files"

# Define experiments: (mean_var, consistency, sched)
experiments=(
  "False False False"  # 1: baseline
  "False False True"   # 2: sched only
  "False True False"   # 3: consistency only
  "False True True"    # 4: consistency + sched
  "True False False"   # 5: mean_var only
  "True False True"    # 6: mean_var + sched
  "True True False"    # 7: mean_var + consistency
  "True True True"     # 8: all enabled
)

pids=()
gpu_idx=0

for i in "${!experiments[@]}"; do
  exp_params="${experiments[$i]}"
  # Parse the parameters properly
  read -r mean_var consistency sched <<< "$exp_params"
  run_experiment "$((i+1))" "$gpu_idx" "$mean_var" "$consistency" "$sched" &
  pids+=($!)

  gpu_idx=$(( (gpu_idx + 1) % GPUS ))
done

log "All experiments started. Waiting for completion..."

# Wait for all experiments to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done

log "All experiments completed!"

########################
# Summary              #
########################
log "\n==== Comparison Summary ===="
log "All experiments have been launched with wandb logging enabled."
log "Monitor progress at: https://wandb.ai"
log "Project: rectified-flow-comparison"

cat << 'EOF'

Experiment configurations:
1. baseline: no mean_var, no consistency, no sched
2. sched: warmup + lr_decay only
3. consistency: consistency_flow_matching only
4. consistency + sched: consistency + warmup + lr_decay
5. mean_var: mean_variance_net only
6. mean_var + sched: mean_var + warmup + lr_decay
7. mean_var + consistency: mean_var + consistency
8. all: mean_var + consistency + warmup + lr_decay

Results and logs are available in wandb: rectified-flow-comparison project.
EOF
