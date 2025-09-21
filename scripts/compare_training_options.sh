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
BATCH=${BATCH:-16}
IMSIZE=${IMSIZE:-64}
WORKERS=${WORKERS:-4}
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
# Output directories    #
########################
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="comparison_logs/$TS"
mkdir -p "$OUT_DIR"

########################
# Helpers               #
########################
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_experiment() {
  local exp_id="$1"
  local gpu_id="$2"
  local mean_var="$3"
  local consistency="$4"
  local sched="$5"

  local exp_name="exp_${exp_id}_mv${mean_var}_cons${consistency}_sched${sched}"
  local log_file="$OUT_DIR/${exp_name}.log"

  log "Starting exp $exp_id on GPU $gpu_id: $exp_name"

  CUDA_VISIBLE_DEVICES="$gpu_id" python "$TRAIN_SCRIPT" \
    --num_train_steps "$STEPS" \
    --batch_size "$BATCH" \
    --image_size "$IMSIZE" \
    --num_workers "$WORKERS" \
    --mean_variance_net "$mean_var" \
    --use_consistency "$consistency" \
    --use_warmup "$sched" \
    --use_lr_decay "$sched" \
    --results_folder "results_${exp_name}" \
    --checkpoints_folder "checkpoints_${exp_name}" \
    --wandb_run_name "$exp_name" \
    > "$log_file" 2>&1 &

  # Return the PID
  echo $!
}

########################
# Main experiment loop #
########################
log "Starting training options comparison with $GPUS GPUs"
log "Logs: $OUT_DIR"

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
  pid=$(run_experiment "$((i+1))" "$gpu_idx" $exp_params)
  pids+=("$pid")

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
for i in "${!experiments[@]}"; do
  exp_params="${experiments[$i]}"
  # Parse parameters
  read -r mean_var consistency sched <<< "$exp_params"
  exp_name="exp_$((i+1))_mv${mean_var}_cons${consistency}_sched${sched}"
  log_file="$OUT_DIR/${exp_name}.log"

  if [[ -f "$log_file" ]]; then
    # Extract final loss from log
    final_loss=$(grep "loss:" "$log_file" | tail -n1 | sed 's/.*loss: \([0-9.]*\).*/\1/')
    if [[ -n "$final_loss" ]]; then
      echo "Experiment $((i+1)): $exp_params -> Final loss: $final_loss"
    else
      echo "Experiment $((i+1)): $exp_params -> No loss found"
    fi
  else
    echo "Experiment $((i+1)): $exp_params -> Log file missing"
  fi
done

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

All logs and results saved under comparison_logs/<timestamp>.
EOF
