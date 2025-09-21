# GPU Utilization Instability Diagnosis (Linux)

This guide helps you find the main cause of “GPU 100% ↘ 0% sawtooth” during multi‑GPU training with `accelerate`.
It’s designed for Linux with bash and focuses on isolating the dominant bottleneck, not just mitigating symptoms.

Applicable repo entry point: `train_oxford.py` (but the approach generalizes to other trainers in this repo).

## TL;DR
Run three controlled passes and compare:
- A. Baseline single‑GPU vs multi‑GPU with all side work disabled
- B. Enable logging only (no sampling/checkpoint)
- C. Re‑enable sampling/checkpoint at intervals

Use the built‑in profiling sections to see time distribution across: data_loading, forward_pass, backward_pass, optimizer_step, ema_update, sampling, checkpointing.

## 0) Preliminaries

- Ensure CUDA/NCCL works and GPUs are visible on the node.
- Prepare a modest run (few hundred steps) to keep experiments quick.
- The profiler is already wired in the trainer; enable it with an environment variable.

```bash
export RECTIFIED_FLOW_PROFILE=1
```

Recommended common flags (adjust sizes to your GPUs):
- `--batch_size 128` (or your usual)
- `--image_size 128`
- `--num_train_steps 300` (short run)
- `--num_workers 8` (try 4/8/12/16 later during data sweep)

## 1) Baseline: no sampling, no checkpoint, no W&B
Purpose: Determine if the sawtooth is intrinsic to multi‑GPU synchronization or due to side work.

Single‑GPU baseline:
```bash
accelerate launch --num_processes 1 \
  train_oxford.py \
  --batch_size 128 --image_size 128 \
  --num_train_steps 300 --num_workers 8 \
  --save_results_every 1000000 --checkpoint_every 1000000
```

Multi‑GPU baseline (8 GPUs):
```bash
accelerate launch --num_processes 8 \
  train_oxford.py \
  --batch_size 128 --image_size 128 \
  --num_train_steps 300 --num_workers 8 \
  --save_results_every 1000000 --checkpoint_every 1000000
```

What to look for:
- The console prints periodic profiling reports. Focus on time shares of:
  - data_loading
  - forward_pass
  - backward_pass
  - optimizer_step
  - ema_update
  - synchronization (if shown)
- If single‑GPU is smooth but multi‑GPU sawtooths, the culprit is often per‑step global synchronization or main‑rank side work.

## 2) Add logging back (optional W&B), still no sampling/checkpoint
Purpose: Evaluate the impact of CPU‑heavy metrics/logging and any implicit sync they trigger.

Without W&B (just console/profiler):
```bash
accelerate launch --num_processes 8 \
  train_oxford.py \
  --batch_size 128 --image_size 128 \
  --num_train_steps 300 --num_workers 8 \
  --save_results_every 1000000 --checkpoint_every 1000000
```

With W&B enabled:
```bash
accelerate launch --num_processes 8 \
  train_oxford.py \
  --batch_size 128 --image_size 128 \
  --num_train_steps 300 --num_workers 8 \
  --save_results_every 1000000 --checkpoint_every 1000000 \
  --use_wandb --wandb_run_name diag_logging_only
```

What to look for:
- Check if profiling shows increased time in metrics/logging or synchronization.
- If GPU utilization dips correlate with log cadence, logging/sync is a main driver.

## 3) Re‑enable sampling and checkpointing
Purpose: Confirm whether periodic drops line up with sampling/checkpoint intervals.

```bash
accelerate launch --num_processes 8 \
  train_oxford.py \
  --batch_size 128 --image_size 128 \
  --num_train_steps 600 --num_workers 8 \
  --save_results_every 100 --checkpoint_every 200
```

What to look for:
- If utilization dips every 100/200 steps exactly, then sampling/checkpointing is the dominant cause for the sawtooth.
- Profiling should show spikes in `sampling` and/or `checkpointing` during those steps.

## 4) Data loader throughput sweep
Purpose: Evaluate if input pipeline is starving GPUs.

Try different worker counts and watch `data_loading` share and GPU smoothness:
```bash
for W in 4 8 12 16; do
  accelerate launch --num_processes 8 \
    train_oxford.py \
    --batch_size 128 --image_size 128 \
    --num_train_steps 300 --num_workers $W \
    --save_results_every 1000000 --checkpoint_every 1000000 \
    --wandb_run_name diag_workers_$W || true
done
```

Interpretation:
- If `data_loading` > 30% or grows with batch size, boost workers, ensure dataset/cache is on fast storage, and consider lighter transforms.
- If changing workers barely moves the needle but the sawtooth persists, main reason is not the dataloader.

## 5) Optional: Nsight Systems deep dive
If you can, profile 1–2 minutes with Nsight Systems to capture CUDA kernels, NCCL sync, and host activity.
Examples (adjust as needed):

```bash
nsys profile -o rf_trace --trace=cuda,nvtx,osrt \
  accelerate launch --num_processes 8 \
  train_oxford.py --batch_size 128 --image_size 128 \
  --num_train_steps 300 --num_workers 8 \
  --save_results_every 1000000 --checkpoint_every 1000000
```

Look for:
- Long host stalls (logging, image save, subprocess eval)
- NCCL barriers aligning across ranks
- DtoH/HtoD spikes from CPU metric computations

## How to conclude “main reason”
- Single‑GPU smooth, multi‑GPU sawtooth → cross‑rank sync / main‑rank side work is primary.
- Drops align with save_results_every / checkpoint_every → sampling/checkpoint is primary.
- `data_loading` dominates → input pipeline is primary.
- Large per‑step CPU metric time (and DtoH) → per‑step metrics/logging is primary.

## Quick mitigations once you’ve identified the main cause
- If sampling/ckpt is the cause: increase intervals or move to a separate evaluation job; only sync when necessary.
- If per‑step metrics/logging is the cause: reduce logging cadence and avoid per‑step CPU norm scans; log loss only per N steps.
- If data loader is the cause: increase `--num_workers`, keep `pin_memory`, and store data on fast disk.

## Notes
- Keep the runs short but representative (300–600 steps). The profiler prints interim and final summaries.
- Re‑run the smallest test that isolates the suspected cause.
- Share the profiling report slices to discuss targeted code changes.
