# Training Pipeline Profiling

This document explains how to use the built-in profiling functionality to identify performance bottlenecks in the rectified flow training pipeline.

## Overview

The profiling system measures the time spent in different components of the training loop:

- **Data Loading**: Time spent loading batches from the dataloader
- **Forward Pass**: Time spent in the model forward pass
- **Backward Pass**: Time spent computing gradients
- **Metrics Computation**: Time spent calculating gradient norms and parameter norms
- **Optimizer Step**: Time spent updating model parameters
- **EMA Updates**: Time spent updating exponential moving averages
- **Sampling**: Time spent generating sample images
- **Checkpointing**: Time spent saving model checkpoints
- **Synchronization**: Time spent in distributed training synchronization

## How to Enable Profiling

### Method 1: Using the Profile Script

Run the dedicated profiling script:

```bash
# Enable profiling and run training
RECTIFIED_FLOW_PROFILE=1 python profile_training.py
```

### Method 2: Manual Environment Variable

Set the environment variable before running any training:

```bash
export RECTIFIED_FLOW_PROFILE=1
python train_oxford.py
```

## Profiling Output

The profiler provides two types of output:

### 1. Periodic Reports

Every 50 steps (configurable), you'll see:

```
=== Profiling Report (Step 50) ===

Detailed Breakdown:
data_loading: count=50, total=2.341s, avg=0.0468s, min=0.0421s, max=0.0512s
forward_pass: count=50, total=15.234s, avg=0.3047s, min=0.2981s, max=0.3123s
backward_pass: count=50, total=8.956s, avg=0.1791s, min=0.1752s, max=0.1834s
metrics_computation: count=50, total=0.123s, avg=0.0025s, min=0.0021s, max=0.0029s
optimizer_step: count=50, total=0.089s, avg=0.0018s, min=0.0016s, max=0.0021s
ema_update: count=50, total=0.045s, avg=0.0009s, min=0.0008s, max=0.0011s

Time Distribution:
data_loading: 10.2%
forward_pass: 66.3%
backward_pass: 19.5%
metrics_computation: 0.5%
optimizer_step: 0.4%
ema_update: 0.2%
```

### 2. Final Summary

At the end of training:

```
=== Final Profiling Summary (1000 steps) ===

Detailed Breakdown:
data_loading: count=1000, total=46.823s, avg=0.0468s, min=0.0412s, max=0.0523s
forward_pass: count=1000, total=304.678s, avg=0.3047s, min=0.2951s, max=0.3156s
backward_pass: count=1000, total=179.123s, avg=0.1791s, min=0.1723s, max=0.1856s
...

Time Distribution:
data_loading: 10.2%
forward_pass: 66.3%
backward_pass: 19.5%
...

Performance Insights:
- Slowest component: forward_pass (0.3047s avg)
- Data loading appears efficient (10.2% of total time)
```

## Interpreting Results

### Key Metrics to Watch

1. **Data Loading**: Should be < 15% of total time. If higher, consider:
   - Increasing `num_workers` in DataLoader
   - Using faster storage (SSD vs HDD)
   - Reducing image preprocessing

2. **Forward Pass**: Usually the most time-consuming. Optimization strategies:
   - Model architecture changes
   - Mixed precision training (FP16)
   - Gradient checkpointing for memory

3. **Backward Pass**: Should be roughly equal to forward pass time. If much slower:
   - Check for memory bottlenecks
   - Consider gradient accumulation

4. **GPU Utilization**: Monitor with `nvidia-smi`. Low utilization may indicate:
   - Small batch sizes
   - Data loading bottlenecks
   - CPU-GPU transfer overhead

### Common Bottlenecks

- **High Data Loading %**: Increase `num_workers`, use pinned memory, optimize preprocessing
- **High Forward Pass %**: Consider model distillation, quantization, or architectural changes
- **High Synchronization %**: Check network bandwidth in distributed training
- **Low GPU Utilization**: Increase batch size or optimize data pipeline

## Configuration

### Changing Report Frequency

Modify the `log_every` parameter in the profiler initialization:

```python
# In training_profiler.py or in the Trainer
start_training_profiling(log_every=100)  # Report every 100 steps
```

### Adding Custom Profiling Sections

Add profiling to any code section:

```python
from rectified_flow_pytorch.training_profiler import profile_section

with profile_section("my_custom_operation"):
    # Your code here
    pass
```

## Performance Tips

1. **Run profiling on a small subset** first (e.g., 100-500 steps) to identify major bottlenecks
2. **Compare different configurations** (batch sizes, num_workers, etc.)
3. **Profile both single-GPU and multi-GPU** setups to understand scaling efficiency
4. **Monitor memory usage** alongside timing to identify memory bottlenecks

## Troubleshooting

- **No profiling output**: Ensure `RECTIFIED_FLOW_PROFILE=1` is set
- **Import errors**: Make sure `training_profiler.py` is in the same directory
- **Performance impact**: Profiling adds minimal overhead (~1-2% of training time)
