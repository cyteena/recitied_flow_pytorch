# Fix for Multi-GPU Sampling Deadlock

## Issue Description

Training stalls during sampling operations (every 100 steps by default) in multi-GPU setups due to dataloader access conflicts in distributed training.

## Root Cause

The issue occurs in the `Trainer.sample()` method:

```python
def sample(self, fname):
    eval_model = default(self.ema_model, self.model)
    dl = cycle(self.dl)  # ❌ Problematic in multi-GPU
    mock_data = next(dl)  # ❌ Can cause communication deadlock
    data_shape = mock_data.shape[1:]
    # ... rest of sampling
```

In DDP (DistributedDataParallel):
- Each process has its own shard of the training data
- Calling `next()` on the distributed dataloader can trigger inter-process communication
- This synchronization can deadlock when processes are in different states (e.g., some training, some sampling)

## Solution

Store the data shape during the first training step and reuse it for sampling, avoiding dataloader access during sampling.

### Changes Made:

1. **Add data_shape storage in Trainer.__init__()**:
```python
self.data_shape = None
```

2. **Store data_shape after first training step**:
```python
# Store data shape for sampling (only once)
if self.data_shape is None:
    self.data_shape = data.shape[1:]
```

3. **Update sample() method to use stored data_shape**:
```python
def sample(self, fname):
    eval_model = default(self.ema_model, self.model)
    
    # Use stored data_shape instead of accessing dataloader
    assert self.data_shape is not None, "data_shape not set. Run at least one training step first."
    data_shape = self.data_shape
    
    # ... rest of sampling without dataloader access
```

4. **Update EMA data_shape setting**:
```python
data_shape = self.data_shape  # Use stored data_shape
self.ema_model.ema_model.data_shape = data_shape
```

## Why This Works

- **Eliminates Dataloader Access**: No more `next(dl)` calls during sampling
- **Avoids Communication**: Prevents inter-process synchronization during sampling
- **Maintains Functionality**: Sampling still works correctly with the stored shape
- **Thread Safe**: No race conditions or deadlocks in multi-GPU setups

## Technical Details

### Before (Problematic)
```python
# In multi-GPU, this can deadlock
dl = cycle(self.dl)
mock_data = next(dl)  # Triggers distributed communication
data_shape = mock_data.shape[1:]
```

### After (Fixed)
```python
# Uses pre-stored shape, no communication needed
data_shape = self.data_shape  # Set during first training step
```

## Benefits

- **Reliable Multi-GPU Training**: No more stalls during sampling
- **Better Performance**: Avoids unnecessary dataloader synchronization
- **Cleaner Code**: Separates data access from sampling logic
- **Robust**: Works consistently across different GPU configurations

## Testing

After applying this fix:
- Multi-GPU training should continue past 100 steps without stalling
- Sampling operations should complete successfully
- No performance degradation in single-GPU setups

The fix ensures that sampling operations are completely independent of the distributed dataloader, eliminating the communication bottleneck that was causing training to stall.
