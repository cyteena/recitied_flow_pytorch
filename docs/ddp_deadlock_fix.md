# Fix for Multi-GPU Training Deadlock

## Issue Description

Training stalls after 100 steps in multi-GPU setups due to a communication deadlock when accessing model attributes on DDP-wrapped models.

## Root Cause

The issue occurs in the EMA update section of the `Trainer.forward()` method:

```python
if hasattr(self.model, 'data_shape'):
    data_shape = self.model.data_shape
else:
    data_shape = self.accelerator.unwrap_model(self.model).data_shape
```

When `hasattr()` is called on a DDP-wrapped model, it attempts to communicate with other processes to check attribute existence, which can cause deadlocks or synchronization issues in distributed training.

## Solution

Always unwrap the model before accessing custom attributes to avoid DDP communication issues:

```python
# Before (problematic)
if hasattr(self.model, 'data_shape'):
    data_shape = self.model.data_shape
else:
    data_shape = self.accelerator.unwrap_model(self.model).data_shape

# After (fixed)
unwrapped_model = self.accelerator.unwrap_model(self.model)
data_shape = unwrapped_model.data_shape
```

## Why This Works

- **Avoids DDP Communication**: `hasattr()` on DDP models can trigger inter-process communication
- **Consistent Access**: Always access attributes from the unwrapped model
- **Thread Safety**: Prevents deadlocks in multi-GPU distributed training
- **Reliability**: Eliminates race conditions in attribute access

## Technical Details

DDP (DistributedDataParallel) wraps the model and intercepts attribute access for synchronization. Certain operations like `hasattr()` can inadvertently trigger communication between processes, leading to deadlocks when not all processes are in the same state.

By explicitly unwrapping the model first, we ensure all attribute access happens on the local model instance without distributed coordination.

## Testing

After applying this fix, multi-GPU training should continue past 100 steps without stalling. The deadlock is eliminated by avoiding `hasattr()` calls on DDP-wrapped models.
