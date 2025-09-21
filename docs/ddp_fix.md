# Fix for DDP AttributeError in Multi-GPU Training

## Issue Description

After fixing the pickleable dataset issue, a new error occurred during multi-GPU training:

```
AttributeError: 'DistributedDataParallel' object has no attribute 'data_shape'
```

This error happens in the `Trainer.forward()` method at line 1110 in `rectified_flow_pytorch/rectified_flow.py`:

```python
self.ema_model.ema_model.data_shape = self.model.data_shape
```

## Root Cause

In multi-GPU setups, Accelerate wraps the model with `DistributedDataParallel` (DDP) for parallel training. The `data_shape` attribute is set on the inner (unwrapped) model during the forward pass of `RectifiedFlow`, but DDP-wrapped models don't automatically expose custom attributes from the inner model.

When trying to access `self.model.data_shape` on the wrapped model, it fails because DDP doesn't have this attribute.

## Solution

To access attributes set on the inner model, we need to handle both wrapped (DDP) and unwrapped models. Use `hasattr` to check if the attribute exists on the model directly, and if not (indicating DDP wrapping), unwrap the model.

### Fixed Code

Change the line in `Trainer.forward()`:

```python
# Before (causes AttributeError in DDP)
self.ema_model.ema_model.data_shape = self.model.data_shape

# After (works for both single GPU and DDP)
if hasattr(self.model, 'data_shape'):
    data_shape = self.model.data_shape
else:
    data_shape = self.accelerator.unwrap_model(self.model).data_shape
self.ema_model.ema_model.data_shape = data_shape
```

## Why This Works

- `hasattr(self.model, 'data_shape')` checks if the attribute exists without raising an exception
- In single GPU setups, the model typically has the attribute directly
- In multi-GPU (DDP) setups, the wrapped model doesn't have the attribute, so it falls back to unwrapping
- This is more efficient than try-except and maintains compatibility with both setups

## Testing

After applying this fix, multi-GPU training should proceed without the AttributeError. The EMA model will correctly inherit the data shape from the unwrapped model.
