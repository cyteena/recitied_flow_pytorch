# MeanFlow Evaluation Architecture Mismatch Analysis

## Problem Description

When running evaluation on MeanFlow checkpoints, the following error occurs:

```
RuntimeError: Error(s) in loading state_dict for RectifiedFlow:
    Unexpected key(s) in state_dict: "model.cond_mlp.1.weight", "model.cond_mlp.1.bias", "model.cond_mlp.3.weight", "model.cond_mlp.3.bias", "model.time_mlp.0.weights".
    size mismatch for model.time_mlp.1.weight: copying a param with shape torch.Size([256, 17]) from checkpoint, the shape in current model is torch.Size([256, 64]).
```

## Root Cause Analysis

### 1. Architecture Mismatch
The evaluation script (`evaluate_checkpoint.py`) is hardcoded to create a basic RectifiedFlow model:

```python
from rectified_flow_pytorch import Unet
model = Unet(dim=config.dim, mean_variance_net=config.mean_variance_net)
rectified_flow = RectifiedFlow(model)
```

However, MeanFlow experiments use different model architectures based on time conditioning options:

- **random_fourier_features**: Adds Fourier feature encoding for time
- **learned_sinusoidal_cond**: Uses learned sinusoidal conditioning
- **none**: Basic time conditioning

### 2. Model Type Mismatch
The evaluation script tries to load a RectifiedFlow model, but the checkpoints are from MeanFlow training. While both use similar UNet architectures, they have different conditioning mechanisms.

### 3. Time Encoding Differences
- **Basic model**: Uses simple time embedding (64 dimensions → 256)
- **Random Fourier Features**: Uses Fourier features (17 dimensions → 256)
- **Learned Sinusoidal**: Uses learned sinusoidal encoding

## Impact

- Evaluation fails for all MeanFlow checkpoints
- Cannot compare different MeanFlow configurations
- Training runs successfully but evaluation is broken

## Solution Requirements

1. **Dynamic Model Creation**: Evaluation script must recreate the exact same model architecture used during training
2. **Configuration Persistence**: Training configuration must be saved with checkpoints
3. **Unified Logging**: Evaluation should log to the same wandb run as training, not create separate runs

## Implementation Plan

1. Save training configuration with checkpoints
2. Modify evaluation script to detect model type and recreate appropriate architecture
3. Update trainer to reuse wandb runs for evaluation instead of creating new ones</content>
<parameter name="filePath">d:\code\rectified-flow-pytorch\MEANFLOW_EVAL_ANALYSIS.md