# Evaluation Issues and Fixes

## Problem Analysis

The evaluation was failing with the following error:

```
RuntimeError: Error(s) in loading state_dict for RectifiedFlow:
        Unexpected key(s) in state_dict: "model.cond_mlp.1.weight", "model.cond_mlp.1.bias", "model.cond_mlp.3.weight", "model.cond_mlp.3.bias", "model.time_mlp.0.weights".
        size mismatch for model.time_mlp.1.weight: copying a param with shape torch.Size([256, 17]) from checkpoint, the shape in current model is torch.Size([256, 64]).
```

## Root Causes

### 1. **Model Architecture Mismatch**
The checkpoint was saved with a model that had different time conditioning architecture than what the evaluation script was trying to recreate:

- **Checkpoint model**: Had time conditioning features (random_fourier_features or learned_sinusoidal_cond)
- **Evaluation model**: Was created without these features, leading to different layer structures

### 2. **Missing Reference Batch**
The evaluation required a reference batch (`VIRTUAL_oxford_flowers256.npz`) that didn't exist, and the image size needed to match the training configuration.

### 3. **Separate Wandb Logging**
Training and evaluation were creating separate wandb runs instead of logging to the same experiment.

## Solutions Implemented

### 1. **Fixed Model Recreation Logic**
Modified `evaluate_checkpoint.py` to properly recreate the model based on saved configuration:

```python
# Add time conditioning options if they exist
if hasattr(saved_config, 'random_fourier_features') and saved_config.random_fourier_features:
    model_kwargs['random_fourier_features'] = True
elif hasattr(saved_config, 'learned_sinusoidal_cond') and saved_config.learned_sinusoidal_cond:
    model_kwargs['learned_sinusoidal_cond'] = True

# Add mean_variance_net if it exists
if hasattr(saved_config, 'mean_variance_net'):
    model_kwargs['mean_variance_net'] = saved_config.mean_variance_net
```

### 2. **Automatic Reference Batch Creation**
Added `ensure_reference_batch_exists()` function that:

- Checks if reference batch exists
- If not, automatically creates it using `create_oxford_reference.py`
- Uses the correct image size from the checkpoint configuration
- Ensures evaluation uses the right reference statistics

### 3. **Unified Wandb Logging**
Modified the evaluation to reuse the existing wandb run from training instead of creating a new one.

## Key Changes Made

### `evaluate_checkpoint.py`
- Added `ensure_reference_batch_exists()` function
- Enhanced model recreation logic to handle time conditioning options
- Modified main() to create reference batch with correct image size
- Updated evaluation to use the dynamically created reference batch

### Training Scripts
- Ensured configuration is saved in checkpoints
- Wandb logging is properly initialized for both training and evaluation

## Usage

The evaluation now works automatically:

```bash
# Training creates checkpoints and initializes wandb
python compare_mean_flow_options.py --use_wandb --wandb_run_name "exp_1"

# Evaluation automatically:
# 1. Recreates the exact same model architecture
# 2. Creates reference batch if needed (with correct image size)
# 3. Logs results to the same wandb run
python evaluate_checkpoint.py --checkpoint checkpoints/exp_1/checkpoint.70000.pt
```

## Benefits

1. **No Manual Setup**: Reference batches are created automatically
2. **Architecture Consistency**: Models are recreated exactly as trained
3. **Unified Logging**: Training and evaluation metrics in same wandb experiment
4. **Flexible**: Works with any image size and model configuration
5. **Robust**: Handles missing configurations gracefully

## Future Improvements

- Add validation that reference batch image size matches checkpoint
- Support for different reference datasets
- Batch evaluation of multiple checkpoints
- Integration with training pipeline for automatic evaluation</content>
<parameter name="filePath">d:\code\rectified-flow-pytorch\EVALUATION_FIXES.md