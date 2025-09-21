# Weights & Biases (wandb) Integration for Training

## Overview

Added comprehensive wandb logging and progress bar support to the `Trainer` class to track training metrics, hyperparameters, and experiments.

## Features Added

### 1. Wandb Configuration Parameters

Added the following parameters to `Trainer.__init__()`:

- `use_wandb` (bool): Enable/disable wandb logging (default: False)
- `wandb_project` (str): Wandb project name (default: 'rectified-flow')
- `wandb_run_name` (str): Custom run name (default: None)
- `wandb_kwargs` (dict): Additional wandb.init() arguments

### 2. Automatic Wandb Initialization

The `init_wandb()` method handles wandb initialization:

```python
def init_wandb(self):
    """Initialize Weights & Biases logging if enabled."""
    if self.is_main and self.use_wandb:
        import wandb
        wandb.init(project=self.wandb_project, name=self.wandb_run_name, **self.wandb_kwargs)
```

This method is called automatically in `__init__` when wandb is enabled.

### 3. Metrics Logging

Logs the following metrics at each training step:

- **loss**: Training loss value
- **learning_rate**: Current learning rate from optimizer
- **grad_norm**: Gradient norm (computed before clipping)
- **param_norm**: Parameter norm of the model
- **loss_breakdown**: If available (total, main, data_match, velocity_match)

### 4. Multi-GPU Compatibility

- Wandb logging only occurs on the main process (`self.is_main`)
- Uses `self.accelerator.unwrap_model(self.model)` for parameter access in DDP

### 5. Training Progress Bar

Added a tqdm-based progress bar that displays training progress with real-time metrics:

- **Progress**: Current step / total steps
- **Loss**: Current training loss value
- **Learning Rate**: Current optimizer learning rate
- **Grad Norm**: L2 norm of gradients (before clipping)
- **Param Norm**: L2 norm of model parameters

The progress bar only appears on the main process in multi-GPU setups to avoid duplicate displays.

## Usage Example

### Basic Usage

```python
trainer = Trainer(
    rectified_flow,
    dataset=flowers_dataset,
    use_wandb=True,
    wandb_project='my-rectified-flow-project',
    wandb_run_name='experiment-1'
)
```

### Advanced Configuration

```python
trainer = Trainer(
    rectified_flow,
    dataset=flowers_dataset,
    use_wandb=True,
    wandb_project='rectified-flow-oxford',
    wandb_run_name='oxford-flowers-training',
    wandb_kwargs={
        'entity': 'my-entity',
        'config': {
            'learning_rate': 3e-4,
            'batch_size': 16,
            'model': 'Unet'
        },
        'tags': ['oxford-flowers', 'rectified-flow']
    }
)
```

## Installation

Ensure wandb is installed:

```bash
pip install wandb
```

Login to wandb:

```bash
wandb login
```

## Logged Metrics

The following metrics are automatically logged:

1. **loss**: The main training loss
2. **learning_rate**: Current optimizer learning rate
3. **grad_norm**: L2 norm of gradients (before clipping)
4. **param_norm**: L2 norm of model parameters
5. **Loss Breakdown** (if using RectifiedFlow):
   - total_loss
   - main_loss
   - data_match_loss
   - velocity_match_loss

## Benefits

- **Progress Monitoring**: Real-time progress bar with current loss, learning rate, gradient norm, and parameter norm
- **Experiment Tracking**: Compare different runs, hyperparameters, and architectures
- **Real-time Monitoring**: Track training progress and detect issues early
- **Metric Visualization**: Automatic plots for loss curves, learning rate schedules, etc.
- **Reproducibility**: Log all relevant metrics and configurations
- **Multi-GPU Support**: Properly handles distributed training scenarios

## Notes

- Wandb logging is disabled by default to avoid requiring wandb installation
- Progress bar uses tqdm and shows on main process only in multi-GPU setups
- Only the main process logs to avoid duplicate entries in multi-GPU setups
- All computations use the unwrapped model for accurate parameter/gradient norms
- Compatible with both single-GPU and multi-GPU training
