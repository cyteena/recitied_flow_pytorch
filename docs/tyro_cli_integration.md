# Tyro CLI Integration for Experiment Management

## Overview

Integrated Tyro CLI to provide a clean command-line interface for managing experiment configurations and hyperparameters.

## Features Added

### 1. Configuration Dataclass

Created a `Config` dataclass that centralizes all training parameters:

```python
@dataclass
class Config:
    # Model parameters
    dim: int = 64
    mean_variance_net: bool = True
    
    # Dataset parameters
    image_size: int = 64
    
    # Training parameters
    num_train_steps: int = 70_000
    learning_rate: float = 3e-4
    batch_size: int = 16
    max_grad_norm: float = 0.5
    
    # Sampling parameters
    sample_temperature: float = 1.5
    num_samples: int = 16
    save_results_every: int = 100
    checkpoint_every: int = 1000
    
    # Wandb parameters
    use_wandb: bool = False
    wandb_project: str = 'rectified-flow-oxford'
    wandb_run_name: Optional[str] = None
    
    # Folders
    results_folder: str = './results'
    checkpoints_folder: str = './checkpoints'
```

### 2. Automatic CLI Generation

Tyro automatically generates command-line arguments from the dataclass:

```python
if __name__ == '__main__':
    config = tyro.cli(Config)
    main(config)
```

### 3. Main Function Structure

Refactored the training logic into a `main(config: Config)` function for better organization.

## Usage Examples

### Basic Training

```bash
python train_oxford.py
```

### Custom Learning Rate

```bash
python train_oxford.py --learning_rate 1e-4
```

### Enable Wandb Logging

```bash
python train_oxford.py --use_wandb True --wandb_project my-experiment
```

### Custom Model Dimensions

```bash
python train_oxford.py --dim 128 --batch_size 8
```

### Full Configuration Override

```bash
python train_oxford.py \
    --dim 128 \
    --image_size 128 \
    --num_train_steps 100000 \
    --learning_rate 2e-4 \
    --batch_size 12 \
    --use_wandb True \
    --wandb_project rectified-flow-custom \
    --wandb_run_name large-model-test \
    --results_folder ./results_large \
    --checkpoints_folder ./checkpoints_large
```

## Installation

Install tyro:

```bash
pip install tyro
```

## Benefits

- **Clean CLI**: Automatically generated command-line interface
- **Type Safety**: Type hints ensure correct parameter types
- **Defaults**: Sensible defaults for all parameters
- **Documentation**: Self-documenting parameters via dataclass
- **Flexibility**: Easy to modify or extend parameters
- **Reproducibility**: Command-line arguments are logged and can be reproduced

## Help System

Tyro provides automatic help:

```bash
python train_oxford.py --help
```

This shows all available parameters with their types and default values.

## Integration with Wandb

The CLI integrates seamlessly with wandb logging. Enable wandb and set project/run names directly from command line:

```bash
python train_oxford.py --use_wandb True --wandb_project my-project --wandb_run_name run-1
```

## Multi-GPU Support

All parameters work with multi-GPU training. Use accelerate as before:

```bash
accelerate launch --multi_gpu --num_processes 8 train_oxford.py --batch_size 32 --learning_rate 1e-4
```

## Notes

- All parameters have sensible defaults
- Type annotations ensure CLI validation
- Compatible with existing training workflows
- Easy to extend with new parameters by adding to the Config dataclass
