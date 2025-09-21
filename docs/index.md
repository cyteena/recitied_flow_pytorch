# Rectified Flow PyTorch - Documentation Index

This directory contains comprehensive documentation for the rectified-flow-pytorch project, covering bug fixes, feature implementations, and technical explanations.

## Multi-GPU Training Fixes

- **[ddp_deadlock_fix.md](ddp_deadlock_fix.md)** - Fixes for DDP (Distributed Data Parallel) communication deadlocks during training
- **[ddp_fix.md](ddp_fix.md)** - Additional DDP-related fixes and improvements
- **[multi_gpu_bug_analysis.md](multi_gpu_bug_analysis.md)** - Detailed analysis of multi-GPU training issues and their solutions
- **[nccl_timeout_fix.md](nccl_timeout_fix.md)** - Fixes for NCCL collective operation timeouts in distributed training
- **[sampling_deadlock_fix.md](sampling_deadlock_fix.md)** - Fixes for deadlocks during the sampling phase in distributed training

## Feature Implementations

- **[wandb_integration.md](wandb_integration.md)** - Integration of Weights & Biases for experiment tracking and logging
- **[tyro_cli_integration.md](tyro_cli_integration.md)** - Command-line interface implementation using Tyro for argument management

## Technical Explanations

- **[mean_variance_net_explanation.md](mean_variance_net_explanation.md)** - Detailed explanation of the mean-variance network architecture and its role in rectified flow
- **[loss_breakdown_explanation.md](loss_breakdown_explanation.md)** - Comprehensive explanation of the loss breakdown components in Rectified Flow training

## Quick Reference

### Training Scripts
- `train_oxford.py` - Main training script with Oxford Flowers dataset
- `train_mean_flow.py` - Training script for mean flow models
- `train_mean_flow_ql.py` - Mean flow with Q-learning
- `train_fpo.py` - Training with FPO (Flow Policy Optimization)
- `train_nano_rf.py` - Nano rectified flow training
- `train_split_mean_flow.py` - Split mean flow training

### Core Modules
- `rectified_flow_pytorch/rectified_flow.py` - Main rectified flow implementation
- `rectified_flow_pytorch/mean_flow.py` - Mean flow implementation
- `rectified_flow_pytorch/nano_flow.py` - Nano flow implementation
- `rectified_flow_pytorch/reflow.py` - Reflow implementation
- `rectified_flow_pytorch/split_mean_flow.py` - Split mean flow implementation

## Getting Started

1. **Installation**: `pip install rectified-flow-pytorch`
2. **Basic Usage**: See the main [README.md](../README.md) in the project root
3. **Multi-GPU Training**: Refer to the DDP fix documents for distributed training setup
4. **Experiment Tracking**: Check [wandb_integration.md](wandb_integration.md) for logging setup
5. **CLI Management**: See [tyro_cli_integration.md](tyro_cli_integration.md) for command-line usage

## Recent Updates

- ✅ Fixed all multi-GPU training deadlocks
- ✅ Fixed NCCL collective operation timeouts
- ✅ Added Weights & Biases integration
- ✅ Implemented Tyro CLI for argument management
- ✅ Added comprehensive documentation
- ✅ Organized all docs in this directory

For questions or issues, please refer to the specific documentation files or check the main project README.
