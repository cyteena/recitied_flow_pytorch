# Oxford Flowers Reference Batch

This document explains the Oxford Flowers reference batch for evaluation.

## Overview

The Oxford Flowers dataset (102 categories, ~8,000 images) doesn't have a pre-computed reference batch in the guided-diffusion repository. To get accurate FID scores for Oxford Flowers evaluation, we've created a custom reference batch using the actual dataset.

## Reference Batch Creation

The `create_oxford_reference.py` script:

1. **Loads the full Oxford Flowers dataset** using Hugging Face datasets
2. **Samples 10,000 representative images** (standard for reference batches)
3. **Processes through InceptionV3** to extract features
4. **Computes statistics** (mean, covariance) for FID calculation
5. **Saves as .npz file** compatible with guided-diffusion evaluator

## Usage

### Create the Reference Batch

```bash
# Install dependencies first
pip install datasets torch torchvision tensorflow-gpu scipy requests tqdm

# Create the reference batch
python create_oxford_reference.py
```

This will create `third_party/guided-diffusion/evaluations/VIRTUAL_oxford_flowers256.npz`

### Custom Parameters

```bash
python create_oxford_reference.py \
    --num_samples 10000 \
    --image_size 256 \
    --batch_size 50 \
    --output custom_reference.npz
```

## Why Oxford Flowers Reference?

**Before**: Using LSUN bedroom as proxy
- Different domain (bedrooms vs flowers)
- Inaccurate FID scores
- Not representative of the target distribution

**After**: Using actual Oxford Flowers data
- Same domain and distribution
- Accurate FID scores
- Proper evaluation of flower generation quality

## File Format

The reference batch contains:
- `arr_0`: Raw images (10,000 × 256 × 256 × 3, uint8)
- `mu`: Pool feature means (2048,)
- `sigma`: Pool feature covariance (2048 × 2048)
- `mu_s`: Spatial feature means (784,)
- `sigma_s`: Spatial feature covariance (784 × 784)

## Integration

The evaluation system automatically uses this reference batch for Oxford Flowers training. All existing commands work without changes:

```bash
# Training with evaluation
python train_oxford.py --use_wandb

# Standalone evaluation
python evaluate_checkpoint.py --checkpoint checkpoint.50000.pt
```
