# Using Guided Diffusion Evaluation Tools for FID Calculation

This guide explains how to use the evaluation tools from the `guided-diffusion` repository to calculate FID (Fréchet Inception Distance) and other metrics for your Rectified Flow model samples.

## Overview

The `guided-diffusion` repository provides evaluation tools that compute:
- **FID**: Fréchet Inception Distance (primary metric for image quality)
- **sFID**: spatial FID (spatial features)
- **Precision**: measures how realistic generated samples are
- **Recall**: measures diversity of generated samples
- **Inception Score**: alternative quality metric

## Prerequisites

### 1. Install Dependencies

Navigate to the third_party/guided-diffusion evaluations directory and install requirements:

```bash
cd third_party/guided-diffusion/evaluations
pip install -r requirements.txt
```

Required packages:
- `tensorflow-gpu>=2.0` (or `tensorflow` for CPU-only)
- `scipy`
- `requests`
- `tqdm`

### 2. Prepare Your Generated Samples

You need to convert your Rectified Flow generated images to `.npz` format. The evaluator expects images in NHWC format (batch, height, width, channels) with values in [0, 255].

#### Option A: Convert from PyTorch tensors (recommended)

If you have samples as PyTorch tensors from your training script:

```python
import numpy as np
import torch
from torchvision.utils import save_image

# Assuming you have samples as torch tensor [N, C, H, W] in [-1, 1] range
samples = torch.randn(50000, 3, 256, 256)  # Replace with your actual samples

# Convert to [0, 255] range and NHWC format
samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
samples_np = samples.permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, C]

# Save as .npz
np.savez('my_samples.npz', arr_0=samples_np)
```

#### Option B: Convert from saved images

If you have images saved as PNG/JPG files:

```python
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def load_images_to_npz(image_dir, output_path, image_size=(256, 256)):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    
    for img_file in tqdm(image_files[:50000]):  # Limit to 50k samples
        img = Image.open(os.path.join(image_dir, img_file))
        img = img.resize(image_size, Image.LANCZOS)
        img = np.array(img)
        if img.shape[-1] == 4:  # Remove alpha channel if present
            img = img[..., :3]
        images.append(img)
    
    images_np = np.stack(images)  # [N, H, W, C]
    np.savez(output_path, arr_0=images_np)
    print(f"Saved {len(images)} images to {output_path}")

# Usage
load_images_to_npz('path/to/your/generated/images', 'my_samples.npz')
```

### 3. Get Reference Batch

Download the appropriate reference batch for your dataset. Reference batches contain pre-computed statistics for real images.

#### For Oxford Flowers (102 categories, 256x256):
```bash
# Download reference batch for Oxford Flowers
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz
# Note: Use bedroom as proxy since Oxford Flowers isn't directly available
```

#### For other datasets:
- **LSUN Bedroom**: `VIRTUAL_lsun_bedroom256.npz`
- **LSUN Cat**: `VIRTUAL_lsun_cat256.npz`
- **LSUN Horse**: `VIRTUAL_lsun_horse256.npz`
- **ImageNet 64x64**: `VIRTUAL_imagenet64_labeled.npz`
- **ImageNet 128x128**: `VIRTUAL_imagenet128_labeled.npz`
- **ImageNet 256x256**: `VIRTUAL_imagenet256_labeled.npz`
- **ImageNet 512x512**: `VIRTUAL_imagenet512.npz`

Download from: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/

## Running Evaluation

### Basic Usage

```bash
cd third_party/guided-diffusion/evaluations
python evaluator.py path/to/reference_batch.npz path/to/your_samples.npz
```

### Example for Oxford Flowers

```bash
# Assuming you downloaded the bedroom reference as proxy
python evaluator.py VIRTUAL_lsun_bedroom256.npz my_samples.npz
```

### Expected Output

```
warming up TensorFlow...
computing reference batch activations...
computing/reading reference batch statistics...
computing sample batch activations...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 215.8370361328125
FID: 3.9425574129223264
sFID: 6.140433703346162
Precision: 0.8265
Recall: 0.5309
```

## Understanding the Metrics

### FID (Fréchet Inception Distance)
- **Lower is better**: Measures similarity between real and generated image distributions
- Typical ranges:
  - Excellent: < 5
  - Good: 5-15
  - Poor: > 15
- Compares InceptionV3 features of real vs generated images

### sFID (spatial FID)
- Similar to FID but uses spatial features
- More sensitive to spatial structure

### Precision and Recall
- **Precision**: How realistic your samples are (higher = more realistic)
- **Recall**: How diverse your samples are (higher = more diverse)
- Ideal balance is both high (close to 1.0)
- Trade-off: high precision often means low recall

### Inception Score
- Alternative quality metric
- Higher scores indicate better quality and diversity

## Integration with Your Training Script

### Option 1: Save samples during training

Modify your training script to save samples periodically:

```python
# In your Trainer class, after sampling
if divisible_by(step, save_results_every):
    sampled = self.sample(fname=str(self.results_folder / f'results.{step}.png'))
    
    # Also save as .npz for evaluation
    sampled_np = ((sampled + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sampled_np = sampled_np.permute(0, 2, 3, 1).cpu().numpy()
    np.savez(str(self.results_folder / f'samples.{step}.npz'), arr_0=sampled_np)
```

### Option 2: Post-training evaluation script

Create a separate script to generate and evaluate samples:

```python
import torch
import numpy as np
from rectified_flow_pytorch import RectifiedFlow

# Load your trained model
model = RectifiedFlow(...)
model.load_state_dict(torch.load('path/to/checkpoint.pt'))
model.eval()

# Generate samples
with torch.no_grad():
    samples = model.sample(batch_size=50000, data_shape=(3, 256, 256))

# Convert and save
samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
samples_np = samples.permute(0, 2, 3, 1).cpu().numpy()
np.savez('evaluation_samples.npz', arr_0=samples_np)
```

## Troubleshooting

### Memory Issues
- Reduce batch size in evaluator if you get OOM errors
- Use CPU-only TensorFlow if GPU memory is limited: `pip install tensorflow` instead of `tensorflow-gpu`

### Sample Size
- Use at least 10,000 samples for reliable metrics
- 50,000 samples is the standard for publication-quality results

### Image Format Issues
- Ensure images are in [0, 255] range
- Use NHWC format: [batch, height, width, channels]
- RGB channels only (no alpha)

### Reference Batch Selection
- Choose reference batch that matches your dataset domain
- For custom datasets, you'll need to create your own reference batch

## Advanced Usage

### Custom Reference Batch Creation

If you need to create a reference batch for your own dataset:

```python
import numpy as np
from evaluator import Evaluator
import tensorflow as tf

# Load your real images as [N, H, W, C] in [0, 255]
real_images = np.load('your_real_images.npz')['arr_0']

# Create evaluator
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
evaluator = Evaluator(tf.Session(config=config))

# Compute statistics
ref_acts = evaluator.compute_activations([real_images])
ref_stats, ref_stats_spatial = [evaluator.compute_statistics(x) for x in ref_acts]

# Save reference batch
np.savez('custom_reference.npz', 
         arr_0=real_images,
         mu=ref_stats.mu,
         sigma=ref_stats.sigma,
         mu_s=ref_stats_spatial.mu,
         sigma_s=ref_stats_spatial.sigma)
```

This guide should help you evaluate your Rectified Flow model's performance using industry-standard metrics!</content>
<parameter name="filePath">d:\code\rectified-flow-pytorch\GUIDED_DIFFUSION_EVALUATION_GUIDE.md
