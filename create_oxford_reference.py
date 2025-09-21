#!/usr/bin/env python3
"""
Create a reference batch for Oxford Flowers dataset evaluation.

This script creates a proper reference batch from the Oxford Flowers dataset
that can be used for FID evaluation of generated samples.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import argparse

# Add the guided-diffusion evaluations path to sys.path
sys.path.append('third_party/guided-diffusion/evaluations')

def create_oxford_flowers_reference_batch(
    output_path: str = 'third_party/guided-diffusion/evaluations/VIRTUAL_oxford_flowers256.npz',
    num_samples: int = 10000,
    image_size: int = 256,
    batch_size: int = 50
):
    """
    Create a reference batch from Oxford Flowers dataset.

    Args:
        output_path: Where to save the reference batch
        num_samples: Number of reference images to use
        image_size: Image size for processing
        batch_size: Batch size for processing
    """
    print(f"Creating Oxford Flowers reference batch with {num_samples} samples...")

    # Import required modules
    try:
        from datasets import load_dataset
        import torchvision.transforms as T
        from torch.utils.data import DataLoader, Dataset
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Install with: pip install datasets torch torchvision")
        return

    # Import evaluation components
    try:
        from evaluator import Evaluator
        import tensorflow as tf
    except ImportError as e:
        print(f"Missing TensorFlow/evaluation dependencies: {e}")
        print("Install guided-diffusion requirements first")
        return

    # Create dataset class
    class OxfordFlowersDataset(Dataset):
        def __init__(self, image_size, split='train'):
            self.ds = list(load_dataset('nelorth/oxford-flowers')[split])
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor()
            ])

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            pil = self.ds[idx]['image']
            tensor = self.transform(pil)
            return tensor

    # Load dataset
    print("Loading Oxford Flowers dataset...")
    dataset = OxfordFlowersDataset(image_size=image_size)

    # Limit to num_samples if dataset is larger
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Processing {len(dataset)} images...")

    # Set up TensorFlow session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        evaluator = Evaluator(sess)

        # Warm up
        print("Warming up TensorFlow...")
        evaluator.warmup()

        # Collect all images for reference batch
        all_images = []
        all_activations = []

        for batch in tqdm(dataloader, desc="Processing images"):
            # Convert to numpy and proper format
            batch_np = batch.numpy()  # [B, C, H, W]
            batch_np = np.transpose(batch_np, (0, 2, 3, 1))  # [B, H, W, C]
            batch_np = (batch_np * 255).astype(np.uint8)  # Convert to 0-255 range

            all_images.append(batch_np)

            # Get activations
            activations = evaluator.compute_activations([batch_np])
            all_activations.append(activations)

        # Concatenate all data
        ref_images = np.concatenate(all_images, axis=0)
        ref_activations = (
            np.concatenate([act[0] for act in all_activations], axis=0),
            np.concatenate([act[1] for act in all_activations], axis=0)
        )

        # Compute statistics
        print("Computing reference statistics...")
        ref_stats = evaluator.compute_statistics(ref_activations[0])
        ref_stats_spatial = evaluator.compute_statistics(ref_activations[1])

        # Save reference batch
        print(f"Saving reference batch to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.savez(
            output_path,
            arr_0=ref_images,  # Raw images
            mu=ref_stats.mu,   # Pool statistics
            sigma=ref_stats.sigma,
            mu_s=ref_stats_spatial.mu,  # Spatial statistics
            sigma_s=ref_stats_spatial.sigma
        )

        print("Reference batch created successfully!")
        print(f"Images shape: {ref_images.shape}")
        print(f"Pool stats - mu: {ref_stats.mu.shape}, sigma: {ref_stats.sigma.shape}")
        print(f"Spatial stats - mu: {ref_stats_spatial.mu.shape}, sigma: {ref_stats_spatial.sigma.shape}")

def main():
    parser = argparse.ArgumentParser(description='Create Oxford Flowers reference batch')
    parser.add_argument('--output', type=str,
                       default='third_party/guided-diffusion/evaluations/VIRTUAL_oxford_flowers256.npz',
                       help='Output path for reference batch')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of reference samples')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for processing')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for processing')

    args = parser.parse_args()

    create_oxford_flowers_reference_batch(
        output_path=args.output,
        num_samples=args.num_samples,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
