#!/usr/bin/env python3
"""
Example usage of the evaluation system for Rectified Flow.

This script demonstrates how to:
1. Sample from a checkpoint
2. Run FID evaluation
3. Log results to wandb

Usage examples:

# Basic evaluation of a checkpoint
python evaluate_checkpoint.py --checkpoint checkpoints/checkpoint.5000.pt --num_samples 5000

# Evaluation with wandb logging
python evaluate_checkpoint.py --checkpoint checkpoints/checkpoint.5000.pt --num_samples 5000 --wandb_project my-project

# Custom reference batch
python evaluate_checkpoint.py --checkpoint checkpoints/checkpoint.5000.pt --reference_batch path/to/custom_reference.npz

# During training (automatic every 5k steps)
python train_oxford.py --use_wandb --wandb_project rectified-flow-oxford
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Rectified Flow checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples to generate (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for sampling (default: 50)')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size (default: 64)')
    parser.add_argument('--temperature', type=float, default=1.5,
                       help='Sampling temperature (default: 1.5)')
    parser.add_argument('--reference_batch', type=str,
                       default='third_party/guided-diffusion/evaluations/VIRTUAL_oxford_flowers256.npz',
                       help='Path to reference batch for evaluation')
    parser.add_argument('--eval_script', type=str,
                       default='third_party/guided-diffusion/evaluations/evaluator.py',
                       help='Path to evaluation script')
    parser.add_argument('--output_dir', type=str, default='./eval_samples',
                       help='Directory to save sample files')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--step', type=int, default=None,
                       help='Training step (for wandb logging)')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name')

    args = parser.parse_args()

    # Import and run evaluation
    from evaluate_checkpoint import main as eval_main

    # Replace sys.argv with our parsed args
    original_argv = sys.argv
    sys.argv = ['evaluate_checkpoint.py'] + [f'--{k.replace("_", "-")}={v}' for k, v in vars(args).items() if v is not None]

    try:
        eval_main()
    finally:
        sys.argv = original_argv

if __name__ == '__main__':
    main()
