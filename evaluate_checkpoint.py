def ensure_reference_batch_exists(
    reference_path: str,
    image_size: int,
    num_samples: int = 10000
) -> str:
    """
    Ensure that the reference batch exists, creating it if necessary.
    
    Args:
        reference_path: Path where reference batch should be
        image_size: Image size to use for reference batch
        num_samples: Number of reference samples
        
    Returns:
        Path to the reference batch (may be different if created with different size)
    """
    import os
    from pathlib import Path
    
    # Check if reference batch exists
    if os.path.exists(reference_path):
        print(f"Reference batch already exists: {reference_path}")
        return reference_path
    
    print(f"Reference batch not found: {reference_path}")
    print(f"Creating new reference batch with image_size={image_size}")
    
    # Create reference batch
    try:
        # Import the creation function
        from create_oxford_reference import create_oxford_flowers_reference_batch
        
        # Create with the specified image size
        create_oxford_flowers_reference_batch(
            output_path=reference_path,
            num_samples=num_samples,
            image_size=image_size,
            batch_size=50
        )
        
        print(f"Successfully created reference batch: {reference_path}")
        return reference_path
        
    except ImportError as e:
        print(f"Failed to import reference creation module: {e}")
        raise
    except Exception as e:
        print(f"Failed to create reference batch: {e}")
        raise

def sample_from_checkpoint(
    checkpoint_path: str,
    num_samples: int = 5000,
    batch_size: int = 50,
    image_size: int = 64,
    temperature: float = 1.5,
    output_path: Optional[str] = None,
    device: str = 'auto'
) -> str:
    """
    Sample images from a checkpoint and save as .npz file for evaluation.

    Args:
        checkpoint_path: Path to the checkpoint file
        num_samples: Number of samples to generate
        batch_size: Batch size for sampling
        image_size: Image size (assumed square)
        temperature: Sampling temperature
        output_path: Where to save the .npz file
        device: Device to use ('auto', 'cpu', 'cuda')

    Returns:
        Path to the saved .npz file
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get saved configuration
    saved_config = checkpoint.get('config', None)
    
    # Import here to avoid circular imports
    from rectified_flow_pytorch import RectifiedFlow, MeanFlow, Unet

    # Create model based on saved configuration
    if saved_config is not None:
        # Recreate model with exact same configuration as training
        model_kwargs = {
            'dim': getattr(saved_config, 'dim', 64),
            'dim_cond': getattr(saved_config, 'dim_cond', 1),
            'accept_cond': True
        }
        
        # Add time conditioning options if they exist
        if hasattr(saved_config, 'random_fourier_features') and saved_config.random_fourier_features:
            model_kwargs['random_fourier_features'] = True
        elif hasattr(saved_config, 'learned_sinusoidal_cond') and saved_config.learned_sinusoidal_cond:
            model_kwargs['learned_sinusoidal_cond'] = True
            
        # Add mean_variance_net if it exists
        if hasattr(saved_config, 'mean_variance_net'):
            model_kwargs['mean_variance_net'] = saved_config.mean_variance_net
            
        model = Unet(**model_kwargs)
        
        # Determine if this is MeanFlow or RectifiedFlow
        if hasattr(saved_config, 'use_logit_normal_sampler'):
            # This is a MeanFlow checkpoint
            flow_model = MeanFlow(
                model,
                normalize_data_fn=lambda t: t * 2. - 1.,
                unnormalize_data_fn=lambda t: (t + 1.) / 2.,
                use_logit_normal_sampler=getattr(saved_config, 'use_logit_normal_sampler', True),
                logit_normal_mean=getattr(saved_config, 'logit_normal_mean', -0.4),
                logit_normal_std=getattr(saved_config, 'logit_normal_std', 1.0),
                adaptive_loss_weight_p=getattr(saved_config, 'adaptive_loss_weight_p', 0.5),
                prob_default_flow_obj=getattr(saved_config, 'prob_default_flow_obj', 0.5)
            )
        else:
            # This is a RectifiedFlow checkpoint
            flow_model = RectifiedFlow(model)
            
        # Override image_size from saved config if available
        if hasattr(saved_config, 'image_size'):
            image_size = saved_config.image_size
            
    else:
        # Fallback: try to create basic models (legacy support)
        print("Warning: No configuration found in checkpoint, using default models")
        from train_oxford import Config
        config = Config()
        config.image_size = image_size
        
        model = Unet(dim=config.dim, mean_variance_net=config.mean_variance_net)
        flow_model = RectifiedFlow(model)

    # Load state dict
    flow_model.load_state_dict(checkpoint['model'])
    flow_model.to(device)
    flow_model.eval()

    print(f"Model type: {type(flow_model).__name__}")
    print(f"Generating {num_samples} samples with batch size {batch_size}")

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(all_samples))
            print(f"Sampling batch {i+1}/{num_batches} ({current_batch_size} samples)")

            # Prepare sampling kwargs based on model type
            sample_kwargs = {
                'batch_size': current_batch_size,
                'data_shape': (3, image_size, image_size),
            }
            
            # Only add temperature for RectifiedFlow models
            if isinstance(flow_model, RectifiedFlow):
                sample_kwargs['temperature'] = temperature

            samples = flow_model.sample(**sample_kwargs)

            all_samples.append(samples)

    # Concatenate all samples
    samples_tensor = torch.cat(all_samples, dim=0)

    # Convert to numpy in the correct format for evaluation
    # From [-1, 1] to [0, 255], and NCHW to NHWC
    samples_np = ((samples_tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    samples_np = samples_np.permute(0, 2, 3, 1).cpu().numpy()

    # Save as .npz
    if output_path is None:
        checkpoint_name = Path(checkpoint_path).stem
        output_path = f"samples_{checkpoint_name}.npz"

    np.savez(output_path, arr_0=samples_np)
    print(f"Saved {len(samples_np)} samples to {output_path}")

    return output_path

def run_evaluation(
    sample_npz_path: str,
    reference_npz_path: str,
    eval_script_path: str = "third_party/guided-diffusion/evaluations/evaluator.py"
) -> dict:
    """
    Run evaluation using guided-diffusion evaluator.

    Args:
        sample_npz_path: Path to generated samples .npz
        reference_npz_path: Path to reference batch .npz
        eval_script_path: Path to evaluator.py script

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Running evaluation: {eval_script_path} {reference_npz_path} {sample_npz_path}")

    # Run the evaluation script
    result = subprocess.run([
        sys.executable, eval_script_path, reference_npz_path, sample_npz_path
    ], capture_output=True, text=True, cwd=os.getcwd())

    if result.returncode != 0:
        print("Evaluation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("Evaluation script failed")

    # Parse the output
    output = result.stdout
    print("Evaluation output:")
    print(output)

    metrics = {}

    # Parse metrics from output
    lines = output.strip().split('\n')
    for line in lines:
        if 'Inception Score:' in line:
            metrics['inception_score'] = float(line.split(':')[1].strip())
        elif 'FID:' in line:
            metrics['fid'] = float(line.split(':')[1].strip())
        elif 'sFID:' in line:
            metrics['sfid'] = float(line.split(':')[1].strip())
        elif 'Precision:' in line:
            metrics['precision'] = float(line.split(':')[1].strip())
        elif 'Recall:' in line:
            metrics['recall'] = float(line.split(':')[1].strip())

    return metrics

def log_to_wandb(metrics: dict, step: int, checkpoint_path: str):
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of metrics
        step: Training step
        checkpoint_path: Path to checkpoint for reference
    """
    try:
        if not wandb.run:
            print("Wandb not initialized, skipping logging")
            return

        # Add step and checkpoint info
        log_data = {
            'eval_step': step,
            'checkpoint': str(checkpoint_path),
            **metrics
        }

        wandb.log(log_data, step=step)
        print(f"Logged evaluation metrics to wandb: {metrics}")
    except Exception as e:
        print(f"Wandb logging failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sample from checkpoint and evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for sampling')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size (square)')
    parser.add_argument('--temperature', type=float, default=1.5,
                       help='Sampling temperature')
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
                       help='Wandb project name (if not already initialized)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize wandb if needed
    try:
        if args.wandb_project and not wandb.run:
            wandb.init(project=args.wandb_project, name=f"eval_{Path(args.checkpoint).stem}")
        elif not wandb.run:
            # Check if wandb is already initialized (e.g., by training script)
            if wandb.run is None:
                print("Wandb not initialized and no project specified, skipping wandb logging")
    except Exception as e:
        print(f"Wandb initialization check failed: {e}, skipping wandb logging")

    # Sample from checkpoint (this will determine the correct image_size)
    sample_path = sample_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        temperature=args.temperature,
        output_path=str(output_dir / f"samples_{Path(args.checkpoint).stem}.npz"),
        device=args.device
    )

    # Now ensure reference batch exists with the correct image size
    # We need to get the image_size from the checkpoint config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_config = checkpoint.get('config', None)
    reference_image_size = getattr(saved_config, 'image_size', args.image_size) if saved_config else args.image_size
    
    reference_path = ensure_reference_batch_exists(
        reference_path=args.reference_batch,
        image_size=reference_image_size,
        num_samples=10000
    )

    # Run evaluation
    try:
        metrics = run_evaluation(
            sample_npz_path=sample_path,
            reference_npz_path=reference_path,
            eval_script_path=args.eval_script
        )

        print("Evaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Log to wandb
        step = args.step
        if step is None and wandb.run:
            # Try to infer step from checkpoint name
            checkpoint_name = Path(args.checkpoint).stem
            if 'checkpoint.' in checkpoint_name:
                try:
                    step = int(checkpoint_name.split('checkpoint.')[-1])
                except ValueError:
                    step = 0

        if step is not None:
            log_to_wandb(metrics, step, args.checkpoint)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        # Still log that evaluation failed
        try:
            if wandb.run:
                wandb.log({
                    'eval_step': step or 0,
                    'checkpoint': str(args.checkpoint),
                    'eval_failed': True,
                    'error': str(e)
                }, step=step)
        except Exception as wandb_error:
            print(f"Wandb error logging failed: {wandb_error}")

if __name__ == '__main__':
    main()
