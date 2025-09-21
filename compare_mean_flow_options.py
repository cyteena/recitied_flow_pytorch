#!/usr/bin/env python3
"""
Comparison script for MeanFlow training options:
- random_fourier_features vs learned_sinusoidal_cond
- warmup + lr_decay vs constant lr
- use_logit_normal_sampler vs uniform sampling
"""

import tyro
from dataclasses import dataclass
from typing import Optional

import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset


@dataclass
class Config:
    # Model parameters
    dim: int = 64
    dim_cond: int = 1
    
    # Time conditioning options
    random_fourier_features: bool = False
    learned_sinusoidal_cond: bool = False
    
    # MeanFlow parameters
    use_logit_normal_sampler: bool = True
    logit_normal_mean: float = -0.4
    logit_normal_std: float = 1.0
    adaptive_loss_weight_p: float = 0.5
    prob_default_flow_obj: float = 0.5
    
    # Training parameters
    num_train_steps: int = 70_000
    learning_rate: float = 1e-4
    batch_size: int = 64  # Increased for better GPU utilization
    max_grad_norm: float = 0.5
    
    # Learning rate scheduling
    use_warmup: bool = False
    warmup_steps: int = 1000
    use_lr_decay: bool = False
    lr_decay_steps: int = 35000  # Half of training steps
    lr_decay_factor: float = 0.1
    
    # Dataset parameters
    image_size: int = 128  # Increased for more computation
    num_workers: int = 8
    prefetch_factor: int = 8
    persistent_workers: bool = True
    
    # Sampling parameters
    sample_temperature: float = 1.5
    num_samples: int = 16
    save_results_every: int = 5000
    checkpoint_every: int = 10000
    sample_during_training: bool = False
    log_every: int = 100
    compute_norms_every: int = 100
    
    # Wandb parameters
    use_wandb: bool = True
    wandb_project: str = "mean-flow-comparison"
    wandb_run_name: Optional[str] = None
    
    # Folders
    results_folder: str = "./results"
    checkpoints_folder: str = "./checkpoints"
    
    # Evaluation parameters
    eval_every: int = 5000
    eval_num_samples: int = 5000
    eval_batch_size: int = 50


class OxfordFlowersDataset(Dataset):
    def __init__(self, image_size):
        # Load dataset once and keep reference, don't convert to list
        self.ds = load_dataset("nelorth/oxford-flowers")["train"]
        
        # Use more efficient transforms with caching
        self.transform = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]["image"]
        tensor = self.transform(pil)
        return tensor.float() / 255.0


def run_experiment(config: Config, exp_name: str):
    """Run a single MeanFlow experiment with the given config."""
    print(f"\n{'='*50}")
    print(f"Running MeanFlow experiment: {exp_name}")
    print(f"Config: random_fourier_features={config.random_fourier_features}, "
          f"learned_sinusoidal_cond={config.learned_sinusoidal_cond}, "
          f"use_logit_normal_sampler={config.use_logit_normal_sampler}, "
          f"use_warmup={config.use_warmup}, use_lr_decay={config.use_lr_decay}")
    print(f"{'='*50}")

    # Update config with experiment-specific paths
    config.wandb_run_name = exp_name

    # Import here to avoid circular imports
    from rectified_flow_pytorch import MeanFlow, Unet, Trainer

    # Create model with time conditioning options
    model_kwargs = {
        'dim': config.dim,
        'dim_cond': config.dim_cond,
        'accept_cond': True
    }
    
    # Add time conditioning based on config
    if config.random_fourier_features:
        model_kwargs['random_fourier_features'] = True
    elif config.learned_sinusoidal_cond:
        model_kwargs['learned_sinusoidal_cond'] = True
    
    model = Unet(**model_kwargs)

    # Create MeanFlow with sampling options
    mean_flow = MeanFlow(
        model,
        normalize_data_fn=lambda t: t * 2. - 1.,
        unnormalize_data_fn=lambda t: (t + 1.) / 2.,
        use_logit_normal_sampler=config.use_logit_normal_sampler,
        logit_normal_mean=config.logit_normal_mean,
        logit_normal_std=config.logit_normal_std,
        adaptive_loss_weight_p=config.adaptive_loss_weight_p,
        prob_default_flow_obj=config.prob_default_flow_obj
    )

    # Create trainer (use simpler parameter set like original train_mean_flow.py)
    trainer = Trainer(
        mean_flow,
        dataset=OxfordFlowersDataset(image_size=config.image_size),
        num_train_steps=config.num_train_steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        max_grad_norm=config.max_grad_norm,
        results_folder=config.results_folder,
        checkpoints_folder=config.checkpoints_folder,
        save_results_every=config.save_results_every,
        checkpoint_every=config.checkpoint_every,
        sample_during_training=config.sample_during_training,
        sample_temperature=config.sample_temperature,
        num_samples=config.num_samples,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_run_name=config.wandb_run_name,
        log_every=config.log_every,
        compute_norms_every=config.compute_norms_every,
        eval_every=config.eval_every,
        eval_num_samples=config.eval_num_samples,
        eval_batch_size=config.eval_batch_size,
        # Learning rate scheduling
        use_warmup=config.use_warmup,
        warmup_steps=config.warmup_steps,
        use_lr_decay=config.use_lr_decay,
        lr_decay_steps=config.lr_decay_steps,
        lr_decay_factor=config.lr_decay_factor,
    )

    trainer()


def main():
    # Parse config
    config = tyro.cli(Config)

    # Create experiment name based on config
    time_cond = "rff" if config.random_fourier_features else ("lsc" if config.learned_sinusoidal_cond else "none")
    sampler = "logit" if config.use_logit_normal_sampler else "uniform"
    sched = "sched" if config.use_warmup or config.use_lr_decay else "const"
    
    exp_name = f"meanflow_{time_cond}_{sampler}_{sched}"

    try:
        run_experiment(config, exp_name)
    except Exception as e:
        print(f"Experiment {exp_name} failed: {e}")
        raise


if __name__ == "__main__":
    main()