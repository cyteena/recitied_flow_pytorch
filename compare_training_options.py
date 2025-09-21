#!/usr/bin/env python3
"""
Comparison script for mean-variance-net, consistency_flow_matching, and warm_up + lr_decay
Runs 8 experiments with all combinations of the three options.
"""

import tyro
from dataclasses import dataclass
from typing import Optional

# hf datasets for easy oxford flowers training

import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset


@dataclass
class Config:
    # Model parameters
    dim: int = 64
    mean_variance_net: bool = False

    # Flow parameters
    use_consistency: bool = False

    # Training parameters
    num_train_steps: int = 10_000  # Shorter for comparison
    learning_rate: float = 3e-4
    batch_size: int = 16
    max_grad_norm: float = 0.5

    # Learning rate scheduling
    use_warmup: bool = False
    warmup_steps: int = 1000
    use_lr_decay: bool = False
    lr_decay_steps: int = 5000
    lr_decay_factor: float = 0.1

    # Dataset parameters
    image_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True

    # Sampling parameters
    sample_temperature: float = 1.5
    num_samples: int = 16
    save_results_every: int = 1000  # Less frequent for comparison
    checkpoint_every: int = 5000
    sample_during_training: bool = False
    log_every: int = 1
    compute_norms_every: int = 1

    # Wandb parameters
    use_wandb: bool = False
    wandb_project: str = "rectified-flow-comparison"
    wandb_run_name: Optional[str] = None

    # Folders
    results_folder: str = "./results"
    checkpoints_folder: str = "./checkpoints"

    # Evaluation parameters
    eval_every: int = 5000
    eval_num_samples: int = 5000
    eval_batch_size: int = 50
    eval_reference_batch: str = (
        "third_party/guided-diffusion/evaluations/VIRTUAL_oxford_flowers256.npz"
    )


class OxfordFlowersDataset(Dataset):
    def __init__(self, image_size):
        self.ds = list(load_dataset("nelorth/oxford-flowers")["train"])

        self.transform = T.Compose(
            [T.Resize((image_size, image_size)), T.PILToTensor()]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]["image"]
        tensor = self.transform(pil)
        return tensor / 255.0


def run_experiment(config: Config, exp_name: str):
    """Run a single experiment with the given config."""
    print(f"\n{'='*50}")
    print(f"Running experiment: {exp_name}")
    print(f"Config: mean_variance_net={config.mean_variance_net}, "
          f"use_consistency={config.use_consistency}, "
          f"use_warmup={config.use_warmup}, use_lr_decay={config.use_lr_decay}")
    print(f"{'='*50}")

    # Create experiment-specific folders
    exp_results = f"{config.results_folder}_{exp_name}"
    exp_checkpoints = f"{config.checkpoints_folder}_{exp_name}"

    # Update config with experiment-specific paths
    config.results_folder = exp_results
    config.checkpoints_folder = exp_checkpoints
    config.wandb_run_name = exp_name

    # Import here to avoid circular imports
    from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

    model = Unet(dim=config.dim, mean_variance_net=config.mean_variance_net)

    rectified_flow = RectifiedFlow(
        model,
        use_consistency=config.use_consistency
    )

    trainer = Trainer(
        rectified_flow,
        dataset=OxfordFlowersDataset(image_size=config.image_size),
        num_train_steps=config.num_train_steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        dataloader_prefetch_factor=config.prefetch_factor,
        dataloader_persistent_workers=config.persistent_workers,
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
        config=config,
        log_every=config.log_every,
        compute_norms_every=config.compute_norms_every,
        eval_every=config.eval_every,
        eval_num_samples=config.eval_num_samples,
        eval_batch_size=config.eval_batch_size,
        eval_reference_batch=config.eval_reference_batch,
        # Learning rate scheduling
        use_warmup=config.use_warmup,
        warmup_steps=config.warmup_steps,
        use_lr_decay=config.use_lr_decay,
        lr_decay_steps=config.lr_decay_steps,
        lr_decay_factor=config.lr_decay_factor,
    )

    trainer()


def main():
    # Parse base config
    base_config = tyro.cli(Config)

    # Define all 8 combinations (mean_var, consistency, warmup_decay)
    combinations = [
        (False, False, False),  # baseline: no mean_var, no consistency, no warmup_decay
        (False, False, True),   # warmup_decay only
        (False, True, False),   # consistency only
        (False, True, True),    # consistency + warmup_decay
        (True, False, False),   # mean_var only
        (True, False, True),    # mean_var + warmup_decay
        (True, True, False),    # mean_var + consistency
        (True, True, True),     # all enabled
    ]

    # Run all experiments
    for i, (mean_var, consistency, warmup_decay) in enumerate(combinations):
        config = base_config.__class__(
            **vars(base_config),
            mean_variance_net=mean_var,
            use_consistency=consistency,
            use_warmup=warmup_decay,
            use_lr_decay=warmup_decay,  # Both warmup and decay together
        )

        exp_name = f"exp_{i+1:02d}_mv{int(mean_var)}_cons{int(consistency)}_sched{int(warmup_decay)}"
        try:
            run_experiment(config, exp_name)
        except Exception as e:
            print(f"Experiment {exp_name} failed: {e}")
            continue

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
