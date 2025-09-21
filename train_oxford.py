import torch
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

    # Dataset parameters
    image_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True

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
    sample_during_training: bool = True  # set False to avoid multi-GPU stalls
    log_every: int = 10  # wandb / pbar update frequency
    compute_norms_every: int = 0  # 0 disables grad/param norm computation

    # Wandb parameters
    use_wandb: bool = False
    wandb_project: str = "rectified-flow-oxford"
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


def main(config: Config):
    flowers_dataset = OxfordFlowersDataset(image_size=config.image_size)

    # models and trainer

    from rectified_flow_pytorch import RectifiedFlow, Unet, Trainer

    model = Unet(dim=config.dim, mean_variance_net=config.mean_variance_net)

    rectified_flow = RectifiedFlow(model)

    trainer = Trainer(
        rectified_flow,
        dataset=flowers_dataset,
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
    )

    trainer()


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
