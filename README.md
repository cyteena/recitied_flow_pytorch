# Rectified Flow PyTorch

A PyTorch implementation of Rectified Flow for generative modeling, with comprehensive evaluation and monitoring tools.

## Features

- 🚀 **Rectified Flow Implementation**: Clean, efficient PyTorch implementation
- 📊 **Automated Evaluation**: FID, Inception Score, Precision/Recall calculation every 5k steps
- 📈 **Wandb Integration**: Automatic logging of metrics, losses, and generated samples
- 🔧 **Evaluation Tools**: Standalone evaluation scripts for any checkpoint
- 🎯 **Oxford Flowers Dataset**: Pre-configured training on the Oxford 102 Flowers dataset

## Project Structure

```
.
├── rectified_flow_pytorch/          # Core implementation
│   ├── rectified_flow.py           # Main model and training logic
│   ├── __init__.py
│   └── ...
├── third_party/                    # External dependencies
│   └── guided-diffusion/           # OpenAI's evaluation tools
│       └── evaluations/            # FID calculation scripts
├── train_oxford.py                 # Training script for Oxford Flowers
├── evaluate_checkpoint.py          # Standalone evaluation script
├── example_evaluation.py           # Usage examples
├── EVALUATION_README.md            # Evaluation system documentation
├── GUIDED_DIFFUSION_EVALUATION_GUIDE.md  # Detailed evaluation guide
└── pyproject.toml                  # Project configuration
```

## Quick Start

### Training

```bash
# Train with automatic evaluation every 5k steps
python train_oxford.py \
    --use_wandb \
    --wandb_project rectified-flow-oxford \
    --num_train_steps 70000
```

### Evaluation

```bash
# Evaluate a trained checkpoint
python evaluate_checkpoint.py \
    --checkpoint checkpoints/checkpoint.50000.pt \
    --num_samples 5000 \
    --wandb_project rectified-flow-oxford
```

## Setup

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/your-repo/rectified-flow-pytorch.git
cd rectified-flow-pytorch
```

### 2. Install Dependencies

```bash
# Install guided-diffusion evaluation requirements
cd third_party/guided-diffusion/evaluations
pip install -r requirements.txt
cd ../../

# Install main dependencies
pip install torch torchvision wandb datasets
```

### 3. Download Reference Data

```bash
# Create Oxford Flowers reference batch for evaluation
python create_oxford_reference.py
```

## Key Components

### Core Implementation (`rectified_flow_pytorch/`)
- Rectified Flow model architecture
- Training loop with automatic evaluation
- Wandb logging integration
- Checkpoint management

### Evaluation System (`third_party/guided-diffusion/`)
- FID calculation using InceptionV3
- Precision/Recall metrics
- Inception Score computation
- Batch processing for efficiency

### Training Scripts
- `train_oxford.py`: Oxford Flowers dataset training
- Evaluation runs automatically every 5,000 steps
- Comprehensive logging and checkpointing

## Metrics

The system tracks:
- **Training**: Loss, learning rate, gradient norms
- **Evaluation**: FID, sFID, Precision, Recall, Inception Score
- **Samples**: Generated images logged to wandb

## Documentation

- `EVALUATION_README.md`: Evaluation system overview
- `GUIDED_DIFFUSION_EVALUATION_GUIDE.md`: Detailed evaluation guide
- `third_party/README.md`: Third-party dependencies documentation

## Contributing

1. The `third_party/` directory contains external dependencies as git submodules
2. Core implementation is in `rectified_flow_pytorch/`
3. Evaluation tools are wrappers around OpenAI's guided-diffusion

## License

See individual component licenses:
- Rectified Flow implementation: MIT
- Guided Diffusion (third_party): OpenAI license
