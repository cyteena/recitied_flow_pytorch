# Rectified Flow Evaluation System

This directory contains an automated evaluation system for Rectified Flow models that calculates FID and other metrics using the guided-diffusion evaluation tools.

## Features

- **Automatic Evaluation**: Runs evaluation every 5,000 training steps
- **FID Calculation**: Computes Fréchet Inception Distance and other metrics
- **Wandb Integration**: Logs all metrics to Weights & Biases
- **Standalone Evaluation**: Can evaluate any checkpoint independently
- **Batch Processing**: Efficiently samples 5,000 images for evaluation

## Quick Start

### 1. Install Dependencies

```bash
# Install guided-diffusion evaluation requirements
cd third_party/guided-diffusion/evaluations
pip install -r requirements.txt
cd ../..

# Make sure you have wandb installed (optional)
pip install wandb
```

### 2. Download Reference Batch

For Oxford Flowers evaluation, download the LSUN bedroom reference batch as a proxy:

```bash
# Download to third_party/guided-diffusion/evaluations/
python create_oxford_reference.py
```

### 3. Training with Automatic Evaluation

```bash
# Train with automatic evaluation every 5k steps
python train_oxford.py \
    --use_wandb \
    --wandb_project rectified-flow-oxford \
    --eval_every 5000 \
    --eval_num_samples 5000
```

### 4. Standalone Evaluation

```bash
# Evaluate a specific checkpoint
python evaluate_checkpoint.py \
    --checkpoint checkpoints/checkpoint.5000.pt \
    --num_samples 5000 \
    --wandb_project rectified-flow-oxford
```

## Configuration Options

### Training Configuration

Add these parameters to `train_oxford.py` or use command line:

```python
@dataclass
class Config:
    # Evaluation parameters
    eval_every: int = 5000          # Run evaluation every N steps
    eval_num_samples: int = 5000    # Number of samples for evaluation
    eval_batch_size: int = 50       # Batch size for sampling
    eval_reference_batch: str = 'third_party/guided-diffusion/evaluations/VIRTUAL_oxford_flowers256.npz'
```

### Command Line Options

```bash
python train_oxford.py \
    --eval_every 5000 \
    --eval_num_samples 5000 \
    --eval_batch_size 50 \
    --eval_reference_batch path/to/reference.npz
```

## Metrics Logged

The system logs the following metrics to wandb:

- **FID**: Fréchet Inception Distance (lower is better)
- **sFID**: Spatial FID
- **Inception Score**: Alternative quality metric
- **Precision**: How realistic samples are
- **Recall**: How diverse samples are
- **eval_step**: Training step when evaluation was run
- **checkpoint**: Path to evaluated checkpoint

## File Structure

```
.
├── evaluate_checkpoint.py          # Main evaluation script
├── example_evaluation.py           # Usage examples
├── train_oxford.py                 # Modified training script
├── rectified_flow_pytorch/
│   └── rectified_flow.py          # Modified trainer with evaluation
├── third_party/
│   └── guided-diffusion/
│       └── evaluations/               # Guided-diffusion evaluation tools
│       ├── evaluator.py
│       ├── VIRTUAL_lsun_bedroom256.npz  # Reference batch
│       └── requirements.txt
└── GUIDED_DIFFUSION_EVALUATION_GUIDE.md  # Detailed usage guide
```

## Understanding Results

### FID Scores
- **Excellent**: < 5.0
- **Good**: 5.0 - 15.0
- **Poor**: > 15.0

### Precision vs Recall
- **High Precision**: Realistic samples, may lack diversity
- **High Recall**: Diverse samples, may include unrealistic ones
- **Balanced**: Both metrics close to 1.0

## Troubleshooting

### Common Issues

1. **Missing Reference Batch**
   ```
   Error: No such file or directory: 'VIRTUAL_lsun_bedroom256.npz'
   ```
   Solution: Run `python create_oxford_reference.py` to create the Oxford Flowers reference batch

2. **Wandb Not Logging**
   ```
   Wandb not initialized, skipping logging
   ```
   Solution: Initialize wandb with `wandb login` or pass `--wandb_project`

3. **Evaluation Script Not Found**
   ```
   Error: third_party/guided-diffusion/evaluations/evaluator.py not found
   ```
   Solution: Ensure guided-diffusion repo is cloned and evaluator.py exists

4. **Memory Issues**
   - Reduce `eval_batch_size` if getting OOM errors
   - Use CPU for evaluation: `--device cpu`

### Performance Tips

- **GPU Memory**: Evaluation requires ~2-4GB GPU memory
- **Time**: Each evaluation takes 5-15 minutes depending on hardware
- **Storage**: Sample files are saved to `results/eval_samples/`

## Advanced Usage

### Custom Reference Batches

Create your own reference batch for custom datasets:

```python
# See GUIDED_DIFFUSION_EVALUATION_GUIDE.md for detailed instructions
```

### Different Datasets

For other datasets, download appropriate reference batches:
- ImageNet 64x64: `VIRTUAL_imagenet64_labeled.npz`
- LSUN Cat: `VIRTUAL_lsun_cat256.npz`
- LSUN Horse: `VIRTUAL_lsun_horse256.npz`

### Integration with Other Models

The evaluation system can be used with any generative model that can output images in the correct format. Modify `sample_from_checkpoint()` to load your model architecture.

## Example Wandb Dashboard

After running training with evaluation, you'll see metrics like:

```
Training Loss: 0.023
Learning Rate: 0.0001
FID: 4.23
sFID: 6.14
Precision: 0.826
Recall: 0.530
Inception Score: 215.8
```

Plus sample images and evaluation timing information.</content>
<parameter name="filePath">d:\code\rectified-flow-pytorch\EVALUATION_README.md
