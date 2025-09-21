# Third-Party Dependencies

This directory contains external repositories and tools that are used as dependencies for this project.

## Structure

- `guided-diffusion/`: OpenAI's guided-diffusion repository (used for FID evaluation)
  - Contains evaluation tools for computing FID, Inception Score, and other metrics
  - Used by the automated evaluation system in `evaluate_checkpoint.py`

## Setup

### Guided Diffusion

The guided-diffusion repository is included as a git submodule. To initialize it:

```bash
git submodule init
git submodule update
```

Or clone with submodules:

```bash
git clone --recursive https://github.com/your-repo/rectified-flow-pytorch.git
```

### Installation

After cloning the submodule, install the evaluation dependencies:

```bash
cd third_party/guided-diffusion/evaluations
pip install -r requirements.txt
```

## Usage

The guided-diffusion tools are used by the evaluation system to compute metrics like:
- FID (Fréchet Inception Distance)
- sFID (spatial FID)
- Precision and Recall
- Inception Score

See `EVALUATION_README.md` and `GUIDED_DIFFUSION_EVALUATION_GUIDE.md` for detailed usage instructions.

## Adding New Dependencies

When adding new third-party dependencies:

1. Add them as git submodules in this directory
2. Update `.gitmodules` accordingly
3. Document the dependency and its purpose in this README
4. Update any relevant installation/setup instructions
