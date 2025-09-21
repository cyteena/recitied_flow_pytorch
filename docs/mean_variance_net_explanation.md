# Understanding the Mean-Variance Network in Rectified Flow

## Overview

The `mean_variance_net` is a probabilistic extension of the standard rectified flow model that predicts a distribution over flows rather than a deterministic flow. This approach can improve sample quality and training stability.

## Core Concept

In standard rectified flow, the model predicts a deterministic flow field `v(x_t, t)` that guides the ODE from noise to data. The mean-variance network instead predicts the parameters of a probability distribution over possible flows.

## Architecture Changes

### Model Output
- **Standard Mode**: Model outputs `channels` values (deterministic flow)
- **Mean-Variance Mode**: Model outputs `2 × channels` values (mean and log-variance of flow distribution)

```python
# In Unet.forward()
if not self.mean_variance_net:
    return out  # Shape: (batch, channels, H, W)

mean, log_var = rearrange(out, 'b (c mean_log_var) h w -> mean_log_var b c h w', mean_log_var=2)
variance = log_var.exp()  # Ensure variance is positive
return stack((mean, variance))  # Shape: (2, batch, channels, H, W)
```

### Loss Function

When `mean_variance_net=True`, the loss automatically switches to `MeanVarianceNetLoss`:

```python
class MeanVarianceNetLoss(Module):
    def forward(self, pred, target, **kwargs):
        dist = Normal(*pred)  # Create Normal distribution from mean and variance
        return -dist.log_prob(target).mean()  # Negative log-likelihood
```

**What this means:**
- `pred` contains `[mean, variance]` from the model
- `target` is the ground-truth flow (data - noise)
- Loss measures how well the predicted distribution matches the true flow
- Equivalent to maximizing the likelihood of the true flow under the predicted distribution

## Training Logic

### During Training
```python
# In RectifiedFlow.forward()
if self.mean_variance_net:
    mean, variance = model_output
    pred_flow = torch.normal(mean, variance)  # Sample from predicted distribution
else:
    pred_flow = model_output  # Use deterministic prediction
```

The model predicts a distribution, but during training we sample from it to get the actual flow used for loss computation.

### Loss Computation
```python
# Target is always the deterministic flow (data - noise)
target = data - noise

# Loss compares predicted flow (sampled or deterministic) to target
main_loss = self.loss_fn(output, target, ...)
```

## Sampling Logic

### During Inference/Sampling
```python
# In RectifiedFlow.sample()
if self.mean_variance_net:
    mean, variance = output
    std = variance.clamp(min=1e-5).sqrt()
    flow = torch.normal(mean, std * temperature)  # Sample with temperature scaling
else:
    flow = output  # Deterministic flow
```

**Key differences:**
- **Temperature scaling**: Variance is scaled by temperature during sampling
- **Stochastic sampling**: Each sampling step uses randomness from the predicted distribution
- **Improved diversity**: Can generate different samples even with same noise input

## Mathematical Loss Computation

### Mean-Variance Network Loss

Given the model's prediction `pred = [μ, σ²]` and target flow `v_true`, the loss is:

**Loss Function:**
```
ℒ = -𝔼[log p(v_true | μ, σ²)]
   = -log p(v_true | μ, σ²)
   = -log [ (2π σ²)^{-1/2} exp( -(v_true - μ)² / (2σ²) ) ]
   = (1/2) log(2π σ²) + (v_true - μ)² / (2σ²)
```

**Vectorized Form:**
```
ℒ(μ, σ², v_true) = (1/2) log(2π σ²) + (v_true - μ)² / (2σ²)
```

**Batch-averaged Loss:**
```
ℒ_batch = (1/B) Σ_{i=1}^B ℒ(μ_i, σ²_i, v_true,i)
```

Where:
- `μ ∈ ℝ^{B×C×H×W}`: Predicted mean flow
- `σ² ∈ ℝ^{B×C×H×W}`: Predicted variance (from log_var.exp())
- `v_true ∈ ℝ^{B×C×H×W}`: Ground truth flow (data - noise)
- `B`: Batch size, `C`: Channels, `H×W`: Spatial dimensions

### Implementation Details

**In Code:**
```python
dist = Normal(mean, variance.sqrt())  # variance = log_var.exp()
loss = -dist.log_prob(target).mean()
```

**Equivalent Manual Computation:**
```python
# pred = [mean, variance], target = v_true
mean, variance = pred
log_prob = -0.5 * torch.log(2 * pi * variance) - (target - mean)**2 / (2 * variance)
loss = -log_prob.mean()
```

### Gradient Flow

**Gradient w.r.t. Mean:**
```
∂ℒ/∂μ = (μ - v_true) / σ²
```

**Gradient w.r.t. Variance:**
```
∂ℒ/∂σ² = (1/(2σ²)) - (v_true - μ)² / (2σ⁴)
```

This encourages the model to:
- Predict means close to true flows
- Predict appropriate variance (not too small to avoid overfitting, not too large to maintain precision)

### Connection to Standard Flow Matching

**Standard Flow Matching Loss:**
```
ℒ_FM = ||v_pred - v_true||²
```

**Mean-Variance as Probabilistic FM:**
```
ℒ_MV = 𝔼_{v~p(v|μ,σ²)} [||v - v_true||²] + KL(p||q)
      ≈ ||μ - v_true||² + trace(σ²)
```

Where the mean-variance loss approximates flow matching with uncertainty quantification.

### Relationship to Cross-Entropy Loss

**No, this is NOT cross-entropy loss.** 

**Cross-Entropy Loss** is used for discrete classification:
```
ℒ_CE = -Σ y_true,i log(y_pred,i)
```
Where `y_pred` is a probability distribution over discrete classes.

**Mean-Variance Loss** is negative log-likelihood for continuous regression:
```
ℒ_MVN = -log p(v_true | μ, σ²)
```
Where `p` is a continuous Normal distribution over flow values.

**Key Differences:**
- **Cross-entropy**: Discrete probabilities over classes
- **Mean-variance**: Continuous probability density over real-valued flows
- **Cross-entropy**: Categorical distribution
- **Mean-variance**: Normal (Gaussian) distribution

The mean-variance loss is more similar to maximum likelihood estimation for regression with uncertainty quantification than to classification losses.

## Benefits

### 1. **Improved Sample Quality**
- Probabilistic modeling captures uncertainty in flow predictions
- Can generate higher quality samples with better diversity

### 2. **Better Training Stability**
- Distributional modeling can be more robust to outliers
- Negative log-likelihood provides smoother loss landscape

### 3. **Temperature Control**
- Sampling temperature allows controlling sample diversity
- `temperature > 1`: More diverse samples
- `temperature < 1`: More conservative samples

### 4. **Uncertainty Estimation**
- Variance prediction provides confidence estimates
- Can be used for adaptive sampling strategies

## Implementation Details

### Output Processing
```python
# Split output into mean and log-variance
mean, log_var = rearrange(out, 'b (c 2) h w -> 2 b c h w')
variance = log_var.exp()  # Convert log-variance to variance
```

### Distribution Sampling
```python
# Reparameterization trick for training
pred_flow = torch.normal(mean, variance.sqrt())

# Temperature scaling for sampling
flow = torch.normal(mean, variance.sqrt() * temperature)
```

### Loss Computation
```python
# Create distribution
dist = Normal(mean, variance.sqrt())

# Compute negative log-likelihood
loss = -dist.log_prob(target).mean()
```

## Usage

Enable mean-variance network in training:

```python
model = Unet(dim=64, mean_variance_net=True)
rectified_flow = RectifiedFlow(model)

# During sampling
samples = rectified_flow.sample(temperature=1.5)  # Higher temperature for diversity
```

## Comparison with Standard Flow

| Aspect | Standard Flow | Mean-Variance Flow |
|--------|---------------|-------------------|
| Output | Deterministic | Distribution parameters |
| Loss | MSE | Negative log-likelihood |
| Sampling | Deterministic | Stochastic |
| Diversity | Limited | High (with temperature) |
| Uncertainty | None | Built-in variance estimation |

## Notes

- Automatically enabled when `mean_variance_net=True` in Unet
- Loss function switches automatically in RectifiedFlow
- Compatible with all other features (EMA, consistency, etc.)
- Increases model parameter count (2x output channels)
- Can be more computationally expensive due to distribution sampling
