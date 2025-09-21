# Loss Breakdown Explanation

## Overview

The `loss_breakdown` is a detailed decomposition of the training loss components in Rectified Flow models. When `return_loss_breakdown=True` is enabled (which happens automatically for `RectifiedFlow` models), the loss is returned as a named tuple containing four key components.

## Loss Components

### 1. `total`
- **Description**: The final combined loss used for backpropagation
- **Formula**: `total = main_loss + consistency_loss * consistency_loss_weight`
- **Purpose**: This is the actual loss value that gets backpropagated through the network

### 2. `main`
- **Description**: The primary training objective loss
- **Purpose**: This is the core rectified flow loss that trains the model to understand the data distribution
- **Computation**: MSE loss between predicted and target values, where the target depends on the prediction objective:
  - **Flow prediction**: MSE(predicted_flow, true_flow) where true_flow = data - noise
  - **Noise prediction**: MSE(predicted_noise, true_noise)

### 3. `data_match` (Consistency Loss Only)
- **Description**: MSE loss between predicted data from main model and EMA model
- **Formula**: `F.mse_loss(pred_data, ema_pred_data)`
- **Purpose**: Measures how well current model predictions match stabilized EMA predictions
- **When Used**: Only computed when `use_consistency=True` in RectifiedFlow

### 4. `velocity_match` (Consistency Loss Only)
- **Description**: MSE loss between predicted flow from main model and EMA model
- **Formula**: `F.mse_loss(pred_flow, ema_pred_flow)`
- **Purpose**: Ensures flow predictions remain stable and consistent across training steps
- **When Used**: Only computed when `use_consistency=True` in RectifiedFlow
- **Weighting**: Multiplied by `consistency_velocity_match_alpha` in consistency loss calculation

## Consistency Loss Computation

When consistency training is enabled, the consistency loss is computed as:

```python
data_match_loss = F.mse_loss(pred_data, ema_pred_data)
velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)
consistency_loss = data_match_loss + velocity_match_loss * consistency_velocity_match_alpha
```

## Named Tuple Definition

```python
LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])
```

## Usage in Training

The loss breakdown is automatically computed and logged when using RectifiedFlow models:

```python
# In trainer forward method
if self.return_loss_breakdown:
    loss, loss_breakdown = self.model(data, return_loss_breakdown=True)
    # loss_breakdown.total, loss_breakdown.main, etc.
```

## Wandb Logging

When wandb logging is enabled, all four components are logged separately:

- `total_loss`: The combined training loss
- `main_loss`: The primary objective loss
- `data_match_loss`: Data consistency loss (if applicable)
- `velocity_match_loss`: Velocity consistency loss (if applicable)

## Why Loss Breakdown Matters

1. **Training Diagnostics**: Monitor if the main objective is learning properly
2. **Consistency Analysis**: Understand the impact of consistency regularization
3. **Debugging**: Identify training issues by examining individual loss components
4. **Research**: Analyze how different loss terms contribute to model performance

## Related Concepts

- **Consistency Flow Matching**: The technique that introduces data_match and velocity_match losses
- **EMA (Exponential Moving Average)**: Stabilized model used for consistency comparisons
- **Rectified Flow**: The base generative modeling framework
- **Flow Matching**: The general class of generative models this belongs to

## Implementation Details

- Only available for `RectifiedFlow` models (not NanoFlow or other variants)
- Automatically enabled when `isinstance(rectified_flow, RectifiedFlow)` in Trainer
- Components are zero when consistency loss is not used
- All losses are computed as scalar values for logging and monitoring
