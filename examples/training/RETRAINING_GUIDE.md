# TiRex Model: Recreation and Retraining Guide

## Executive Summary

**YES, it is technically possible to recreate the TiRex model architecture and retrain it.** The repository contains all the necessary architectural components, and the checkpoint file reveals the complete model configuration. However, there are important considerations regarding feasibility, licensing, and practical challenges.

---

## Model Architecture Details

### Extracted Configuration from Checkpoint

```json
{
  "input_patch_size": 32,
  "output_patch_size": 32,
  "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  "block_kwargs": {
    "all_slstm": true,
    "embedding_dim": 512,
    "num_blocks": 12,
    "num_heads": 4,
    "return_last_states": true,
    "vocab_size": 0
  },
  "input_ff_dim": 2048
}
```

### Architecture Components

The TiRex model consists of:

1. **Input Processing**
   - `PatchedUniTokenizer`: Converts time series into patches (size: 32)
   - `StandardScaler`: Normalizes input data
   - `ResidualBlock`: Input embedding (64 → 2048 → 512 dimensions)

2. **Core Architecture: xLSTM Block Stack**
   - 12 sLSTM (scalar LSTM) blocks
   - Each block contains:
     - RMS Layer Normalization
     - sLSTM layer with 4 heads (embedding_dim: 512, head_dim: 128)
     - Feed-forward network (512 → 1408 → 512)
     - Residual connections

3. **Output Processing**
   - `ResidualBlock`: Output embedding (512 → 2048 → 288)
   - Generates 9 quantile predictions simultaneously
   - Output patch size: 32

4. **Total Parameters**: 35,291,200 (~35M parameters)

---

## What You Have Available

### ✅ Complete Architecture Code

All model components are present in the repository:

- `tirex/models/tirex.py`: Main TiRexZero model class
- `tirex/models/components.py`: ResidualBlock, PatchedUniTokenizer, StandardScaler
- `tirex/models/mixed_stack.py`: xLSTMMixedLargeBlockStack (12 sLSTM blocks)
- Dependencies: `xlstm` package (available on PyPI)

### ✅ Pre-trained Weights

- Located at: `./model/model.ckpt`
- Contains both hyperparameters and state_dict
- Can be used for transfer learning or as initialization

### ❌ Training Code (Not Included)

The repository does **NOT** include:
- Training loop implementation
- Loss function definition
- Data loading/preprocessing for training
- Optimizer configuration
- Training hyperparameters (learning rate, batch size, etc.)
- Dataset used for pre-training

---

## How to Recreate and Retrain

### Step 1: Instantiate the Model from Scratch

```python
import torch
from tirex.models.tirex import TiRexZero

# Define model configuration (from checkpoint inspection)
model_config = {
    "input_patch_size": 32,
    "output_patch_size": 32,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "block_kwargs": {
        "all_slstm": True,
        "embedding_dim": 512,
        "num_blocks": 12,
        "num_heads": 4,
        "return_last_states": True,
        "vocab_size": 0,
        "norm_eps": 1e-5,
        "use_bias": False,
        "norm_reduction_force_float32": True,
        "add_out_norm": True,
    },
    "input_ff_dim": 2048
}

# Create model
train_ctx_len = 512  # Context length during training (adjust as needed)
model = TiRexZero(model_config=model_config, train_ctx_len=train_ctx_len)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Step 2: Implement Training Components

You need to implement the following components that are NOT in the repository:

#### A. Quantile Loss Function

```python
import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""
    
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, num_quantiles, prediction_length]
            targets: [batch_size, prediction_length]
        """
        # Expand targets to match quantile dimension
        targets_expanded = targets.unsqueeze(1)  # [batch, 1, pred_len]
        
        # Calculate residuals
        residuals = targets_expanded - predictions  # [batch, num_quantiles, pred_len]
        
        # Quantile loss (pinball loss)
        quantiles = self.quantiles.view(1, -1, 1).to(predictions.device)
        loss = torch.where(
            residuals >= 0,
            quantiles * residuals,
            (quantiles - 1) * residuals
        )
        
        return loss.mean()
```

#### B. Data Loader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, data, context_length=512, prediction_length=32):
        """
        Args:
            data: List of time series arrays or pandas Series
            context_length: Length of input context
            prediction_length: Length of forecast horizon
        """
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        
    def __len__(self):
        # Calculate number of valid windows in all time series
        total_windows = 0
        for ts in self.data:
            if len(ts) >= self.context_length + self.prediction_length:
                total_windows += len(ts) - self.context_length - self.prediction_length + 1
        return total_windows
    
    def __getitem__(self, idx):
        # Find which time series and window position
        # (Simplified - you'd need to implement proper indexing)
        ts = self.data[idx % len(self.data)]
        
        # Extract random window
        max_start = len(ts) - self.context_length - self.prediction_length
        start_idx = torch.randint(0, max_start, (1,)).item()
        
        context = torch.tensor(ts[start_idx:start_idx + self.context_length], dtype=torch.float32)
        target = torch.tensor(
            ts[start_idx + self.context_length:start_idx + self.context_length + self.prediction_length],
            dtype=torch.float32
        )
        
        return context, target

# Example usage
# dataset = TimeSeriesDataset(your_time_series_list)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### C. Training Loop (PyTorch Lightning)

```python
import lightning as L
from torch.optim import AdamW

class TiRexTrainer(L.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = QuantileLoss(model.model_config.quantiles)
    
    def forward(self, x, prediction_length):
        return self.model._forecast_tensor(x, prediction_length=prediction_length)
    
    def training_step(self, batch, batch_idx):
        context, target = batch
        # Forward pass
        predictions = self(context, prediction_length=target.shape[-1])
        # Calculate loss
        loss = self.loss_fn(predictions, target)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        context, target = batch
        predictions = self(context, prediction_length=target.shape[-1])
        loss = self.loss_fn(predictions, target)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]

# Training execution
# trainer_module = TiRexTrainer(model)
# trainer = L.Trainer(max_epochs=100, accelerator='gpu', devices=1)
# trainer.fit(trainer_module, train_dataloader, val_dataloader)
```

### Step 3: Prepare Training Data

The original TiRex was likely trained on massive datasets. For retraining, you need:

1. **Large-scale time series data** (millions of samples recommended)
2. **Diverse domains** (to maintain zero-shot capabilities)
3. **Various frequencies** (minute, hourly, daily data)

**Data Sources:**
- Monash Time Series Forecasting Archive
- Financial market data (stocks, crypto, forex)
- Energy consumption datasets
- Weather data
- Your proprietary cryptocurrency data

**Preprocessing:**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_time_series(df, column='close'):
    """Prepare time series for training."""
    # Extract values
    values = df[column].values
    
    # Handle missing values
    values = pd.Series(values).interpolate(method='linear').values
    
    # Optional: Remove outliers or apply smoothing
    # (The model's tokenizer handles normalization)
    
    return values
```

### Step 4: Training Hyperparameters (Estimated)

Since the original training hyperparameters aren't provided, you'll need to experiment:

```python
training_config = {
    "batch_size": 32,  # Adjust based on GPU memory
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "max_epochs": 100,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 4,  # Effective batch size = 32 * 4 = 128
    "context_length": 512,
    "prediction_length": 32,
    "num_gpus": 1,  # Or more for distributed training
}
```

### Step 5: Transfer Learning (Recommended Approach)

Instead of training from scratch, use the pre-trained weights:

```python
from tirex import load_model

# Load pre-trained model
pretrained_model = load_model("./model/model.ckpt")

# Fine-tune on your data
trainer_module = TiRexTrainer(pretrained_model, learning_rate=1e-5)  # Lower LR for fine-tuning
trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=1)
trainer.fit(trainer_module, your_dataloader, val_dataloader)

# Save fine-tuned model
trainer.save_checkpoint("tirex_finetuned.ckpt")
```

---

## Practical Challenges

### 1. **Computational Requirements**

- **Memory**: 35M parameters require ~140MB for fp32 weights, but training requires much more (gradients, optimizer states, activations)
- **GPU**: Recommended NVIDIA A100/H100 with 40GB+ VRAM
- **Training Time**: Weeks to months for full retraining on large datasets
- **Cost**: Potentially $10,000+ in cloud GPU costs for full retraining

### 2. **Data Requirements**

- Original TiRex was trained on massive, diverse datasets
- You need **millions of time series samples** for comparable performance
- Data quality is crucial for good zero-shot capabilities

### 3. **CUDA Kernels**

The sLSTM implementation uses custom CUDA kernels:
- Requires GPU with CUDA compute capability ≥ 8.0
- Falls back to slower PyTorch implementation otherwise
- May affect training speed significantly

### 4. **No Training Documentation**

You'll need to:
- Determine optimal hyperparameters through experimentation
- Implement data augmentation strategies
- Design curriculum learning schedules
- Monitor training stability

---

## Legal Considerations (NXAI Community License)

### ✅ You CAN:
- Use and modify the NXAI Materials (including model architecture)
- Create derivative works (including retrained models)
- Use for research and development
- Use commercially if your organization has <€100M annual revenue

### ⚠️ You MUST:
- Include the license agreement with any distribution
- Display "Built with technology from NXAI" attribution
- Retain copyright notices in distributed copies

### ❌ You CANNOT (without commercial license):
- Use in commercial products/services if your org has >€100M revenue
- Remove or modify attribution requirements
- Use NXAI trademarks without permission

---

## Recommended Approach

Given the challenges, here's the recommended strategy:

### Option 1: **Fine-tune on Pre-trained Weights** (Easiest)
- Start with pre-trained checkpoint
- Fine-tune on your cryptocurrency data
- Requires less data and compute
- Preserves zero-shot capabilities

### Option 2: **Domain Adaptation** (Moderate)
- Train additional adapter layers
- Freeze most of the pre-trained model
- Only update small task-specific components
- Much faster than full retraining

### Option 3: **Full Retraining** (Hardest)
- Train from scratch on massive datasets
- Requires significant resources
- Only necessary if pre-trained model is unsuitable

### Option 4: **Wait for Official Fine-tuning Support** (Safest)
- NXAI plans to release fine-tuning capabilities
- Will be officially supported and tested
- May include optimal hyperparameters

---

## Quick Start: Minimal Training Example

```python
import torch
import lightning as L
from tirex import load_model
from torch.utils.data import DataLoader

# 1. Load pre-trained model
model = load_model("./model/model.ckpt")

# 2. Prepare your data
train_dataset = TimeSeriesDataset(your_training_data)
val_dataset = TimeSeriesDataset(your_validation_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. Setup trainer
trainer_module = TiRexTrainer(model, learning_rate=1e-5)

# 4. Train
trainer = L.Trainer(
    max_epochs=10,
    accelerator='gpu',
    devices=1,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
)

trainer.fit(trainer_module, train_loader, val_loader)

# 5. Save fine-tuned model
trainer.save_checkpoint("tirex_crypto_finetuned.ckpt")
```

---

## Conclusion

**Yes, you can recreate and retrain the TiRex model architecture.** All necessary components are available in the repository, and the checkpoint reveals the complete configuration. However:

1. **Training from scratch** is resource-intensive and requires massive datasets
2. **Fine-tuning** the pre-trained model is more practical and recommended
3. You'll need to implement training infrastructure (loss function, data loaders, training loop)
4. Consider waiting for official fine-tuning support from NXAI

The pre-trained model already achieves state-of-the-art zero-shot performance. Your current optimization approach (tuning preprocessing parameters) may be sufficient without retraining the model itself.

