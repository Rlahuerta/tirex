# Fine-tuning TiRex with Cryptocurrency Data

A complete, practical guide for fine-tuning TiRex on your Bitcoin/cryptocurrency price data.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding Your Data](#understanding-your-data)
3. [Data Preparation](#data-preparation)
4. [Fine-tuning Setup](#fine-tuning-setup)
5. [Training Process](#training-process)
6. [Evaluation](#evaluation)
7. [Using the Fine-tuned Model](#using-the-fine-tuned-model)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

```bash
# Ensure you have the tirex environment activated
conda activate tirex

# Install additional dependencies for training
pip install lightning tensorboard
```

### 5-Minute Test Run

```bash
cd /home/hephaestus/Repositories/tirex/examples/training
python crypto_finetune_simple.py --quick-test
```

This will run a minimal fine-tuning test to verify everything works.

---

## Understanding Your Data

### Current Data Format

Your cryptocurrency data is in this format (from `btcusd_2022-06-01.joblib`):

```python
{
    15: DataFrame,  # 15-minute intervals
    60: DataFrame,  # 60-minute (hourly) intervals
    ...
}
```

Each DataFrame has these columns:
- `timestamp` (index)
- `open`, `high`, `low`, `close` (OHLC prices)
- `volume`

### What TiRex Expects

TiRex processes **univariate time series** (single values over time). For crypto, we'll use:
- Primary: `close` prices (most common)
- Alternative: `volume`, `high-low spread`, or derived features

---

## Data Preparation

### Step 1: Extract and Preprocess Data

Create `prepare_crypto_data.py`:

```python
"""
Prepare cryptocurrency data for TiRex fine-tuning.
"""
import numpy as np
import pandas as pd
from joblib import load, dump
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import RobustScaler


class CryptoDataPreparator:
    """Prepare cryptocurrency data for time series forecasting."""
    
    def __init__(self, data_path: str, time_interval: int = 60):
        """
        Args:
            data_path: Path to joblib file with crypto data
            time_interval: Time interval in minutes (15, 60, etc.)
        """
        self.data_path = Path(data_path)
        self.time_interval = time_interval
        self.data = None
        self.scaler = RobustScaler()  # Better for financial data with outliers
        
    def load_data(self) -> pd.DataFrame:
        """Load data from joblib file."""
        dict_data = load(self.data_path)
        self.data = dict_data[self.time_interval]
        print(f"Loaded {len(self.data)} samples at {self.time_interval}-minute intervals")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        return self.data
    
    def create_features(self) -> pd.DataFrame:
        """Create additional features from OHLC data."""
        df = self.data.copy()
        
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=24).std()
        df['price_momentum'] = df['close'] - df['close'].shift(24)
        
        # Technical indicators
        df['sma_short'] = df['close'].rolling(window=12).mean()
        df['sma_long'] = df['close'].rolling(window=48).mean()
        df['ema'] = df['close'].ewm(span=24).mean()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price range
        df['high_low_spread'] = df['high'] - df['low']
        df['close_open_spread'] = df['close'] - df['open']
        
        # Drop NaN values from rolling computations
        df = df.dropna()
        
        return df
    
    def prepare_training_sequences(
        self,
        feature_column: str = 'close',
        context_length: int = 512,
        prediction_length: int = 32,
        stride: int = 16,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Prepare data as sequences for training.
        
        Args:
            feature_column: Which column to use as time series
            context_length: Input sequence length
            prediction_length: Forecast horizon
            stride: Step size for sliding window
            train_split: Fraction for training
            val_split: Fraction for validation (rest is test)
        
        Returns:
            Tuple of (train_sequences, val_sequences, test_sequences)
        """
        df = self.create_features()
        values = df[feature_column].values
        
        # Normalize data
        values_scaled = self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        # Create sequences
        sequences = []
        total_length = context_length + prediction_length
        
        for i in range(0, len(values_scaled) - total_length + 1, stride):
            seq = values_scaled[i:i + total_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        print(f"Created {len(sequences)} sequences of length {total_length}")
        
        # Split data
        n = len(sequences)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        train_seqs = sequences[:train_end]
        val_seqs = sequences[train_end:val_end]
        test_seqs = sequences[val_end:]
        
        print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
        
        return train_seqs, val_seqs, test_seqs
    
    def save_prepared_data(self, output_dir: str):
        """Save prepared sequences to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        train, val, test = self.prepare_training_sequences()
        
        dump({
            'train': train,
            'val': val,
            'test': test,
            'scaler': self.scaler,
            'time_interval': self.time_interval,
        }, output_path / 'crypto_prepared.joblib')
        
        print(f"Saved prepared data to {output_path / 'crypto_prepared.joblib'}")
        
        return train, val, test


if __name__ == "__main__":
    # Example usage
    preparator = CryptoDataPreparator(
        data_path="../../tests/data/btcusd_2022-06-01.joblib",
        time_interval=60
    )
    
    preparator.load_data()
    train, val, test = preparator.save_prepared_data("./prepared_data")
```

### Step 2: Run Data Preparation

```bash
python prepare_crypto_data.py
```

This creates `prepared_data/crypto_prepared.joblib` with train/val/test splits.

---

## Fine-tuning Setup

### Create Training Script

Create `crypto_finetune.py`:

```python
"""
Fine-tune TiRex on cryptocurrency data.
"""
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from joblib import load
from typing import Tuple, Optional
import argparse
from datetime import datetime

from tirex import load_model


# ============================================================================
# DATASET
# ============================================================================

class CryptoDataset(Dataset):
    """Dataset for cryptocurrency time series."""
    
    def __init__(self, sequences: np.ndarray, context_length: int = 512, prediction_length: int = 32):
        """
        Args:
            sequences: Array of shape (n_sequences, context_length + prediction_length)
            context_length: Length of input context
            prediction_length: Length of forecast horizon
        """
        self.sequences = sequences
        self.context_length = context_length
        self.prediction_length = prediction_length
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        context = seq[:self.context_length]
        target = seq[self.context_length:self.context_length + self.prediction_length]
        
        return (
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting."""
    
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles))
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_quantiles, prediction_length]
            targets: [batch_size, prediction_length]
        """
        targets_expanded = targets.unsqueeze(1)  # [batch, 1, pred_len]
        residuals = targets_expanded - predictions
        
        quantiles = self.quantiles.view(1, -1, 1)
        loss = torch.where(
            residuals >= 0,
            quantiles * residuals,
            (quantiles - 1) * residuals
        )
        
        return loss.mean()


# ============================================================================
# TRAINING MODULE
# ============================================================================

class CryptoFineTuner(L.LightningModule):
    """Fine-tune TiRex for cryptocurrency forecasting."""
    
    def __init__(
        self,
        model,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Loss function
        self.loss_fn = QuantileLoss(model.model_config.quantiles)
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        return self.model._forecast_tensor(
            context,
            prediction_length=prediction_length,
            max_accelerated_rollout_steps=1
        )
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        context, target = batch
        predictions = self(context, prediction_length=target.shape[-1])
        loss = self.loss_fn(predictions, target)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log median prediction MAE
        median_pred = predictions[:, 4, :]  # 0.5 quantile
        mae = torch.abs(median_pred - target).mean()
        self.log('train_mae', mae, prog_bar=False, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        context, target = batch
        predictions = self(context, prediction_length=target.shape[-1])
        loss = self.loss_fn(predictions, target)
        
        # Calculate additional metrics
        median_pred = predictions[:, 4, :]  # 0.5 quantile
        mae = torch.abs(median_pred - target).mean()
        mse = ((median_pred - target) ** 2).mean()
        
        # Calculate coverage for 80% prediction interval (0.1 to 0.9 quantiles)
        lower_bound = predictions[:, 0, :]  # 0.1 quantile
        upper_bound = predictions[:, -1, :]  # 0.9 quantile
        coverage = ((target >= lower_bound) & (target <= upper_bound)).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mae', mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mse', mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_coverage', coverage, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with warmup and cosine decay."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate schedule with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(1, self.trainer.max_steps - self.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_crypto_model(
    data_path: str = "./prepared_data/crypto_prepared.joblib",
    model_path: str = "../../model/model.ckpt",
    output_dir: str = "./crypto_finetuned",
    context_length: int = 512,
    prediction_length: int = 32,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    max_epochs: int = 20,
    num_gpus: int = 1,
    quick_test: bool = False,
):
    """
    Main training function for fine-tuning TiRex on crypto data.
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print(f"TiRex Cryptocurrency Fine-tuning - {timestamp}")
    print("=" * 80)
    
    # Load pre-trained model
    print(f"\n1. Loading pre-trained TiRex model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() and num_gpus > 0 else 'cpu'
    model = load_model(model_path, device=device)
    print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Device: {device}")
    
    # Load prepared data
    print(f"\n2. Loading prepared crypto data from {data_path}...")
    data_dict = load(data_path)
    train_sequences = data_dict['train']
    val_sequences = data_dict['val']
    test_sequences = data_dict['test']
    
    # Quick test mode: use subset
    if quick_test:
        print("   QUICK TEST MODE: Using subset of data")
        train_sequences = train_sequences[:100]
        val_sequences = val_sequences[:20]
        max_epochs = 2
        batch_size = 16
    
    print(f"   Train: {len(train_sequences)} sequences")
    print(f"   Val: {len(val_sequences)} sequences")
    print(f"   Test: {len(test_sequences)} sequences")
    
    # Create datasets and dataloaders
    print("\n3. Creating dataloaders...")
    train_dataset = CryptoDataset(train_sequences, context_length, prediction_length)
    val_dataset = CryptoDataset(val_sequences, context_length, prediction_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create training module
    print("\n4. Setting up training module...")
    finetuner = CryptoFineTuner(
        model=model,
        learning_rate=learning_rate,
        warmup_steps=min(500, len(train_loader) * 2),
    )
    
    # Setup trainer
    print("\n5. Configuring PyTorch Lightning trainer...")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if num_gpus > 0 else 'cpu',
        devices=num_gpus if num_gpus > 0 else 'auto',
        precision='16-mixed' if device == 'cuda' else '32',
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        default_root_dir=str(output_path),
        enable_checkpointing=True,
        callbacks=[
            L.callbacks.ModelCheckpoint(
                dirpath=output_path / 'checkpoints',
                filename=f'crypto-tirex-{timestamp}-{{epoch:02d}}-{{val_loss:.4f}}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True,
            ),
            L.callbacks.LearningRateMonitor(logging_interval='step'),
            L.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                verbose=True,
            ),
            L.callbacks.RichProgressBar(),
        ],
        logger=L.pytorch.loggers.TensorBoardLogger(
            save_dir=str(output_path),
            name='logs',
        ),
    )
    
    # Train
    print("\n6. Starting training...\n")
    print("=" * 80)
    
    trainer.fit(finetuner, train_loader, val_loader)
    
    # Save final model
    print("\n" + "=" * 80)
    print("Training completed!")
    final_model_path = output_path / f'crypto_tirex_final_{timestamp}.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"  Best val_loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
    print(f"  Best val_mae: {trainer.callback_metrics.get('val_mae', 'N/A')}")
    print(f"  Total epochs: {trainer.current_epoch + 1}")
    print(f"  Checkpoints saved in: {output_path / 'checkpoints'}")
    print(f"  TensorBoard logs: {output_path / 'logs'}")
    print("\nView training logs with: tensorboard --logdir " + str(output_path / 'logs'))
    print("=" * 80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TiRex on cryptocurrency data")
    
    parser.add_argument('--data-path', type=str, default='./prepared_data/crypto_prepared.joblib',
                        help='Path to prepared data file')
    parser.add_argument('--model-path', type=str, default='../../model/model.ckpt',
                        help='Path to pre-trained TiRex model')
    parser.add_argument('--output-dir', type=str, default='./crypto_finetuned',
                        help='Output directory for checkpoints')
    parser.add_argument('--context-length', type=int, default=512,
                        help='Input context length')
    parser.add_argument('--prediction-length', type=int, default=32,
                        help='Forecast horizon length')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--max-epochs', type=int, default=20,
                        help='Maximum number of training epochs')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with subset of data')
    
    args = parser.parse_args()
    
    # Set random seed
    L.seed_everything(42)
    
    # Run training
    train_crypto_model(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        num_gpus=args.num_gpus,
        quick_test=args.quick_test,
    )
```

---

## Training Process

### Step 1: Prepare Data

```bash
python prepare_crypto_data.py
```

### Step 2: Quick Test (Recommended First)

```bash
python crypto_finetune.py --quick-test
```

This runs 2 epochs on a small subset to verify everything works.

### Step 3: Full Fine-tuning

```bash
# Basic fine-tuning with default settings
python crypto_finetune.py

# Custom settings
python crypto_finetune.py \
    --batch-size 64 \
    --learning-rate 2e-5 \
    --max-epochs 30 \
    --prediction-length 64
```

### Step 4: Monitor Training

```bash
# In a separate terminal
tensorboard --logdir ./crypto_finetuned/logs
```

Open http://localhost:6006 in your browser to monitor:
- Training/validation loss
- MAE (Mean Absolute Error)
- Coverage (prediction interval accuracy)
- Learning rate schedule

### Expected Training Time

- **Quick test**: 2-5 minutes
- **Full training (20 epochs)**: 2-4 hours on GPU, 12-24 hours on CPU
- **GPU recommendation**: RTX 3090, RTX 4090, or A4000+

---

## Evaluation

### Create Evaluation Script

Create `evaluate_crypto_model.py`:

```python
"""
Evaluate fine-tuned TiRex model on test data.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
from scipy.stats import pearsonr

from tirex import load_model


def evaluate_model(
    model_path: str,
    data_path: str,
    output_dir: str = "./evaluation_results",
    num_samples: int = 50,
):
    """Evaluate fine-tuned model on test data."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Loading model and data...")
    model = load_model(model_path)
    data_dict = load(data_path)
    test_sequences = data_dict['test']
    scaler = data_dict['scaler']
    
    context_length = 512
    prediction_length = 32
    
    # Select random test samples
    indices = np.random.choice(len(test_sequences), min(num_samples, len(test_sequences)), replace=False)
    
    results = []
    
    print(f"Evaluating {len(indices)} test samples...")
    
    for idx in indices:
        seq = test_sequences[idx]
        context = seq[:context_length]
        target = seq[context_length:context_length + prediction_length]
        
        # Make prediction
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predictions = model.forecast(context=context_tensor, prediction_length=prediction_length)
        
        # Extract quantiles (shape: [1, num_quantiles, prediction_length])
        pred_array = predictions.cpu().numpy()[0]
        
        # Calculate metrics
        median_pred = pred_array[4, :]  # 0.5 quantile
        mae = np.abs(median_pred - target).mean()
        mse = ((median_pred - target) ** 2).mean()
        rmse = np.sqrt(mse)
        
        # Correlation
        corr, _ = pearsonr(median_pred, target)
        
        # Coverage
        lower = pred_array[0, :]  # 0.1 quantile
        upper = pred_array[-1, :]  # 0.9 quantile
        coverage = np.mean((target >= lower) & (target <= upper))
        
        results.append({
            'mae': mae,
            'rmse': rmse,
            'correlation': corr,
            'coverage': coverage,
        })
    
    # Aggregate results
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nMetrics (averaged over {len(results)} samples):")
    print(f"  MAE:         {df_results['mae'].mean():.6f} ± {df_results['mae'].std():.6f}")
    print(f"  RMSE:        {df_results['rmse'].mean():.6f} ± {df_results['rmse'].std():.6f}")
    print(f"  Correlation: {df_results['correlation'].mean():.4f} ± {df_results['correlation'].std():.4f}")
    print(f"  Coverage:    {df_results['coverage'].mean():.2%} (target: 80%)")
    print("=" * 80)
    
    # Save results
    df_results.to_csv(output_path / 'evaluation_results.csv', index=False)
    print(f"\nResults saved to {output_path / 'evaluation_results.csv'}")
    
    # Plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    df_results['mae'].hist(bins=20, ax=axes[0, 0])
    axes[0, 0].set_title('MAE Distribution')
    axes[0, 0].set_xlabel('MAE')
    
    df_results['correlation'].hist(bins=20, ax=axes[0, 1])
    axes[0, 1].set_title('Correlation Distribution')
    axes[0, 1].set_xlabel('Correlation')
    
    df_results['coverage'].hist(bins=20, ax=axes[1, 0])
    axes[1, 0].set_title('Coverage Distribution')
    axes[1, 0].set_xlabel('Coverage')
    axes[1, 0].axvline(0.8, color='r', linestyle='--', label='Target 80%')
    axes[1, 0].legend()
    
    # Box plot
    df_results[['mae', 'rmse', 'correlation', 'coverage']].boxplot(ax=axes[1, 1])
    axes[1, 1].set_title('Metrics Box Plot')
    
    plt.tight_layout()
    plt.savefig(output_path / 'evaluation_plots.png', dpi=300)
    print(f"Plots saved to {output_path / 'evaluation_plots.png'}")
    
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--data-path', type=str, default='./prepared_data/crypto_prepared.joblib')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    parser.add_argument('--num-samples', type=int, default=50)
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
```

### Run Evaluation

```bash
python evaluate_crypto_model.py \
    --model-path ./crypto_finetuned/checkpoints/best_model.ckpt \
    --num-samples 100
```

---

## Using the Fine-tuned Model

### Integration with Your Existing Code

Update your `opt_price_forecast.py`:

```python
def _load_forecast_model(self):
    """Load fine-tuned or pre-trained model."""
    try:
        # Try loading fine-tuned model first
        finetuned_path = Path(__file__).parent / "training" / "crypto_finetuned" / "checkpoints" / "best_model.ckpt"
        
        if finetuned_path.exists():
            print(f"Loading FINE-TUNED model from {finetuned_path}")
            self._model = load_model(str(finetuned_path))
        else:
            # Fall back to pre-trained model
            model_file_path = (Path(__file__).parent.parent.parent / "model" / "model.ckpt").resolve()
            print(f"Loading PRE-TRAINED model from {model_file_path}")
            self._model = load_model(str(model_file_path))
        
        self._forecast = self._model.forecast
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
```

### Direct Usage

```python
from tirex import load_model
import torch
import numpy as np

# Load your fine-tuned model
model = load_model("./crypto_finetuned/checkpoints/best_model.ckpt")

# Prepare your context data (last 512 points)
context = your_crypto_prices[-512:]
context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

# Make forecast
predictions = model.forecast(context=context_tensor, prediction_length=32)

# Extract quantiles
quantiles = predictions.cpu().numpy()[0]
median_forecast = quantiles[4, :]  # 0.5 quantile
lower_bound = quantiles[0, :]      # 0.1 quantile  
upper_bound = quantiles[-1, :]     # 0.9 quantile

print(f"Median forecast: {median_forecast}")
print(f"80% prediction interval: [{lower_bound}, {upper_bound}]")
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python crypto_finetune.py --batch-size 16

# Reduce context length
python crypto_finetune.py --context-length 256

# Use gradient accumulation
# (Edit script to add: trainer = L.Trainer(..., accumulate_grad_batches=4))
```

#### 2. Model Not Learning

**Symptoms**: Loss not decreasing, high MAE

**Solutions**:
- Increase learning rate: `--learning-rate 5e-5`
- More epochs: `--max-epochs 50`
- Check data quality: Ensure no NaN values
- Verify data scaling is appropriate

#### 3. Overfitting

**Symptoms**: Train loss decreases but val loss increases

**Solutions**:
- Early stopping (already enabled)
- Add dropout (requires model modification)
- More training data
- Reduce model capacity (train only final layers)

#### 4. Slow Training

**Solutions**:
```bash
# Use mixed precision (enabled by default on GPU)
# Increase batch size if memory allows
python crypto_finetune.py --batch-size 64

# Reduce workers if CPU bottleneck
# (Edit script: num_workers=2)

# Use multiple GPUs
python crypto_finetune.py --num-gpus 2
```

### Performance Optimization Tips

1. **Data Loading**: Use SSD storage for faster I/O
2. **Batch Size**: Larger batches = faster training (if memory allows)
3. **Mixed Precision**: Enabled by default on GPU, 2x speedup
4. **Multiple GPUs**: Near-linear scaling with more GPUs
5. **Compiled Model**: Use `torch.compile()` in PyTorch 2.0+

---

## Advanced: Fine-tuning Only Specific Layers

If you want to freeze most of the model and only train the output layers:

```python
class CryptoFineTuner(L.LightningModule):
    def __init__(self, model, freeze_encoder=True, **kwargs):
        super().__init__()
        self.model = model
        
        if freeze_encoder:
            # Freeze encoder layers
            for name, param in model.named_parameters():
                if 'block_stack' in name:
                    param.requires_grad = False
                    
            print("Frozen encoder layers, training only input/output layers")
        
        # ... rest of initialization
```

This approach:
- Trains much faster
- Requires less data
- Prevents catastrophic forgetting
- Good for domain adaptation

---

## Next Steps

After fine-tuning:

1. **Compare Performance**: Run your optimization with both pre-trained and fine-tuned models
2. **Ensemble Methods**: Combine predictions from multiple checkpoints
3. **Continual Learning**: Periodically retrain on new data
4. **Multi-asset**: Fine-tune on multiple cryptocurrencies
5. **Feature Engineering**: Add technical indicators as additional context

---

## Summary

You now have a complete pipeline for fine-tuning TiRex on your cryptocurrency data:

✅ Data preparation script  
✅ Training script with monitoring  
✅ Evaluation framework  
✅ Integration with existing code  
✅ Troubleshooting guide  

**Quick Start Checklist**:
- [ ] Run `prepare_crypto_data.py`
- [ ] Test with `--quick-test` flag
- [ ] Monitor with TensorBoard
- [ ] Evaluate on test set
- [ ] Integrate best checkpoint into your workflow

Good luck with your fine-tuning! 🚀

