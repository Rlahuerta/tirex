"""
Fine-tune TiRex on cryptocurrency data - Complete working example.

Usage:
    # Quick test (2 epochs, small data)
    python training_example.py --quick-test

    # Full training
    python training_example.py

    # Custom settings
    python training_example.py --batch-size 64 --learning-rate 2e-5 --max-epochs 30
"""
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from joblib import load
from typing import Tuple
import argparse
from datetime import datetime
import sys

# Add parent directory to path to import tirex
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tirex import load_model as load_tirex_model


# ============================================================================
# DATASET
# ============================================================================

class CryptoDataset(Dataset):
    """Dataset for cryptocurrency time series."""

    def __init__(self, sequences: np.ndarray, context_length: int = 512, prediction_length: int = 32):
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

    def __init__(self, model, learning_rate: float = 1e-5, weight_decay: float = 0.01, warmup_steps: int = 500):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.loss_fn = QuantileLoss(model.model_config.quantiles)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        return self.model._forecast_tensor(context, prediction_length=prediction_length, max_accelerated_rollout_steps=1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        context, target = batch
        predictions = self(context, prediction_length=target.shape[-1])
        loss = self.loss_fn(predictions, target)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        median_pred = predictions[:, 4, :]
        mae = torch.abs(median_pred - target).mean()
        self.log('train_mae', mae, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        context, target = batch
        predictions = self(context, prediction_length=target.shape[-1])
        loss = self.loss_fn(predictions, target)

        median_pred = predictions[:, 4, :]
        mae = torch.abs(median_pred - target).mean()
        mse = ((median_pred - target) ** 2).mean()

        lower_bound = predictions[:, 0, :]
        upper_bound = predictions[:, -1, :]
        coverage = ((target >= lower_bound) & (target <= upper_bound)).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mae', mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mse', mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_coverage', coverage, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(1, self.trainer.max_steps - self.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}


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
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print(f"TiRex Cryptocurrency Fine-tuning - {timestamp}")
    print("=" * 80)

    # Load model
    print(f"\n1. Loading pre-trained TiRex model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() and num_gpus > 0 else 'cpu'
    model = load_tirex_model(model_path, device=device)
    print(f"   ✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   ✓ Device: {device}")

    # Load data
    print(f"\n2. Loading prepared crypto data from {data_path}...")
    data_dict = load(data_path)
    train_sequences = data_dict['train']
    val_sequences = data_dict['val']
    test_sequences = data_dict['test']

    if quick_test:
        print("   ⚠ QUICK TEST MODE: Using subset of data")
        train_sequences = train_sequences[:100]
        val_sequences = val_sequences[:20]
        max_epochs = 2
        batch_size = 16

    print(f"   ✓ Train: {len(train_sequences)} sequences")
    print(f"   ✓ Val: {len(val_sequences)} sequences")
    print(f"   ✓ Test: {len(test_sequences)} sequences")

    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_dataset = CryptoDataset(train_sequences, context_length, prediction_length)
    val_dataset = CryptoDataset(val_sequences, context_length, prediction_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches: {len(val_loader)}")

    # Create training module
    print("\n4. Setting up training module...")
    finetuner = CryptoFineTuner(model=model, learning_rate=learning_rate, warmup_steps=min(500, len(train_loader) * 2))

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
            L.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=True),
        ],
        logger=L.pytorch.loggers.TensorBoardLogger(save_dir=str(output_path), name='logs'),
    )

    # Train
    print("\n6. Starting training...\n")
    print("=" * 80)

    trainer.fit(finetuner, train_loader, val_loader)

    # Save final model
    print("\n" + "=" * 80)
    print("✓ Training completed!")
    final_model_path = output_path / f'crypto_tirex_final_{timestamp}.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    print("\nTraining Summary:")
    print(f"  Best val_loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
    print(f"  Best val_mae: {trainer.callback_metrics.get('val_mae', 'N/A')}")
    print(f"  Total epochs: {trainer.current_epoch + 1}")
    print(f"  Checkpoints: {output_path / 'checkpoints'}")
    print(f"  Logs: {output_path / 'logs'}")
    print("\nView logs with: tensorboard --logdir " + str(output_path / 'logs'))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TiRex on cryptocurrency data")

    parser.add_argument('--data-path', type=str, default='./prepared_data/crypto_prepared.joblib')
    parser.add_argument('--model-path', type=str, default='../../model/model.ckpt')
    parser.add_argument('--output-dir', type=str, default='./crypto_finetuned')
    parser.add_argument('--context-length', type=int, default=512)
    parser.add_argument('--prediction-length', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--quick-test', action='store_true')

    args = parser.parse_args()

    L.seed_everything(42)

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

