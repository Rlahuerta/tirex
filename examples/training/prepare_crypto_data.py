"""
Prepare cryptocurrency data for TiRex fine-tuning.

This script loads your crypto data and prepares it into sequences suitable for training.
"""
import numpy as np
import pandas as pd
from joblib import load, dump
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import RobustScaler
import argparse


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
        print(f"✓ Loaded {len(self.data)} samples at {self.time_interval}-minute intervals")
        print(f"  Date range: {self.data.index[0]} to {self.data.index[-1]}")
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        print(f"\n✓ Creating features...")
        df = self.create_features()
        values = df[feature_column].values
        
        print(f"✓ Normalizing data using RobustScaler...")
        values_scaled = self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        # Create sequences
        print(f"✓ Creating sequences (context={context_length}, forecast={prediction_length}, stride={stride})...")
        sequences = []
        total_length = context_length + prediction_length
        
        for i in range(0, len(values_scaled) - total_length + 1, stride):
            seq = values_scaled[i:i + total_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        print(f"  Created {len(sequences)} sequences of length {total_length}")
        
        # Split data chronologically (important for time series!)
        n = len(sequences)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        train_seqs = sequences[:train_end]
        val_seqs = sequences[train_end:val_end]
        test_seqs = sequences[val_end:]
        
        print(f"\n✓ Data split:")
        print(f"  Train: {len(train_seqs)} sequences ({len(train_seqs)/n*100:.1f}%)")
        print(f"  Val:   {len(val_seqs)} sequences ({len(val_seqs)/n*100:.1f}%)")
        print(f"  Test:  {len(test_seqs)} sequences ({len(test_seqs)/n*100:.1f}%)")
        
        return train_seqs, val_seqs, test_seqs
    
    def save_prepared_data(
        self,
        output_dir: str,
        feature_column: str = 'close',
        context_length: int = 512,
        prediction_length: int = 32,
    ):
        """Save prepared sequences to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        train, val, test = self.prepare_training_sequences(
            feature_column=feature_column,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        
        output_file = output_path / 'crypto_prepared.joblib'
        
        dump({
            'train': train,
            'val': val,
            'test': test,
            'scaler': self.scaler,
            'time_interval': self.time_interval,
            'feature_column': feature_column,
            'context_length': context_length,
            'prediction_length': prediction_length,
        }, output_file)
        
        print(f"\n✓ Saved prepared data to {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare cryptocurrency data for TiRex fine-tuning")
    
    parser.add_argument('--data-path', type=str,
                        default='../../tests/data/btcusd_2022-06-01.joblib',
                        help='Path to input data file')
    parser.add_argument('--output-dir', type=str,
                        default='./prepared_data',
                        help='Output directory for prepared data')
    parser.add_argument('--time-interval', type=int, default=60,
                        help='Time interval in minutes (15, 60, etc.)')
    parser.add_argument('--feature', type=str, default='close',
                        help='Feature column to use (close, volume, etc.)')
    parser.add_argument('--context-length', type=int, default=512,
                        help='Input context length')
    parser.add_argument('--prediction-length', type=int, default=32,
                        help='Forecast horizon length')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TiRex Cryptocurrency Data Preparation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input:              {args.data_path}")
    print(f"  Output:             {args.output_dir}")
    print(f"  Time interval:      {args.time_interval} minutes")
    print(f"  Feature:            {args.feature}")
    print(f"  Context length:     {args.context_length}")
    print(f"  Prediction length:  {args.prediction_length}")
    print()
    
    # Create preparator and process data
    preparator = CryptoDataPreparator(
        data_path=args.data_path,
        time_interval=args.time_interval
    )
    
    preparator.load_data()
    preparator.save_prepared_data(
        output_dir=args.output_dir,
        feature_column=args.feature,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )
    
    print("\n" + "=" * 80)
    print("✓ Data preparation completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Test: python crypto_finetune.py --quick-test")
    print("  2. Train: python crypto_finetune.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

