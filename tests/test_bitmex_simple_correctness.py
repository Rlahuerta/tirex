"""
Comprehensive correctness tests for BitMEX refactored implementation.

These tests verify:
1. Data structure is correct (DataFrame format, columns, index)
2. OHLCV constraints are maintained (financial data validity)
3. Time intervals match expected dt values (data continuity)
4. Resampling preserves data trends
5. No data loss or corruption
"""

import pytest
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta

from tirex.utils.bitmex import BitMEX

matplotlib.use('Agg')
plot_path = (Path(__file__).parent / 'plots' / 'ticker').resolve()


def create_valid_mock_data(n_samples=100, base_price=40000, interval_minutes=5):
    """Create valid OHLCV mock data."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=None)
    data = []
    
    current_price = base_price
    for i in range(n_samples):
        timestamp = base_time + timedelta(minutes=interval_minutes*i)
        
        # Realistic price movement
        price_change = np.random.randn() * 50
        current_price += price_change
        
        open_price = current_price
        close_price = current_price + np.random.randn() * 30
        
        # Ensure OHLCV constraints
        high_price = max(open_price, close_price) + abs(np.random.randn() * 20)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 20)
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'open': float(open_price),
            'high': float(high_price),
            'low': float(low_price),
            'close': float(close_price),
            'volume': float(np.random.randint(500000, 1500000)),
            'trades': float(np.random.randint(100, 500))
        })
    
    return data


class TestBitMEXOutputStructure:
    """Test that BitMEX returns correct data structures."""
    
    def test_get_net_chart_returns_dataframe(self):
        """Verify get_net_chart returns a pandas DataFrame."""
        mock_data = create_valid_mock_data(n_samples=50)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=1.0, cpair='XBTUSD', dt=1)
            
            assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"
            print(f"✅ get_net_chart returns DataFrame with shape {result.shape}")
    
    def test_dataframe_has_required_columns(self):
        """Verify DataFrame has all required OHLCV columns."""
        mock_data = create_valid_mock_data(n_samples=50)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=1.0, cpair='XBTUSD', dt=1)
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                assert col in result.columns, f"Missing required column: {col}"
            
            print(f"✅ DataFrame has all required columns: {list(result.columns)}")
    
    def test_dataframe_index_is_datetime(self):
        """Verify DataFrame index is DatetimeIndex."""
        mock_data = create_valid_mock_data(n_samples=50)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=1.0, cpair='XBTUSD', dt=1)
            
            assert isinstance(result.index, pd.DatetimeIndex), \
                f"Expected DatetimeIndex, got {type(result.index)}"
            
            print(f"✅ DataFrame index is DatetimeIndex with {len(result)} entries")


class TestOHLCVConstraints:
    """Test that OHLCV data maintains financial constraints."""
    
    def test_ohlcv_constraints_maintained(self):
        """Verify that High >= max(Open, Close) and Low <= min(Open, Close)."""
        mock_data = create_valid_mock_data(n_samples=100)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=2.0, cpair='XBTUSD', dt=1)
            
            df = result
            
            # Check OHLCV constraints
            violations_high = (df['high'] < df['open']) | (df['high'] < df['close'])
            violations_low = (df['low'] > df['open']) | (df['low'] > df['close'])
            
            print(f"\n📊 OHLCV Constraint Verification:")
            print(f"  Total samples: {len(df)}")
            print(f"  High violations: {violations_high.sum()}")
            print(f"  Low violations: {violations_low.sum()}")
            
            # Visual verification
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Candlestick-like plot
            for idx in range(min(50, len(df))):  # Plot first 50 samples
                row = df.iloc[idx]
                color = 'g' if row['close'] >= row['open'] else 'r'
                
                # Draw high-low line
                axes[0].plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
                # Draw open-close box
                height = abs(row['close'] - row['open'])
                bottom = min(row['open'], row['close'])
                axes[0].add_patch(plt.Rectangle((idx-0.3, bottom), 0.6, height, 
                                                facecolor=color, alpha=0.5))
            
            axes[0].set_title('OHLC Candlestick Visualization (First 50 samples)')
            axes[0].set_ylabel('Price')
            axes[0].grid(True, alpha=0.3)
            
            # Check relationships
            axes[1].plot(df.index, df['high'] - df['open'], label='High - Open', alpha=0.7)
            axes[1].plot(df.index, df['high'] - df['close'], label='High - Close', alpha=0.7)
            axes[1].plot(df.index, df['open'] - df['low'], label='Open - Low', alpha=0.7)
            axes[1].plot(df.index, df['close'] - df['low'], label='Close - Low', alpha=0.7)
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1].set_title('OHLC Relationships (should all be >= 0)')
            axes[1].set_ylabel('Price Difference')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            local_plot_path = plot_path / 'ohlcv_constraints.png'
            plt.savefig(local_plot_path, dpi=100)
            plt.close()
            
            print(f"📈 Plot saved to: {local_plot_path}")
            
            # Assertions
            assert violations_high.sum() == 0, f"Found {violations_high.sum()} high price violations"
            assert violations_low.sum() == 0, f"Found {violations_low.sum()} low price violations"
            
            # Additional checks
            assert np.all(df['high'] >= df['low']), "High should always be >= Low"
            assert np.all(df['volume'] >= 0), "Volume should be non-negative"
            
            print("✅ OHLCV constraints test passed")


class TestTimeIntervals:
    """Test that time intervals match expected dt values."""
    
    def test_dt1_gives_5minute_intervals(self, tmp_path):
        """Verify dt=1 produces 5-minute intervals."""
        mock_data = create_valid_mock_data(n_samples=100, interval_minutes=5)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=2.0, cpair='XBTUSD', dt=1)
            
            # Calculate time differences
            time_diffs = result.index.to_series().diff()[1:]
            time_diffs_minutes = time_diffs.dt.total_seconds() / 60
            
            print(f"\n📊 dt=1 Time interval analysis:")
            print(f"  Samples: {len(result)}")
            print(f"  Mean interval: {time_diffs_minutes.mean():.2f} minutes")
            print(f"  Median interval: {time_diffs_minutes.median():.2f} minutes")
            print(f"  Min interval: {time_diffs_minutes.min():.2f} minutes")
            print(f"  Max interval: {time_diffs_minutes.max():.2f} minutes")
            
            # Plot for visual verification
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(time_diffs_minutes, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=5, color='r', linestyle='--', linewidth=2, label='Expected (5min)')
            ax.set_xlabel('Time Interval (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'dt=1: Time Interval Distribution\n(Mean: {time_diffs_minutes.mean():.2f}min)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            local_plot_path = plot_path / 'dt1_intervals.png'
            plt.savefig(local_plot_path, dpi=100)
            plt.close()
            
            print(f"📈 Histogram saved to: {local_plot_path}")
            
            # Assert median is close to 5 minutes
            assert abs(time_diffs_minutes.median() - 5) < 1, \
                f"Median interval {time_diffs_minutes.median()} not close to 5 minutes"
            
            print("✅ dt=1 produces ~5-minute intervals")
    
    def test_dt3_gives_15minute_intervals(self):
        """Verify dt=3 produces 15-minute intervals after resampling."""
        # Create fine-grained data that will be resampled
        mock_data = create_valid_mock_data(n_samples=200, interval_minutes=5)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=4.0, cpair='XBTUSD', dt=3)
            
            # Calculate time differences
            time_diffs = result.index.to_series().diff()[1:]
            time_diffs_minutes = time_diffs.dt.total_seconds() / 60
            
            print(f"\n📊 dt=3 Time interval analysis:")
            print(f"  Samples: {len(result)}")
            print(f"  Mean interval: {time_diffs_minutes.mean():.2f} minutes")
            print(f"  Median interval: {time_diffs_minutes.median():.2f} minutes")
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(time_diffs_minutes, bins=20, alpha=0.7, edgecolor='black', color='green')
            ax.axvline(x=15, color='r', linestyle='--', linewidth=2, label='Expected (15min)')
            ax.set_xlabel('Time Interval (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'dt=3: Time Interval Distribution\n(Mean: {time_diffs_minutes.mean():.2f}min)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            local_plot_path = plot_path / 'dt3_intervals.png'
            plt.savefig(local_plot_path, dpi=100)
            plt.close()
            
            print(f"📈 Histogram saved to: {local_plot_path}")
            
            # Assert median is close to 15 minutes
            assert abs(time_diffs_minutes.median() - 15) < 2, \
                f"Median interval {time_diffs_minutes.median()} not close to 15 minutes"
            
            print("✅ dt=3 produces ~15-minute intervals")
    
    def test_dt12_gives_60minute_intervals(self, tmp_path):
        """Verify dt=12 produces 60-minute intervals."""
        # dt=12 fetches 60m data directly
        mock_data = create_valid_mock_data(n_samples=100, interval_minutes=60)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=24.0, cpair='XBTUSD', dt=12)
            
            # Calculate time differences
            time_diffs = result.index.to_series().diff()[1:]
            time_diffs_minutes = time_diffs.dt.total_seconds() / 60
            
            print(f"\n📊 dt=12 Time interval analysis:")
            print(f"  Samples: {len(result)}")
            print(f"  Mean interval: {time_diffs_minutes.mean():.2f} minutes")
            print(f"  Median interval: {time_diffs_minutes.median():.2f} minutes")
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(time_diffs_minutes, bins=20, alpha=0.7, edgecolor='black', color='purple')
            ax.axvline(x=60, color='r', linestyle='--', linewidth=2, label='Expected (60min)')
            ax.set_xlabel('Time Interval (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'dt=12: Time Interval Distribution\n(Mean: {time_diffs_minutes.mean():.2f}min)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            local_plot_path = plot_path / 'dt12_intervals.png'
            plt.savefig(local_plot_path, dpi=100)
            plt.close()
            
            print(f"📈 Histogram saved to: {local_plot_path}")
            
            # Assert median is close to 60 minutes
            assert abs(time_diffs_minutes.median() - 60) < 5, \
                f"Median interval {time_diffs_minutes.median()} not close to 60 minutes"
            
            print("✅ dt=12 produces ~60-minute intervals")


class TestDataVisualization:
    """Create visual plots to inspect data quality."""
    
    def test_price_chart_looks_reasonable(self, tmp_path):
        """Generate and save OHLC chart for visual inspection."""
        mock_data = create_valid_mock_data(n_samples=200)
        
        with patch('tirex.utils.bitmex_client.BitMEXHttpClient.get') as mock_get:
            mock_get.return_value = mock_data
            
            bitmex = BitMEX()
            result = bitmex.get_net_chart(hours=4.0, cpair='XBTUSD', dt=1)
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Plot 1: OHLC with candlestick representation
            ax1 = axes[0]
            for i in range(len(result)):
                row = result.iloc[i]
                color = 'g' if row['close'] >= row['open'] else 'r'
                
                # High-Low line
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1, alpha=0.8)
                
                # Open-Close box
                height = abs(row['close'] - row['open'])
                bottom = min(row['open'], row['close'])
                rect = plt.Rectangle((i-0.3, bottom), 0.6, height, 
                                    facecolor=color, edgecolor=color, alpha=0.6)
                ax1.add_patch(rect)
            
            ax1.set_title(f'OHLC Candlestick Chart ({len(result)} candles)')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Close price with moving average
            ax2 = axes[1]
            ax2.plot(result.index, result['close'], 'b-', label='Close', linewidth=1.5)
            ax2.plot(result.index, result['close'].rolling(window=20).mean(), 
                    'r--', label='MA(20)', linewidth=1.5, alpha=0.7)
            ax2.set_ylabel('Close Price')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Volume
            ax3 = axes[2]
            ax3.bar(result.index, result['volume'], alpha=0.6, color='green', width=0.8/24)
            ax3.set_ylabel('Volume')
            ax3.set_xlabel('Time')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            local_plot_path = plot_path / 'comprehensive_chart.png'
            plt.savefig(local_plot_path, dpi=150)
            plt.close()
            
            print(f"\n📊 Comprehensive chart saved to: {local_plot_path}")
            print(f"📈 Price range: {result['low'].min():.2f} - {result['high'].max():.2f}")
            print(f"📈 Total volume: {result['volume'].sum():.0f}")
            print("✅ Chart generated successfully - inspect visually")


class TestResamplingCorrectness:
    """Test resampling operations."""
    
    def test_resample_preserves_trends(self):
        """Verify that resampling preserves overall price trends."""
        # Create synthetic data with clear trend
        n_samples = 200
        timestamps = np.array([datetime(2024, 1, 1, 0, 0, 0) + timedelta(minutes=5*i) 
                               for i in range(n_samples)])
        
        # Create upward trend
        trend = np.linspace(40000, 45000, n_samples)
        noise = np.random.randn(n_samples) * 100
        closes = trend + noise
        
        # Create OHLC based on close
        opens = closes + np.random.randn(n_samples) * 50
        highs = np.maximum(opens, closes) + abs(np.random.randn(n_samples) * 20)
        lows = np.minimum(opens, closes) - abs(np.random.randn(n_samples) * 20)
        volumes = np.random.randint(500000, 1500000, n_samples)
        
        # Resample to half the samples
        from tirex.utils.bitmex_transforms import resample_ohlcv, create_ohlcv_dict
        
        # Create OHLCV dictionary
        ohlcv_dict = create_ohlcv_dict(
            opens, closes, highs, lows, volumes, 
            np.zeros(n_samples), timestamps
        )
        
        resampled = resample_ohlcv(ohlcv_dict, interval_minutes=15)  # Resample from 5min to 15min
        
        # Visual comparison
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Original data
        axes[0].plot(timestamps, closes, 'b-', label='Original Close', linewidth=1, alpha=0.7)
        axes[0].set_title(f'Original Data ({n_samples} samples, 5min intervals)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Resampled data
        axes[1].plot(resampled.index, resampled['close'], 'r-', 
                    label='Resampled Close', linewidth=2, marker='o', markersize=4)
        axes[1].set_title(f'Resampled Data ({len(resampled)} samples, 15min intervals)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Overlay
        axes[2].plot(timestamps, closes, 'b-', label='Original', linewidth=1, alpha=0.5)
        axes[2].plot(resampled.index, resampled['close'], 'r-', 
                    label='Resampled', linewidth=2, marker='o', markersize=5)
        axes[2].set_title('Overlay Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        local_plot_path = plot_path / 'resampling_trend_preservation.png'
        plt.savefig(local_plot_path, dpi=100)
        plt.close()
        
        print(f"\n📈 Plot saved to: {local_plot_path}")
        print(f"📊 Original samples: {n_samples}")
        print(f"📊 Resampled samples: {len(resampled)}")
        print(f"📊 Original trend: {closes[0]:.2f} -> {closes[-1]:.2f}")
        print(f"📊 Resampled trend: {resampled['close'].iloc[0]:.2f} -> {resampled['close'].iloc[-1]:.2f}")
        
        # Check that trend is preserved
        original_trend = closes[-1] - closes[0]
        resampled_trend = resampled['close'].iloc[-1] - resampled['close'].iloc[0]
        
        assert np.sign(original_trend) == np.sign(resampled_trend), \
            "Resampling should preserve trend direction"
        
        print("✅ Resampling trend preservation test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
