"""Unit tests for dt parameter in get_net_chart method."""

import pytest
import pandas as pd
# import numpy as np

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from tirex.utils.bitmex import BitMEX

plot_path = (Path(__file__).parent / 'plots' / 'ticker').resolve()


class TestGetNetChartDtParameter:
    """Test that dt parameter correctly maps to time resolutions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.bitmex = BitMEX(base_url='https://testnet.bitmex.com/api/v1/')
    
    @pytest.mark.parametrize("dt,expected_bin_size,expected_interval", [
        (1, '5m', 5),   # 5-minute bars, no resampling
        (3, '5m', 15),  # 15-minute bars, resampled from 5m
        (6, '5m', 30),  # 30-minute bars, resampled from 5m
        (12, '1h', 60), # 60-minute bars, no resampling
    ])
    def test_dt_parameter_mapping(self, dt, expected_bin_size, expected_interval):
        """Test that dt parameter correctly maps to API bin size and final interval."""
        # Mock the HTTP client to avoid actual API calls
        with patch.object(self.bitmex, 'get_trade_bucketed') as mock_get:
            # Create mock data with proper time intervals
            # Start at a well-aligned time (12:00)
            start_time = pd.Timestamp('2024-01-01 12:00:00')
            mock_data = []
            
            # Determine the base interval from bin_size
            base_interval = 5 if expected_bin_size == '5m' else 60
            
            # Generate 100 sample data points with proper alignment
            for i in range(100):
                timestamp = start_time + pd.Timedelta(minutes=base_interval * i)
                mock_data.append({
                    'timestamp': timestamp.isoformat(),
                    'open': 50000.0 + i,
                    'high': 50100.0 + i,
                    'low': 49900.0 + i,
                    'close': 50050.0 + i,
                    'volume': 1000000,
                    'trades': 100
                })
            
            mock_get.return_value = mock_data
            
            # Call get_net_chart
            result = self.bitmex.get_net_chart(hours=8, cpair='XBTUSD', dt=dt)
            
            # Verify the result is a DataFrame
            assert isinstance(result, pd.DataFrame)
            assert not result.empty, f"Result is empty for dt={dt}"
            
            # Verify the time interval between consecutive rows
            if len(result) > 1:
                time_diffs = result.index.to_series().diff().dropna()
                # Convert to minutes
                time_diffs_minutes = time_diffs.dt.total_seconds() / 60
                
                # Check that intervals are close to expected (within 1 minute tolerance)
                # For dt=1 and dt=12, we expect base_interval
                # For dt=3 and dt=6, we expect resampled interval
                if dt in [1, 12]:
                    expected_diff = base_interval
                else:
                    expected_diff = expected_interval
                
                # Allow some tolerance for edge cases
                mean_diff = time_diffs_minutes.mean()
                assert abs(mean_diff - expected_diff) < 5, \
                    f"Expected ~{expected_diff} min intervals, got ~{mean_diff:.1f} min"
    
    def test_dt_1_uses_5minute_data(self):
        """Test that dt=1 fetches 5-minute bars without resampling."""
        with patch.object(self.bitmex, 'get_trade_bucketed') as mock_get:
            # Setup mock
            mock_get.return_value = [{
                'timestamp': pd.Timestamp.now().isoformat(),
                'open': 50000, 'high': 50100, 'low': 49900, 'close': 50050,
                'volume': 1000000, 'trades': 100
            }]
            
            # Call with dt=1
            self.bitmex.get_net_chart(hours=1, cpair='XBTUSD', dt=1)
            
            # Verify bin_size parameter
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['bin_size'] == '5m'
    
    def test_dt_3_uses_5minute_data_resampled_to_15(self):
        """Test that dt=3 fetches 5-minute bars and resamples to 15 minutes."""
        with patch.object(self.bitmex, 'get_trade_bucketed') as mock_get:
            # Setup mock with data at 5-minute intervals
            now = pd.Timestamp('2024-01-01 12:00:00')
            mock_data = []
            for i in range(12):  # 1 hour of 5-minute data
                timestamp = now + pd.Timedelta(minutes=5 * i)
                mock_data.append({
                    'timestamp': timestamp.isoformat(),
                    'open': 50000 + i, 'high': 50100 + i, 'low': 49900 + i, 
                    'close': 50050 + i, 'volume': 1000000, 'trades': 100
                })
            
            mock_get.return_value = mock_data
            
            # Call with dt=3
            result = self.bitmex.get_net_chart(hours=1, cpair='XBTUSD', dt=3)
            
            # Verify bin_size parameter
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['bin_size'] == '5m'
            
            # Verify resampling occurred (should have 4 bars: 12 5-min bars -> 4 15-min bars)
            # Due to alignment, may have 3-4 bars
            assert len(result) <= 4
    
    def test_dt_12_uses_1hour_data(self):
        """Test that dt=12 fetches 1-hour bars without resampling."""
        with patch.object(self.bitmex, 'get_trade_bucketed') as mock_get:
            # Setup mock
            mock_get.return_value = [{
                'timestamp': pd.Timestamp.now().isoformat(),
                'open': 50000, 'high': 50100, 'low': 49900, 'close': 50050,
                'volume': 1000000, 'trades': 100
            }]
            
            # Call with dt=12
            self.bitmex.get_net_chart(hours=1, cpair='XBTUSD', dt=12)
            
            # Verify bin_size parameter
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['bin_size'] == '1h'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
