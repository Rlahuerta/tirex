#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script for BitMEX latest data fetcher.
This performs a simple smoke test without making actual API calls.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
# from pathlib import Path

from tirex.utils.bitmex_latest import (
    get_latest_bitmex_data,
    fetch_and_plot_latest_btc,
    plot_ticker_with_thick_lines
)


class TestBitMEXLatest(unittest.TestCase):
    """Test suite for BitMEX latest data fetcher."""
    
    def setUp(self):
        """Create mock data for testing."""
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=96, freq='15min')
        self.mock_data = pd.DataFrame({
            'open': np.random.uniform(45000, 46000, 96),
            'high': np.random.uniform(45500, 46500, 96),
            'low': np.random.uniform(44500, 45500, 96),
            'close': np.random.uniform(45000, 46000, 96),
            'volume': np.random.uniform(1000000, 2000000, 96),
            'trades': np.random.uniform(500, 1000, 96)
        }, index=dates)
        
        # Ensure OHLC consistency
        self.mock_data['high'] = self.mock_data[['open', 'high', 'close']].max(axis=1)
        self.mock_data['low'] = self.mock_data[['open', 'low', 'close']].min(axis=1)
    
    @patch('tirex.utils.bitmex_latest.BitMEX')
    def test_get_latest_bitmex_data_basic(self, mock_bitmex_class):
        """Test basic data fetching."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.get_net_chart.return_value = self.mock_data
        mock_bitmex_class.return_value = mock_instance
        
        # Test
        df, fig = get_latest_bitmex_data(
            symbol='XBTUSD',
            hours=24,
            dt=15,
            plot=False
        )
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsNone(fig)
        self.assertEqual(len(df), 96)
        self.assertIn('close', df.columns)
        
        # Verify BitMEX was called correctly
        mock_instance.get_net_chart.assert_called_once()
        mock_instance.close.assert_called_once()
    
    @patch('tirex.utils.bitmex_latest.BitMEX')
    def test_get_latest_bitmex_data_with_plot(self, mock_bitmex_class):
        """Test data fetching with plot generation."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.get_net_chart.return_value = self.mock_data
        mock_bitmex_class.return_value = mock_instance
        
        # Test
        df, fig = get_latest_bitmex_data(
            symbol='XBTUSD',
            hours=24,
            dt=15,
            plot=True
        )
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsNotNone(fig)
        self.assertEqual(len(df), 96)
    
    @patch('tirex.utils.bitmex_latest.BitMEX')
    def test_fetch_and_plot_latest_btc(self, mock_bitmex_class):
        """Test convenience function for Bitcoin."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.get_net_chart.return_value = self.mock_data
        mock_bitmex_class.return_value = mock_instance
        
        # Test
        df, fig = fetch_and_plot_latest_btc(hours=24)
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsNotNone(fig)
        self.assertEqual(len(df), 96)
        
        # Verify correct parameters were used
        call_args = mock_instance.get_net_chart.call_args
        self.assertEqual(call_args[1]['cpair'], 'XBTUSD')
        self.assertEqual(call_args[1]['dt'], 3)  # dt=15 maps to dt_internal=3
    
    def test_plot_ticker_with_thick_lines(self):
        """Test plot generation with thick lines."""
        # Test
        fig = plot_ticker_with_thick_lines(
            ticker_data=self.mock_data,
            dt=15,
            title="Test Plot",
            linewidth=3.0
        )
        
        # Assertions
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_suptitle(), "Test Plot")
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid dt value
        with self.assertRaises(AssertionError):
            get_latest_bitmex_data('XBTUSD', hours=24, dt=10)
        
        # Invalid hours
        with self.assertRaises(AssertionError):
            get_latest_bitmex_data('XBTUSD', hours=-1, dt=15)
    
    def test_dataframe_validation(self):
        """Test DataFrame validation in plotting."""
        # Missing required columns
        invalid_df = pd.DataFrame({'close': [100, 101, 102]})
        
        with self.assertRaises(AssertionError):
            plot_ticker_with_thick_lines(invalid_df, dt=15)
        
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        with self.assertRaises(AssertionError):
            plot_ticker_with_thick_lines(empty_df, dt=15)


def run_tests():
    """Run all tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBitMEXLatest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*70)
    print("BitMEX Latest Data Fetcher - Unit Tests")
    print("="*70)
    print("\nRunning tests...")
    print()
    
    success = run_tests()
    
    print()
    print("="*70)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*70)
