# -*- coding: utf-8 -*-
"""
Unit tests for Gap Filling and Interpolation

Tests the fill_missing_data function and assert_data_continuity with fill_gaps=True.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from tirex.utils.bitmex_transforms import (
    fill_missing_data,
    assert_data_continuity,
    create_ohlcv_dict
)


class TestFillMissingData(unittest.TestCase):
    """Test fill_missing_data function."""
    
    def test_continuous_data_no_filling(self):
        """Test that continuous data is not modified."""
        dates = pd.date_range('2021-01-01', periods=10, freq='1min')
        data = create_ohlcv_dict(
            list_open=[100.0] * 10,
            list_close=[101.0] * 10,
            list_high=[102.0] * 10,
            list_low=[99.0] * 10,
            list_volume=[1000.0] * 10,
            list_trades=[50.0] * 10,
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1)
        
        # Should be identical
        self.assertEqual(len(filled['date']), 10)
        np.testing.assert_array_equal(filled['date'], data['date'])
        np.testing.assert_array_equal(filled['open'], data['open'])
    
    def test_single_gap_linear_interpolation(self):
        """Test linear interpolation with single gap."""
        # Create data with one missing point
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:03',  # Missing 00:02
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 102.0, 106.0],
            list_close=[101.0, 103.0, 107.0],
            list_high=[102.0, 104.0, 108.0],
            list_low=[99.0, 101.0, 105.0],
            list_volume=[1000.0, 1100.0, 1200.0],
            list_trades=[50.0, 55.0, 60.0],
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1, method='linear')
        
        # Should have 4 points (00:00, 00:01, 00:02, 00:03)
        self.assertEqual(len(filled['date']), 4)
        
        # Check interpolated values at 00:02 (index 2)
        # Linear interpolation: (102 + 106) / 2 = 104
        self.assertAlmostEqual(filled['open'][2], 104.0, places=1)
        self.assertAlmostEqual(filled['close'][2], 105.0, places=1)
        self.assertAlmostEqual(filled['high'][2], 106.0, places=1)
        self.assertAlmostEqual(filled['low'][2], 103.0, places=1)
        
        # Volume and trades should be 0 for interpolated points
        self.assertEqual(filled['volume'][2], 0.0)
        self.assertEqual(filled['trades'][2], 0.0)
    
    def test_multiple_gaps_linear_interpolation(self):
        """Test linear interpolation with multiple gaps."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # Missing 00:02, 00:03, 00:04
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 101.0, 105.0],
            list_close=[100.5, 101.5, 105.5],
            list_high=[101.0, 102.0, 106.0],
            list_low=[99.5, 100.5, 104.5],
            list_volume=[1000.0, 1100.0, 1200.0],
            list_trades=[50.0, 55.0, 60.0],
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1, method='linear')
        
        # Should have 6 points (00:00 to 00:05)
        self.assertEqual(len(filled['date']), 6)
        
        # Check that all missing points are filled
        expected_timestamps = pd.date_range('2021-01-01 00:00', periods=6, freq='1min')
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(filled['date']),
            expected_timestamps
        )
        
        # Verify interpolated values are between original values
        self.assertTrue(filled['open'][2] > 101.0 and filled['open'][2] < 105.0)
        self.assertTrue(filled['open'][3] > 101.0 and filled['open'][3] < 105.0)
        self.assertTrue(filled['open'][4] > 101.0 and filled['open'][4] < 105.0)
    
    def test_forward_fill_method(self):
        """Test forward fill interpolation."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:03',  # Missing 00:02
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 102.0, 106.0],
            list_close=[101.0, 103.0, 107.0],
            list_high=[102.0, 104.0, 108.0],
            list_low=[99.0, 101.0, 105.0],
            list_volume=[1000.0, 1100.0, 1200.0],
            list_trades=[50.0, 55.0, 60.0],
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1, method='forward')
        
        # Should have 4 points
        self.assertEqual(len(filled['date']), 4)
        
        # Forward fill: 00:02 should have same values as 00:01
        self.assertEqual(filled['open'][2], 102.0)
        self.assertEqual(filled['close'][2], 103.0)
        self.assertEqual(filled['high'][2], 104.0)
        self.assertEqual(filled['low'][2], 101.0)
    
    def test_backward_fill_method(self):
        """Test backward fill interpolation."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:02',  # Missing 00:01
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 106.0],
            list_close=[101.0, 107.0],
            list_high=[102.0, 108.0],
            list_low=[99.0, 105.0],
            list_volume=[1000.0, 1200.0],
            list_trades=[50.0, 60.0],
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1, method='backward')
        
        # Should have 3 points
        self.assertEqual(len(filled['date']), 3)
        
        # Backward fill: 00:01 should have same values as 00:02
        self.assertEqual(filled['open'][1], 106.0)
        self.assertEqual(filled['close'][1], 107.0)
    
    def test_nearest_fill_method(self):
        """Test nearest neighbor interpolation."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:03',  # Missing 00:01, 00:02
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 110.0],
            list_close=[101.0, 111.0],
            list_high=[102.0, 112.0],
            list_low=[99.0, 109.0],
            list_volume=[1000.0, 1200.0],
            list_trades=[50.0, 60.0],
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1, method='nearest')
        
        # Should have 4 points
        self.assertEqual(len(filled['date']), 4)
        
        # Nearest neighbor: 00:01 closer to 00:00, 00:02 closer to 00:03
        self.assertEqual(filled['open'][1], 100.0)  # Closer to 00:00
        self.assertEqual(filled['open'][2], 110.0)  # Closer to 00:03
    
    def test_volume_trades_set_to_zero(self):
        """Test that volume and trades are set to 0 for filled points."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:03',  # Missing 00:01, 00:02
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 110.0],
            list_close=[101.0, 111.0],
            list_high=[102.0, 112.0],
            list_low=[99.0, 109.0],
            list_volume=[1000.0, 1200.0],
            list_trades=[50.0, 60.0],
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1)
        
        # Original points should keep their volume/trades
        self.assertEqual(filled['volume'][0], 1000.0)
        self.assertEqual(filled['volume'][3], 1200.0)
        self.assertEqual(filled['trades'][0], 50.0)
        self.assertEqual(filled['trades'][3], 60.0)
        
        # Filled points should have 0
        self.assertEqual(filled['volume'][1], 0.0)
        self.assertEqual(filled['volume'][2], 0.0)
        self.assertEqual(filled['trades'][1], 0.0)
        self.assertEqual(filled['trades'][2], 0.0)
    
    def test_preserves_data_types(self):
        """Test that data types are preserved."""
        dates = pd.date_range('2021-01-01', periods=3, freq='1min')
        data = create_ohlcv_dict(
            list_open=[100.0, 101.0, 102.0],
            list_close=[100.5, 101.5, 102.5],
            list_high=[101.0, 102.0, 103.0],
            list_low=[99.5, 100.5, 101.5],
            list_volume=[1000.0, 1100.0, 1200.0],
            list_trades=[50.0, 55.0, 60.0],
            list_time=dates.tolist()
        )
        
        filled = fill_missing_data(data, expected_interval_minutes=1)
        
        # Check data types
        self.assertTrue(np.issubdtype(filled['open'].dtype, np.floating))
        self.assertTrue(np.issubdtype(filled['close'].dtype, np.floating))
        self.assertTrue(np.issubdtype(filled['high'].dtype, np.floating))
        self.assertTrue(np.issubdtype(filled['low'].dtype, np.floating))
        self.assertTrue(np.issubdtype(filled['volume'].dtype, np.floating))
        self.assertTrue(np.issubdtype(filled['trades'].dtype, np.floating))
        self.assertTrue(np.issubdtype(filled['date'].dtype, np.datetime64))


class TestAssertDataContinuityWithFilling(unittest.TestCase):
    """Test assert_data_continuity with fill_gaps=True."""
    
    def test_continuous_data_no_filling_needed(self):
        """Test that continuous data returns None for filled_data."""
        dates = pd.date_range('2021-01-01', periods=10, freq='1min')
        data = create_ohlcv_dict(
            list_open=[100.0] * 10,
            list_close=[101.0] * 10,
            list_high=[102.0] * 10,
            list_low=[99.0] * 10,
            list_volume=[1000.0] * 10,
            list_trades=[50.0] * 10,
            list_time=dates.tolist()
        )
        
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            fill_gaps=True
        )
        
        self.assertTrue(report['is_continuous'])
        self.assertIsNone(report['filled_data'])
        self.assertEqual(report['filled_count'], 0)
    
    def test_fill_gaps_automatically(self):
        """Test that gaps are filled when fill_gaps=True."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # Missing 00:02, 00:03, 00:04
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 101.0, 105.0],
            list_close=[100.5, 101.5, 105.5],
            list_high=[101.0, 102.0, 106.0],
            list_low=[99.5, 100.5, 104.5],
            list_volume=[1000.0, 1100.0, 1200.0],
            list_trades=[50.0, 55.0, 60.0],
            list_time=dates.tolist()
        )
        
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            fill_gaps=True,
            interpolation_method='linear'
        )
        
        # Original data was not continuous
        self.assertFalse(report['is_continuous'])
        self.assertEqual(len(report['gaps']), 1)
        
        # But filled data should be provided
        self.assertIsNotNone(report['filled_data'])
        self.assertEqual(len(report['filled_data']['date']), 6)
        self.assertEqual(report['filled_count'], 3)
    
    def test_no_assertion_error_when_filling(self):
        """Test that no error is raised when fill_gaps=True even with raise_on_gaps=True."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:05',  # Gap
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 105.0],
            list_close=[101.0, 106.0],
            list_high=[102.0, 107.0],
            list_low=[99.0, 104.0],
            list_volume=[1000.0, 1200.0],
            list_trades=[50.0, 60.0],
            list_time=dates.tolist()
        )
        
        # Should NOT raise even though raise_on_gaps=True
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            raise_on_gaps=True,
            fill_gaps=True
        )
        
        self.assertIsNotNone(report['filled_data'])
        self.assertEqual(len(report['filled_data']['date']), 6)
    
    def test_different_interpolation_methods(self):
        """Test that different interpolation methods work."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:03',
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 110.0],
            list_close=[101.0, 111.0],
            list_high=[102.0, 112.0],
            list_low=[99.0, 109.0],
            list_volume=[1000.0, 1200.0],
            list_trades=[50.0, 60.0],
            list_time=dates.tolist()
        )
        
        # Test linear
        report_linear = assert_data_continuity(
            data, expected_interval_minutes=1,
            fill_gaps=True, interpolation_method='linear'
        )
        self.assertIsNotNone(report_linear['filled_data'])
        
        # Test forward
        report_forward = assert_data_continuity(
            data, expected_interval_minutes=1,
            fill_gaps=True, interpolation_method='forward'
        )
        self.assertIsNotNone(report_forward['filled_data'])
        
        # Test backward
        report_backward = assert_data_continuity(
            data, expected_interval_minutes=1,
            fill_gaps=True, interpolation_method='backward'
        )
        self.assertIsNotNone(report_backward['filled_data'])
    
    def test_filled_data_is_continuous(self):
        """Test that filled data passes continuity validation."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # Gap
            '2021-01-01 00:06',
        ])
        
        data = create_ohlcv_dict(
            list_open=[100.0, 101.0, 105.0, 106.0],
            list_close=[100.5, 101.5, 105.5, 106.5],
            list_high=[101.0, 102.0, 106.0, 107.0],
            list_low=[99.5, 100.5, 104.5, 105.5],
            list_volume=[1000.0, 1100.0, 1200.0, 1300.0],
            list_trades=[50.0, 55.0, 60.0, 65.0],
            list_time=dates.tolist()
        )
        
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            fill_gaps=True
        )
        
        # Verify filled data
        filled_data = report['filled_data']
        self.assertIsNotNone(filled_data)
        
        # Validate filled data is continuous
        filled_report = assert_data_continuity(
            filled_data,
            expected_interval_minutes=1,
            fill_gaps=False,
            raise_on_gaps=True
        )
        
        # Should be continuous now
        self.assertTrue(filled_report['is_continuous'])
        self.assertEqual(len(filled_report['gaps']), 0)


class TestIntegrationGapFilling(unittest.TestCase):
    """Integration tests for gap filling."""
    
    def test_realistic_data_with_gaps(self):
        """Test with realistic Bitcoin-like data with gaps."""
        # Create realistic data with gaps
        dates_list = []
        base = pd.Timestamp('2021-01-01 00:00')
        
        # Add continuous data for first 10 minutes
        for i in range(10):
            dates_list.append(base + pd.Timedelta(minutes=i))
        
        # Skip 5 minutes (gap)
        
        # Add continuous data for next 10 minutes
        for i in range(15, 25):
            dates_list.append(base + pd.Timedelta(minutes=i))
        
        dates = pd.DatetimeIndex(dates_list)
        n = len(dates)
        
        # Generate realistic OHLC data
        base_price = 50000
        opens = base_price + np.random.randn(n) * 100
        closes = opens + np.random.randn(n) * 50
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 20)
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 20)
        
        data = create_ohlcv_dict(
            list_open=opens.tolist(),
            list_close=closes.tolist(),
            list_high=highs.tolist(),
            list_low=lows.tolist(),
            list_volume=(np.random.uniform(10000, 50000, n)).tolist(),
            list_trades=(np.random.uniform(50, 200, n)).tolist(),
            list_time=dates.tolist()
        )
        
        # Fill gaps
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            fill_gaps=True,
            interpolation_method='linear'
        )
        
        # Should have filled 5 missing points
        self.assertEqual(report['filled_count'], 5)
        self.assertEqual(len(report['filled_data']['date']), 25)
        
        # Verify filled data is continuous
        filled_report = assert_data_continuity(
            report['filled_data'],
            expected_interval_minutes=1,
            raise_on_gaps=True
        )
        self.assertTrue(filled_report['is_continuous'])


if __name__ == '__main__':
    unittest.main()
