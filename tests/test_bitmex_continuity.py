# -*- coding: utf-8 -*-
"""
Unit tests for Data Continuity Validation

Tests timestamp continuity checking, gap detection, and duplicate detection
in BitMEX data transformation module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from tirex.utils.bitmex_transforms import (
    validate_timestamp_continuity,
    validate_no_duplicates,
    assert_data_continuity,
    create_ohlcv_dict
)


class TestValidateTimestampContinuity(unittest.TestCase):
    """Test validate_timestamp_continuity function."""
    
    def test_continuous_data_no_gaps(self):
        """Test that continuous data with no gaps passes validation."""
        timestamps = pd.date_range('2021-01-01', periods=100, freq='1min')
        expected_interval = pd.Timedelta(minutes=1)
        
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval
        )
        
        self.assertTrue(is_continuous)
        self.assertEqual(len(gaps), 0)
    
    def test_single_timestamp(self):
        """Test that single timestamp is considered continuous."""
        timestamps = pd.DatetimeIndex(['2021-01-01 00:00'])
        expected_interval = pd.Timedelta(minutes=1)
        
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval
        )
        
        self.assertTrue(is_continuous)
        self.assertEqual(len(gaps), 0)
    
    def test_empty_timestamps(self):
        """Test that empty timestamps are considered continuous."""
        timestamps = pd.DatetimeIndex([])
        expected_interval = pd.Timedelta(minutes=1)
        
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval
        )
        
        self.assertTrue(is_continuous)
        self.assertEqual(len(gaps), 0)
    
    def test_single_gap_detected(self):
        """Test that a single gap is detected."""
        timestamps = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # 3-minute gap
            '2021-01-01 00:06',
        ])
        expected_interval = pd.Timedelta(minutes=1)
        
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval
        )
        
        self.assertFalse(is_continuous)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0][2], pd.Timedelta(minutes=4))
    
    def test_multiple_gaps_detected(self):
        """Test that multiple gaps are detected."""
        timestamps = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # Gap 1
            '2021-01-01 00:06',
            '2021-01-01 00:10',  # Gap 2
            '2021-01-01 00:11',
        ])
        expected_interval = pd.Timedelta(minutes=1)
        
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval
        )
        
        self.assertFalse(is_continuous)
        self.assertEqual(len(gaps), 2)
    
    def test_tolerance_allows_small_variations(self):
        """Test that tolerance allows small timing variations."""
        # Create timestamps with small variations (within 10% tolerance)
        base = pd.Timestamp('2021-01-01 00:00')
        timestamps = pd.DatetimeIndex([
            base,
            base + pd.Timedelta(seconds=59),  # 1 second early
            base + pd.Timedelta(seconds=119),  # 1 second early
            base + pd.Timedelta(seconds=181),  # 1 second late
        ])
        expected_interval = pd.Timedelta(minutes=1)
        
        # With 10% tolerance (6 seconds), these should be OK
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval, tolerance=0.1
        )
        
        self.assertTrue(is_continuous)
        self.assertEqual(len(gaps), 0)
    
    def test_tight_tolerance_detects_small_gaps(self):
        """Test that tight tolerance detects small variations."""
        base = pd.Timestamp('2021-01-01 00:00')
        timestamps = pd.DatetimeIndex([
            base,
            base + pd.Timedelta(seconds=70),  # 10 seconds late
        ])
        expected_interval = pd.Timedelta(minutes=1)
        
        # With 1% tolerance (0.6 seconds), 10 seconds should be detected
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval, tolerance=0.01
        )
        
        self.assertFalse(is_continuous)
        self.assertEqual(len(gaps), 1)
    
    def test_five_minute_intervals(self):
        """Test validation with 5-minute intervals."""
        timestamps = pd.date_range('2021-01-01', periods=100, freq='5min')
        expected_interval = pd.Timedelta(minutes=5)
        
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval
        )
        
        self.assertTrue(is_continuous)
        self.assertEqual(len(gaps), 0)
    
    def test_gap_information_correct(self):
        """Test that gap information contains correct timestamps."""
        timestamps = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:10',  # Gap after 00:01
        ])
        expected_interval = pd.Timedelta(minutes=1)
        
        is_continuous, gaps = validate_timestamp_continuity(
            timestamps, expected_interval
        )
        
        self.assertFalse(is_continuous)
        self.assertEqual(len(gaps), 1)
        
        # Check gap details
        before, after, gap_size = gaps[0]
        self.assertEqual(before, pd.Timestamp('2021-01-01 00:01'))
        self.assertEqual(after, pd.Timestamp('2021-01-01 00:10'))
        self.assertEqual(gap_size, pd.Timedelta(minutes=9))


class TestValidateNoDuplicates(unittest.TestCase):
    """Test validate_no_duplicates function."""
    
    def test_no_duplicates(self):
        """Test that data without duplicates passes."""
        timestamps = pd.date_range('2021-01-01', periods=100, freq='1min')
        
        is_unique, duplicates = validate_no_duplicates(timestamps)
        
        self.assertTrue(is_unique)
        self.assertEqual(len(duplicates), 0)
    
    def test_single_duplicate_detected(self):
        """Test that a single duplicate is detected."""
        timestamps = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:01',  # Duplicate
            '2021-01-01 00:02',
        ])
        
        is_unique, duplicates = validate_no_duplicates(timestamps)
        
        self.assertFalse(is_unique)
        self.assertEqual(len(duplicates), 1)
        self.assertIn(pd.Timestamp('2021-01-01 00:01'), duplicates)
    
    def test_multiple_duplicates_detected(self):
        """Test that multiple duplicates are detected."""
        timestamps = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:00',  # Duplicate 1
            '2021-01-01 00:01',
            '2021-01-01 00:01',  # Duplicate 2
            '2021-01-01 00:01',  # Duplicate 2 again
            '2021-01-01 00:02',
        ])
        
        is_unique, duplicates = validate_no_duplicates(timestamps)
        
        self.assertFalse(is_unique)
        self.assertEqual(len(duplicates), 2)
    
    def test_empty_timestamps(self):
        """Test that empty timestamps have no duplicates."""
        timestamps = pd.DatetimeIndex([])
        
        is_unique, duplicates = validate_no_duplicates(timestamps)
        
        self.assertTrue(is_unique)
        self.assertEqual(len(duplicates), 0)
    
    def test_single_timestamp(self):
        """Test that single timestamp has no duplicates."""
        timestamps = pd.DatetimeIndex(['2021-01-01 00:00'])
        
        is_unique, duplicates = validate_no_duplicates(timestamps)
        
        self.assertTrue(is_unique)
        self.assertEqual(len(duplicates), 0)


class TestAssertDataContinuity(unittest.TestCase):
    """Test assert_data_continuity function."""
    
    def setUp(self):
        """Set up test data."""
        # Create continuous 1-minute data
        dates = pd.date_range('2021-01-01', periods=100, freq='1min')
        self.continuous_data = create_ohlcv_dict(
            list_open=[100.0] * 100,
            list_close=[101.0] * 100,
            list_high=[102.0] * 100,
            list_low=[99.0] * 100,
            list_volume=[1000.0] * 100,
            list_trades=[50.0] * 100,
            list_time=dates.tolist()
        )
    
    def test_continuous_data_passes(self):
        """Test that continuous data passes validation."""
        report = assert_data_continuity(
            self.continuous_data,
            expected_interval_minutes=1,
            raise_on_gaps=True
        )
        
        self.assertTrue(report['is_continuous'])
        self.assertFalse(report['has_duplicates'])
        self.assertEqual(len(report['gaps']), 0)
        self.assertEqual(len(report['duplicates']), 0)
        self.assertEqual(report['total_points'], 100)
    
    def test_report_without_raising(self):
        """Test that report can be generated without raising errors."""
        # Create data with a gap
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # Gap
        ])
        data = create_ohlcv_dict(
            list_open=[100.0] * 3,
            list_close=[101.0] * 3,
            list_high=[102.0] * 3,
            list_low=[99.0] * 3,
            list_volume=[1000.0] * 3,
            list_trades=[50.0] * 3,
            list_time=dates.tolist()
        )
        
        # Should not raise even with gaps
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            raise_on_gaps=False
        )
        
        self.assertFalse(report['is_continuous'])
        self.assertEqual(len(report['gaps']), 1)
    
    def test_raises_on_gap(self):
        """Test that gaps cause AssertionError when raise_on_gaps=True."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # Gap
        ])
        data = create_ohlcv_dict(
            list_open=[100.0] * 3,
            list_close=[101.0] * 3,
            list_high=[102.0] * 3,
            list_low=[99.0] * 3,
            list_volume=[1000.0] * 3,
            list_trades=[50.0] * 3,
            list_time=dates.tolist()
        )
        
        with self.assertRaises(AssertionError) as cm:
            assert_data_continuity(
                data,
                expected_interval_minutes=1,
                raise_on_gaps=True
            )
        
        self.assertIn('gaps', str(cm.exception).lower())
    
    def test_raises_on_duplicates(self):
        """Test that duplicates cause AssertionError when raise_on_gaps=True."""
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:01',  # Duplicate
            '2021-01-01 00:02',
        ])
        data = create_ohlcv_dict(
            list_open=[100.0] * 4,
            list_close=[101.0] * 4,
            list_high=[102.0] * 4,
            list_low=[99.0] * 4,
            list_volume=[1000.0] * 4,
            list_trades=[50.0] * 4,
            list_time=dates.tolist()
        )
        
        with self.assertRaises(AssertionError) as cm:
            assert_data_continuity(
                data,
                expected_interval_minutes=1,
                raise_on_gaps=True
            )
        
        self.assertIn('duplicate', str(cm.exception).lower())
    
    def test_expected_points_calculation(self):
        """Test that expected points are calculated correctly."""
        # 10 minutes of data with 1-minute intervals = 11 points (inclusive)
        dates = pd.date_range('2021-01-01 00:00', periods=11, freq='1min')
        data = create_ohlcv_dict(
            list_open=[100.0] * 11,
            list_close=[101.0] * 11,
            list_high=[102.0] * 11,
            list_low=[99.0] * 11,
            list_volume=[1000.0] * 11,
            list_trades=[50.0] * 11,
            list_time=dates.tolist()
        )
        
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            raise_on_gaps=False
        )
        
        self.assertEqual(report['total_points'], 11)
        self.assertEqual(report['expected_points'], 11)
    
    def test_missing_points_calculation_with_gap(self):
        """Test that missing points are calculated correctly."""
        # Create data missing 3 minutes in the middle
        dates = pd.DatetimeIndex([
            '2021-01-01 00:00',
            '2021-01-01 00:01',
            '2021-01-01 00:05',  # Missing 00:02, 00:03, 00:04
            '2021-01-01 00:06',
        ])
        data = create_ohlcv_dict(
            list_open=[100.0] * 4,
            list_close=[101.0] * 4,
            list_high=[102.0] * 4,
            list_low=[99.0] * 4,
            list_volume=[1000.0] * 4,
            list_trades=[50.0] * 4,
            list_time=dates.tolist()
        )
        
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            raise_on_gaps=False
        )
        
        self.assertEqual(report['total_points'], 4)
        self.assertEqual(report['expected_points'], 7)  # 00:00 to 00:06 = 7 points
        self.assertEqual(len(report['gaps']), 1)


class TestDataContinuityIntegration(unittest.TestCase):
    """Integration tests for data continuity validation."""
    
    def test_realistic_bitcoin_data(self):
        """Test with realistic Bitcoin minute data."""
        # Simulate 1 hour of Bitcoin data with realistic timestamps
        dates = pd.date_range('2021-01-01 00:00', periods=60, freq='1min')
        opens = 50000 + np.random.randn(60) * 100
        closes = opens + np.random.randn(60) * 50
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(60) * 20)
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(60) * 20)
        
        data = create_ohlcv_dict(
            list_open=opens.tolist(),
            list_close=closes.tolist(),
            list_high=highs.tolist(),
            list_low=lows.tolist(),
            list_volume=(np.random.uniform(10000, 50000, 60)).tolist(),
            list_trades=(np.random.uniform(50, 200, 60)).tolist(),
            list_time=dates.tolist()
        )
        
        report = assert_data_continuity(
            data,
            expected_interval_minutes=1,
            raise_on_gaps=True
        )
        
        self.assertTrue(report['is_continuous'])
        self.assertEqual(report['total_points'], 60)
    
    def test_multi_hour_continuous_data(self):
        """Test with multiple hours of continuous data."""
        # 24 hours of 5-minute data = 288 candles
        dates = pd.date_range('2021-01-01', periods=288, freq='5min')
        
        data = create_ohlcv_dict(
            list_open=[50000.0] * 288,
            list_close=[50100.0] * 288,
            list_high=[50200.0] * 288,
            list_low=[49900.0] * 288,
            list_volume=[25000.0] * 288,
            list_trades=[100.0] * 288,
            list_time=dates.tolist()
        )
        
        report = assert_data_continuity(
            data,
            expected_interval_minutes=5,
            raise_on_gaps=True
        )
        
        self.assertTrue(report['is_continuous'])
        self.assertEqual(report['total_points'], 288)


if __name__ == '__main__':
    unittest.main()
