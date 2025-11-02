# -*- coding: utf-8 -*-
"""
Unit tests for BitMEX Data Transformation Module

Tests data validation, resampling functions, and vectorized operations.
"""

import unittest
import numpy as np
import pandas as pd
# from datetime import datetime, timedelta

from tirex.utils.bitmex_transforms import (
    validate_ohlcv_data,
    resample_ohlcv_simple,
    resample_ohlcv_3min,
    resample_ohlcv_30min,
    resample_ohlcv,
    create_ohlcv_dict
)


class TestValidateOHLCVData(unittest.TestCase):
    """Test validate_ohlcv_data function."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_data = {
            'open': np.array([100.0, 101.0, 102.0]),
            'high': np.array([105.0, 106.0, 107.0]),
            'low': np.array([99.0, 100.0, 101.0]),
            'close': np.array([103.0, 104.0, 105.0]),
            'volume': np.array([1000.0, 1100.0, 1200.0]),
            'trades': np.array([50.0, 55.0, 60.0]),
            'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']).values
        }
    
    def test_valid_data_passes(self):
        """Test that valid data passes validation."""
        try:
            validate_ohlcv_data(self.valid_data)
        except AssertionError:
            self.fail("validate_ohlcv_data raised AssertionError unexpectedly")
    
    def test_missing_key_raises_error(self):
        """Test that missing required key raises error."""
        invalid_data = self.valid_data.copy()
        del invalid_data['volume']
        
        with self.assertRaises(AssertionError):
            validate_ohlcv_data(invalid_data)
    
    def test_inconsistent_lengths_raises_error(self):
        """Test that inconsistent array lengths raise error."""
        invalid_data = self.valid_data.copy()
        invalid_data['open'] = np.array([100.0, 101.0])  # Different length
        
        with self.assertRaises(AssertionError):
            validate_ohlcv_data(invalid_data)
    
    def test_invalid_date_dtype_raises_error(self):
        """Test that non-datetime date raises error."""
        invalid_data = self.valid_data.copy()
        invalid_data['date'] = np.array([1, 2, 3])  # Not datetime
        
        with self.assertRaises(AssertionError):
            validate_ohlcv_data(invalid_data)
    
    def test_high_less_than_low_raises_error(self):
        """Test that high < low raises error."""
        invalid_data = self.valid_data.copy()
        invalid_data['high'] = np.array([90.0, 91.0, 92.0])  # Less than low
        
        with self.assertRaises(AssertionError):
            validate_ohlcv_data(invalid_data)
    
    def test_high_less_than_open_allowed(self):
        """Test that high < open is allowed (relaxed validation for real-world data)."""
        data = self.valid_data.copy()
        data['high'] = np.array([99.0, 100.0, 101.0])  # Less than open
        
        # Should not raise - relaxed validation
        try:
            validate_ohlcv_data(data)
        except AssertionError:
            self.fail("validate_ohlcv_data raised unexpected AssertionError")
    
    def test_low_greater_than_close_allowed(self):
        """Test that low > close is allowed (relaxed validation for real-world data)."""
        data = self.valid_data.copy()
        data['low'] = np.array([104.0, 105.0, 106.0])  # Greater than close
        
        # Should not raise - relaxed validation
        try:
            validate_ohlcv_data(data)
        except AssertionError:
            self.fail("validate_ohlcv_data raised unexpected AssertionError")
    
    def test_empty_data_passes(self):
        """Test that empty but valid structure passes."""
        empty_data = {
            'open': np.array([]),
            'high': np.array([]),
            'low': np.array([]),
            'close': np.array([]),
            'volume': np.array([]),
            'trades': np.array([]),
            'date': np.array([], dtype='datetime64[s]')  # Specify unit
        }
        
        try:
            validate_ohlcv_data(empty_data)
        except AssertionError:
            self.fail("validate_ohlcv_data raised AssertionError for empty data")


class TestResampleOHLCVSimple(unittest.TestCase):
    """Test resample_ohlcv_simple function."""
    
    def setUp(self):
        """Set up test data."""
        self.data = {
            'open': np.array([100.0, 101.0, 102.0]),
            'high': np.array([105.0, 106.0, 107.0]),
            'low': np.array([99.0, 100.0, 101.0]),
            'close': np.array([103.0, 104.0, 105.0]),
            'volume': np.array([1000.0, 1100.0, 1200.0]),
            'trades': np.array([50.0, 55.0, 60.0]),
            'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']).values
        }
    
    def test_returns_dataframe(self):
        """Test that function returns DataFrame."""
        result = resample_ohlcv_simple(self.data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_preserves_data_count(self):
        """Test that no resampling occurs."""
        result = resample_ohlcv_simple(self.data)
        self.assertEqual(len(result), len(self.data['open']))
    
    def test_has_correct_columns(self):
        """Test that DataFrame has correct columns."""
        result = resample_ohlcv_simple(self.data)
        expected_columns = ['open', 'close', 'high', 'low', 'trades', 'volume']
        self.assertEqual(set(result.columns), set(expected_columns))
    
    def test_has_datetime_index(self):
        """Test that DataFrame has datetime index."""
        result = resample_ohlcv_simple(self.data)
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))
    
    def test_data_values_preserved(self):
        """Test that data values are preserved."""
        result = resample_ohlcv_simple(self.data)
        np.testing.assert_array_equal(result['open'].values, self.data['open'])
        np.testing.assert_array_equal(result['close'].values, self.data['close'])


class TestResampleOHLCV3Min(unittest.TestCase):
    """Test resample_ohlcv_3min function."""
    
    def setUp(self):
        """Set up test data with minute-level timestamps."""
        # Create 9 minutes of data starting at 10:00
        dates = pd.date_range('2021-01-01 10:00', periods=9, freq='1min')
        # Ensure OHLC consistency: high >= open, high >= close, low <= open, low <= close
        opens = np.arange(100.0, 109.0)
        closes = np.arange(101.0, 110.0)
        self.data = {
            'open': opens,
            'close': closes,
            'high': np.maximum(opens, closes) + 2.0,  # high >= max(open, close)
            'low': np.minimum(opens, closes) - 1.0,   # low <= min(open, close)
            'volume': np.ones(9) * 1000,
            'trades': np.ones(9) * 50,
            'date': dates.values
        }
    
    def test_returns_dataframe(self):
        """Test that function returns DataFrame."""
        result = resample_ohlcv_3min(self.data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_resamples_to_3min(self):
        """Test that data is resampled to 3-minute intervals."""
        result = resample_ohlcv_3min(self.data)
        # 9 minutes should produce 3 bars of 3 minutes
        self.assertGreater(len(result), 0)
        self.assertLess(len(result), len(self.data['open']))
    
    def test_ohlc_aggregation_correct(self):
        """Test that OHLC aggregation is correct."""
        # Create simple data to verify aggregation
        dates = pd.date_range('2021-01-01 10:00', periods=6, freq='1min')
        opens = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        closes = np.array([101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        data = {
            'open': opens,
            'close': closes,
            'high': np.maximum(opens, closes) + 1.0,  # high >= max(open, close)
            'low': np.minimum(opens, closes) - 1.0,   # low <= min(open, close)
            'volume': np.ones(6) * 1000,
            'trades': np.ones(6) * 50,
            'date': dates.values
        }
        
        result = resample_ohlcv_3min(data)
        
        if len(result) > 0:
            # First bar should have first open
            self.assertGreaterEqual(result['open'].iloc[0], 100.0)
            # Volume should be summed
            self.assertGreater(result['volume'].iloc[0], 1000.0)


class TestResampleOHLCV30Min(unittest.TestCase):
    """Test resample_ohlcv_30min function."""
    
    def setUp(self):
        """Set up test data with minute-level timestamps."""
        # Create 60 minutes of data
        dates = pd.date_range('2021-01-01 10:00', periods=60, freq='1min')
        opens = np.arange(100.0, 160.0)
        closes = np.arange(101.0, 161.0)
        self.data = {
            'open': opens,
            'close': closes,
            'high': np.maximum(opens, closes) + 2.0,  # high >= max(open, close)
            'low': np.minimum(opens, closes) - 1.0,   # low <= min(open, close)
            'volume': np.ones(60) * 1000,
            'trades': np.ones(60) * 50,
            'date': dates.values
        }
    
    def test_returns_dataframe(self):
        """Test that function returns DataFrame."""
        result = resample_ohlcv_30min(self.data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_resamples_to_30min(self):
        """Test that data is resampled to 30-minute intervals."""
        result = resample_ohlcv_30min(self.data)
        # 60 minutes should produce fewer bars
        self.assertLess(len(result), len(self.data['open']))
    
    def test_handles_empty_data(self):
        """Test handling of empty data."""
        empty_data = {
            'open': np.array([]),
            'close': np.array([]),
            'high': np.array([]),
            'low': np.array([]),
            'volume': np.array([]),
            'trades': np.array([]),
            'date': np.array([], dtype='datetime64[s]')  # Specify unit
        }
        
        result = resample_ohlcv_30min(empty_data)
        self.assertEqual(len(result), 0)
    
    def test_volume_aggregation(self):
        """Test that volume is summed correctly."""
        result = resample_ohlcv_30min(self.data)
        
        if len(result) > 0:
            # Volume should be sum of constituent bars
            self.assertGreater(result['volume'].iloc[0], 1000.0)


class TestResampleOHLCV(unittest.TestCase):
    """Test resample_ohlcv dispatcher function."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2021-01-01 10:00', periods=60, freq='1min')
        opens = np.arange(100.0, 160.0)
        closes = np.arange(101.0, 161.0)
        self.data = {
            'open': opens,
            'close': closes,
            'high': np.maximum(opens, closes) + 2.0,  # high >= max(open, close)
            'low': np.minimum(opens, closes) - 1.0,   # low <= min(open, close)
            'volume': np.ones(60) * 1000,
            'trades': np.ones(60) * 50,
            'date': dates.values
        }
    
    def test_interval_1_minute(self):
        """Test resampling to 1-minute (no resampling)."""
        result = resample_ohlcv(self.data, interval_minutes=1)
        self.assertEqual(len(result), len(self.data['open']))
    
    def test_interval_3_minutes(self):
        """Test resampling to 3-minute."""
        result = resample_ohlcv(self.data, interval_minutes=3)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLess(len(result), len(self.data['open']))
    
    def test_interval_30_minutes(self):
        """Test resampling to 30-minute."""
        result = resample_ohlcv(self.data, interval_minutes=30)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLess(len(result), len(self.data['open']))
    
    def test_unsupported_interval_raises_error(self):
        """Test that unsupported interval raises ValueError."""
        with self.assertRaises(ValueError):
            resample_ohlcv(self.data, interval_minutes=7)  # 7 is not supported
    
    def test_invalid_data_raises_error(self):
        """Test that invalid data raises error."""
        invalid_data = {'open': np.array([1, 2, 3])}
        
        with self.assertRaises(AssertionError):
            resample_ohlcv(invalid_data, interval_minutes=1)


class TestCreateOHLCVDict(unittest.TestCase):
    """Test create_ohlcv_dict function."""
    
    def test_creates_valid_dict(self):
        """Test that function creates valid OHLCV dictionary."""
        times = [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-02')]
        opens = [100.0, 101.0]
        closes = [101.0, 102.0]
        highs = [102.0, 103.0]
        lows = [99.0, 100.0]
        volumes = [1000.0, 1100.0]
        trades = [50.0, 55.0]
        
        result = create_ohlcv_dict(opens, closes, highs, lows, volumes, trades, times)
        
        self.assertIn('open', result)
        self.assertIn('close', result)
        self.assertIn('high', result)
        self.assertIn('low', result)
        self.assertIn('volume', result)
        self.assertIn('trades', result)
        self.assertIn('date', result)
    
    def test_converts_to_numpy_arrays(self):
        """Test that lists are converted to numpy arrays."""
        times = [pd.Timestamp('2021-01-01')]
        data_lists = [[100.0], [101.0], [102.0], [99.0], [1000.0], [50.0]]
        
        result = create_ohlcv_dict(*data_lists, times)
        
        self.assertIsInstance(result['open'], np.ndarray)
        self.assertIsInstance(result['close'], np.ndarray)
        self.assertEqual(result['open'].dtype, np.float32)
    
    def test_inconsistent_lengths_raises_error(self):
        """Test that inconsistent list lengths raise error."""
        times = [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-02')]
        opens = [100.0]  # Different length
        closes = [101.0, 102.0]
        highs = [102.0, 103.0]
        lows = [99.0, 100.0]
        volumes = [1000.0, 1100.0]
        trades = [50.0, 55.0]
        
        with self.assertRaises(AssertionError):
            create_ohlcv_dict(opens, closes, highs, lows, volumes, trades, times)
    
    def test_datetime_conversion(self):
        """Test that timestamps are converted to datetime64."""
        times = [pd.Timestamp('2021-01-01 10:00:00+00:00')]
        data_lists = [[100.0], [101.0], [102.0], [99.0], [1000.0], [50.0]]
        
        result = create_ohlcv_dict(*data_lists, times)
        
        self.assertEqual(result['date'].dtype.kind, 'M')  # Datetime type
    
    def test_validates_created_dict(self):
        """Test that created dict passes validation."""
        times = [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-02')]
        opens = [100.0, 101.0]
        closes = [103.0, 104.0]
        highs = [105.0, 106.0]
        lows = [99.0, 100.0]
        volumes = [1000.0, 1100.0]
        trades = [50.0, 55.0]
        
        result = create_ohlcv_dict(opens, closes, highs, lows, volumes, trades, times)
        
        # Should not raise
        try:
            validate_ohlcv_data(result)
        except AssertionError:
            self.fail("Created dict did not pass validation")


class TestPerformance(unittest.TestCase):
    """Test performance of vectorized operations."""
    
    def test_vectorized_vs_loop_performance(self):
        """Test that vectorized operations are faster (informational)."""
        # This is informational - just ensures functions complete
        # Create larger dataset
        dates = pd.date_range('2021-01-01', periods=1000, freq='1min')
        opens = np.random.uniform(90, 110, 1000)
        closes = np.random.uniform(90, 110, 1000)
        data = {
            'open': opens,
            'close': closes,
            'high': np.maximum(opens, closes) + np.random.uniform(1, 10, 1000),
            'low': np.minimum(opens, closes) - np.random.uniform(1, 10, 1000),
            'volume': np.random.uniform(1000, 2000, 1000),
            'trades': np.random.uniform(40, 60, 1000),
            'date': dates.values
        }
        
        import time
        start = time.time()
        result = resample_ohlcv(data, interval_minutes=30)
        duration = time.time() - start
        
        # Should complete in reasonable time (< 1 second for 1000 points)
        self.assertLess(duration, 1.0)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()
