# -*- coding: utf-8 -*-
"""
Unit tests for BitMEX Plotting Module

Tests plotting functions with mocked matplotlib to avoid rendering.
"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np

from tirex.utils.bitmex_plot import (
    configure_matplotlib,
    get_date_formatters,
    plot_ohlcv_candlestick,
    plot_price_series,
    plot_multiple_series
)


class TestConfigureMatplotlib(unittest.TestCase):
    """Test configure_matplotlib function."""
    
    @patch('tirex.utils.bitmex_plot.plt.ion')
    @patch('tirex.utils.bitmex_plot.plt.ioff')
    def test_interactive_mode_on(self, mock_ioff, mock_ion):
        """Test that interactive mode can be enabled."""
        configure_matplotlib(interactive=True)
        mock_ion.assert_called_once()
        mock_ioff.assert_not_called()
    
    @patch('tirex.utils.bitmex_plot.plt.ion')
    @patch('tirex.utils.bitmex_plot.plt.ioff')
    def test_interactive_mode_off(self, mock_ioff, mock_ion):
        """Test that interactive mode can be disabled."""
        configure_matplotlib(interactive=False)
        mock_ioff.assert_called_once()
        mock_ion.assert_not_called()


class TestGetDateFormatters(unittest.TestCase):
    """Test get_date_formatters function."""
    
    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        result = get_date_formatters()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
    
    def test_returns_locators_and_formatter(self):
        """Test that correct matplotlib objects are returned."""
        years, months, weeks, fmt = get_date_formatters()
        
        # Verify they have expected attributes (duck typing)
        self.assertTrue(callable(years))
        self.assertTrue(callable(months))
        self.assertTrue(callable(weeks))
        self.assertTrue(hasattr(fmt, 'fmt'))


class TestPlotOHLCVCandlestick(unittest.TestCase):
    """Test plot_ohlcv_candlestick function."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2021-01-01', periods=10, freq='1D')
        self.df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 10),
            'high': np.random.uniform(100, 120, 10),
            'low': np.random.uniform(80, 100, 10),
            'close': np.random.uniform(90, 110, 10),
            'volume': np.random.uniform(1000, 2000, 10)
        }, index=dates)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_creates_figure(self, mock_tight_layout, mock_subplots):
        """Test that function creates a figure."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        result = plot_ohlcv_candlestick(self.df, show=False)
        
        self.assertIsNotNone(result)
        mock_subplots.assert_called_once()
    
    def test_requires_all_columns(self):
        """Test that function requires all OHLCV columns."""
        incomplete_df = self.df.drop('volume', axis=1)
        
        with self.assertRaises(AssertionError):
            plot_ohlcv_candlestick(incomplete_df, show=False)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    @patch('tirex.utils.bitmex_plot.plt.show')
    def test_show_parameter(self, mock_show, mock_tight_layout, mock_subplots):
        """Test that show parameter controls display."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        # Test with show=False
        plot_ohlcv_candlestick(self.df, show=False)
        mock_show.assert_not_called()
        
        # Test with show=True
        plot_ohlcv_candlestick(self.df, show=True)
        mock_show.assert_called_once()
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_save_path_parameter(self, mock_tight_layout, mock_subplots):
        """Test that save_path parameter saves figure."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        from pathlib import Path
        save_path = Path("/tmp/test_chart.png")
        
        plot_ohlcv_candlestick(self.df, save_path=save_path, show=False)
        
        mock_fig.savefig.assert_called_once()
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_custom_title(self, mock_tight_layout, mock_subplots):
        """Test that custom title is used."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        custom_title = "Custom BTC/USD Chart"
        plot_ohlcv_candlestick(self.df, title=custom_title, show=False)
        
        # Verify set_title was called with custom title
        mock_ax1.set_title.assert_called_once_with(custom_title)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_custom_figsize(self, mock_tight_layout, mock_subplots):
        """Test that custom figsize is used."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        custom_figsize = (16, 8)
        plot_ohlcv_candlestick(self.df, figsize=custom_figsize, show=False)
        
        # Verify subplots was called with custom figsize
        call_kwargs = mock_subplots.call_args[1]
        self.assertEqual(call_kwargs['figsize'], custom_figsize)


class TestPlotPriceSeries(unittest.TestCase):
    """Test plot_price_series function."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2021-01-01', periods=10, freq='1D')
        self.df = pd.DataFrame({
            'close': np.random.uniform(90, 110, 10),
            'open': np.random.uniform(90, 110, 10)
        }, index=dates)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_creates_figure(self, mock_tight_layout, mock_subplots):
        """Test that function creates a figure."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        result = plot_price_series(self.df, show=False)
        
        self.assertIsNotNone(result)
        mock_subplots.assert_called_once()
    
    def test_requires_price_column(self):
        """Test that function requires specified price column."""
        with self.assertRaises(AssertionError):
            plot_price_series(self.df, price_column='nonexistent', show=False)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_custom_price_column(self, mock_tight_layout, mock_subplots):
        """Test plotting with custom price column."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_price_series(self.df, price_column='open', show=False)
        
        # Verify plot was called
        mock_ax.plot.assert_called_once()
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    @patch('tirex.utils.bitmex_plot.plt.show')
    def test_show_parameter(self, mock_show, mock_tight_layout, mock_subplots):
        """Test that show parameter controls display."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Test with show=False
        plot_price_series(self.df, show=False)
        mock_show.assert_not_called()
        
        # Test with show=True
        plot_price_series(self.df, show=True)
        mock_show.assert_called_once()


class TestPlotMultipleSeries(unittest.TestCase):
    """Test plot_multiple_series function."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2021-01-01', periods=10, freq='1D')
        self.df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 10),
            'close': np.random.uniform(90, 110, 10),
            'high': np.random.uniform(100, 120, 10)
        }, index=dates)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_creates_figure(self, mock_tight_layout, mock_subplots):
        """Test that function creates a figure."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        result = plot_multiple_series(self.df, ['open', 'close'], show=False)
        
        self.assertIsNotNone(result)
        mock_subplots.assert_called_once()
    
    def test_requires_all_columns(self):
        """Test that function requires all specified columns."""
        with self.assertRaises(AssertionError):
            plot_multiple_series(self.df, ['open', 'nonexistent'], show=False)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_plots_multiple_series(self, mock_tight_layout, mock_subplots):
        """Test that multiple series are plotted."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        columns = ['open', 'close']
        plot_multiple_series(self.df, columns, show=False)
        
        # Verify plot was called for each column
        self.assertEqual(mock_ax.plot.call_count, len(columns))
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_legend_added(self, mock_tight_layout, mock_subplots):
        """Test that legend is added."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plot_multiple_series(self.df, ['open', 'close'], show=False)
        
        # Verify legend was called
        mock_ax.legend.assert_called_once()
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    @patch('tirex.utils.bitmex_plot.plt.show')
    def test_show_parameter(self, mock_show, mock_tight_layout, mock_subplots):
        """Test that show parameter controls display."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Test with show=False
        plot_multiple_series(self.df, ['open'], show=False)
        mock_show.assert_not_called()
        
        # Test with show=True
        plot_multiple_series(self.df, ['open'], show=True)
        mock_show.assert_called_once()


class TestPlottingIntegration(unittest.TestCase):
    """Integration tests for plotting module."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2021-01-01', periods=100, freq='1h')  # Use lowercase 'h'
        self.df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(100, 120, 100),
            'low': np.random.uniform(80, 100, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.uniform(1000, 2000, 100)
        }, index=dates)
    
    @patch('tirex.utils.bitmex_plot.plt.subplots')
    @patch('tirex.utils.bitmex_plot.plt.tight_layout')
    def test_all_plot_functions_work(self, mock_tight_layout, mock_subplots):
        """Test that all plot functions work with realistic data."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_ax = Mock()
        
        # Test candlestick (returns tuple of axes)
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        try:
            plot_ohlcv_candlestick(self.df, show=False)
        except Exception as e:
            self.fail(f"plot_ohlcv_candlestick failed: {e}")
        
        # Reset mock
        mock_subplots.reset_mock()
        
        # Test price series (returns single ax)
        mock_subplots.return_value = (mock_fig, mock_ax)
        try:
            plot_price_series(self.df, show=False)
        except Exception as e:
            self.fail(f"plot_price_series failed: {e}")
        
        # Reset mock
        mock_subplots.reset_mock()
        
        # Test multiple series (returns single ax)
        mock_subplots.return_value = (mock_fig, mock_ax)
        try:
            plot_multiple_series(self.df, ['open', 'close'], show=False)
        except Exception as e:
            self.fail(f"plot_multiple_series failed: {e}")


if __name__ == '__main__':
    unittest.main()
