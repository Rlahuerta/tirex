# -*- coding: utf-8 -*-
"""
BitMEX Plotting Module

This module provides plotting functions for BitMEX OHLCV data visualization.
It isolates matplotlib configuration and provides optional plotting capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple
from pathlib import Path


# Matplotlib configuration - isolated from global scope
def configure_matplotlib(interactive: bool = False) -> None:
    """
    Configure matplotlib settings for BitMEX plots.
    
    Parameters
    ----------
    interactive : bool, default=False
        If True, enables interactive plotting mode
        If False, disables interactive mode (for batch processing)
        
    Examples
    --------
    >>> configure_matplotlib(interactive=False)
    >>> # Matplotlib is now in non-interactive mode
    """
    if interactive:
        plt.ion()
    else:
        plt.ioff()


# Date formatting helpers
def get_date_formatters() -> Tuple:
    """
    Get matplotlib date formatters for different time scales.
    
    Returns
    -------
    tuple
        (years_locator, months_locator, weeks_locator, year_formatter)
        
    Examples
    --------
    >>> years, months, weeks, fmt = get_date_formatters()
    >>> # Use these with matplotlib axis formatting
    """
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    weeks = mdates.WeekdayLocator()
    year_fmt = mdates.DateFormatter('%Y')
    
    return years, months, weeks, year_fmt


def plot_ohlcv_candlestick(
    df: pd.DataFrame,
    title: str = "OHLCV Chart",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot OHLCV data as candlestick chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data and datetime index
        Must contain columns: 'open', 'high', 'low', 'close', 'volume'
    title : str, default="OHLCV Chart"
        Chart title
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
    save_path : Path, optional
        If provided, saves the figure to this path
    show : bool, default=False
        If True, displays the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Raises
    ------
    AssertionError
        If required columns are missing from DataFrame
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'open': [100, 101, 102],
    ...     'high': [105, 106, 107],
    ...     'low': [99, 100, 101],
    ...     'close': [103, 104, 105],
    ...     'volume': [1000, 1100, 1200]
    ... }, index=pd.date_range('2021-01-01', periods=3))
    >>> fig = plot_ohlcv_candlestick(df, show=False)
    >>> plt.close(fig)
    
    Notes
    -----
    This function creates a simplified candlestick representation using
    matplotlib's standard plotting functions.
    """
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    assert required_cols.issubset(df.columns), \
        f"DataFrame must contain columns: {required_cols}"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Price chart
    ax1.plot(df.index, df['close'], label='Close', linewidth=1)
    ax1.fill_between(df.index, df['low'], df['high'], alpha=0.2, label='High-Low Range')
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    ax2.bar(df.index, df['volume'], width=0.8, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    years, months, weeks, year_fmt = get_date_formatters()
    ax2.xaxis.set_major_locator(months)
    ax2.xaxis.set_major_formatter(year_fmt)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_price_series(
    df: pd.DataFrame,
    price_column: str = 'close',
    title: str = "Price Chart",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot a simple price time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data and datetime index
    price_column : str, default='close'
        Column name to plot
    title : str, default="Price Chart"
        Chart title
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
    save_path : Path, optional
        If provided, saves the figure to this path
    show : bool, default=False
        If True, displays the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'close': [100, 101, 102, 103]
    ... }, index=pd.date_range('2021-01-01', periods=4))
    >>> fig = plot_price_series(df, show=False)
    >>> plt.close(fig)
    """
    assert price_column in df.columns, \
        f"Column '{price_column}' not found in DataFrame"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df.index, df[price_column], linewidth=1.5)
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    years, months, weeks, year_fmt = get_date_formatters()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(year_fmt)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_multiple_series(
    df: pd.DataFrame,
    columns: list,
    title: str = "Multi-Series Chart",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot multiple price series on the same chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data and datetime index
    columns : list of str
        Column names to plot
    title : str, default="Multi-Series Chart"
        Chart title
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
    save_path : Path, optional
        If provided, saves the figure to this path
    show : bool, default=False
        If True, displays the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'open': [100, 101, 102],
    ...     'close': [103, 104, 105]
    ... }, index=pd.date_range('2021-01-01', periods=3))
    >>> fig = plot_multiple_series(df, ['open', 'close'], show=False)
    >>> plt.close(fig)
    """
    missing_cols = set(columns) - set(df.columns)
    assert not missing_cols, f"Columns not found: {missing_cols}"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        ax.plot(df.index, df[col], label=col, linewidth=1.5)
    
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    years, months, weeks, year_fmt = get_date_formatters()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(year_fmt)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# Initialize matplotlib configuration (non-interactive by default)
configure_matplotlib(interactive=False)
