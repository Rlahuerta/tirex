# -*- coding: utf-8 -*-
"""
BitMEX Latest Data Fetcher with Visualization

This module provides convenience functions to fetch the latest data from BitMEX
and create plots with thick lines for 15-minute intervals.
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from tirex.utils.bitmex import BitMEX
from tirex.utils.plot import _add_candlestick

logger = logging.getLogger(__name__)


def plot_ticker_with_thick_lines(
    ticker_data: pd.DataFrame,
    dt: int = 15,
    title: str = "BitMEX Latest Data",
    linewidth: float = 3.0,
    figsize: Tuple[int, int] = (24, 8),
    dpi: int = 300,
    save_path: Optional[Path] = None,
    close: bool = False
) -> Figure:
    """
    Plot ticker data with thick lines for better visibility.
    
    This function creates a candlestick chart with thicker lines than standard
    plots, optimized for 15-minute intervals.
    
    Parameters
    ----------
    ticker_data : pd.DataFrame
        DataFrame with OHLCV data and datetime index.
        Must contain columns: 'open', 'high', 'low', 'close', 'volume'
    dt : int, default=15
        Time increment in minutes (15 for 15-minute bars)
    title : str, default="BitMEX Latest Data"
        Chart title
    linewidth : float, default=3.0
        Line width for plot elements (thicker for visibility)
    figsize : tuple, default=(24, 8)
        Figure size (width, height) in inches
    dpi : int, default=300
        Resolution in dots per inch
    save_path : Path, optional
        If provided, saves figure to this path
    close : bool, default=False
        If True, closes the figure after saving
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'open': [100, 101, 102],
    ...     'high': [105, 106, 107],
    ...     'low': [99, 100, 101],
    ...     'close': [103, 104, 105],
    ...     'volume': [1000, 1100, 1200]
    ... }, index=pd.date_range('2021-01-01', periods=3, freq='15min'))
    >>> fig = plot_ticker_with_thick_lines(df)
    
    Notes
    -----
    Uses thicker lines (linewidth=3.0) compared to standard plots for
    enhanced visibility in presentations and reports.
    """
    assert isinstance(ticker_data, pd.DataFrame), "ticker_data must be a DataFrame"
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    assert required_cols.issubset(ticker_data.columns), \
        f"DataFrame must contain columns: {required_cols}"
    assert len(ticker_data) > 0, "DataFrame cannot be empty"
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add candlestick chart
    _add_candlestick(ax, ticker_data, dt=dt)
    
    # Add thick close price line for visibility and legend
    ax.plot(
        ticker_data.index,
        ticker_data['close'],
        color='blue',
        label='Close Price',
        linewidth=linewidth,
        alpha=0.7,
        zorder=3
    )
    
    # Format x-axis for time display
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    
    if dt == 15:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    elif dt == 60:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Labels and grid
    ax.set_xlabel(
        f"Date - {ticker_data.index[-1].strftime('%B')}/{ticker_data.index[-1].year}",
        fontsize=12
    )
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(which='minor', alpha=0.2, axis='x')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    if close:
        plt.close(fig)
    
    return fig


def get_latest_bitmex_data(
        symbol: str = 'XBTUSD',
        hours: float = 24,
        dt: int = 15,
        base_url: str = 'https://www.bitmex.com/api/v1/',
        plot: bool = True,
        plot_len: Optional[int] = 240,
        plot_title: Optional[str] = None,
        save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Optional[Figure]]:
    """
    Fetch latest data from BitMEX and optionally create a plot with thick lines.
    
    This function provides a convenient interface to fetch recent OHLCV data
    from BitMEX and visualize it with enhanced line thickness.
    
    Parameters
    ----------
    symbol : str, default='XBTUSD'
        Trading pair symbol (e.g., 'XBTUSD', 'ETHUSD')
    hours : float, default=24
        Number of hours of historical data to fetch
    dt : int, default=15
        Time resolution in minutes:
        - 15: 15-minute bars (recommended)
        - 5: 5-minute bars
        - 30: 30-minute bars
        - 60: 60-minute bars (1 hour)
    base_url : str, default='https://www.bitmex.com/api/v1/'
        BitMEX API base URL
    plot : bool, default=True
        If True, creates a plot with thick lines
    plot_title : str, optional
        Custom plot title (defaults to auto-generated title)
    plot_len : int, optional
        Number of data points to include in the plot (default: 240)
    save_path : Path, optional
        If provided, saves plot to this path
        
    Returns
    -------
    data : pd.DataFrame
        OHLCV data with datetime index
        Columns: ['open', 'high', 'low', 'close', 'volume', 'trades']
    figure : matplotlib.figure.Figure or None
        Plot figure if plot=True, otherwise None
        
    Examples
    --------
    >>> # Fetch 24 hours of 15-minute data
    >>> df, fig = get_latest_bitmex_data('XBTUSD', hours=24, dt=15)
    >>> print(df.head())
    >>> # fig contains the matplotlib figure
    
    >>> # Fetch without plotting
    >>> df, _ = get_latest_bitmex_data('XBTUSD', hours=48, plot=False)
    
    >>> # Save plot to file
    >>> df, fig = get_latest_bitmex_data(
    ...     'XBTUSD',
    ...     hours=24,
    ...     save_path=Path('btc_chart.png')
    ... )
    
    Notes
    -----
    The dt parameter internally maps to BitMEX API bin sizes:
    - dt=5: Fetches 5-minute bars (dt_internal=1)
    - dt=15: Fetches 5-minute bars and resamples to 15min (dt_internal=3)
    - dt=30: Fetches 5-minute bars and resamples to 30min (dt_internal=6)
    - dt=60: Fetches 1-hour bars (dt_internal=12)
    
    Raises
    ------
    AssertionError
        If parameters are invalid
    BitMEXError
        If API request fails after retries
    """
    assert hours > 0, "hours must be positive"
    assert dt in [5, 15, 30, 60], f"dt must be one of [5, 15, 30, 60], got {dt}"
    
    # Map external dt (minutes) to internal dt parameter for BitMEX API
    dt_mapping = {
        5: 1,    # 5-minute bars
        15: 3,   # 15-minute bars (resample 3x5min)
        30: 6,   # 30-minute bars (resample 6x5min)
        60: 12   # 60-minute bars (1-hour)
    }
    dt_internal = dt_mapping[dt]
    
    logger.info(f"Fetching {hours} hours of {symbol} data at {dt}-minute intervals...")
    
    # Initialize BitMEX client
    bitmex = BitMEX(base_url=base_url, symbol=symbol)
    
    try:
        # Fetch data
        data = bitmex.get_net_chart(
            hours=hours,
            cpair=symbol,
            fn_time=None,  # Use current time
            dt=dt_internal
        )
        
        logger.info(f"Successfully fetched {len(data)} data points")
        
        # Validate data
        assert len(data) > 0, "No data returned from BitMEX"
        assert 'close' in data.columns, "Data missing 'close' column"
        
        # Create plot if requested
        figure = None
        if plot:
            if plot_title is None:
                plot_title = f"{symbol} - Last {hours}h ({dt}min intervals)"
            
            figure = plot_ticker_with_thick_lines(
                ticker_data=data[-plot_len:],
                dt=dt,
                title=plot_title,
                save_path=save_path
            )
            logger.info("Plot created successfully")
        
        return data, figure
    
    finally:
        bitmex.close()


def fetch_and_plot_latest_btc(
    hours: float = 24,
    save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Figure]:
    """
    Quick convenience function to fetch and plot latest Bitcoin data.
    
    This is a simplified wrapper around get_latest_bitmex_data() with
    sensible defaults for Bitcoin (XBTUSD) with 15-minute intervals.
    
    Parameters
    ----------
    hours : float, default=24
        Number of hours of historical data (default: 24 hours)
    save_path : Path, optional
        If provided, saves plot to this path
        
    Returns
    -------
    data : pd.DataFrame
        OHLCV data for Bitcoin
    figure : matplotlib.figure.Figure
        Plot with thick lines
        
    Examples
    --------
    >>> # Fetch last 24 hours and plot
    >>> df, fig = fetch_and_plot_latest_btc()
    >>> print(f"Latest price: ${df['close'].iloc[-1]:.2f}")
    
    >>> # Fetch last 48 hours and save
    >>> df, fig = fetch_and_plot_latest_btc(
    ...     hours=48,
    ...     save_path=Path('btc_48h.png')
    ... )
    
    Notes
    -----
    This function always creates a plot with thick lines optimized for
    15-minute interval visualization.
    """
    logger.info(f"Fetching latest Bitcoin data ({hours} hours)...")
    
    data, figure = get_latest_bitmex_data(
        symbol='XBTUSD',
        hours=hours,
        dt=15,
        plot=True,
        plot_title=f"Bitcoin (XBTUSD) - Last {hours} hours (15min intervals)",
        save_path=save_path
    )
    
    # Log summary statistics
    if len(data) > 0:
        latest_price = data['close'].iloc[-1]
        price_change = data['close'].iloc[-1] - data['close'].iloc[0]
        pct_change = (price_change / data['close'].iloc[0]) * 100
        
        logger.info(f"Latest price: ${latest_price:.2f}")
        logger.info(f"Change: ${price_change:.2f} ({pct_change:+.2f}%)")
        logger.info(f"High: ${data['high'].max():.2f}")
        logger.info(f"Low: ${data['low'].min():.2f}")
    
    return data, figure


if __name__ == '__main__':
    # Example usage
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Fetching latest Bitcoin data from BitMEX...")
    print("This will take a few moments...")
    
    try:
        # Fetch 24 hours of 15-minute data
        df, fig = fetch_and_plot_latest_btc(hours=24)
        
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Data points: {len(df)}")
        print(f"Time range: {df.index[0]} to {df.index[-1]}")
        print(f"\nLatest prices:")
        print(df[['open', 'high', 'low', 'close', 'volume']].tail())
        print("="*60)
        
        # Optionally save
        # save_path = Path('bitcoin_latest.png')
        # fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"\nPlot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
