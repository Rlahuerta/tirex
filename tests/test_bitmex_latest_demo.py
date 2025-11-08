#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo script to test the new BitMEX latest data fetcher.

This script demonstrates how to use the new functions to fetch
and visualize recent Bitcoin data with thick lines.
"""

import logging
from pathlib import Path

from tirex.utils.bitmex_latest import (
    get_latest_bitmex_data,
    fetch_and_plot_latest_btc
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_fetch():
    """Demonstrate basic data fetching."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Data Fetch (24 hours, 15-minute intervals)")
    print("="*70)
    
    df, fig = get_latest_bitmex_data(
        symbol='XBTUSD',
        hours=24,
        dt=15,
        plot=True
    )
    
    print(f"\nFetched {len(df)} data points")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"\nLatest data:")
    print(df[['open', 'high', 'low', 'close', 'volume']].tail(5))
    print(f"\nLatest price: ${df['close'].iloc[-1]:.2f}")
    
    return df, fig


def demo_convenience_function():
    """Demonstrate convenience function."""
    print("\n" + "="*70)
    print("DEMO 2: Convenience Function (Quick Bitcoin Fetch)")
    print("="*70)
    
    df, fig = fetch_and_plot_latest_btc(hours=24)
    
    # Calculate statistics
    latest_price = df['close'].iloc[-1]
    first_price = df['close'].iloc[0]
    price_change = latest_price - first_price
    pct_change = (price_change / first_price) * 100
    
    print(f"\nBitcoin Statistics (24h):")
    print(f"  Latest price: ${latest_price:,.2f}")
    print(f"  24h change: ${price_change:+,.2f} ({pct_change:+.2f}%)")
    print(f"  24h high: ${df['high'].max():,.2f}")
    print(f"  24h low: ${df['low'].min():,.2f}")
    print(f"  24h volume: {df['volume'].sum():,.0f}")
    
    return df, fig


def demo_without_plot():
    """Demonstrate data-only fetch (no plot)."""
    print("\n" + "="*70)
    print("DEMO 3: Data Only (No Plot)")
    print("="*70)
    
    df, fig = get_latest_bitmex_data(
        symbol='XBTUSD',
        hours=6,
        dt=15,
        plot=False
    )
    
    assert fig is None, "Figure should be None when plot=False"
    print(f"\nFetched {len(df)} data points (no plot)")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def demo_different_timeframes():
    """Demonstrate different timeframes."""
    print("\n" + "="*70)
    print("DEMO 4: Different Timeframes")
    print("="*70)
    
    timeframes = [5, 15, 30, 60]
    results = {}
    
    for dt in timeframes:
        print(f"\nFetching {dt}-minute data...")
        df, _ = get_latest_bitmex_data(
            symbol='XBTUSD',
            hours=12,
            dt=dt,
            plot=False
        )
        results[dt] = df
        print(f"  {dt}min: {len(df)} data points")
    
    return results


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BitMEX Latest Data Fetcher - Demo Script")
    print("="*70)
    print("\nThis script demonstrates the new BitMEX data fetching functions.")
    print("Note: This will make real API calls to BitMEX (public data).")
    print("="*70)
    
    try:
        # Run demos
        df1, fig1 = demo_basic_fetch()
        df2, fig2 = demo_convenience_function()
        df3 = demo_without_plot()
        results = demo_different_timeframes()
        
        print("\n" + "="*70)
        print("All demos completed successfully!")
        print("="*70)
        
        # Optionally save plots
        output_dir = Path(__file__).parent / "plots" / "ticker"
        output_dir.mkdir(exist_ok=True)
        fig1.savefig(output_dir / 'demo1_basic.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / 'demo2_convenience.png', dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to {output_dir}/")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise
