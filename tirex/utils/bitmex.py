# -*- coding: utf-8 -*-
"""
Refactored BitMEX API Module

This is the main orchestrator module that coordinates authentication, HTTP client,
data transformations, and plotting using the decomposed modules.
"""

import os
import json
import time
import logging
from typing import Optional
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Import decomposed modules
from tirex.utils.bitmex_auth import AccessTokenAuth, APIKeyAuthWithExpires, BitMEXAuthenticator
from tirex.utils.bitmex_client import BitMEXHttpClient, BitMEXError, BitMEXAuthenticationError
from tirex.utils.bitmex_transforms import create_ohlcv_dict, resample_ohlcv
from tirex.utils.bitmex_plot import configure_matplotlib

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Configuration
BITMEX_API_KEY = os.getenv('BITMEX_API_KEY', '')
BITMEX_API_SECRET = os.getenv('BITMEX_API_SECRET', '')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure matplotlib
configure_matplotlib(interactive=False)


class BitMEX:
    """
    Refactored BitMEX API Connector using modular architecture.
    
    This class orchestrates various components (authentication, HTTP client,
    data transformation, plotting) to provide a high-level interface for
    interacting with the BitMEX API.
    
    Parameters
    ----------
    base_url : str, optional
        Base URL for BitMEX API
    symbol : str, optional
        Trading symbol (e.g., 'XBTUSD')
    api_key : str, optional
        API key (defaults to environment variable)
    api_secret : str, optional
        API secret (defaults to environment variable)
    http_client : BitMEXHttpClient, optional
        Custom HTTP client (for dependency injection)
    authenticator : BitMEXAuthenticator, optional
        Custom authenticator (for dependency injection)
        
    Attributes
    ----------
    base_url : str
        The API base URL
    symbol : str
        The trading symbol
    client : BitMEXHttpClient
        The HTTP client instance
        
    Examples
    --------
    >>> # Basic usage with environment variables
    >>> bitmex = BitMEX(base_url='https://www.bitmex.com/api/v1/')
    >>> data = bitmex.get_net_chart(hours=24, cpair='XBTUSD', dt=1)
    
    >>> # Advanced usage with dependency injection
    >>> from tirex.utils.bitmex_auth import APIKeyAuthWithExpires
    >>> from tirex.utils.bitmex_client import BitMEXHttpClient
    >>> auth = APIKeyAuthWithExpires("key", "secret")
    >>> client = BitMEXHttpClient(base_url="https://api.example.com", authenticator=auth)
    >>> bitmex = BitMEX(http_client=client)
    
    Notes
    -----
    This refactored version follows SOLID principles:
    - Single Responsibility: Each component handles one concern
    - Open/Closed: Extensible through dependency injection
    - Liskov Substitution: Protocol-based authenticators
    - Interface Segregation: Minimal, focused interfaces
    - Dependency Inversion: Depends on abstractions, not concretions
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        symbol: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        http_client: Optional[BitMEXHttpClient] = None,
        authenticator: Optional[BitMEXAuthenticator] = None
    ):
        """
        Initialize BitMEX connector with optional dependency injection.
        
        Parameters
        ----------
        base_url : str, optional
            API base URL
        symbol : str, optional
            Trading symbol
        api_key : str, optional
            API key (overrides environment variable)
        api_secret : str, optional
            API secret (overrides environment variable)
        http_client : BitMEXHttpClient, optional
            Pre-configured HTTP client
        authenticator : BitMEXAuthenticator, optional
            Pre-configured authenticator
        """
        self.base_url = base_url or 'https://www.bitmex.com/api/v1/'
        self.symbol = symbol
        
        # Use provided credentials or fall back to environment variables
        self.api_key = api_key or BITMEX_API_KEY
        self.api_secret = api_secret or BITMEX_API_SECRET
        
        # Setup authenticator (dependency injection or create new)
        if authenticator:
            self.authenticator = authenticator
        elif self.api_key and self.api_secret:
            self.authenticator = APIKeyAuthWithExpires(self.api_key, self.api_secret)
        else:
            logger.warning("No API credentials provided. Some operations may fail.")
            self.authenticator = None
        
        # Setup HTTP client (dependency injection or create new)
        if http_client:
            self.client = http_client
        else:
            self.client = BitMEXHttpClient(
                base_url=self.base_url,
                authenticator=self.authenticator
            )
    
    def get_instrument(self, symbol: Optional[str] = None) -> dict:
        """
        Get instrument details.
        
        Parameters
        ----------
        symbol : str, optional
            Trading symbol (uses instance symbol if not provided)
            
        Returns
        -------
        dict
            Instrument details
            
        Raises
        ------
        BitMEXError
            If instrument not found or not open
        """
        symbol = symbol or self.symbol
        assert symbol, "Symbol must be provided"
        
        params = {'filter': json.dumps({'symbol': symbol})}
        instruments = self.client.get('/instrument', params=params)
        
        if not instruments:
            raise BitMEXError(f"Instrument not found: {symbol}")
        
        instrument = instruments[0]
        
        if instrument.get("state") != "Open":
            raise BitMEXError(
                f"Instrument {symbol} is not open. State: {instrument.get('state')}"
            )
        
        return instrument
    
    def get_trade_bucketed(
        self,
        symbol: str,
        bin_size: str = '1m',
        count: int = 100,
        start_time: Optional[pd.Timestamp] = None
    ) -> list:
        """
        Get bucketed trade data (OHLCV candles).
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        bin_size : str, default='1m'
            Bin size ('1m', '5m', '1h', '1d')
        count : int, default=100
            Number of candles to fetch
        start_time : pd.Timestamp, optional
            Start time for data
            
        Returns
        -------
        list
            List of OHLCV candle dictionaries
        """
        params = {
            'symbol': symbol,
            'binSize': bin_size,
            'count': count,
            'reverse': True
        }
        
        if start_time:
            params['startTime'] = start_time.isoformat()
        
        return self.client.get('/trade/bucketed', params=params)
    
    def get_net_chart(
        self,
        hours: float,
        cpair: str,
        fn_time: Optional[pd.Timestamp] = None,
        dt: int = 1
    ) -> pd.DataFrame:
        """
        Get and process historical chart data with resampling.
        
        This method fetches historical OHLCV data and optionally resamples it
        to different time intervals using vectorized operations.
        
        Parameters
        ----------
        hours : float
            Number of hours of historical data to fetch
        cpair : str
            Trading pair symbol (e.g., 'XBTUSD', 'ETHUSD')
        fn_time : pd.Timestamp, optional
            End time for data (defaults to now)
        dt : int, default=1
            Resampling interval in minutes (1, 3, or 30)
            - dt=1 or 2: No resampling (1-minute data)
            - dt=3: Resample to 3-minute bars
            - dt=6: Resample to 30-minute bars
            
        Returns
        -------
        pd.DataFrame
            OHLCV data with datetime index
            
        Examples
        --------
        >>> bitmex = BitMEX(base_url='https://www.bitmex.com/api/v1/')
        >>> # Get 24 hours of 1-minute data
        >>> df = bitmex.get_net_chart(24, 'XBTUSD', dt=1)
        >>> # Get 48 hours of 3-minute data
        >>> df = bitmex.get_net_chart(48, 'XBTUSD', dt=3)
        
        Notes
        -----
        The function makes multiple API calls in batches of 500 candles to
        retrieve the requested time period. Progress is shown via tqdm.
        """
        assert hours > 0, "Hours must be positive"
        assert cpair, "Currency pair must be specified"
        
        fn_time = fn_time or pd.Timestamp.now()
        
        # Calculate number of API calls needed (500 candles per call)
        dt_loop = int(hours * 60 / 500) + 1
        
        # Initialize data lists
        list_open, list_close, list_high, list_low = [], [], [], []
        list_vol, list_ntrades, list_time = [], [], []
        
        logger.info(f"Fetching {hours} hours of {cpair} data ({dt_loop} API calls)...")
        
        # Fetch data in batches
        for _ in tqdm(range(dt_loop), desc="Fetching data"):
            try:
                # Get data batch
                list_data = self.get_trade_bucketed(
                    symbol=cpair,
                    bin_size='1m',
                    count=500,
                    start_time=fn_time - pd.Timedelta(minutes=500)
                )
                
                if not list_data:
                    logger.warning("No data returned, retrying in 70s...")
                    time.sleep(70)
                    continue
                
                # Parse data in reverse chronological order
                for data_point in reversed(list_data):
                    list_open.append(data_point['open'])
                    list_close.append(data_point['close'])
                    list_high.append(data_point['high'])
                    list_low.append(data_point['low'])
                    list_time.append(pd.to_datetime(data_point['timestamp'], errors='coerce'))
                    list_ntrades.append(data_point.get('trades', 0))
                    list_vol.append(data_point.get('volume', 0))
                
                # Update time window
                fn_time -= pd.Timedelta(minutes=500)
                
            except BitMEXError as e:
                logger.error(f"API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                continue
            
            time.sleep(0.5)  # Rate limiting
        
        # Create OHLCV dictionary
        ohlcv_dict = create_ohlcv_dict(
            list_open, list_close, list_high, list_low,
            list_vol, list_ntrades, list_time
        )
        
        # Determine resampling interval
        if dt in [1, 2]:
            interval_minutes = 1  # No resampling
        elif dt == 3:
            interval_minutes = 3  # 3-minute bars
        elif dt == 6:
            interval_minutes = 30  # 30-minute bars
        else:
            logger.warning(f"Unsupported dt={dt}, defaulting to 1-minute")
            interval_minutes = 1
        
        # Resample data
        df = resample_ohlcv(ohlcv_dict, interval_minutes)
        
        logger.info(f"Retrieved {len(df)} candles after resampling to {interval_minutes}min")
        
        return df
    
    def close(self):
        """Close the HTTP client and release resources."""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class GetDataPair:
    """
    Convenience wrapper for fetching data for specific trading pairs.
    
    This class provides a simplified interface for common trading pairs.
    
    Parameters
    ----------
    base_url : str, optional
        BitMEX API base URL
        
    Examples
    --------
    >>> get_data = GetDataPair()
    >>> df = get_data('XBTUSD', hours=24, dt=1)
    """
    
    def __init__(self, base_url: str = 'https://www.bitmex.com/api/v1/'):
        """
        Initialize data fetcher.
        
        Parameters
        ----------
        base_url : str, optional
            BitMEX API base URL
        """
        self.bitmex = BitMEX(base_url=base_url)
    
    def __call__(
        self,
        cpair: str,
        hours: float,
        dt: int,
        fn_time: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Fetch data for a trading pair.
        
        Parameters
        ----------
        cpair : str
            Currency pair ('XBTUSD', 'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD')
        hours : float
            Hours of historical data
        dt : int
            Resampling interval
        fn_time : pd.Timestamp, optional
            End time
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        # Normalize pair names
        if cpair in ['XBTUSD', 'BTCUSD']:
            cpair = 'XBTUSD'
        
        return self.bitmex.get_net_chart(hours, cpair, fn_time=fn_time, dt=dt)


def save_ticker(dt: int = 60, size: int = 80000, hours: int = 168):
    """
    Save ticker data to parquet file.
    
    Parameters
    ----------
    dt : int, default=60
        Time interval in minutes
    size : int, default=80000
        Number of data points to save
    hours : int, default=168
        Number of hours of historical data to fetch (default: 1 week)
        
    Examples
    --------
    >>> # Fetch 1 week of 15-minute data
    >>> save_ticker(dt=15, hours=168)
    
    >>> # Fetch 1 day of 5-minute data
    >>> save_ticker(dt=5, hours=24)
    """

    current_file_path = Path(__file__).parent.parent.parent
    parquet_path = current_file_path / "Signals/data"
    parquet_path.mkdir(parents=True, exist_ok=True)
    
    get_data_pair = GetDataPair()
    
    pd_time = pd.Timestamp.now()
    dt_inc = pd_time.minute // dt
    
    end_date = pd.Timestamp(
        year=pd_time.year,
        month=pd_time.month,
        day=pd_time.day,
        hour=pd_time.hour,
        minute=dt_inc * dt
    )
    
    fn_name = f"btcusd_{dt}m_{end_date.strftime('%Y-%m-%d')}"
    
    logger.info(f"Fetching {hours} hours of data for {fn_name}...")
    logger.info(f"This will make ~{int(hours * 60 / 500) + 1} API calls...")
    
    pd_chart = get_data_pair('XBTUSD', hours, dt // 5, fn_time=end_date)
    pd_chart = pd_chart[-size:]
    
    output_path = parquet_path / f"{fn_name}.parquet"
    pd_chart.to_parquet(output_path)
    logger.info(f"Saved {len(pd_chart)} rows to {output_path}")
    logger.info(f"Output file: {output_path}")


if __name__ == '__main__':
    # Example: Fetch 24 hours of 15-minute data for testing
    # For production, increase hours parameter (e.g., 168 for 1 week, 720 for 1 month)

    save_ticker(dt=15, hours=40000)  # Just 24 hours for testing
    # save_ticker(dt=60, hours=40000)   # Just 24 hours for testing
