# -*- coding: utf-8 -*-

import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone


def fetch_bitcoin_data(hours=3000, interval='15m', symbol='BTCUSDT'):
    """
    Fetches continuous Bitcoin (BTC/USDT) OHLCV data from Binance for the last N hours.

    :param hours: Number of hours of history to retrieve.
    :param interval: Candle interval (default '15m').
    :param symbol: Trading pair symbol (default 'BTCUSDT').

    :return: Pandas DataFrame containing the data.
    """
    base_url = "https://api.binance.com/api/v3/klines"

    # Use UTC for consistent calculations
    now_utc = datetime.now(timezone.utc)

    # Calculate timestamps in milliseconds (Binance API requirement)
    # End time is now
    end_time_ms = int(now_utc.timestamp() * 1000)

    # Start time is N hours ago
    start_time_ms = int((now_utc - timedelta(hours=hours)).timestamp() * 1000)

    print(f"--- Initialization ---")
    print(f"Target:     {hours} hours of {interval} data for {symbol}")
    print(f"Start Time: {datetime.fromtimestamp(start_time_ms / 1000, timezone.utc)}")
    print(f"End Time:   {datetime.fromtimestamp(end_time_ms / 1000, timezone.utc)}")
    print(f"Fetching data in batches...")

    all_data = []
    current_start_ms = start_time_ms

    # Pagination loop
    while current_start_ms < end_time_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start_ms,
            'endTime': end_time_ms,
            'limit': 1000  # Maximum allowed by Binance per request
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                print("Warning: No data received for this batch.")
                break

            all_data.extend(data)

            # Logic for next batch:
            # data[-1][6] is the 'Close Time' of the last candle in the batch
            # We start the next batch 1ms after the last candle closes
            last_close_time = data[-1][6]
            current_start_ms = last_close_time + 1

            # Optional: Progress indicator
            last_candle_date = datetime.fromtimestamp(last_close_time / 1000, timezone.utc)
            print(f"Fetched {len(data)} candles... Reached {last_candle_date}")

            # Sleep to respect API rate limits (Weight 1 per request)
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    if not all_data:
        return None

    # --- Data Processing ---

    columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ]

    df = pd.DataFrame(all_data, columns=columns)

    # Optimized numeric conversion (Vectorized)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    # Convert Open Time to datetime object
    df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')

    # Set index and drop raw timestamp columns
    df.set_index('Date', inplace=True)

    # Select only the ticker columns usually needed
    final_df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    return final_df


def validate_data(df, expected_hours):
    """
    Performs a self-test on the downloaded data to ensure continuity.
    """
    print(f"\n--- Data Integrity Test ---")

    # 1. Check shape
    rows = len(df)
    print(f"Total Rows: {rows}")

    # 2. Check time continuity (approximate)
    # 3000 hours * 4 (15m intervals per hour) = 12,000 rows expected
    expected_rows = expected_hours * 4
    # We allow a small margin of error (e.g., +/- 5 rows) due to 'now' moving
    diff = abs(rows - expected_rows)

    if diff < 20:
        print(f"✅ Row count match: {rows} (Expected ~{expected_rows})")
    else:
        print(f"⚠️ Row count mismatch: {rows} (Expected ~{expected_rows})")

    # 3. Check for duplicate indices
    if df.index.is_unique:
        print("✅ No duplicate timestamps found.")
    else:
        print("❌ Error: Duplicate timestamps detected.")

    # 4. Check for gaps
    # Calculate the difference between consecutive index entries
    time_diffs = df.index.to_series().diff().dropna()
    # 15 minutes is 900 seconds
    gaps = time_diffs[time_diffs > timedelta(minutes=15)]

    if gaps.empty:
        print("✅ Time sequence is continuous (no gaps > 15m).")
    else:
        print(f"❌ Gaps detected! Found {len(gaps)} gaps.")
        print(gaps.head())


if __name__ == "__main__":
    HOURS_TO_FETCH = 3000

    print("Running Fetcher...")
    df = fetch_bitcoin_data(hours=HOURS_TO_FETCH)

    if df is not None:
        # Run validation
        validate_data(df, HOURS_TO_FETCH)

        print("\n--- Sample Data (Head) ---")
        print(df.head())

        print("\n--- Sample Data (Tail) ---")
        print(df.tail())

        # Save
        filename = f"btc_15m_{HOURS_TO_FETCH}h.csv"
        df.to_csv(filename)
        print(f"\nSaved to {filename}")
    else:
        print("Failed to retrieve data.")