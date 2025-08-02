
# import os
# os.environ["TIREX_NO_CUDA"] = "1"
# os.environ['TORCH_CUDA_ARCH_LIST']
import sys
from pathlib import Path
import numpy as np
# import torch
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt

# Add the project root to the Python path
project_local_path = Path(__file__).resolve().parent
project_root = project_local_path.parent.parent
sys.path.append(str(project_root))

from tirex import ForecastModel, load_model

def plot_fc(ctx, quantile_fc, real_future_values=None, start_date=None):
    """
    Plots the forecast against the ground truth.
    """
    median_forecast = quantile_fc[:, 4]
    lower_bound = quantile_fc[:, 0]
    upper_bound = quantile_fc[:, 8]

    if start_date:
        ctx_dates = pd.to_datetime(start_date) - pd.to_timedelta(np.arange(len(ctx), 0, -1), unit='D')
        forecast_dates = pd.to_datetime(start_date) + pd.to_timedelta(np.arange(len(median_forecast)), unit='D')
        if real_future_values is not None:
            future_dates = pd.to_datetime(start_date) + pd.to_timedelta(np.arange(len(real_future_values)), unit='D')
    else:
        ctx_dates = np.arange(len(ctx))
        forecast_dates = np.arange(len(ctx), len(ctx) + len(median_forecast))
        if real_future_values is not None:
            future_dates = np.arange(len(ctx), len(ctx) + len(real_future_values))

    plt.figure(figsize=(12, 6))
    plt.plot(ctx_dates, ctx, label="Ground Truth Context", color="#4a90d9")
    if real_future_values is not None:
        plt.plot(future_dates, real_future_values, label="Ground Truth Future", color="#4a90d9", linestyle=":")
    plt.plot(forecast_dates, median_forecast, label="Forecast (Median)", color="#d94e4e", linestyle="--")
    plt.fill_between(
        forecast_dates, lower_bound, upper_bound, color="#d94e4e", alpha=0.1, label="Forecast 10% - 90% Quantiles"
    )
    plt.xlim(left=ctx_dates[0])
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to run the Nasdaq forecast.
    """
    # --- Parameters ---
    input_window = 120
    prediction_length = 14
    start_date_str = "2025-06-15"
    
    # --- Load Data ---
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date - pd.DateOffset(days=1)
    start_fetch_date = end_date - pd.DateOffset(days=input_window * 5) # Fetch more data to ensure we have enough trading days

    try:
        nasdaq_data = yf.download('^IXIC', start=start_fetch_date, end=end_date, interval='1h')
        nasdaq_data = nasdaq_data['Close']
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    if len(nasdaq_data) < input_window:
        print(f"Not enough historical data available. Required: {input_window}, Downloaded: {len(nasdaq_data)}")
        return

    # --- Load Model ---
    try:
        model: ForecastModel = load_model("NX-AI/TiRex")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    inp_window = nasdaq_data[:-15]
    out_window = nasdaq_data[-15:]
    inp_len = 650
    out_len = 15

    # --- Generate Forecast ---
    try:
        quantiles, mean = model.forecast(inp_window.values[-inp_len:], prediction_length=out_len, output_type="numpy")
    except Exception as e:
        print(f"Error during forecast: {e}")
        return

    # --- Plot Results ---
    plot_fc(inp_window.values[-inp_len:][-120:], quantiles[0], real_future_values=out_window, start_date=start_date)

if __name__ == "__main__":
    main()
