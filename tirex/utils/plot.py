# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from matplotlib import pyplot as plt


def plot_fc(ctx, quantile_fc, real_future_values=None, save_path=None):
    """
    Plots the forecast against the historical context and, optionally, the ground truth future values.

    Args:
        ctx (array-like): The historical context data.
        quantile_fc (array-like): The quantile forecast data, expected to have 9 quantiles.
        real_future_values (array-like, optional): The ground truth future values. Defaults to None.
        save_path (str, optional): If provided, the plot will be saved to this path instead of being displayed.
                                   Defaults to None.
    """
    median_forecast = quantile_fc[:, 4]
    lower_bound = quantile_fc[:, 0]
    upper_bound = quantile_fc[:, 8]

    original_x = range(len(ctx))
    forecast_x = range(len(ctx), len(ctx) + len(median_forecast))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(original_x, ctx, label="Ground Truth Context", color="#4a90d9")
    if real_future_values is not None:
        original_fut_x = range(len(ctx), len(ctx) + len(real_future_values))
        plt.plot(original_fut_x, real_future_values, label="Ground Truth Future", color="#4a90d9", linestyle=":")
    plt.plot(forecast_x, median_forecast, label="Forecast (Median)", color="#d94e4e", linestyle="--")
    plt.fill_between(
        forecast_x, lower_bound, upper_bound, color="#d94e4e", alpha=0.1, label="Forecast 10% - 90% Quantiles"
    )
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
