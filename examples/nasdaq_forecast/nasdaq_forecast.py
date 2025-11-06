
# import os
# os.environ["TIREX_NO_CUDA"] = "1"
# os.environ['TORCH_CUDA_ARCH_LIST']
# import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from tirex import ForecastModel, load_model
from tirex.utils.filters import ConvolutionFilter
from tirex.utils.ceemdan import ICEEMDAN
from tirex.utils.ewt import EmpiricalWaveletTransform
from tirex.utils.plot import plot_fc, emd_plot

# Add the project root to the Python path
project_local_path = Path(__file__).resolve().parent


def iceemdan_forecast():
    """
    Main function to run the Nasdaq forecast.
    """

    local_plot_path = (project_local_path / "emd_plots").resolve()
    local_plot_path.mkdir(exist_ok=True)

    # --- Parameters ---
    input_window = 120
    start_date_str = "2025-06-15"
    
    # --- Load Data ---
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date - pd.DateOffset(days=1)

    # Fetch more data to ensure we have enough trading days
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

    inp_len = 600
    out_len = 18
    clen = inp_len // 2
    bclen = 1

    full_idx = np.arange(inp_len + out_len)
    inp_idx = full_idx[:inp_len]
    out_idx = full_idx[inp_len:]

    convolution_filter = ConvolutionFilter(adim=inp_len, window=3)

    config = {"processes": 1, "spline_kind": 'akima', "DTYPE": float}
    iceemdan = ICEEMDAN(trials=20, max_imf=-1, **config)

    np_x = np.linspace(0, 1, inp_len)

    list_inp_win = []
    list_out_win = []

    for i in range(20):
        inc_i = i * 50
        inp_idx = inp_idx + inc_i
        out_idx = out_idx + inc_i

        sr_y_i = nasdaq_data.iloc[inp_idx, :]
        sr_y_ref_i = nasdaq_data.iloc[out_idx, :]

        np_y_i = sr_y_i.values[:, 0]
        np_y_ft_i = convolution_filter(np_y_i)
        np_y_ref_i = sr_y_ref_i.values[:, 0]

        local_plot_path_i = (local_plot_path / f"trial_{i}").resolve()
        local_plot_path_i.mkdir(exist_ok=True)

        list_inp_win.append(np_y_i)
        list_out_win.append(np_y_ref_i)

        # First run
        c_imfs_i = iceemdan.iceemdan(np_y_ft_i, T=np_x)

        # Plot results
        emd_plot(np_x[-clen:], np_y_i[-clen:], c_imfs_i[:, -clen:],
                 plot_title=f"ICEEMDAN Unit Test Case i: {i}",
                 plot_name=f'{local_plot_path_i}/iceemdan_full_case_{i}.png')

        list_quantiles_i = []
        list_mean_i = []

        select_imfs_i = np.arange(0, c_imfs_i.shape[0])

        for k in select_imfs_i[1:]:
            signal_k = c_imfs_i[k, :]

            quantiles_k, mean_k = model.forecast(signal_k[:-bclen],
                                                 prediction_length=out_len + bclen,
                                                 output_type="numpy",
                                                 )
            plot_fc(signal_k[-120:], quantiles_k[0][bclen:], save_path=f'{local_plot_path_i}/iceemdan_signal_pred_{k}.png')

            list_quantiles_i.append(quantiles_k[0][bclen:])
            list_mean_i.append(mean_k[0][bclen:])

        quantiles_i = np.asarray(list_quantiles_i).sum(axis=0)
        mean_i = np.asarray(list_mean_i).sum(axis=0)

        plot_fc(np_y_i[-120:], quantiles_i,
                real_future_values=np_y_ref_i,
                save_path=f'{local_plot_path_i}/iceemdan_sum_signal_pred.png')


def ewt_forecast():
    """
    Main function to run the Nasdaq forecast.
    """

    local_plot_path = (project_local_path / "ewt_plots").resolve()
    local_plot_path.mkdir(exist_ok=True)

    # --- Parameters ---
    input_window = 120
    start_date_str = "2025-08-4"

    # --- Load Data ---
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date - pd.DateOffset(days=1)

    # Fetch more data to ensure we have enough trading days
    start_fetch_date = end_date - pd.DateOffset(
        days=input_window * 6)  # Fetch more data to ensure we have enough trading days

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

    inp_len = 600
    out_len = 12
    clen = inp_len // 2
    bclen = 5
    plt_len = 120

    full_idx = np.arange(inp_len + out_len + bclen)
    inp_idx = full_idx[:inp_len]
    out_idx = full_idx[inp_len:] - bclen

    convolution_filter = ConvolutionFilter(adim=inp_len, window=3)
    convolution_filter_smooth = ConvolutionFilter(adim=inp_len, window=20)

    # Initialize EWT
    ewt = EmpiricalWaveletTransform()
    np_ewt_x = np.linspace(0, 1, inp_len)

    list_inp_win = []
    list_out_win = []

    for i in range(30):
        print(f"Forecast trial: {i}")
        inc_i = i * 10
        inp_idx = inp_idx + inc_i
        out_idx = out_idx + inc_i
        full_idx_i = np.arange(inp_idx[0], out_idx[-1] + 1)[-(plt_len + out_len):]

        if out_idx[-1] < nasdaq_data.shape[0]:
            sr_x_i = nasdaq_data.iloc[inp_idx, :]
            sr_y_i = nasdaq_data.iloc[out_idx, :]

            np_x_i = sr_x_i.values[:, 0]
            np_x_ft_i = convolution_filter(np_x_i)
            np_x_ft2_i = convolution_filter_smooth(np_x_i)
            np_y_i = sr_y_i.values[:, 0]

            local_plot_path_i = (local_plot_path / f"trial_{i}").resolve()
            local_plot_path_i.mkdir(exist_ok=True)

            list_inp_win.append(np_x_i)
            list_out_win.append(np_y_i)

            # First run
            ewt_res_i, np_mwvlt_i, np_bcs_i = ewt(np_x_ft_i, 6)
            ewt_comps_i = ewt_res_i.T

            # Plot results
            emd_plot(np_ewt_x[-clen:], np_x_i[-clen:], ewt_comps_i[:, -clen:],
                     plot_title=f"EWT Unit Test Case i: {i}",
                     plot_name=f'{local_plot_path_i}/full_decomposition.png')

            list_quantiles_i = []
            list_mean_i = []

            select_imfs_i = np.arange(0, ewt_comps_i.shape[0])

            for k in select_imfs_i:
                np_signal_x_k = ewt_comps_i[k, :]
                quantiles_y_k, mean_y_k = model.forecast(np_signal_x_k[:-bclen],
                                                     prediction_length=bclen + out_len,
                                                     output_type="numpy",
                                                     )
                plot_fc(np_signal_x_k[-plt_len:], quantiles_y_k[0][bclen:],
                        save_path=f'{local_plot_path_i}/ewt_signal_{k}_pred.png')

                list_quantiles_i.append(quantiles_y_k[0][bclen:])
                list_mean_i.append(mean_y_k[0][bclen:])

            ewt_quantiles_i = np.asarray(list_quantiles_i).sum(axis=0)
            ewt_mean_i = np.asarray(list_mean_i).sum(axis=0)

            plot_fc(np_x_i[-plt_len:], ewt_quantiles_i,
                    real_future_values=np_y_i[bclen:],
                    full_timeseries=nasdaq_data.iloc[full_idx_i, :].values[:, 0],
                    title=f"End Time: {sr_x_i.index[bclen:][-1]}",
                    save_path=f'{local_plot_path_i}/ewt_signal_sum_pred.png')

            # trying to forescat using full data
            quantiles_i, mean_i = model.forecast(np_x_ft2_i[:-bclen],
                                                 prediction_length=out_len + bclen,
                                                 output_type="numpy",
                                                 )

            plot_fc(np_x_i[-plt_len:], quantiles_i[0][bclen:],
                    real_future_values=np_y_i[bclen:],
                    full_timeseries=nasdaq_data.iloc[full_idx_i, :].values[:, 0],
                    title=f"End Time: {sr_x_i.index[bclen:][-1]}",
                    save_path=f'{local_plot_path_i}/full_signal_pred.png')

        else:
            break


if __name__ == "__main__":
    # iceemdan_forecast()
    ewt_forecast()
