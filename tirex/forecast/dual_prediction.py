# -*- coding: utf-8 -*-
import time
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
from sklearn.preprocessing import MinMaxScaler

from tirex import ForecastModel, load_model
from tirex.utils.ewt import EmpiricalWaveletTransform
from tirex.utils.ceemdan import ICEEMDAN
from tirex.utils.ssa import ssa
from tirex.utils.filters import ConvolutionFilter, quadratic_fit_series
from tirex.utils.time import create_time_index
from tirex.utils.plot import plot_fc, dual_plot_mpl_ticker
from tirex.utils.bitmex_latest import get_latest_bitmex_data, fetch_and_plot_latest_btc
from tirex.utils.rescale import apply_minmax_inverse_scaler


# FIXME
# Add the project root to the Python path
project_local_path = Path(__file__).resolve().parent.parent.parent
local_plot_path = (project_local_path / "predictions").resolve()
local_plot_path.mkdir(exist_ok=True)


class DualOptForecast:

    def __init__(self,
                 opt_dsvars: pd.DataFrame,
                 plt_len: int = 120,
                 inc_len: int = 500,
                 run_size: int = 50,
                 dtype: str = "ewt",
                 ftype: str = "backward",
                 plot_flag: bool = False,
                 seed: int = 42,
                 **kwargs,
                 ):
        """
        Initialize the OptForecast instance.

        Args:
            input_data (pd.DataFrame): Input data frame with a 'close' column.
            out_len (int):      Number of future time steps to forecast.
            plt_len (int):      Window length for plotting historical data.
            inc_len (int):      Increment length for sliding evaluation windows.
            run_size (int):     Number of evaluation runs to perform.
            dtype (str):        Decomposition type ('ewt', 'xwt', 'emd', 'ssa').
            ftype (str):        The type of filter to use:
                                    - 'full': Symmetric/non-causal filter (looks both directions)
                                    - 'forward': Causal filter (looks only forward, no past leakage)
                                    - 'backward': Anti-causal filter (looks only backward, no future leakage)
            plot_flag (bool): Whether to enable plotting of results.
            seed (int): Random seed for reproducibility.
            debug (bool): If True, use mock forecast model.
        """

        # Get the number of hours
        hdt15 = opt_dsvars.loc[15, "decomplen"] // 4
        hdt60 = opt_dsvars.loc[60, "decomplen"]
        nhours = int(1.1 * max(hdt15, hdt60))

        df_price_data, fig = get_latest_bitmex_data(symbol='XBTUSD', hours=nhours, dt=15, plot=False)
        self.input_data = df_price_data

        # Forecast Variables
        self.opt_dsvars = opt_dsvars

        self.out_len = self.opt_dsvars.loc[60, "outlen"] * 4
        self.plt_len = plt_len
        self.inc_len = inc_len
        self.seed = seed
        self.run_size = run_size

        self.plot_flag = plot_flag
        self._plot_local_path = None

        self.dtype = dtype
        self.ftype = ftype

        self._ft15_len = 1651
        self._ft15_win = 44
        self._ft60_len = 963
        self._ft60_win = 44

        self.convolve_ft15 = ConvolutionFilter(adim=self._ft15_len, window=self._ft15_win, ftype=self.ftype)
        self.convolve_ft60 = ConvolutionFilter(adim=self._ft60_len, window=self._ft60_win, ftype=self.ftype)

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": float}
        self.emd = ICEEMDAN(trials=20, max_imf=-1, **config)
        self.ewt = EmpiricalWaveletTransform()

        self.ssa_wlen = 20  # Window length for SSA

        # Forecast Model and function
        self._model = None
        self._model_path = (Path(__file__).parent.parent.parent / "model" / "model.ckpt").resolve()
        self._forecast = None

        self.df_data = self.input_data["close"]
        self.scaler_data = MinMaxScaler(feature_range=(0., 100.))

        self.np_data_idx = np.arange(len(self.df_data))
        self.sr_data_rs = None

        self.np_idx_inc_eval = np.array([])

        dt_object = datetime.datetime.fromtimestamp(time.time())
        self.str_timestamp = dt_object.strftime("%Y%m%d_%H%M%S")

        self._preprocess_data()
        self._load_forecast_model()

    def _preprocess_data(self):
        """
        Preprocess and rescale input data, and generate shuffled evaluation indices.

        Populates:
            self.np_idx_inc_eval (np.ndarray): Shuffled indices for evaluation.
            self.sr_data_rs (pd.Series): Rescaled time series data.
        """
        # preprocessing the joblib data

        np_data_rs = self.scaler_data.fit_transform(self.df_data.values.reshape(-1, 1)).flatten()
        self.sr_data_rs = pd.Series(np_data_rs, index=self.input_data.index)

    def _load_forecast_model(self):
        """
        Load the forecasting model. If debug is enabled, set the forecast method to mock.
        Otherwise, load the pretrained TiRex model.
        """
        try:
            self._model: ForecastModel = load_model(str(self._model_path))
            self._forecast = self._model.forecast

        except Exception as e:
            print(f"Error loading model: {e}")
            return

    def update_input_data(self):
        """
        Update the input data and re-preprocess.
        """

        new_data, _ = get_latest_bitmex_data(symbol='XBTUSD', hours=1, dt=15, plot=False)

        # Concatenate and remove duplicates by index
        combined_data = pd.concat([self.input_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()

        self.input_data = combined_data
        self.df_data = self.input_data["close"]

        np_data_rs = self.scaler_data.transform(self.df_data.values.reshape(-1, 1)).flatten()
        self.sr_data_rs = pd.Series(np_data_rs, index=self.input_data.index)

    def _signal_decomposition(self,
                              input_signal: pd.Series,
                              nsignal: int,
                              ) -> pd.DataFrame:
        """
        Decompose the input signal using selected method (EWT, XWT, EMD, SSA).

        Args:
            input_signal (pd.Series): Signal to decompose.
            nsignal (int): Number of modes/components.

        Returns:
            pd.DataFrame: Decomposed signal components.
        """
        # TODO: Add option to use ceemdan as signal decomp method
        assert nsignal > 1, "nsignal should be greater than 1"

        if self.dtype == "ewt":
            # First EWT Decomposition
            np_ewt_res, np_mwvlt, np_bcs = self.ewt(input_signal.values, nsignal)
            pd_ewt_res = pd.DataFrame(np_ewt_res[-input_signal.size:], index=input_signal.index)
            return pd_ewt_res

        elif self.dtype == "xwt":
            np_ewt_res, np_mwvlt, np_bcs = self.ewt(input_signal.values, nsignal)
            np_ewt_res_fix = np_ewt_res[-input_signal.size:]
            np_ewt_sum = np_ewt_res_fix.sum(axis=1)
            np_sign_res = (input_signal.values - np_ewt_sum).reshape(-1, 1)
            np_xwt = np.concatenate((np_ewt_res_fix, np_sign_res), axis=1)
            pd_ewt_res = pd.DataFrame(np_xwt, index=input_signal.index)
            return pd_ewt_res

        elif self.dtype == "swt":
            np_ssa_res = ssa(input_signal.values, 2, wlen=8)

            np_ewt_res, np_mwvlt, np_bcs = self.ewt(np_ssa_res[:, 0], nsignal)
            np_swt = np.concatenate((np_ewt_res[-input_signal.size:], np_ssa_res[:, 1:]), axis=1)
            np_swt_sum = np_swt.sum(axis=1)
            np_swt_res = input_signal.values - np_swt_sum
            np_swt_fnl = np.concatenate((np_swt, np_swt_res.reshape(-1, 1)), axis=1)
            pd_swt_res = pd.DataFrame(np_swt_fnl, index=input_signal.index)
            return pd_swt_res

        elif self.dtype == "emd":
            # First EMD Decomposition
            np_emd_res = self.emd.iceemdan(input_signal.values, max_imf=nsignal)
            pd_emd_res = pd.DataFrame(np_emd_res[:, -input_signal.size:].T, index=input_signal.index)
            return pd_emd_res

        elif self.dtype == "ssa":
            return pd.DataFrame(ssa(input_signal.values, nsignal, wlen=self.ssa_wlen), index=input_signal.index)

        else:
            raise NotImplementedError(f"Unsupported decomposition type: {self.dtype}. Use 'ewt' or 'emd'.")

    def _forecast_signal(
            self,
            signal: pd.Series,
            out_len: int,
            plot_path: Optional[Path] = None,
            **kwargs,
    ) -> pd.DataFrame:

        np_index = np.arange(0, signal.shape[0] + out_len)
        np_output_idx = np_index[signal.shape[0]:]

        dt_time = signal.index[1] - signal.index[0]
        num_dt = int(dt_time.total_seconds() / 60)
        list_output_datetime_idx = create_time_index(dt_time, signal.index[-1], np_output_idx.size)

        sr_signal_x_full = self.sr_data_rs.loc[:signal.index[-1]]

        if num_dt == 15:
            sr_signal_x = sr_signal_x_full.iloc[-self._ft15_len:]
            sr_ft_signal_x = self.convolve_ft15(sr_signal_x)
        elif num_dt == 60:
            sr_signal_x = sr_signal_x_full.iloc[-self._ft60_len:]
            sr_ft_signal_x = self.convolve_ft60(sr_signal_x)
        else:
            raise NotImplementedError(f"Unsupported dt_time: {dt_time}. Use 15 or 60 minutes.")

        np_quantiles_y, np_mean_y = self._forecast(sr_ft_signal_x.values,
                                                   prediction_length=out_len,
                                                   output_type="numpy",
                                                   )

        pd_quantiles_y = pd.DataFrame(np_quantiles_y[0], index=list_output_datetime_idx)
        sr_mean_y = pd.Series(np_mean_y[0], index=list_output_datetime_idx, name="ft_mean")

        sr_poly2_y = quadratic_fit_series(sr_mean_y)
        sr_poly2_y.name = "ft_poly2"

        if self.plot_flag and plot_path is not None:
            num_dt = int(dt_time.total_seconds() / 60)
            plt_kwargs = dict(save_path=f'{plot_path}/conv_filter_dt{num_dt}_pred.png')
            plot_fc(ctx=sr_ft_signal_x.iloc[-self.plt_len:], quantile_fc=pd_quantiles_y, **plt_kwargs)

        return pd.concat([pd_quantiles_y, sr_mean_y, sr_poly2_y], axis=1)

    def _forecast_multi_signals(self,
                                decomp_signals: pd.DataFrame,
                                out_len: int,
                                bc_len: int,
                                plot_path: Optional[Path] = None,
                                ) -> pd.DataFrame:
        """
        Forecast each decomposed component, aggregate quantiles and mean, and optionally plot.

        Args:
            decomp_signals (pd.DataFrame): DataFrame of decomposed components.
            bc_len (int): Back-cast length for each component.
            plot_path (Optional[Path]): Directory to save forecast plots.

        Returns:
            pd.DataFrame: Concatenated quantiles and mean forecasts.
        """

        np_index = np.arange(0, decomp_signals.shape[0] + out_len)
        np_input_idx = np_index[:decomp_signals.shape[0] - bc_len]
        np_output_idx = np_index[decomp_signals.shape[0] - bc_len:]
        dt_time = decomp_signals.index[1] - decomp_signals.index[0]
        num_dt = int(dt_time.total_seconds() / 60)

        list_output_datetime_idx = create_time_index(dt_time,
                                                     decomp_signals.index[np_input_idx][-1],
                                                     np_output_idx.size)
        list_quantiles = []
        list_mean = []
        pred_len = out_len + bc_len

        for k, (signal_id_k, sr_signal_val_k) in enumerate(decomp_signals.items()):
            sr_signal_x_k = sr_signal_val_k.iloc[np_input_idx]
            np_quantiles_y_k, np_mean_y_k = self._forecast(sr_signal_x_k.values,
                                                           prediction_length=pred_len,
                                                           output_type="numpy",
                                                           )

            pd_quantiles_y_k = pd.DataFrame(np_quantiles_y_k[0], index=list_output_datetime_idx)
            pd_mean_y_k = pd.Series(np_mean_y_k[0], index=list_output_datetime_idx)

            list_quantiles.append(pd_quantiles_y_k)
            list_mean.append(pd_mean_y_k)

            if self.plot_flag and plot_path is not None:
                plt_kwargs = dict(save_path=f'{plot_path}/decomp_signal_dt{num_dt}_i{k}_pred.png')
                plt_kwargs["rescale"] = self.scaler_data

                if bc_len > 0:
                    plt_kwargs["bcs"] = pd_mean_y_k.iloc[:bc_len]

                plot_fc(sr_signal_val_k.iloc[-self.plt_len:],
                        pd_quantiles_y_k.iloc[bc_len:, :], **plt_kwargs)

        pd_quantiles = sum(list_quantiles)
        pd_quantiles.columns = [f"quantile_{i}" for i in pd_quantiles.columns]

        sr_mean = sum(list_mean)
        sr_mean.name = "mean"

        sr_poly2 = quadratic_fit_series(sr_mean)
        sr_poly2.name = "poly2"

        return pd.concat([pd_quantiles, sr_mean, sr_poly2], axis=1)

    def _rescaling(self, pd_signal: pd.DataFrame) -> pd.DataFrame:

        if pd_signal.shape[1] > 1:
            dict_signal = {}
            for key_i, signal_i in pd_signal.items():
                dict_signal[key_i] = self.scaler_data.transform(signal_i.values.reshape(-1, 1)).flatten()

            return pd.DataFrame(dict_signal, index=pd_signal.index)
        else:
            raise NotImplementedError(f"Unsupported rescaling with shape: {pd_signal.shape[1]}.")

    def _get_support_points(self,
                            signal: pd.Series,
                            ) -> pd.Series:

        np_signal_ssa = ssa(signal.values, 2, wlen=8)
        np_signal_diff = np.diff(np_signal_ssa.sum(axis=1))

        list_idx = []
        for i, val_i in enumerate(np_signal_diff):
            if i > 0:
                if np.sign(val_i) != np.sign(np_signal_diff[i-1]):
                    list_idx.append(i - 1)

        np_support_idx = np.asarray(list_idx)
        min_idx = np_signal_ssa.shape[0] - self.plt_len
        plt_chk = min_idx <= np_support_idx

        return signal.iloc[np_support_idx[plt_chk]]

    def _processing_signals(self,
                            dtm_x: Dict[int, pd.Series],
                            plot_path: str = None,
                            ) -> Dict[int, Dict[str, Any]]:

        signal_process = dict()
        for dt_k, sr_x_dt_k in dtm_x.items():

            # First Signal Decomposition
            pd_signal_decomp_k = self._signal_decomposition(sr_x_dt_k, self.opt_dsvars.loc[dt_k, "nsignal"])
            sr_signal_sum_k = pd_signal_decomp_k.iloc[self.opt_dsvars.loc[dt_k, "bclen"]:, :].sum(axis=1)

            bc_win_k = self.opt_dsvars.loc[dt_k, ["window", "bclen"]].sum()

            # Forecast the signal
            pd_forecast_multi_signals_k = self._forecast_multi_signals(pd_signal_decomp_k[-bc_win_k:],
                                                         out_len=self.opt_dsvars.loc[dt_k, "outlen"],
                                                         bc_len=self.opt_dsvars.loc[dt_k, "bclen"],
                                                         plot_path=plot_path)

            pd_ft_forecast_signal_k = self._forecast_signal(sr_x_dt_k,
                                                            out_len=self.opt_dsvars.loc[dt_k, "outlen"],
                                                            plot_path=plot_path)

            pd_forecast_multi_signals_k = pd.concat([pd_forecast_multi_signals_k,
                                                     pd_ft_forecast_signal_k['ft_mean']], axis=1)

            if dt_k == 60:
                pd_forecast_multi_signals_k.index -= pd.Timedelta(minutes=45)

            # Trade Parameters
            fc_mean_k = pd_forecast_multi_signals_k["mean"]

            dy0_k = fc_mean_k.iloc[0].item() - sr_x_dt_k.iloc[-1].item()
            dy1_k = fc_mean_k.iloc[-1].item() - fc_mean_k.iloc[0].item()

            min_mean_val_k = (fc_mean_k - fc_mean_k.iloc[0].item()).min().item()
            max_mean_val_k = (fc_mean_k - fc_mean_k.iloc[0].item()).max().item()

            if min_mean_val_k < 0. and min_mean_val_k < dy1_k:
                dyf_k = min_mean_val_k
            elif max_mean_val_k > 0. and max_mean_val_k > dy1_k:
                dyf_k = max_mean_val_k
            else:
                dyf_k = (fc_mean_k - fc_mean_k.iloc[0].item()).mean().item()

            # Store Values
            signal_process[dt_k] = dict(signal_decomp=pd_signal_decomp_k,
                                        signal_sum=sr_signal_sum_k,
                                        bc_win=bc_win_k,
                                        forecast=pd_forecast_multi_signals_k,
                                        dy0=dy0_k,
                                        dy1=dy1_k,
                                        dym=dyf_k,
                                        )

        # FIXME: forecast correction using dt15 latest value
        return signal_process

    def prediction(self):

        local_obj_plot_path = (project_local_path / f"{self.dtype}_plots").resolve()
        local_obj_plot_path.mkdir(exist_ok=True)

        dt15_bclen = self.opt_dsvars.loc[15, "bclen"]
        dt60_bclen = self.opt_dsvars.loc[60, "bclen"]

        ticker_keys = ['open', 'close', 'high', 'low']

        signal15_len = self.opt_dsvars.loc[15, ["decomplen", "bclen"]].sum()
        signal60_len = 4 * self.opt_dsvars.loc[60, ["decomplen", "bclen"]].sum()

        dt60_decomp_idx = np.arange(0, signal60_len)
        dt15_to_dt60 = np.flip(np.arange(dt60_decomp_idx[-1], 0, step=-4))

        sr_dt15_x = self.sr_data_rs[-signal15_len:]
        index_dt60_x = self.sr_data_rs[-signal60_len:].index[dt15_to_dt60]
        sr_dt60_x = self.sr_data_rs.loc[index_dt60_x]

        dtm_x = {15: sr_dt15_x, 60: sr_dt60_x}

        # Decompose and process the signals, where Y =: F(X)
        dict_signal_process_i = self._processing_signals(dtm_x)

        # Get Support Points (FIXME)
        # sr_support_points_i = self._get_support_points(dtm_x[15])

        # Config for plotting
        dt_object = datetime.datetime.fromtimestamp(time.time())
        self.str_timestamp = dt_object.strftime("%Y%m%d_%H%M%S")
        file_path = f'{local_plot_path}/{self.str_timestamp}_dual_forecast.png'

        # Prepare data for plotting
        df_inp_tickers = self.input_data.loc[dtm_x[15].index[-self.plt_len:], ticker_keys]

        ## Real
        sr_dt15_y_signal_decomp_sum = dict_signal_process_i[15]["signal_sum"][-self.plt_len:]
        sr_dt60_y_signal_decomp_sum = dict_signal_process_i[60]["signal_sum"][-(self.plt_len // 4):]

        dt15_output = dict_signal_process_i[15]["forecast"].iloc[dt15_bclen:, :]
        dt60_output = dict_signal_process_i[60]["forecast"].iloc[dt60_bclen:, :]

        pkwargs = dict(
                swt15=apply_minmax_inverse_scaler(self.scaler_data, sr_dt15_y_signal_decomp_sum),
                swt60=apply_minmax_inverse_scaler(self.scaler_data, sr_dt60_y_signal_decomp_sum),
                dt15_output_tickers=apply_minmax_inverse_scaler(self.scaler_data, dt15_output),
                dt60_output_tickers=apply_minmax_inverse_scaler(self.scaler_data, dt60_output),
                plot_name=file_path,
                dpi=500,
                )

        dual_plot_mpl_ticker(df_inp_tickers, **pkwargs)


def main_opt_prediction():

    # Get the latest data
    # df_price_data, fig = get_latest_bitmex_data(symbol='XBTUSD', hours=900, dt=15, plot=False)
    # fig.savefig(local_plot_path / 'situation.png', dpi=300, bbox_inches='tight')

    seed = 42
    # dtype = "swt"
    # dtype = "emd"
    dtype = "ewt"
    # dtype = "ssa"

    # run_size = 50
    run_size = 300

    dict_cfg = dict(
        swt={
            15: pd.Series(dict(window=334, decomplen=430, bclen=3, nsignal=13, outlen=12, dtype="swt"), name=15),
            60: pd.Series(dict(window=328, decomplen=863, bclen=1, nsignal=3, outlen=8, dtype="swt"), name=60),
        },
        emd={
            15: pd.Series(dict(window=160, decomplen=1280, bclen=0, nsignal=13, outlen=12, dtype="emd"), name=15),
            # 15: pd.Series(dict(window=135, decomplen=649, bclen=1, nsignal=11, outlen=12, dtype="emd"), name=15),
            60: pd.Series(dict(window=796, decomplen=1126, bclen=7, nsignal=8, outlen=8, dtype="emd"), name=60),
        },
        ewt={
            # 15: pd.Series(dict(window=861, decomplen=3361, bclen=0, nsignal=18, outlen=12, dtype="ewt"), name=15),
            # 15: pd.Series(dict(window=446, decomplen=1317, bclen=0, nsignal=12, outlen=12, dtype="ewt"), name=15),
            # 15: pd.Series(dict(window=124, decomplen=707, bclen=0, nsignal=13, outlen=12, dtype="ewt"), name=15),
            15: pd.Series(dict(window=116, decomplen=1281, bclen=0, nsignal=17, outlen=12, dtype="ewt"), name=15),
            60: pd.Series(dict(window=106, decomplen=831, bclen=2, nsignal=16, outlen=8, dtype="ewt"), name=60),
        },
        ssa={
            15: pd.Series(dict(window=1794, decomplen=2228, bclen=9, nsignal=17, outlen=12, dtype="ssa"), name=15),
            60: pd.Series(dict(window=445, decomplen=4133, bclen=0, nsignal=18, outlen=8, dtype="ssa"), name=60),
        },
    )

    pd_opt_forecast_cfg = pd.concat([dict_cfg[dtype][15], dict_cfg[dtype][60]], axis=1).T

    opt_forecst = DualOptForecast(opt_dsvars=pd_opt_forecast_cfg,
                                  inc_len=50,
                                  plt_len=120,
                                  seed=seed,
                                  dtype=dtype,
                                  ftype="backward",
                                  plot_flag=True,
                                  run_size=run_size,
                                  )

    # ftime = 1800        # 30 minutes
    # ftime = 900         # 15 minutes

    for i in range(100):
        opt_forecst.update_input_data()
        opt_forecst.prediction()

        if i < 99:
            # Get the timestamp of the latest ticker
            ticker_minute_i = opt_forecst.input_data.index[-1].minute
            current_time_i = datetime.datetime.now().minute
            min_diff_i = 15 - (current_time_i - ticker_minute_i) + 0.2
            adjusted_wait = int(abs(min_diff_i) * 60.)

            for remaining in tqdm(
                    range(adjusted_wait, 0, -1),
                    desc=f"Trial {i}, ⏳ Next prediction in",
                    unit="s",
                    colour="green",
                    bar_format="{l_bar}{bar}| {remaining}s left"
            ):
                time.sleep(1)

            print("\n")


if __name__ == "__main__":
    main_opt_prediction()