from pathlib import Path
# import time
# import datetime
# import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import load
# from scipy.optimize import differential_evolution, direct, minimize
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

from tirex import ForecastModel, load_model
from tirex.utils.filters import ConvolutionFilter, quadratic_fit_series
from tirex.utils.ewt import EmpiricalWaveletTransform
from tirex.utils.ceemdan import ICEEMDAN
from tirex.utils.ssa import ssa
from tirex.utils.time import create_time_index
from tirex.utils.path import cleanup_directory
from tirex.utils.plot import plot_fc, dual_plot_mpl_ticker
# from tirex.utils.trade import TrailingStopOrder

# Add the project root to the Python path
project_local_path = Path(__file__).resolve().parent

local_plot_path = (project_local_path / "ewt_plots").resolve()
local_plot_path.mkdir(exist_ok=True)


class DualOptForecast:

    def __init__(self,
                 input_data: pd.DataFrame,
                 opt_dsvars: pd.DataFrame,
                 plt_len: int = 120,
                 inc_len: int = 500,
                 run_size: int = 50,
                 dtype: str = "ewt",
                 ftype: str = "convolution",
                 plot_flag: bool = False,
                 seed: int = 42,
                 debug: bool = False,
                 **kwargs,
                 ):
        """
        Initialize the OptForecast instance.

        Args:
            input_data (pd.DataFrame): Input data frame with a 'close' column.
            out_len (int): Number of future time steps to forecast.
            plt_len (int): Window length for plotting historical data.
            inc_len (int): Increment length for sliding evaluation windows.
            run_size (int): Number of evaluation runs to perform.
            dtype (str): Decomposition type ('ewt', 'xwt', 'emd', 'ssa').
            ftype (str): Filter type ('convolution' or none).
            plot_flag (bool): Whether to enable plotting of results.
            seed (int): Random seed for reproducibility.
            debug (bool): If True, use mock forecast model.
        """

        self.input_data = input_data

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
        self.ewt = EmpiricalWaveletTransform()

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": float}
        self.emd = ICEEMDAN(trials=20, max_imf=-1, **config)

        self.ssa_wlen = 20  # Window length for SSA

        # Forecast Model and function
        self._model = None
        self._forecast = None

        self.df_data = input_data["close"]
        self.scaler_data = MinMaxScaler(feature_range=(0., 100.))

        self.np_data_idx = np.arange(len(self.df_data))
        self.sr_data_rs = None

        self.np_idx_inc_eval = np.array([])

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

        list_incs = []
        inc_i = 3000
        max_length = self.df_data.shape[0] - (self.out_len + 10)
        for i in range(1000):
            if inc_i > max_length:
                break
            list_incs.append(inc_i)
            inc_i += self.inc_len

        self.np_idx_inc_eval = np.array(list_incs, dtype=int)
        np.random.seed(self.seed)
        np.random.shuffle(self.np_idx_inc_eval)

        np_data_rs = self.scaler_data.fit_transform(self.df_data.values.reshape(-1, 1)).flatten()
        self.sr_data_rs = pd.Series(np_data_rs, index=self.input_data.index)

    def _load_forecast_model(self):
        """
        Load the forecasting model. If debug is enabled, set the forecast method to mock.
        Otherwise, load the pretrained TiRex model.
        """
        try:
            model_file_path = (Path(__file__).parent.parent.parent / "model" / "model.ckpt").resolve()
            self._model: ForecastModel = load_model(str(model_file_path))
            self._forecast = self._model.forecast

        except Exception as e:
            print(f"Error loading model: {e}")
            return

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

    def _forecast_signal(self,
                         decomp_signals: pd.DataFrame,
                         out_len: int,
                         bclen: int,
                         plot_path: Optional[Path] = None,
                         ) -> pd.DataFrame:
        """
        Forecast each decomposed component, aggregate quantiles and mean, and optionally plot.

        Args:
            decomp_signals (pd.DataFrame): DataFrame of decomposed components.
            bclen (int): Backcast length for each component.
            plot_path (Optional[Path]): Directory to save forecast plots.

        Returns:
            pd.DataFrame: Concatenated quantiles and mean forecasts.
        """

        np_index = np.arange(0, decomp_signals.shape[0] + out_len)
        np_input_idx = np_index[:decomp_signals.shape[0] - bclen]
        np_output_idx = np_index[decomp_signals.shape[0] - bclen:]
        dt_time = decomp_signals.index[1] - decomp_signals.index[0]
        num_dt = int(dt_time.total_seconds() / 60)

        list_output_datetime_idx = create_time_index(dt_time,
                                                     decomp_signals.index[np_input_idx][-1],
                                                     np_output_idx.size)
        list_quantiles = []
        list_mean = []

        for k, (signal_id_k, sr_signal_val_k) in enumerate(decomp_signals.items()):
            sr_signal_x_k = sr_signal_val_k.iloc[np_input_idx]
            np_quantiles_y_k, np_mean_y_k = self._forecast(sr_signal_x_k.values,
                                                     prediction_length=out_len + bclen,
                                                     output_type="numpy",
                                                     )

            pd_quantiles_y_k = pd.DataFrame(np_quantiles_y_k[0], index=list_output_datetime_idx)
            pd_mean_y_k = pd.Series(np_mean_y_k[0], index=list_output_datetime_idx)

            list_quantiles.append(pd_quantiles_y_k)
            list_mean.append(pd_mean_y_k)

            if self.plot_flag and plot_path is not None:
                # TODO: add dt to the name
                plt_kwargs = dict(save_path=f'{plot_path}/decomp_signal_dt{num_dt}_i{k}_pred.png')
                plt_kwargs["rescale"] = self.scaler_data

                if bclen > 0:
                    plt_kwargs["bcs"] = pd_mean_y_k.iloc[:bclen]

                plot_fc(sr_signal_val_k.iloc[-self.plt_len:],
                        pd_quantiles_y_k.iloc[bclen:, :], **plt_kwargs)

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

    def _get_input_signals(self, dt15_index: np.ndarray, dt60_index: np.ndarray) -> Dict[int, pd.Series]:
        """
        Get the input signals for the 15-minute and 60-minute timeframes.

        Args:
            dt15_index (np.ndarray): Indices for the 15-minute timeframe.

        Returns:
            Tuple[pd.Series, pd.Series]: Input signals for 15-minute and 60-minute timeframes.
        """
        sr_dt15_x = self.sr_data_rs.iloc[dt15_index]
        index_dt60_x = self.sr_data_rs.index[dt60_index]
        index_dt60_x += sr_dt15_x.index[-1] - index_dt60_x[-1]

        sr_dt60_x = self.sr_data_rs.loc[index_dt60_x]

        return {15: sr_dt15_x, 60: sr_dt60_x}

    def _get_support_points(self, signal: pd.Series) -> pd.Series:

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

    def _processing_signals(self, dtm_x: Dict[int, pd.Series], plot_path: str = None) -> Dict[int, Dict[str, Any]]:

        signal_process = dict()
        for dt_k, sr_x_dt_k in dtm_x.items():

            # First Signal Decomposition
            pd_signal_decomp_k = self._signal_decomposition(sr_x_dt_k, self.opt_dsvars.loc[dt_k, "nsignal"])
            sr_signal_sum_k = pd_signal_decomp_k.iloc[self.opt_dsvars.loc[dt_k, "bclen"]:, :].sum(axis=1)

            bc_win_k = self.opt_dsvars.loc[dt_k, ["window", "bclen"]].sum()

            # Forecast the signal
            pd_forecast_k = self._forecast_signal(pd_signal_decomp_k[-bc_win_k:],
                                                  out_len=self.opt_dsvars.loc[dt_k, "outlen"],
                                                  bclen=self.opt_dsvars.loc[dt_k, "bclen"],
                                                  plot_path=plot_path)

            if dt_k == 60:
                pd_forecast_k.index -= pd.Timedelta(minutes=45)

            # Trade Parameters
            dy0_k = pd_forecast_k["mean"].iloc[0].item() - sr_x_dt_k.iloc[-1].item()
            dy1_k = pd_forecast_k["mean"].iloc[-1].item() - pd_forecast_k["mean"].iloc[0].item()

            min_mean_val_k = (pd_forecast_k["mean"] - pd_forecast_k["mean"].iloc[0].item()).min().item()
            max_mean_val_k = (pd_forecast_k["mean"] - pd_forecast_k["mean"].iloc[0].item()).max().item()

            if min_mean_val_k < 0. and min_mean_val_k < dy1_k:
                dyf_k = min_mean_val_k
            elif max_mean_val_k > 0. and max_mean_val_k > dy1_k:
                dyf_k = max_mean_val_k
            else:
                dyf_k = (pd_forecast_k["mean"] - pd_forecast_k["mean"].iloc[0].item()).mean().item()

            # Store Values
            signal_process[dt_k] = dict(signal_decomp=pd_signal_decomp_k,
                                               signal_sum=sr_signal_sum_k,
                                               bc_win=bc_win_k,
                                               forecast=pd_forecast_k,
                                               dy0=dy0_k,
                                               dy1=dy1_k,
                                               dym=dyf_k,
                                               )

        # FIXME: forecast correction using dt15 latest value
        return signal_process

    def opt_trade(self, td_dsvars: np.ndarray) -> float:

        local_obj_plot_path = (project_local_path / f"{self.dtype}_plots").resolve()
        local_obj_plot_path.mkdir(exist_ok=True)
        local_plot_path_i = None

        dt15_bclen = self.opt_dsvars.loc[15, "bclen"]
        dt60_bclen = self.opt_dsvars.loc[60, "bclen"]

        dt15_time = self.sr_data_rs.index[1] - self.sr_data_rs.index[0]

        dt15_decomp_idx = np.arange(0, self.opt_dsvars.loc[15, ["decomplen", "bclen"]].sum())
        dt60_decomp_idx = np.arange(0, 4 * self.opt_dsvars.loc[60, ["decomplen", "bclen"]].sum())
        dt15_to_dt60 = np.flip(np.arange(dt60_decomp_idx[-1], 0, step=-4))

        # list_trade_ops = []
        list_trade_gain = []

        for i, idx_i in enumerate(tqdm(self.np_idx_inc_eval[:self.run_size], desc="Processing")):
            dt15_decomp_idx_i = dt15_decomp_idx + (idx_i - self.opt_dsvars.loc[15, "decomplen"])
            dt60_decomp_idx_i = (dt60_decomp_idx + (idx_i - 4 * self.opt_dsvars.loc[60, "decomplen"]))[dt15_to_dt60]

            if self.plot_flag:
                local_plot_path_i = (local_obj_plot_path / f"trial_{i}").resolve()
                if not local_plot_path_i.is_dir():
                    local_plot_path_i.mkdir(exist_ok=True)
                else:
                    cleanup_directory(local_plot_path_i)

            # Get inputs (X) from 15 and 60 minutes tickers
            try:
                dtm_x_i = self._get_input_signals(dt15_decomp_idx_i, dt60_decomp_idx_i)

                # Decompose and process the signals, where Y =: F(X)
                dict_signal_process_i = self._processing_signals(dtm_x_i, plot_path=local_plot_path_i)

                # Get Support Points
                sr_support_points_i = self._get_support_points(dtm_x_i[15])

                ## Reference (Future Values)
                end_dt15_i = dict_signal_process_i[15]["forecast"].index[dt15_bclen:][0] - dt15_time
                list_dt15_output_idx_i = create_time_index(dt15_time, end_dt15_i, 160)

                # #############################################################################
                # TODO: Calculate A-B-C for the prediction
                # trailing order parameter
                ## tolerance parameter (to enter into the trade)
                dx1_15_i = 100. * dict_signal_process_i[15]["dy1"] / dtm_x_i[15].iloc[-1].item()
                dx1_60_i = 100. * dict_signal_process_i[60]["dy1"] / dtm_x_i[15].iloc[-1].item()
                dxm_60_i = 100. * dict_signal_process_i[60]["dym"] / dtm_x_i[15].iloc[-1].item()

                df_trade_bounds_i = pd.DataFrame()
                if abs(dx1_15_i) >= 0.3 and (abs(dx1_60_i) >= 1. or abs(dxm_60_i) >= 1.):
                    list_dt15_td_output_idx_i = create_time_index(dt15_time, end_dt15_i, 64)
                    np_ones = np.ones(len(list_dt15_td_output_idx_i), dtype=float)

                    if dx1_15_i > 0. and (dx1_60_i >= 0. or dxm_60_i >= 0.):
                        # Buy
                        np_lwr_i = 0.99 * dict_signal_process_i[15]["forecast"].iloc[:, 0].min()
                        np_upp_i = dict_signal_process_i[15]["forecast"]["mean"].iloc[-1].item()

                    else:
                        # Sell
                        np_lwr_i = 1.01 * dict_signal_process_i[15]["forecast"].iloc[:, 8].max()
                        np_upp_i = dict_signal_process_i[15]["forecast"]["mean"].iloc[-1].item()

                    sr_stop_loss_i = pd.Series(np_lwr_i * np_ones, index=list_dt15_td_output_idx_i, name="stop_loss")
                    sr_take_profit_i = pd.Series(np_upp_i * np_ones, index=list_dt15_td_output_idx_i, name="take_profit")

                    df_trade_bounds_i = pd.concat([sr_stop_loss_i, sr_take_profit_i], axis=1)

                # #############################################################################
                if self.plot_flag:
                    # plot the trade operation
                    file_path_i = f'{local_plot_path_i}/dual_forecast.png'
                    df_inp_tickers_i = self.input_data.loc[dtm_x_i[15].index[-self.plt_len:], ['open', 'close', 'high', 'low']]
                    df_inp_tickers_rs_i = self._rescaling(df_inp_tickers_i)

                    ## Real
                    df_dt15_y_i = self.input_data.loc[list_dt15_output_idx_i, ['open', 'close', 'high', 'low']]
                    df_dt15_y_rs_i = self._rescaling(df_dt15_y_i)

                    sr_dt15_y_signal_decomp_sum_i = dict_signal_process_i[15]["signal_sum"][-self.plt_len:]
                    sr_dt60_y_signal_decomp_sum_i = dict_signal_process_i[60]["signal_sum"][-(self.plt_len // 4):]

                    pkwargs = dict(forward_tickers=df_dt15_y_rs_i,
                                   swt15=sr_dt15_y_signal_decomp_sum_i,
                                   swt60=sr_dt60_y_signal_decomp_sum_i,
                                   dt15_output_tickers=dict_signal_process_i[15]["forecast"].iloc[dt15_bclen:, :],
                                   dt60_output_tickers=dict_signal_process_i[60]["forecast"].iloc[dt60_bclen:, :],
                                   trade_bounds=df_trade_bounds_i,
                                   support=sr_support_points_i,
                                   plot_name=file_path_i,
                                   dpi=500,
                                   rescale=self.scaler_data,
                                   )

                    dual_plot_mpl_ticker(df_inp_tickers_rs_i, **pkwargs)

            except Exception as m_err:
                print(f"Error in processing signals for trial {i}: {m_err}")
                continue

            else:
                list_trade_gain.append(0.)

        # Calculate the total trade gain
        fval = -np.asarray(list_trade_gain).sum()
        print(f"Trade gain: {fval}, Parameters: {td_dsvars}")

        return fval


def main_opt_trade():

    # input_data_path = (Path(__file__).parent.parent.parent / "tests" / "data" / "btcusd_2022-06-01.joblib").resolve()
    # input_data_path = (Path(__file__).parent / "data" / "btcusd_15m_2025-10-26.parquet").resolve()
    input_data_path = (Path(__file__).parent.parent.parent / "Signals/data/btcusd_15m_2025-11-01.parquet").resolve()
    # input_data_path = (Path(__file__).parent.parent.parent / "Signals/data/btcusd_15m_2025-11-02.parquet").resolve()

    dt = 15
    # dt = 60

    if not input_data_path.is_file():
        raise FileNotFoundError(f"Input data file not found: {input_data_path}")

    if input_data_path.suffix == ".joblib":
        dict_price_data = load(input_data_path)
        df_price_data = dict_price_data[dt]

    elif input_data_path.suffix == ".parquet":
        df_price_data = pd.read_parquet(input_data_path)

    else:
        df_price_data = pd.DataFrame()

    seed = 42
    # dtype = "swt"
    # dtype = "emd"
    # dtype = "ewt"
    dtype = "ssa"

    # run_size = 50
    run_size = 300

    dict_cfg = dict(
        swt={
            15: pd.Series(dict(window=334, decomplen=430, bclen=3, nsignal=13, outlen=12, dtype="swt"), name=15),
            60: pd.Series(dict(window=328, decomplen=863, bclen=1, nsignal=3, outlen=8, dtype="swt"), name=60),
        },
        emd={
            15: pd.Series(dict(window=160, decomplen=1280, bclen=0, nsignal=13, outlen=12, dtype="emd"), name=15),
            60: pd.Series(dict(window=796, decomplen=1126, bclen=7, nsignal=8, outlen=8, dtype="emd"), name=60),
        },
        ewt={
            # 15: pd.Series(dict(window=861, decomplen=3361, bclen=0, nsignal=18, outlen=12, dtype="ewt"), name=15),
            15: pd.Series(dict(window=446, decomplen=1317, bclen=0, nsignal=12, outlen=12, dtype="ewt"), name=15),
            60: pd.Series(dict(window=106, decomplen=831, bclen=2, nsignal=16, outlen=8, dtype="ewt"), name=60),
        },
        ssa={
            15: pd.Series(dict(window=1794, decomplen=2228, bclen=9, nsignal=17, outlen=12, dtype="ssa"), name=15),
            60: pd.Series(dict(window=445, decomplen=4133, bclen=0, nsignal=18, outlen=8, dtype="ssa"), name=60),
        },
    )

    pd_opt_forecast = pd.concat([dict_cfg[dtype][15], dict_cfg[dtype][60]], axis=1).T

    opt_ewt_forecst = DualOptForecast(input_data=df_price_data,
                                      opt_dsvars=pd_opt_forecast,
                                      inc_len=50,
                                      plt_len=120,
                                      seed=seed,
                                      dtype=dtype,
                                      ftype="",
                                      plot_flag=True,
                                      run_size=run_size,
                                      )

    # [0]: tolerance parameter (to enter into the trade) [buy]
    # [1]: trailing stop value (percentage) [buy]
    # [2]: tolerance parameter (to enter into the trade) [sell]
    # [3]: trailing stop value (percentage) [sell]
    # np_opt_trade = np.array([0.00628666, 0.55794069, 0.78817656, 1.84162112], dtype=float)
    np_opt_trade = np.array([0.0718302, 6.62697033, 0.10341756, 7.19575578], dtype=float)
    fval = opt_ewt_forecst.opt_trade(np_opt_trade)

    bounds = [(0.0001, 1.), (0.0001, 10.), (0.0001, 1.), (0.0001, 10.)]

    # result = differential_evolution(
    #     func=opt_ewt_forecst.opt_trade,
    #     bounds=bounds,
    #     x0=np_opt_trade,
    #     strategy='best1bin',
    #     maxiter=100,    # Number of generations
    #     popsize=15,     # Population size
    #     tol=0.001,      # Tolerance for convergence
    #     disp=True,      # Display optimization progress
    #     seed=42         # Uncomment for reproducibility
    # )

    # result = direct(opt_ewt_forecst.opt_trade, bounds)
    # result = minimize(opt_ewt_forecst.opt_trade, np_opt_trade, method='Nelder-Mead', bounds=bounds)

    # print(f"Optimization result: {result}")
    test = 1.


if __name__ == "__main__":
    main_opt_trade()