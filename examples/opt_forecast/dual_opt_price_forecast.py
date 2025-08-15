from pathlib import Path
import time
import datetime
# import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import load
from scipy.optimize import differential_evolution, direct, minimize
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
from tirex.utils.trade import TrailingStopOrder

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

        self.out_len = self.opt_dsvars.loc[60, "outlen"].item() * 4
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

    def opt_trade(self, td_dsvars: np.ndarray) -> float:

        local_obj_plot_path = (project_local_path / f"{self.dtype}_plots").resolve()
        local_obj_plot_path.mkdir(exist_ok=True)
        local_plot_path_i = None

        dt15_time = self.sr_data_rs.index[1] - self.sr_data_rs.index[0]
        # dt60_time = self.sr_data_rs.index[4] - self.sr_data_rs.index[0]

        dt15_bclen = self.opt_dsvars.loc[15, "bclen"].item()
        dt60_bclen = self.opt_dsvars.loc[60, "bclen"].item()

        dt15_decomp_idx = np.arange(0, self.opt_dsvars.loc[15, ["decomplen", "bclen"]].sum().item())
        dt60_decomp_idx = np.arange(0, 4 * self.opt_dsvars.loc[60, ["decomplen", "bclen"]].sum().item())
        dt15_to_dt60 = np.flip(np.arange(dt60_decomp_idx[-1], 0, step=-4))

        list_trade_ops = []
        list_trade_gain = []

        for i, idx_i in enumerate(tqdm(self.np_idx_inc_eval[:self.run_size], desc="Processing")):
            dt15_decomp_idx_i = dt15_decomp_idx + (idx_i - self.opt_dsvars.loc[15, "decomplen"].item())
            dt60_decomp_idx_i = (dt60_decomp_idx + (idx_i - 4 * self.opt_dsvars.loc[60, "decomplen"].item()))[dt15_to_dt60]
            dt60_decomp_idx_i -= dt60_decomp_idx_i[-1] - dt15_decomp_idx_i[-1] + 4

            # Input Signal (X), where Y := F(X)
            sr_dt15_x_ft_i = self.sr_data_rs.iloc[dt15_decomp_idx_i]
            sr_dt60_x_ft_i = self.sr_data_rs.iloc[dt60_decomp_idx_i]

            if self.plot_flag:
                local_plot_path_i = (local_obj_plot_path / f"trial_{i}").resolve()
                if not local_plot_path_i.is_dir():
                    local_plot_path_i.mkdir(exist_ok=True)
                else:
                    cleanup_directory(local_plot_path_i)

            # First Signal Decomposition
            pd_dt15_signal_decomp_i = self._signal_decomposition(sr_dt15_x_ft_i, self.opt_dsvars.loc[15, "nsignal"].item())
            pd_dt60_signal_decomp_i = self._signal_decomposition(sr_dt60_x_ft_i, self.opt_dsvars.loc[60, "nsignal"].item())

            # Forecast the signal
            dt15_bc_win_i = self.opt_dsvars.loc[15, ["window", "bclen"]].sum().item()
            dt60_bc_win_i = self.opt_dsvars.loc[60, ["window", "bclen"]].sum().item()

            pd_dt15_forecast_i = self._forecast_signal(pd_dt15_signal_decomp_i[-dt15_bc_win_i:],
                                                       out_len=self.opt_dsvars.loc[15, "outlen"].item(),
                                                       bclen=dt15_bclen,
                                                       plot_path=local_plot_path_i)

            pd_dt60_forecast_i = self._forecast_signal(pd_dt60_signal_decomp_i[-dt60_bc_win_i:],
                                                       out_len=self.opt_dsvars.loc[60, "outlen"].item(),
                                                       bclen=dt60_bclen,
                                                       plot_path=local_plot_path_i)

            ## Reference (Future Values)
            end_dt15_i = pd_dt15_forecast_i.index[dt15_bclen:][0] - dt15_time
            # end_dt60_i = pd_dt60_forecast_i.index[dt60_bclen:][0] - dt60_time

            list_dt15_output_idx_i = create_time_index(dt15_time, end_dt15_i, 160)
            # list_dt60_output_idx_i = create_time_index(dt60_time, end_dt60_i, 40)

            # Future Values
            ## Prediction

            # TODO: Calculate A-B-C for the prediction
            # trailing order parameter
            ## tolerance parameter (to enter into the trade)

            if self.plot_flag:
                # plot the trade operation
                file_path_i = f'{local_plot_path_i}/dual_forecast.png'
                df_inp_tickers_i = self.input_data.loc[sr_dt15_x_ft_i.index[-self.plt_len:], ['open', 'close', 'high', 'low']]
                df_inp_tickers_rs_i = self._rescaling(df_inp_tickers_i)

                ## Real
                df_dt15_y_i = self.input_data.loc[list_dt15_output_idx_i, ['open', 'close', 'high', 'low']]
                df_dt15_y_rs_i = self._rescaling(df_dt15_y_i)

                sr_dt15_y_signal_decomp_sum_i = pd_dt15_signal_decomp_i.iloc[dt15_bclen:, :].sum(axis=1)[-self.plt_len:]
                sr_dt60_y_signal_decomp_sum_i = pd_dt60_signal_decomp_i.iloc[dt60_bclen:, :].sum(axis=1)[-(self.plt_len // 4):]

                pkwargs = dict(forward_tickers=df_dt15_y_rs_i,
                               swt15=sr_dt15_y_signal_decomp_sum_i,
                               swt60=sr_dt60_y_signal_decomp_sum_i,
                               dt15_output_tickers=pd_dt15_forecast_i.iloc[dt15_bclen:, :],
                               dt60_output_tickers=pd_dt60_forecast_i.iloc[dt60_bclen:, :],
                               plot_name=file_path_i,
                               dpi=500,
                               )

                dual_plot_mpl_ticker(df_inp_tickers_rs_i, **pkwargs)

            else:
                list_trade_gain.append(0.)

        # Calculate the total trade gain
        fval = -np.asarray(list_trade_gain).sum()
        print(f"Trade gain: {fval}, Parameters: {td_dsvars}")

        return fval


def main_opt_trade():

    input_data_path = (Path(__file__).parent.parent.parent / "tests" / "data" / "btcusd_2022-06-01.joblib").resolve()
    dict_price_data = load(input_data_path)
    dt = 15

    seed = 42
    dtype = "swt"
    # dtype = "emd"

    # run_size = 50
    run_size = 300

    # dt 15
    # sr_opt_forecast = pd.Series(dict(window=1600, decomplen=1900, bclen=3, nsignal=6))

    # dt 60
    # sr_opt_forecast = pd.Series(dict(window=200, decomplen=600, bclen=1, nsignal=8))
    # sr_opt_forecast = pd.Series(dict(window=700, decomplen=5200, bclen=4, nsignal=7))

    pd_opt_forecast = pd.DataFrame(dict(window=[1300, 300],
                                        decomplen=[4700, 600],
                                        bclen=[3, 1],
                                        nsignal=[11, 14],
                                        outlen=[12, 8],
                                        ),
                                   index=[15, 60])

    opt_ewt_forecst = DualOptForecast(input_data=dict_price_data[dt],
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