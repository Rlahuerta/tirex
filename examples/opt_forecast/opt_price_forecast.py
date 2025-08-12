from pathlib import Path
import time
import datetime
# import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import load
from scipy.linalg import svd
from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

from tirex import ForecastModel, load_model
from tirex.utils.filters import ConvolutionFilter
from tirex.utils.ewt import EmpiricalWaveletTransform
from tirex.utils.ceemdan import ICEEMDAN
from tirex.utils.plot import plot_fc
from tirex.utils.trade import TrailingStopOrder

# Add the project root to the Python path
project_local_path = Path(__file__).resolve().parent

local_plot_path = (project_local_path / "ewt_plots").resolve()
local_plot_path.mkdir(exist_ok=True)


def cleanup_directory(directory_path):
    # Create a Path object for the directory
    path = Path(directory_path)

    # Ensure the directory exists
    if not path.exists() or not path.is_dir():
        print(f"The directory {directory_path} does not exist.")
        return

    # Iterate through each item in the directory
    for item in path.iterdir():
        # Check if it's a file and delete it
        if item.is_file():
            try:
                item.unlink()  # Delete the file
                print(f"Deleted file: {item}")
            except Exception as e:
                print(f"Error deleting file {item}: {e}")


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    sl, sk = matrix.shape
    nv = sl + sk - 1
    series = np.zeros(nv)
    for i in range(nv):
        values = []
        for j in range(max(0, i + 1 - sk), min(i + 1, sl)):
            values.append(matrix[j, i - j])
        series[i] = np.mean(values)
    return series


def create_time_index(delta_time, ini_time, size: int) -> List[datetime.datetime]:

    list_output_datetime_idx = []
    dt_current = ini_time
    for i in range(size):
        dt_current += delta_time
        list_output_datetime_idx.append(dt_current)

    return list_output_datetime_idx


class OptForecast:

    def __init__(self,
                 input_data: pd.DataFrame,
                 out_len: int,
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
        self.out_len = out_len
        self.plt_len = plt_len
        self.inc_len = inc_len
        self.seed = seed
        self.run_size = run_size

        self.plot_flag = plot_flag
        self._plot_local_path = None

        # FIXME
        self._debug = debug

        self.dtype = dtype
        self.ftype = ftype
        self.ewt = EmpiricalWaveletTransform()

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": float}
        self.emd = ICEEMDAN(trials=20, max_imf=-1, **config)

        self.ssa_wlen = 20  # Window length for SSA
        self._prd_dsvars = pd.Series()

        # Forecast Model and function
        self._model = None
        self._forecast = None

        self.df_data = input_data["close"]
        self.scaler_data = MinMaxScaler(feature_range=(0., 100.))

        self.np_data_idx = np.arange(len(self.df_data))
        self.sr_data_rs = None

        self.convolution_filter = None
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

    @staticmethod
    def _mock_forecast_model(input_x: np.ndarray,
                             prediction_length: int = 10,
                             **kwargs,
                             ) -> (np.ndarray, np.ndarray):
        """
        Generate a mock forecast for testing and debugging.

        Args:
            input_x (np.ndarray): Input time series values.
            prediction_length (int): Length of the forecast to generate.

        Returns:
            tuple: (quantiles array, mean forecast array).
        """

        x = np.linspace(0., 300., prediction_length)
        y_base = np.sin(x / 2.)

        list_quantiles = []
        np_noise = np.linspace(0.01, 0.2, 9)

        for noise_i in np_noise:
            y_main_i = 5. * noise_i * np.cumsum(np.random.random(prediction_length) - 0.5)
            y_noisy_i = y_base + y_main_i
            y_noisy_i += 1.1 * (np.abs(y_noisy_i)).max()
            y_noisy_i /= (np.abs(y_noisy_i)).max()
            y_noisy_i *= input_x.mean()
            y_noisy_i -= y_noisy_i[-1]
            y_noisy_i += input_x[-1]

            list_quantiles.append(y_noisy_i)

        np_quantiles = np.array([list_quantiles])
        np_mean = np_quantiles.mean(axis=1)

        return np_quantiles, np_mean

    def _load_forecast_model(self):
        """
        Load the forecasting model. If debug is enabled, set the forecast method to mock.
        Otherwise, load the pretrained TiRex model.
        """
        try:
            if not self._debug:
                model_file_path = (Path(__file__).parent.parent.parent / "model" / "model.ckpt").resolve()
                self._model: ForecastModel = load_model(str(model_file_path))
                self._forecast = self._model.forecast
            else:
                self._forecast = self._mock_forecast_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    @staticmethod
    def _ssa(input_signal: pd.Series,
             nsignal: int,
             wlen: int = 20,
             ) -> np.ndarray:
        """
        Perform Singular Spectrum Analysis (SSA) decomposition on a signal.

        Args:
            input_signal (pd.Series): Time series to decompose.
            nsignal (int): Number of components to extract.

        Returns:
            pd.DataFrame: Reconstructed components as columns.
        """

        # Step 1: Embedding
        k = input_signal.size - wlen + 1
        trajectory_matrix = np.zeros((wlen, k))
        for i in range(k):
            trajectory_matrix[:, i] = input_signal.iloc[i:i + wlen]

        # Step 2: SVD
        ut, sigma, vt = svd(trajectory_matrix)

        # Step 3: Grouping and Reconstruction
        component_indices = range(nsignal)  # Adjust based on how many components you expect
        reconstructed_components = []

        for idx in component_indices:
            component = ut[:, idx:idx + 1] @ np.diag(sigma[idx:idx + 1]) @ vt[idx:idx + 1, :]
            reconstructed_component = diagonal_averaging(component)
            reconstructed_components.append(reconstructed_component)

        return np.asarray(reconstructed_components).T

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
            np_ssa_res = self._ssa(input_signal, 2, wlen=8)

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
            return pd.DataFrame(self._ssa(input_signal, nsignal, wlen=self.ssa_wlen), index=input_signal.index)

        else:
            raise NotImplementedError(f"Unsupported decomposition type: {self.dtype}. Use 'ewt' or 'emd'.")

    def _filter(self, input_signal: pd.Series, flen: int = 3) -> pd.Series:
        """
        Apply filtering to the input signal based on configuration.

        Args:
            input_signal (pd.Series): Series to filter.
            flen (int): Filter length for convolution.

        Returns:
            pd.Series: Filtered signal.
        """

        if self.ftype == "convolution":
            if self.convolution_filter is None:
                self.convolution_filter = ConvolutionFilter(adim=input_signal.size, length=flen)
            elif self.convolution_filter.adim != input_signal.shape[0]:
                self.convolution_filter = ConvolutionFilter(adim=input_signal.size, length=flen)

            return pd.Series(self.convolution_filter(input_signal.values), index=input_signal.index)

        else:
            return input_signal

    def _forecast_signal(self,
                         decomp_signals: pd.DataFrame,
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

        np_index = np.arange(0, decomp_signals.shape[0] + self.out_len)
        np_input_idx = np_index[:decomp_signals.shape[0] - bclen]
        np_output_idx = np_index[decomp_signals.shape[0] - bclen:]
        dt_time = decomp_signals.index[1] - decomp_signals.index[0]

        list_output_datetime_idx = create_time_index(dt_time,
                                                     decomp_signals.index[np_input_idx][-1],
                                                     np_output_idx.size)
        list_quantiles = []
        list_mean = []

        for k, (signal_id_k, sr_signal_val_k) in enumerate(decomp_signals.items()):
            sr_signal_x_k = sr_signal_val_k.iloc[np_input_idx]
            np_quantiles_y_k, np_mean_y_k = self._forecast(sr_signal_x_k.values,
                                                     prediction_length=self.out_len + bclen,
                                                     output_type="numpy",
                                                     )

            pd_quantiles_y_k = pd.DataFrame(np_quantiles_y_k[0], index=list_output_datetime_idx)
            pd_mean_y_k = pd.Series(np_mean_y_k[0], index=list_output_datetime_idx)

            list_quantiles.append(pd_quantiles_y_k)
            list_mean.append(pd_mean_y_k)

            if self.plot_flag and plot_path is not None:
                plt_kwargs = dict(save_path=f'{plot_path}/decomp_signal_{k}_pred.png')
                if bclen > 0:
                    plt_kwargs["bcs"] = pd_mean_y_k.iloc[:bclen]

                plot_fc(sr_signal_val_k.iloc[-self.plt_len:],
                        pd_quantiles_y_k.iloc[bclen:, :], **plt_kwargs)

        pd_quantiles = sum(list_quantiles)
        sr_mean = sum(list_mean)
        sr_mean.name = "mean"

        return pd.concat([pd_quantiles, sr_mean], axis=1)

    def objective(self, dsvars: pd.Series) -> Dict[str, np.ndarray]:
        """
        Compute objective metrics for a set of design variables.

        Args:
            dsvars (pd.Series): Series containing 'window', 'decomplen', 'bclen', 'nsignal'.

        Returns:
            Dict[str, np.ndarray]: Dictionary with metrics:
                'lsq': least-squares errors,
                'efficiency': efficiency ratios,
                'performance': performance score,
                'dy_ref': reference slopes,
                'dy_prd': predicted slopes,
                'reference': list of reference series,
                'forecast': list of forecast series.
        """

        # Optimization Variables
        window = dsvars["window"]
        decomplen = dsvars["decomplen"]
        bclen = dsvars["bclen"]
        nsignal = dsvars["nsignal"]

        assert window > 50, f"window should be greater than 50, window is {window}"
        assert decomplen >= window, f"decomplen should be greater or equal than window, decomplen is {decomplen}"
        assert bclen >= 0, f"bclen should be greater or equal than 0, bclen is {bclen}"
        assert nsignal > 3, f"nsignal should be greater than 3, nsignal is {nsignal}"

        local_obj_plot_path = (project_local_path / f"{self.dtype}_plots").resolve()
        local_obj_plot_path.mkdir(exist_ok=True)
        local_plot_path_i = None

        decomp_idx = np.arange(0, decomplen + bclen)

        list_lsq = []
        list_y_ref = []
        list_y_prd = []
        list_dy_ref = []
        list_dy_prd = []
        list_eff_pred = []

        for i, idx_i in enumerate(tqdm(self.np_idx_inc_eval[:self.run_size], desc="Processing")):
            decomp_idx_i = decomp_idx + (idx_i - decomplen)

            # Input Signal (X), where Y := F(X)
            sr_data_rs_i = self.sr_data_rs.iloc[decomp_idx_i]
            sr_x_ft_i = self._filter(sr_data_rs_i)

            if self.plot_flag:
                local_plot_path_i = (local_obj_plot_path / f"trial_{i}").resolve()
                if not local_plot_path_i.is_dir():
                    local_plot_path_i.mkdir(exist_ok=True)
                else:
                    cleanup_directory(local_plot_path_i)

            # First Signal Decomposition
            pd_signal_decomp_i = self._signal_decomposition(sr_x_ft_i, nsignal)
            sr_y_decomp_sum_i = pd_signal_decomp_i.iloc[bclen:, :].sum(axis=1)

            # Forecast the signal
            pd_forecast_i = self._forecast_signal(pd_signal_decomp_i[-(window + bclen):],
                                                  bclen,
                                                  plot_path=local_plot_path_i)

            # Reference
            sr_y_i = self.sr_data_rs[pd_forecast_i.index[bclen:]]
            dy_i = (sr_y_i.values[-1] - sr_x_ft_i.values[-1]) / (sr_y_i.size + 1)
            list_dy_ref.append(dy_i)

            # Prediction
            sr_y_prd_i = pd_forecast_i["mean"][bclen:]
            dy_prd_i = (sr_y_prd_i.values[-1] - sr_y_prd_i.values[0]) / sr_y_prd_i.size
            list_dy_prd.append(dy_prd_i)

            y_eff_prd_i = dy_prd_i / (dy_i + 1.e-9)

            # Calculate LSQ cost function
            list_eff_pred.append(y_eff_prd_i)
            np_diff_i = (sr_y_i.values - sr_y_i.values[0]) - (sr_y_prd_i.values - sr_y_prd_i.values[0])
            list_lsq.append(np.dot(np_diff_i, np_diff_i))

            if self.plot_flag:
                list_y_ref.append(sr_y_i)
                list_y_prd.append(sr_y_prd_i)

                plot_fc(sr_x_ft_i[-self.plt_len:], pd_forecast_i.iloc[bclen:, :-1],
                        bcs=pd_forecast_i["mean"][:bclen],
                        real_future_values=sr_y_i,
                        decomp_sum=sr_y_decomp_sum_i[-self.plt_len:],
                        title=f"End Time: {sr_x_ft_i.index[-1]}",
                        save_path=f'{local_plot_path_i}/ewt_signal_sum_pred.png')

        np_lsq = np.asarray(list_lsq)
        np_eff_pred = np.asarray(list_eff_pred)

        np_dy_prd_idx = np.abs(np.asarray(list_dy_prd)) > 0.002
        np_eff_pred_cl = np_eff_pred[np_dy_prd_idx]
        neff_cl = (np_eff_pred_cl > 0.).sum() / np_eff_pred_cl.size

        return {"lsq": np_lsq,
                "efficiency": np_eff_pred,
                "performance": neff_cl,
                "dy_ref": np.asarray(list_dy_ref),
                "dy_prd": np.asarray(list_dy_prd),
                "reference": list_dy_ref,
                "forecast": list_y_prd,
                }

    def opt_trade(self, td_dsvars: np.ndarray) -> float:

        # Forecast Variables
        window = self._prd_dsvars["window"]
        decomplen = self._prd_dsvars["decomplen"]
        bclen = self._prd_dsvars["bclen"]
        nsignal = self._prd_dsvars["nsignal"]

        # Output length for trailing stop order
        output_ref = 180

        local_obj_plot_path = (project_local_path / f"{self.dtype}_plots").resolve()
        local_obj_plot_path.mkdir(exist_ok=True)
        local_plot_path_i = None

        dt_time = self.sr_data_rs.index[1] - self.sr_data_rs.index[0]
        decomp_idx = np.arange(0, decomplen + bclen)
        list_trade_ops = []
        list_trade_gain = []

        for i, idx_i in enumerate(tqdm(self.np_idx_inc_eval[:self.run_size], desc="Processing")):
            decomp_idx_i = decomp_idx + (idx_i - decomplen)

            # Input Signal (X), where Y := F(X)
            sr_data_rs_i = self.sr_data_rs.iloc[decomp_idx_i]
            sr_x_ft_i = self._filter(sr_data_rs_i)

            if self.plot_flag:
                local_plot_path_i = (local_obj_plot_path / f"trial_{i}").resolve()
                if not local_plot_path_i.is_dir():
                    local_plot_path_i.mkdir(exist_ok=True)
                else:
                    cleanup_directory(local_plot_path_i)

            # First Signal Decomposition
            pd_signal_decomp_i = self._signal_decomposition(sr_x_ft_i, nsignal)

            # Forecast the signal
            pd_forecast_i = self._forecast_signal(pd_signal_decomp_i[-(window + bclen):],
                                                  bclen,
                                                  plot_path=local_plot_path_i)

            # Prediction
            sr_y_prd_i = pd_forecast_i["mean"][bclen:]
            dy_prd_i = (sr_y_prd_i.values[-1] - sr_y_prd_i.values[0]) / sr_y_prd_i.size
            dy_perc_i = (dy_prd_i / sr_x_ft_i.iloc[-1]).item() * 100.
            list_trade_ops.append(dy_perc_i)

            # Reference
            list_output_idx_i = create_time_index(dt_time, pd_forecast_i.index[bclen:][0] - dt_time, output_ref)
            sr_y_i = self.sr_data_rs[list_output_idx_i]

            # trailing order parameter
            ## tolerance parameter (to enter into the trade)
            if np.abs(dy_perc_i) >= td_dsvars[0]:
                trail_value_i = td_dsvars[1] * dy_perc_i

                trailing_stop_order_i = TrailingStopOrder(
                    size=10.,
                    initial_price=sr_y_i.values[0].item(),
                    trail_value=trail_value_i,
                    trail_type='percentage',
                )

                for k, price_k in enumerate(sr_y_i.items()):
                    # Add full candle stick price
                    if trailing_stop_order_i.check_order_trigger(price_k[1]):
                        # print(f"Order triggered at market price: {price_k[1]} (iter: {k}), with stop price: {trailing_stop_order_i.current_stop_price}")
                        break
                    else:
                        trailing_stop_order_i.update_stop_price(price_k[1])
                        # print(f"Market price: {price_k[1]} (iter: {k}), Current stop price: {trailing_stop_order_i.current_stop_price:.2f}")

                list_trade_gain.append(trailing_stop_order_i.gain)

                if self.plot_flag:
                    # plot the trade operation
                    sr_y_signal_decomp_sum_i = pd_signal_decomp_i.iloc[bclen:, :].sum(axis=1)

            else:
                list_trade_gain.append(0.)

        fval = -np.asarray(list_trade_gain).sum()
        print(f"Trade gain: {fval}")

        return fval


def get_design_sample(seed: int = 42) -> np.ndarray:

    # Create a sample variables
    np_window = np.arange(100, 2000, step=100)
    np_decomplen = np.arange(500, 8000, step=100)
    np_bclen = np.arange(0, 10)
    np_nsignal = np.arange(4, 20)

    list_window = []
    list_decomplen = []
    list_bclen = []
    list_nsignal = []

    for win_i in np_window:
        for decomplen_i in np_decomplen:
            if decomplen_i >= win_i:
                for bclen_i in np_bclen:
                    for nsignal_i in np_nsignal:
                        list_window.append(win_i)
                        list_decomplen.append(decomplen_i)
                        list_bclen.append(bclen_i)
                        list_nsignal.append(nsignal_i)

    dict_dsvars = dict(window=list_window, decomplen=list_decomplen, bclen=list_bclen, nsignal=list_nsignal)
    np_dsvars_idx = np.arange(len(list_window))

    pd_dsvars = pd.DataFrame(dict_dsvars, index=np_dsvars_idx.copy())

    np.random.seed(seed)
    np.random.shuffle(np_dsvars_idx)

    return pd_dsvars.iloc[np_dsvars_idx, :]


def main_opt_forecast():

    input_data_path = (Path(__file__).parent.parent.parent / "tests" / "data" / "btcusd_2022-06-01.joblib").resolve()
    dict_price_data = load(input_data_path)
    # dt = 15
    # out_len = 12

    dt = 60
    out_len = 8

    seed = 42
    # dtype = "ewt"
    # dtype = "xwt"
    dtype = "swt"
    # dtype = "emd"
    # dtype = "ssa"

    run_size = 100
    # run_size = 200
    opt_ewt_forecst = OptForecast(input_data=dict_price_data[dt],
                                  out_len=out_len,
                                  inc_len=50,
                                  plt_len=120,
                                  seed=seed,
                                  dtype=dtype,
                                  ftype="",
                                  # plot_flag=True,
                                  run_size=run_size,
                                  )

    # Create a sample variables
    pd_dsvars = get_design_sample(seed=seed)

    # Convert timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(time.time())

    # Format the datetime object into a string suitable for a filename.
    timestamp_str = dt_object.strftime("%Y%m%d_%H%M%S")     # YYYYMMDD_HHMMSS is a good format for chronological sorting.
    opt_file_info = (project_local_path / f"opt_ifo_{dtype}_dt_{dt}_forlen_{out_len}_rsize_{run_size}_{timestamp_str}.csv").resolve()

    list_output = []
    for i, row_i in enumerate(pd_dsvars.iterrows()):
        # row_i[1]['window'] = 1600
        # row_i[1]['decomplen'] = 4600
        # row_i[1]['bclen'] = 3
        # row_i[1]['nsignal'] = 14

        res_i = opt_ewt_forecst.objective(row_i[1])

        input_i = row_i[1].to_dict()
        input_i["iter"] = i
        input_i["lsq"] = res_i["lsq"].sum().item()
        input_i["efficiency"] = (res_i["efficiency"] > 0.1).sum().item() / res_i["efficiency"].size
        input_i["performance"] = res_i["performance"]
        list_output.append(input_i)

        print(f" >>> Iter.: {i}, Processing DSVARS: {row_i[1].to_dict()}, LSQ Value: {input_i['lsq']}")
        if i % 5 == 0 and i > 0:
            df_opt_info = pd.DataFrame(list_output)
            df_opt_info.to_csv(opt_file_info, index=False)


def main_opt_trade():

    input_data_path = (Path(__file__).parent.parent.parent / "tests" / "data" / "btcusd_2022-06-01.joblib").resolve()
    dict_price_data = load(input_data_path)
    dt = 15
    # dt = 60

    out_len = 12
    # out_len = 8

    seed = 42
    dtype = "swt"
    # dtype = "emd"

    run_size = 50
    # run_size = 300
    opt_ewt_forecst = OptForecast(input_data=dict_price_data[dt],
                                  out_len=out_len,
                                  inc_len=50,
                                  plt_len=120,
                                  seed=seed,
                                  dtype=dtype,
                                  ftype="",
                                  # plot_flag=True,
                                  run_size=run_size,
                                  )

    sr_opt_forecast = pd.Series(dict(window=1600, decomplen=1900, bclen=3, nsignal=6))
    opt_ewt_forecst._prd_dsvars = sr_opt_forecast


    # [0]: tolerance parameter (to enter into the trade)
    # [1]: trailing stop value (percentage)
    np_opt_trade = np.array([0.05, 1.], dtype=float)
    # fval = opt_ewt_forecst.opt_trade(np_opt_trade)

    bounds = [(0.0001, 2.), (0.0001, 10.)]

    result = differential_evolution(
        func=opt_ewt_forecst.opt_trade,
        bounds=bounds,
        x0=np_opt_trade,
        strategy='best1bin',
        maxiter=100,    # Number of generations
        popsize=15,     # Population size
        tol=0.001,      # Tolerance for convergence
        disp=True,      # Display optimization progress
        seed=42         # Uncomment for reproducibility
    )

    print(f"Optimization result: {result}")
    test = 1.


if __name__ == "__main__":
    # main_opt_forecast()
    main_opt_trade()