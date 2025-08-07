
from pathlib import Path
import time
import datetime
import warnings
import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, Optional, Union, Callable

from tirex import ForecastModel, load_model
from tirex.utils.filters import ConvolutionFilter
from tirex.utils.ewt import EmpiricalWaveletTransform
from tirex.utils.plot import plot_fc, emd_plot

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


class OptEWTForecast:

    def __init__(self,
                 input_data: pd.DataFrame,
                 out_len: int,
                 plt_len: int = 120,
                 inc_len: int = 500,
                 plot_flag: bool = False,
                 debug: bool = False,
                 **kwargs,
                 ):

        self.input_data = input_data
        self.out_len = out_len
        self.plt_len = plt_len
        self.inc_len = inc_len
        self.plot_flag = plot_flag
        self._plot_local_path = None
        self._debug = debug

        self.ewt = EmpiricalWaveletTransform()
        self._model = None
        self._forecast = None

        # Optimization Variables
        self.dsvars = ['window', 'bclen', 'nsignal']
        self.dsvars_ini = np.array([600, 2, 6], dtype=int)
        self.dsvars_bounds = [(80, 2000), (0, 10), (4, 15)]

        self.df_data = input_data["close"]
        self.scaler_data = MinMaxScaler(feature_range=(0., 100.))

        self.np_data_idx = np.arange(len(self.df_data))
        self.sr_data_rs = None

        self.convolution_filter = None

        self.np_idx_inc_eval = np.array([])

        self._preprocess_data()
        self._load_forecast_model()

    def _preprocess_data(self):
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

        np_data_rs = self.scaler_data.fit_transform(self.df_data.values.reshape(-1, 1)).flatten()
        self.sr_data_rs = pd.Series(np_data_rs, index=self.input_data.index)

    @staticmethod
    def _mock_forecast_model(input_x: np.ndarray,
                             prediction_length: int = 10,
                             **kwargs,
                             ) -> np.ndarray:

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
        try:
            if not self._debug:
                self._model: ForecastModel = load_model("NX-AI/TiRex")
                self._forecast = self._model.forecast
            else:
                self._forecast = self._mock_forecast_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    def _signal_decomposition(self,
                              input_signal: pd.Series,
                              nsignal: int,
                              ) -> pd.DataFrame:
        # TODO: Add option to use ceemdan as signal decomp method
        assert nsignal > 1, "nsignal should be greater than 1"

        # First EWT Decomposition
        np_ewt_res, np_mwvlt, np_bcs = self.ewt(input_signal.values, nsignal)
        pd_ewt_res = pd.DataFrame(np_ewt_res[-input_signal.size:], index=input_signal.index)

        return pd_ewt_res

    def _forecast_signal(self,
                         decomp_signals: pd.DataFrame,
                         bclen: int,
                         plot_path: Optional[Path] = None,
                         ) -> pd.DataFrame:

        np_index = np.arange(0, decomp_signals.shape[0] + self.out_len)
        np_input_idx = np_index[:decomp_signals.shape[0] - bclen]
        np_output_idx = np_index[decomp_signals.shape[0] - bclen:]
        dt_time = decomp_signals.index[1] - decomp_signals.index[0]

        list_output_datetime_idx = []
        dt_current = decomp_signals.index[np_input_idx][-1]
        for i in range(np_output_idx.size):
            dt_current += dt_time
            list_output_datetime_idx.append(dt_current)

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
                plt_kwargs = dict(save_path=f'{plot_path}/ewt_signal_{k}_pred.png')
                if bclen > 0:
                    plt_kwargs["bcs"] = pd_mean_y_k.iloc[:bclen]

                plot_fc(sr_signal_val_k.iloc[-self.plt_len:],
                        pd_quantiles_y_k.iloc[bclen:, :], **plt_kwargs)

        pd_quantiles = sum(list_quantiles)
        sr_mean = sum(list_mean)
        sr_mean.name = "mean"

        return pd.concat([pd_quantiles, sr_mean], axis=1)

    def objective(self, dsvars: pd.Series) -> float:

        window = dsvars["window"]
        decomplen = dsvars["decomplen"]
        bclen = dsvars["bclen"]
        nsignal = dsvars["nsignal"]

        assert window > 50, f"window should be greater than 50, window is {window}"
        assert decomplen >= window, f"decomplen should be greater or equal than window, decomplen is {decomplen}"
        assert bclen >= 0, f"bclen should be greater or equal than 0, bclen is {bclen}"
        assert nsignal > 3, f"nsignal should be greater than 3, nsignal is {nsignal}"

        if self.convolution_filter is None:
            self.convolution_filter = ConvolutionFilter(adim=decomplen + bclen, length=5)
        elif self.convolution_filter.adim != decomplen:
            self.convolution_filter = ConvolutionFilter(adim=decomplen + bclen, length=5)

        local_plot_path_i = Path()
        decomp_idx = np.arange(0, decomplen + bclen)

        list_y_ref = []
        list_y_prd = []
        list_lsq = []

        for i, idx_i in enumerate(self.np_idx_inc_eval):
            decomp_idx_i = decomp_idx + (idx_i - decomplen)
            sr_data_rs_i = self.sr_data_rs.iloc[decomp_idx_i]

            if self.convolution_filter.adim == sr_data_rs_i.size:
                sr_x_ft_i = pd.Series(self.convolution_filter(sr_data_rs_i.values), index=sr_data_rs_i.index)
            else:
                warnings.warn(f"Problem with convolution filter, dimension error: {self.convolution_filter.adim} != {sr_data_rs_i.size}")
                sr_x_ft_i = sr_data_rs_i.copy()

            if self.plot_flag:
                local_plot_path_i = (local_plot_path / f"trial_{i}").resolve()
                if not local_plot_path_i.is_dir():
                    local_plot_path_i.mkdir(exist_ok=True)
                else:
                    cleanup_directory(local_plot_path_i)

            # First EWT Decomposition
            pd_ewt_comps_i = self._signal_decomposition(sr_x_ft_i, nsignal)
            pd_forecast_i = self._forecast_signal(pd_ewt_comps_i, bclen, local_plot_path_i)

            sr_y_i = self.sr_data_rs[pd_forecast_i.index[bclen:]]
            sr_y_prd_i = pd_forecast_i["mean"][bclen:]

            list_y_ref.append(sr_y_i)
            list_y_prd.append(sr_y_prd_i)

            np_diff_i = (sr_y_i.values - sr_y_i.values[0]) - (sr_y_prd_i.values - sr_y_prd_i.values[0])
            list_lsq.append(np.dot(np_diff_i, np_diff_i))

            if self.plot_flag:
                sr_y_ewt_sum_i = pd_ewt_comps_i.iloc[bclen:, :-1].sum(axis=1)

                plot_fc(sr_x_ft_i[-self.plt_len:], pd_forecast_i.iloc[bclen:, :-1],
                        bcs=pd_forecast_i["mean"][:bclen],
                        real_future_values=sr_y_i,
                        decomp_sum=sr_y_ewt_sum_i,
                        title=f"End Time: {sr_x_ft_i[-1]}",
                        save_path=f'{local_plot_path_i}/ewt_signal_sum_pred.png')

        np_lsq = np.asarray(list_lsq)
        return np_lsq.sum().item()


def get_design_sample(seed: int = 42) -> np.ndarray:

    # Create a sample variables
    np_window = np.arange(100, 2000, step=100)
    np_decomplen = np.arange(100, 3000, step=100)
    np_bclen = np.arange(0, 10)
    np_nsignal = np.arange(4, 15)

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

def main():

    input_data_path = (Path(__file__).parent.parent.parent / "tests" / "data" / "btcusd_2022-06-01.joblib").resolve()
    dict_price_data = load(input_data_path)
    # dt = 15
    # out_len = 12

    dt = 60
    out_len = 8

    opt_ewt_forecst = OptEWTForecast(input_data=dict_price_data[dt],
                                     out_len=out_len,
                                     plt_len=120,
                                     plot_flag=True,
                                     # debug=True,
                                     )

    # Create a sample variables
    pd_dsvars = get_design_sample()

    # Convert timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(time.time())

    # Format the datetime object into a string suitable for a filename.
    timestamp_str = dt_object.strftime("%Y%m%d_%H%M%S")     # YYYYMMDD_HHMMSS is a good format for chronological sorting.
    opt_file_info = (project_local_path / f"opt_ifo_dt_{dt}_forlen_{out_len}_{timestamp_str}.csv").resolve()

    list_output = []
    for i, row_i in enumerate(pd_dsvars.iterrows()):
        lsq_i = opt_ewt_forecst.objective(row_i[1])

        input_i = row_i[1].to_dict()
        input_i["lsq"] = lsq_i
        list_output.append(input_i)

        print(f" >>> Iter.: {i}, Processing DSVARS: {row_i[1].to_dict()}, LSQ Value: {lsq_i}")
        if i % 10 == 0 and i > 0:
            df_opt_info = pd.DataFrame(list_output)
            df_opt_info.to_csv(opt_file_info, index=False)


if __name__ == "__main__":
    main()