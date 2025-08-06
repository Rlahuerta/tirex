
from pathlib import Path
import time
import datetime
import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import milp

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
                 **kwargs,
                 ):

        self.input_data = input_data
        self.out_len = out_len
        self.plt_len = plt_len
        self.inc_len = inc_len
        self.plot_flag = plot_flag

        self.ewt = EmpiricalWaveletTransform()
        self.forecast_model = None

        # Optimization Variables
        self.dsvars = ['window', 'bclen', 'nsignal']
        self.dsvars_ini = np.array([600, 2, 6], dtype=int)
        self.dsvars_bounds = [(80, 2000), (0, 10), (4, 15)]

        self.df_data = input_data["close"]
        self.np_data_idx = np.arange(len(self.df_data))
        self.scaler_data = MinMaxScaler(feature_range=(0., 100.))
        self.np_data_rs = self.scaler_data.fit_transform(self.df_data.values.reshape(-1, 1)).flatten()
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

    def objective(self, dsvars: pd.Series) -> float:

        window = dsvars["window"]
        decomplen = dsvars["decomplen"]
        bclen = dsvars["bclen"]
        nsignal = dsvars["nsignal"]

        if self.convolution_filter is None:
            self.convolution_filter = ConvolutionFilter(adim=decomplen, length=5)
        elif self.convolution_filter.adim != window:
            self.convolution_filter = ConvolutionFilter(adim=decomplen, length=5)

        decomp_idx = np.arange(0, decomplen)
        out_idx = np.arange(0, self.out_len + bclen)

        inp_idx_bc = decomp_idx[-window:] - bclen
        out_idx_bc = np.arange(bclen, self.out_len + bclen)
        y_idx_bc = np.arange(0, self.out_len)

        list_y_ref = []
        list_y_prd = []
        list_lsq = []

        for i, idx_i in enumerate(self.np_idx_inc_eval):
            decomp_idx_i = decomp_idx + (idx_i - decomplen)
            out_idx_i = out_idx + idx_i
            out_len_i = bclen + self.out_len

            np_x_i = self.np_data_rs[decomp_idx_i]
            np_x_ft_i = self.convolution_filter(np_x_i)
            np_y_i = self.np_data_rs[out_idx_i][y_idx_bc]

            if self.plot_flag:
                full_idx_i = np.concatenate((decomp_idx_i[-self.plt_len:], out_idx_i[y_idx_bc]))
                local_plot_path_i = (local_plot_path / f"trial_{i}").resolve()

                if not local_plot_path_i.is_dir():
                    local_plot_path_i.mkdir(exist_ok=True)
                else:
                    cleanup_directory(local_plot_path_i)

            # First EWT Decomposition
            ewt_res_i, np_mwvlt_i, np_bcs_i = self.ewt(np_x_ft_i, nsignal)
            ewt_comps_i = ewt_res_i.T

            list_quantiles_i = []
            list_mean_i = []

            select_ewt_i = np.arange(0, ewt_comps_i.shape[0])
            for k in select_ewt_i:
                np_ewt_signal_x_k = ewt_comps_i[k, inp_idx_bc]
                quantiles_y_k, mean_y_k = self.forecast_model.forecast(np_ewt_signal_x_k,
                                                     prediction_length=out_len_i,
                                                     output_type="numpy",
                                                     )

                list_quantiles_i.append(quantiles_y_k[0])
                list_mean_i.append(mean_y_k[0])

                if self.plot_flag:
                    plot_fc(np_ewt_signal_x_k[-self.plt_len:], quantiles_y_k[0][out_idx_bc],
                            bcs=mean_y_k[0][:bclen],
                            save_path=f'{local_plot_path_i}/ewt_signal_{k}_pred.png')

            np_ewt_quantiles_i = np.asarray(list_quantiles_i).sum(axis=0)
            np_ewt_mean_i = np.asarray(list_mean_i).sum(axis=0)

            list_y_ref.append(np_y_i)
            list_y_prd.append(np_ewt_mean_i[out_idx_bc])

            np_diff_i = (np_y_i - np_y_i[0]) - (np_ewt_mean_i[out_idx_bc] - np_ewt_mean_i[out_idx_bc][0])
            list_lsq.append(np.dot(np_diff_i, np_diff_i))

            if self.plot_flag:
                plot_fc(np_x_i[-self.plt_len:], np_ewt_quantiles_i[out_idx_bc, :],
                        bcs=np_ewt_mean_i[:bclen],
                        real_future_values=np_y_i,
                        full_timeseries=self.np_data_rs[full_idx_i],
                        title=f"End Time: {self.df_data.index[decomp_idx_i][-1]}",
                        save_path=f'{local_plot_path_i}/ewt_signal_sum_pred.png')

        np_lsq = np.asarray(list_lsq)
        return np_lsq.sum().item()

    def _load_forecast_model(self):
        try:
            self.forecast_model: ForecastModel = load_model("NX-AI/TiRex")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

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
    dt = 15

    opt_ewt_forecst = OptEWTForecast(input_data=dict_price_data[dt],
                                     out_len=12,
                                     plt_len=120,
                                     # plot_flag=True,
                                     )

    # Create a sample variables
    pd_dsvars = get_design_sample()

    # Convert timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(time.time())

    # Format the datetime object into a string suitable for a filename.
    timestamp_str = dt_object.strftime("%Y%m%d_%H%M%S")     # YYYYMMDD_HHMMSS is a good format for chronological sorting.
    opt_file_info = (project_local_path / f"opt_ifo_dt{dt}_{timestamp_str}.csv").resolve()

    list_output = []
    for i, row_i in enumerate(pd_dsvars.iterrows()):
        lsq_i = opt_ewt_forecst.objective(row_i[1])

        input_i = row_i[1].to_dict()
        input_i["lsq"] = lsq_i
        list_output.append(input_i)

        print(f" >>> Iter.: {i}, Processing DSVARS: {row_i[1]}, LSQ Value: {lsq_i}")
        if i % 10 == 0 and i > 0:
            df_opt_info = pd.DataFrame(list_output)
            df_opt_info.to_csv(opt_file_info, index=False)


if __name__ == "__main__":
    main()