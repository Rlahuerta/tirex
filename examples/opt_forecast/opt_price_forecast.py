
from pathlib import Path
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

    def objective(self, dsvars: np.ndarray) -> float:

        window = dsvars[0]
        bclen = dsvars[1]
        nsignal = dsvars[2]

        if self.convolution_filter is None:
            self.convolution_filter = ConvolutionFilter(adim=window, length=3)
        elif self.convolution_filter.adim != window:
            self.convolution_filter = ConvolutionFilter(adim=window, length=3)

        inp_idx = np.arange(0, window)
        out_idx = np.arange(0, self.out_len + bclen)

        inp_idx_bc = np.arange(0, window - bclen)
        out_idx_bc = np.arange(bclen, self.out_len + bclen)
        y_idx_bc = np.arange(0, self.out_len)

        list_y_ref = []
        list_y_prd = []
        list_lsq = []

        for i, idx_i in enumerate(self.np_idx_inc_eval):
            inp_idx_i = inp_idx + (idx_i - window)
            out_idx_i = out_idx + idx_i
            out_len_i = bclen + self.out_len

            np_x_i = self.np_data_rs[inp_idx_i]
            np_x_ft_i = self.convolution_filter(np_x_i)
            np_y_i = self.np_data_rs[out_idx_i][y_idx_bc]

            if self.plot_flag:
                full_idx_i = np.concatenate((inp_idx_i[-self.plt_len:], out_idx_i[y_idx_bc]))
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
                np_ewt_signal_x_k = ewt_comps_i[k, :]
                quantiles_y_k, mean_y_k = self.forecast_model.forecast(np_ewt_signal_x_k[inp_idx_bc],
                                                     prediction_length=out_len_i,
                                                     output_type="numpy",
                                                     )


                list_quantiles_i.append(quantiles_y_k[0][out_idx_bc])
                list_mean_i.append(mean_y_k[0][out_idx_bc])

                if self.plot_flag:
                    plot_fc(np_ewt_signal_x_k[-self.plt_len:], quantiles_y_k[0][out_idx_bc],
                            save_path=f'{local_plot_path_i}/ewt_signal_{k}_pred.png')

            ewt_quantiles_i = np.asarray(list_quantiles_i).sum(axis=0)
            ewt_mean_i = np.asarray(list_mean_i).sum(axis=0)

            list_y_ref.append(np_y_i)
            list_y_prd.append(ewt_mean_i)

            np_diff_i = (np_y_i - np_y_i[0]) - (ewt_mean_i - ewt_mean_i[0])
            list_lsq.append(np.dot(np_diff_i, np_diff_i))

            if self.plot_flag:
                plot_fc(np_x_i[-self.plt_len:], ewt_quantiles_i,
                        real_future_values=np_y_i,
                        full_timeseries=self.np_data_rs[full_idx_i],
                        title=f"End Time: {self.df_data.index[inp_idx_i][-1]}",
                        save_path=f'{local_plot_path_i}/ewt_signal_sum_pred.png')

        np_lsq = np.asarray(list_lsq)
        return np_lsq.sum().item()

    def _load_forecast_model(self):
        try:
            self.forecast_model: ForecastModel = load_model("NX-AI/TiRex")
        except Exception as e:
            print(f"Error loading model: {e}")
            return


def main():

    input_data_path = (Path(__file__).parent.parent.parent / "tests" / "data" / "btcusd_2022-06-01.joblib").resolve()
    dict_price_data = load(input_data_path)

    opt_ewt_forecst = OptEWTForecast(input_data=dict_price_data[15],
                                     out_len=12,
                                     plt_len=120,
                                     # plot_flag=True,
                                     )

    # Create a sample variables
    np_window = np.arange(100, 2000, step=100)
    np_bclen = np.arange(0, 10)
    np_nsignal = np.arange(4, 15)

    list_dsvars = []
    for win_i in np_window:
        for bclen_i in np_bclen:
            for nsignal_i in np_nsignal:
                list_dsvars.append((win_i, bclen_i, nsignal_i))

    np_dsvars = np.array(list_dsvars)
    np_dsvars_idx = np.arange(len(np_dsvars))

    np.random.seed(42)
    np.random.shuffle(np_dsvars_idx)

    dict_output = {"dsvars": [], "lsq": []}
    opt_file_info = (project_local_path / "opt_ifo.csv").resolve()

    for i, idx_i in enumerate(np_dsvars_idx):
        lsq_i = opt_ewt_forecst.objective(np_dsvars[idx_i])

        dict_output["dsvars"].append(np_dsvars[idx_i])
        dict_output["lsq"].append(lsq_i)

        print(f"Processing DSVARS: {np_dsvars[idx_i]}, LSQ Value: {lsq_i}")

        if i % 10 == 0:
            df_opt_info = pd.DataFrame(dict_output)
            df_opt_info.to_csv(opt_file_info, index=False)


if __name__ == "__main__":
    main()