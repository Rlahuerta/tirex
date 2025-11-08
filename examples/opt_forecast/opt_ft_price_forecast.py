# -*- coding: utf-8 -*-
from pathlib import Path
import time
import datetime
# import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import load
from scipy.optimize import differential_evolution, LinearConstraint
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

from tirex import ForecastModel, load_model
from tirex.utils.filters import ConvolutionFilter, quadratic_fit_series
from tirex.utils.time import create_time_index
from tirex.utils.path import cleanup_directory
from tirex.utils.plot import plot_fc, plot_mpl_ticker
from tirex.utils.rescale import apply_minmax_inverse_scaler

# Add the project root to the Python path
project_local_path = Path(__file__).resolve().parent


class OptSignalFilterForecast:

    def __init__(self,
                 input_data: pd.DataFrame,
                 out_len: int,
                 plt_len: int = 120,
                 inc_len: int = 500,
                 run_size: int = 50,
                 ftype: str = 'backward',
                 plot_flag: bool = False,
                 seed: int = 42,
                 debug: bool = False,
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
            ftype (str):        Type of convolution filter to use.
            plot_flag (bool):   Whether to enable plotting of results.
            seed (int):         Random seed for reproducibility.
            debug (bool):       If True, use mock forecast model.
        """

        self.input_data = input_data
        self.out_len = out_len
        self.plt_len = plt_len
        self.inc_len = inc_len
        self.seed = seed
        self.run_size = run_size

        self.plot_flag = plot_flag
        self._plot_local_path = None

        self._iter = 0
        self._debug = debug
        self._ftype = ftype
        self._prd_dsvars = pd.Series()

        # Save csv file with optimization results

        ## Convert timestamp to a datetime object
        dt_object = datetime.datetime.fromtimestamp(time.time())
        timestamp_str = dt_object.strftime("%Y%m%d_%H%M%S")
        dt_time = input_data.index[1] - input_data.index[0]
        num_dt = int(dt_time.total_seconds() / 60)
        self.csv_file_info = (
                    project_local_path / f"opt_ft_dt_{num_dt}_forlen_{out_len}_rsize_{run_size}_{timestamp_str}.csv")

        self.loss_info = {"iter": [], "length": [], "window": [], "penal": [], "cost": [], "efficiency": [],
                          "performance": [], }

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
            if not self._debug:
                model_file_path = (Path(__file__).parent.parent.parent / "model" / "model.ckpt").resolve()
                self._model: ForecastModel = load_model(str(model_file_path))
                self._forecast = self._model.forecast
            else:
                self._forecast = self._mock_forecast_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    def _forecast_signal(
            self,
            signal: pd.Series,
            plot_path: Optional[Path] = None,
            **kwargs,
    ) -> pd.DataFrame:

        np_index = np.arange(0, signal.shape[0] + self.out_len)
        np_output_idx = np_index[signal.shape[0]:]

        dt_time = signal.index[1] - signal.index[0]
        list_output_datetime_idx = create_time_index(dt_time, signal.index[-1], np_output_idx.size)

        np_quantiles_y, np_mean_y = self._forecast(signal.values,
                                                   prediction_length=self.out_len,
                                                   output_type="numpy",
                                                   )

        pd_quantiles_y = pd.DataFrame(np_quantiles_y[0], index=list_output_datetime_idx)
        sr_mean_y = pd.Series(np_mean_y[0], index=list_output_datetime_idx, name="mean")

        sr_poly2_y = quadratic_fit_series(sr_mean_y)
        sr_poly2_y.name = "ft_poly2"

        if self.plot_flag and plot_path is not None:
            num_dt = int(dt_time.total_seconds() / 60)
            plt_kwargs = dict(save_path=f'{plot_path}/conv_filter_dt{num_dt}_pred.png')

            plot_fc(ctx=apply_minmax_inverse_scaler(self.scaler_data, signal.iloc[-self.plt_len:]),
                    quantile_fc=apply_minmax_inverse_scaler(self.scaler_data, pd_quantiles_y),
                    **plt_kwargs,
                    )

        return pd.concat([pd_quantiles_y, sr_mean_y, sr_poly2_y], axis=1)

    def _rescaling(self, pd_signal: pd.DataFrame) -> pd.DataFrame:
        # TODO: create a general rescaling function @rescale.py

        if pd_signal.shape[1] > 1:
            dict_signal = {}
            for key_i, signal_i in pd_signal.items():
                dict_signal[key_i] = self.scaler_data.transform(signal_i.values.reshape(-1, 1)).flatten()

            return pd.DataFrame(dict_signal, index=pd_signal.index)
        else:
            raise NotImplementedError(f"Unsupported rescaling with shape: {pd_signal.shape[1]}.")

    def _write_csv(self):
        if self._iter % 5 == 0 and self._iter > 0:
            df_opt_info = pd.DataFrame(self.loss_info)
            df_opt_info.to_csv(self.csv_file_info, index=False)

    def objective(self, dsvars: pd.Series) -> Dict[str, np.ndarray]:
        """
        Compute objective metrics for a set of design variables.

        Args:
            dsvars (pd.Series): Series containing 'window', 'length', 'penal'

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

        length = int(dsvars["length"])
        window = int(dsvars["window"])
        penal = float(dsvars["penal"])

        assert length > window, f"convolution filter length must be greater than window, length is {length}"
        assert window > 1, f"window should be greater than 100, window is {window}"
        assert penal >= 1., f"penalization should be greater or equal than 1., penal is {penal}"

        local_obj_plot_path = (project_local_path / f"conv_ft_plots").resolve()
        local_obj_plot_path.mkdir(exist_ok=True)
        local_plot_path_i = None

        list_lsq = []
        list_y_ref = []
        list_y_prd = []
        list_dy_ref = []
        list_dy_prd = []
        list_eff_pred = []
        ticker_keys = ['open', 'close', 'high', 'low']

        ft_idx = np.arange(0, length, dtype=int)
        conv_filter = ConvolutionFilter(adim=length, window=window, penal=penal, ftype=self._ftype)

        for i, idx_i in enumerate(tqdm(self.np_idx_inc_eval[:self.run_size], desc="Processing")):
            ft_idx_i = ft_idx + (idx_i - length)

            # Input Signal (X), where Y := F(X)
            sr_data_rs_i = self.sr_data_rs.iloc[ft_idx_i]
            sr_ft_x_i = conv_filter(sr_data_rs_i)

            if self.plot_flag:
                local_plot_path_i = (local_obj_plot_path / f"ft_trial_{i}").resolve()
                if not local_plot_path_i.is_dir():
                    local_plot_path_i.mkdir(exist_ok=True)
                else:
                    cleanup_directory(local_plot_path_i)

            # Forecast the signal
            pd_ft_forecast_i = self._forecast_signal(sr_ft_x_i, plot_path=local_plot_path_i)

            # Reference
            sr_y_i = self.sr_data_rs[pd_ft_forecast_i.index]
            dy_i = (sr_y_i.values[-1] - sr_ft_x_i.values[-1]) / (sr_y_i.size + 1)
            list_dy_ref.append(dy_i)

            # Prediction
            sr_y_prd_i = pd_ft_forecast_i["mean"]
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
                out_idx_i = ft_idx_i[-1] + np.arange(1, 31)

                # For verification only
                plot_mpl_ticker(input_tickers=self.input_data[ticker_keys].iloc[ft_idx_i, :][-self.plt_len:],
                                output_tickers=self.input_data[ticker_keys].iloc[out_idx_i, :],
                                ft=apply_minmax_inverse_scaler(self.scaler_data, sr_ft_x_i)[-self.plt_len:],
                                quantile=apply_minmax_inverse_scaler(self.scaler_data, pd_ft_forecast_i.loc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]),
                                Prediction=apply_minmax_inverse_scaler(self.scaler_data, pd_ft_forecast_i["mean"]),
                                plot_name=f'{local_plot_path_i}/ft_signal_pred.png',
                                dpi=400,
                                )

        np_lsq = np.asarray(list_lsq)
        np_eff_pred = np.asarray(list_eff_pred)

        np_dy_prd_idx = np.abs(np.asarray(list_dy_prd)) > 0.002
        np_eff_pred_cl = np_eff_pred[np_dy_prd_idx]
        neff_cl = (np_eff_pred_cl > 0.).sum() / np_eff_pred_cl.size

        return {"lsq": np_lsq,
                "efficiency": (np_eff_pred > 0.1).sum().item() / np_eff_pred.size,
                "performance": neff_cl.item(),
                "dy_ref": np.asarray(list_dy_ref),
                "dy_prd": np.asarray(list_dy_prd),
                "reference": list_dy_ref,
                "forecast": list_y_prd,
                }

    def objective_scipy(self, dsvars: np.ndarray) -> float:
        """
        Compute objective metrics for a set of design variables.

        Args:
            dsvars (pd.Series): Series containing 'window', 'length', 'penal'

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
        out_res = self.objective(pd.Series({'length': dsvars[0], 'window': dsvars[1], 'penal': dsvars[2]}))

        eqv_loss = out_res["lsq"].sum() / 10. - out_res["performance"]

        self.loss_info["iter"].append(self._iter)
        self.loss_info["length"].append(dsvars[0])
        self.loss_info["window"].append(dsvars[1])
        self.loss_info["penal"].append(dsvars[2])

        self.loss_info["cost"].append(eqv_loss.item())
        self.loss_info["efficiency"].append(out_res["efficiency"])
        self.loss_info["performance"].append(out_res["performance"])

        print(f" >>> Iter.: {self._iter}, Processing DSVARS: {dsvars}, LSQ Value: {eqv_loss}, Efficiency {out_res['efficiency']}, Performance: {out_res['performance']}")

        self._write_csv()
        self._iter += 1

        return eqv_loss


def get_design_ft_param_sample(seed: int = 42) -> np.ndarray:
    # Create a sample variables
    np_conv_ft_len = np.arange(100, 2000, step=100)
    np_conv_ft_win = np.arange(500, 8000, step=100)
    np_conv_ft_penal = np.linspace(1., 3., num=10)

    list_window = []
    list_length = []
    list_penal = []

    for win_i in np_conv_ft_win:
        for len_i in np_conv_ft_len:
            if len_i > win_i + 100:
                for penal_i in np_conv_ft_penal:
                    list_window.append(win_i)
                    list_length.append(len_i)
                    list_penal.append(penal_i)

    dict_dsvars = dict(length=list_length, window=list_window, penal=list_penal)
    np_dsvars_idx = np.arange(len(list_window))

    pd_dsvars = pd.DataFrame(dict_dsvars, index=np_dsvars_idx.copy())

    np.random.seed(seed)
    np.random.shuffle(np_dsvars_idx)

    return pd_dsvars.iloc[np_dsvars_idx, :]


def main_opt_forecast(opt: bool = True):

    input_data_path = (Path(__file__).parent.parent.parent / "tests" / "data" / "btcusd_2022-06-01.joblib").resolve()
    dict_price_data = load(input_data_path)
    seed = 100

    dt = 15
    # dt = 60

    # run_size = 30
    # run_size = 100
    # run_size = 200
    # run_size = 300
    run_size = 500

    if dt == 15:
        out_len = 12
        # len, win, penal
        sr_dsvars = pd.Series({'length': 1651, 'window': 44, 'penal': 1.})     # backward
        sr_dsvars = pd.Series({'length': 723, 'window': 27, 'penal': 1.})     # full
        bounds = [(100, 2000), (2, 50), (1, 3)]  # dt15

    elif dt == 60:
        out_len = 8
        # len, win, penal
        # sr_dsvars = pd.Series({'length': 100, 'window': 8, 'penal': 1.})     # random sample
        sr_dsvars = pd.Series({'length': 963, 'window': 44, 'penal': 1.})     # backward
        # sr_dsvars = pd.Series({'length': 156, 'window': 22, 'penal': 1.})     # full
        bounds = [(100, 1500), (2, 50), (1, 3)]  # dt60

    else:
        raise NotImplementedError

    opt_ft_forecst = OptSignalFilterForecast(input_data=dict_price_data[dt],
                                             out_len=out_len,
                                             inc_len=50,
                                             plt_len=120,
                                             seed=seed,
                                             ftype="full",
                                             plot_flag=not opt,
                                             run_size=run_size,
                                             )

    # res = opt_ft_forecst.objective(sr_dsvars)
    res = opt_ft_forecst.objective_scipy(sr_dsvars.values)

    if opt:
        integrality = [True, True, True]

        # Define the linear constraint: x1 - x2 >= 100
        A = [[1, -1, 0]]  # Coefficients for x1 and x2
        b = [100]  # Lower bound for the inequality

        # Create the linear constraint
        constraint = LinearConstraint(A, lb=b, ub=np.inf)

        # Run the differential evolution solver
        result = differential_evolution(
            opt_ft_forecst.objective_scipy,
            bounds,
            x0=sr_dsvars.values,
            constraints=constraint,
            integrality=integrality,
            strategy='best1bin',        # A robust and commonly used strategy
            maxiter=1000,               # Maximum number of iterations
            popsize=20,                 # Population size (5-10 times the number of dimensions)
            recombination=0.7,          # Recombination probability
            init='latinhypercube',      # Initialization method for better space coverage
            seed=seed,
            disp=True,
        )

        print("Optimal solution:", result.x)
        print("Optimal value:", result.fun)


if __name__ == "__main__":
    main_opt_forecast(opt=False)
    # main_opt_forecast(opt=True)