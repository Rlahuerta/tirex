import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from tirex.utils.ceemdan import filt6, pade6


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


def emd_plot(inp_time: np.ndarray,
             inp_signal: np.ndarray,
             emd_signal: np.ndarray,
             plot_title: str,
             plot_name: str,
             res_signal: Union[np.ndarray, Dict[str, np.ndarray], None] = None,
             ):
    """Helper function to plot EMD/CEEMDAN results."""

    num_plots = emd_signal.shape[0] + 1
    ini_idx = 1

    if res_signal is not None:
        num_plots += 1
        ini_idx = 2

    fig, ax = plt.subplots(num_plots, 1, figsize=(15, 12))
    fig.suptitle(plot_title, fontsize=12)

    ax[0].plot(inp_time, inp_signal, 'r')
    ax[0].plot(inp_time, emd_signal.sum(axis=0), 'blue', dashes=[2, 4])

    ax[0].set_ylabel("Original signal")
    ax[0].grid(which='minor', alpha=0.2)
    ax[0].grid(which='major', alpha=0.5)

    if res_signal is not None:
        ax[1].set_ylabel("Residuum")
        ax[1].grid(which='minor', alpha=0.2)
        ax[1].grid(which='major', alpha=0.5)

        if isinstance(res_signal, np.ndarray):
            ax[1].plot(inp_time, res_signal, 'r')

        elif isinstance(res_signal, Dict):
            for i, (key_i, val_i) in enumerate(res_signal.items()):
                if i == 0:
                    ax[1].plot(inp_time, val_i, 'r', label=key_i)
                else:
                    ax[1].plot(inp_time, val_i, '.', label=key_i)

            ax[1].legend(loc='best')

    for i, n in enumerate(range(ini_idx, num_plots)):
        ax[n].plot(inp_time, emd_signal[i, :], 'g')
        ax[n].set_ylabel("eIMF %i" % (i + 1))
        ax[n].locator_params(axis='y', nbins=5)
        ax[n].grid(which='minor', alpha=0.2)
        ax[n].grid(which='major', alpha=0.5)

    ax[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(f'{plot_name}', dpi=200)


class EMDVisualisation:
    """Simple visualisation helper.

    This class is for quick and simple result visualisation.
    """

    PLOT_WIDTH = 12
    PLOT_HEIGHT_PER_IMF = 3

    def __init__(self,
                 emd_instance=None,
                 ):

        self.emd_instance = emd_instance

        self.imfs = None
        self.residue = None

        if emd_instance is not None:
            self.imfs, self.residue = self.emd_instance.get_imfs_and_residue()

    def _check_imfs(self, imfs, residue, include_residue):
        """Checks for passed imfs and residue."""
        imfs = imfs if imfs is not None else self.imfs
        residue = residue if residue is not None else self.residue

        if imfs is None:
            raise AttributeError("No imfs passed to plot")

        if include_residue and residue is None:
            raise AttributeError("Requested to plot residue but no residue provided")

        return imfs, residue

    def plot_imfs(self,
                  imfs=None,
                  residue=None,
                  t: np.ndarray = None,
                  include_residue: bool = True,
                  plot_name: str = None,
                  ):
        """Plots and shows all IMFs.

        All parameters are optional since the `emd` object could have been passed when instantiating this object.

        The residual is an optional and can be excluded by setting `include_residue=False`.
        """
        imfs, residue = self._check_imfs(imfs, residue, include_residue)

        num_rows, t_length = imfs.shape
        num_rows += include_residue is True

        t = t if t is not None else range(t_length)

        fig, axes = plt.subplots(num_rows, 1, figsize=(self.PLOT_WIDTH, num_rows * self.PLOT_HEIGHT_PER_IMF))

        if num_rows == 1:
            axes = list(axes)

        axes[0].set_title("Time series")

        for num, imf in enumerate(imfs):
            ax = axes[num]
            ax.plot(t, imf)
            ax.set_ylabel("IMF " + str(num + 1))

        if include_residue:
            ax = axes[-1]
            ax.plot(t, residue)
            ax.set_ylabel("Res")

        # Making the layout a bit more pleasant to the eye
        plt.tight_layout()

        if plot_name is not None:
            fig.savefig(plot_name, dpi=120)


    def plot_instant_freq(self,
                          t: np.ndarray,
                          imfs=None,
                          order: bool = False,
                          alpha=None,
                          plot_name: str = None,
                          ):
        """Plots and shows instantaneous frequencies for all provided imfs.

        The necessary parameter is `t` which is the time array used to compute the EMD.
        One should pass `imfs` if no `emd` instances is passed when creating the Visualisation object.

        Parameters
        ----------
        t:
        imfs:

        order : bool (default: False)
            Represents whether the finite difference scheme is
            low-order (1st order forward scheme) or high-order (6th order
            compact scheme). The default value is False (low-order)

        alpha : float (default: None)
            Filter intensity. Default value is None, which
            is equivalent to `alpha` = 0.5, meaning that no filter is applied.
            The `alpha` values must be in between -0.5 (fully active) and 0.5
            (no filter).

        plot_name: str
        """
        if alpha is not None:
            assert -0.5 < alpha < 0.5, "`alpha` must be in between -0.5 and 0.5"

        imfs, _ = self._check_imfs(imfs, None, False)
        num_rows = imfs.shape[0]

        imfs_inst_freqs = self._calc_inst_freq(imfs, t, order=order, alpha=alpha)

        fig, axes = plt.subplots(num_rows, 1, figsize=(self.PLOT_WIDTH, num_rows * self.PLOT_HEIGHT_PER_IMF))

        if num_rows == 1:
            axes = fig.axes

        axes[0].set_title("Instantaneous frequency")

        for num, imf_inst_freq in enumerate(imfs_inst_freqs):
            ax = axes[num]
            ax.plot(t, imf_inst_freq)
            ax.set_ylabel("IMF {} [Hz]".format(num + 1))

        # Making the layout a bit more pleasant to the eye
        plt.tight_layout()

        if plot_name is not None:
            fig.savefig(plot_name, dpi=120)

    @staticmethod
    def _calc_inst_phase(sig, alpha):
        """Extract analytical signal through the Hilbert Transform."""
        # Apply Hilbert transform to each row
        analytic_signal = hilbert(sig)

        if alpha is not None:
            assert -0.5 < alpha < 0.5, "`alpha` must be in between -0.5 and 0.5"
            real_part = np.array([filt6(row.real, alpha) for row in analytic_signal])
            imag_part = np.array([filt6(row.imag, alpha) for row in analytic_signal])
            analytic_signal = real_part + 1j * imag_part

        phase = np.unwrap(np.angle(analytic_signal))  # Compute angle between img and real

        if alpha is not None:
            phase = np.array([filt6(row, alpha) for row in phase])  # Filter phase

        return phase

    def _calc_inst_freq(self, sig, t, order, alpha):
        """Extracts instantaneous frequency through the Hilbert Transform."""
        inst_phase = self._calc_inst_phase(sig, alpha=alpha)

        if order is False:
            inst_freqs = np.diff(inst_phase) / (2 * np.pi * (t[1] - t[0]))
            inst_freqs = np.concatenate((inst_freqs, inst_freqs[:, -1].reshape(inst_freqs[:, -1].shape[0], 1)), axis=1)
        else:
            inst_freqs = [pade6(row, t[1] - t[0]) / (2.0 * np.pi) for row in inst_phase]
        if alpha is None:
            return np.array(inst_freqs)
        else:
            return np.array([filt6(row, alpha) for row in inst_freqs])  # Filter freqs

    def show(self):
        plt.show()


def decomp_plot(x: np.ndarray, y_original: np.ndarray, decomp_signal: [np.ndarray], file_path: str):

    fig_size = (40, 20)

    ndecomp = len(decomp_signal)

    fig, ax = plt.subplots((ndecomp + 1), 1, figsize=fig_size, sharex=True, dpi=180)
    fig.suptitle(f'nvars ({x.shape[0]}), signal ({len(decomp_signal)}), EWT-Decomp', fontsize=12)

    ax[0].plot(x, y_original, label='Signal')

    ax[0].grid(which='minor', alpha=0.2)
    ax[0].grid(which='major', alpha=0.5)
    ax[0].legend()

    for i, signal_i in enumerate(decomp_signal):
        idx = i + 1
        ax[idx].plot(x, signal_i, label=f'D{i}')

        ax[idx].grid(which='minor', alpha=0.2)
        ax[idx].grid(which='major', alpha=0.5)
        ax[idx].legend()

    fig.savefig(file_path)
    plt.close()


def save_plot(x: np.ndarray, y_original: np.ndarray, y_filtered: np.ndarray, title: str, file_path: str):

    fig_size = (40, 20)
    fsize = 20

    fig, ax = plt.subplots(figsize=fig_size, dpi=140)
    fig.suptitle(title, fontsize=fsize)

    ax.plot(x, y_original, label="Original Signal", alpha=0.7)
    ax.plot(x, y_filtered, label="Filtered Signal", linewidth=2)
    ax.set_xlabel("Time", fontsize=fsize)
    ax.set_ylabel("Amplitude", fontsize=fsize)
    ax.legend()

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    fig.savefig(file_path)
    plt.close()
