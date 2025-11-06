# -*- coding: utf-8 -*-
import numpy as np
from enum import Enum

import pywt
import pandas as pd
import numpy.typing as npt
from typing import List


class FilterType(Enum):
    GAUSSIAN = 'gaussian'
    AVERAGE = 'average'
    CONVOLVE = 'convolve'
    WAVELET = 'wavelet'


class ConvolutionFilter:
    __name__ = 'ConvolutionFilter'

    def __init__(self, adim: int, window: int, penal: float = 1., ftype: str = "full"):
        """
        Initialize the ConvolutionFilter class.

        :param adim:        The dimension length of the input data.
        :param window:      The window length of the filter.
        :param penal:       The penalization factor for the filter weights (default=1.0).
                           Higher values create sharper weight distributions.
        :param ftype:       The type of filter to use:
                           - 'full': Symmetric/non-causal filter (looks both directions)
                           - 'forward': Causal filter (looks only forward, no past leakage)
                           - 'backward': Anti-causal filter (looks only backward, no future leakage)
        
        The weight function follows: w(d) = |1 - d/dist|^penal where d is the distance
        from the current position. At array boundaries, filters use only available data.
        """

        dist = window / 2
        np_arg_idx = np.arange(0, adim)

        self.length = int(2. * (window // 2) + 1)
        self.adim = adim

        self.list_search = []
        self.list_np_wg = []
        self.list_sum_wg = []

        for adim_i in range(self.adim):
            if ftype == "full":
                # symmetric filter
                np_dist_i = np.abs(np_arg_idx - adim_i)
                np_idx_i = np.where(np_dist_i <= dist)[0]
                np_dist_nrm_i = np_dist_i[np_idx_i]
                np_wg_i = np.abs((np_dist_nrm_i - dist) / dist) ** penal

            elif ftype == "forward":
                np_dist_i = np_arg_idx - adim_i
                np_idx_i = np.where((np_dist_i >= 0) & (np_dist_i <= dist))[0]
                np_dist_nrm_i = np_dist_i[np_idx_i] - adim_i
                np_wg_i = np.abs((np_dist_nrm_i - self.length) / self.length) ** penal

            elif ftype == "backward":
                np_idx_i = np.where(np_arg_idx <= adim_i)[0][-self.length:]
                np_dist_nrm_i = adim_i - np_arg_idx[np_idx_i]
                np_wg_i = np.abs((np_dist_nrm_i - self.length) / self.length) ** penal

            else:
                raise ValueError("Invalid filter type!")

            self.list_search.append(np_idx_i)
            self.list_np_wg.append(np_wg_i)
            self.list_sum_wg.append(np_wg_i.sum())

    def __call__(self, data: np.ndarray | pd.Series):
        """
        Apply the convolution filter to the input data.

        :param data: The input data as a NumPy array.

        :return: The filtered data as a NumPy array.
        """

        assert (self.adim == data.shape[0]), f'Array dim is wrong! {self.adim} != {data.shape[0]}'

        np_data_ou = np.zeros(self.adim, dtype=float)

        if isinstance(data, np.ndarray):
            np_data_in = data
        elif isinstance(data, pd.Series):
            np_data_in = data.values
        else:
            raise ValueError("Input data must be a NumPy array or a Pandas Series.")

        for adim_i in range(self.adim):
            np_data_i = np_data_in[self.list_search[adim_i]]
            np_data_ou[adim_i] = np.dot(np_data_i, self.list_np_wg[adim_i]) / self.list_sum_wg[adim_i]

        if isinstance(data, np.ndarray):
            return np_data_ou
        else:
            return pd.Series(np_data_ou, index=data.index, name="filtered")

    def ticker(self, data: pd.DataFrame, std: bool = False) -> (npt.NDArray, npt.NDArray):
        """
        Apply the convolution filter to the input data.

        :param data: The input data as a NumPy array.
        :param std: The input data as a NumPy array.

        :return: The filtered data as a NumPy array.
        """

        assert (self.adim == data.shape[0]), 'Array dim is wrong!'

        np_data_ou = np.zeros(self.adim, dtype=float)
        np_data_std = np.zeros(self.adim, dtype=float)

        for adim_i in range(self.adim):
            data_i = data.iloc[self.list_search[adim_i][0], :]
            np_data_i = data_i['close'].values
            np_data_ou[adim_i] = np.dot(np_data_i, self.list_np_wg[adim_i]) / self.list_sum_wg[adim_i]

            if std:
                np_data_std[adim_i] = data_i['close'].std()

        return np_data_ou, np_data_std


class WaveletFilter:
    """
    A class to perform wavelet-based filtering on time-series data.
    """
    def __init__(self, wavelet: str = 'db5', mode: str = "symmetric"):
        """
        Initialize the WaveletFilter class.

        :param wavelet:
            The wavelet type to use for filtering (default: 'db5').
        :param mode:
            The signal extension mode to use (default: 'antireflect').
        """

        self.wavelet = wavelet
        self.mode = mode

    def remove_white_noise(self, signal: npt.NDArray, level: int = None, threshold_method: str = 'soft') -> npt.NDArray:
        """
        Remove white noise from a time-series signal using wavelet thresholding.

        :param signal:
            A numpy array representing the input time-series signal.
        :param level:
            The maximum decomposition level to use (default: None, which uses the maximum level possible).
        :param threshold_method:
            The thresholding method to use (either 'soft' or 'hard'; default: 'soft').

        :return: A numpy array representing the denoised time-series signal.
        """

        # Perform the wavelet decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)

        # Estimate the noise level (using the first detail coefficients)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Determine the threshold value
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # Apply thresholding to the detail coefficients
        thresholded_coeffs = [coeffs[0]]  # Keep the approximation coefficients
        for detail_coeff in coeffs[1:]:
            if threshold_method == 'soft':
                thresholded_coeffs.append(pywt.threshold(detail_coeff, threshold, mode='soft'))
            else:
                thresholded_coeffs.append(pywt.threshold(detail_coeff, threshold, mode='hard'))

        # Reconstruct the denoised signal
        denoised_signal = pywt.waverec(thresholded_coeffs, self.wavelet)

        return denoised_signal

    def extract_white_noise(self, signal: npt.NDArray, level: int = None, threshold_method: str = 'soft') \
            -> npt.NDArray:
        """
        Extract the white noise component from a given signal using Discrete Wavelet Transform (DWT).

        This function decomposes the input signal using DWT, estimates the noise variance from the wavelet coefficients,
        applies a thresholding technique to remove small coefficients likely to represent noise, and reconstructs the
        signal from the threshold coefficients to obtain the white noise component.

        Parameters:
            signal (NDArray):    The input signal from which to extract the white noise component.
            level (int, optional):  The level of wavelet decomposition. Default is None, which means it will be
                                    determined automatically.
            threshold_method (str, optional):
                        The thresholding technique to apply to the wavelet coefficients. Default is 'soft'.

        Returns:
            ndarray: The extracted white noise component of the input signal.
        """

        # Decompose the signal using DWT
        coeffs = pywt.wavedec(signal, self.wavelet, mode='per', level=level)

        # Estimate noise variance from the detail coefficients at the finest scale
        detail_coeffs = coeffs[-1]
        noise_variance = np.median(np.abs(detail_coeffs)) / 0.6745

        # Apply thresholding to the wavelet coefficients
        threshold = noise_variance * np.sqrt(2 * np.log(len(signal)))

        thresholded_coeffs = [coeffs[0]]
        for i, (approx, detail) in enumerate(zip(coeffs[:-1], coeffs[1:])):
            if i != 0:
                wvlt_i = pywt.threshold(detail, threshold, mode=threshold_method)
            else:
                wvlt_i = approx

            thresholded_coeffs.append(wvlt_i)

        # Reconstruct the signal from the thresholded coefficients
        denoised_signal = pywt.waverec(thresholded_coeffs, self.wavelet, mode='per')

        # Extract the white noise component
        white_noise = signal - denoised_signal

        return white_noise

    def denoise(self, signal: npt.NDArray, level: int = 5, threshold_type: str = 'soft') -> npt.NDArray:
        """
        Denoise a time-series signal using wavelet thresholding.

        :param signal:
            A numpy array representing the input time-series signal.
        :param level:
            The maximum decomposition level to use (default: 5).
        :param threshold_type:
            The thresholding type to use (either 'soft' or 'hard'; default: 'soft').

        :return: A numpy array representing the denoised time-series signal.
        """

        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)

        # Estimate noise standard deviation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Determine the threshold value
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # Apply thresholding
        new_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Keep the approximation coefficients as they are
                new_coeffs.append(coeff)
            else:
                # Apply soft or hard thresholding to detail coefficients
                if threshold_type == 'soft':
                    new_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))
                elif threshold_type == 'hard':
                    new_coeffs.append(pywt.threshold(coeff, threshold, mode='hard'))

        # Perform wavelet reconstruction
        np_denoised_data = pywt.waverec(new_coeffs, self.wavelet, mode='symmetric')

        return np_denoised_data

    def long_term_trend(self, signal: npt.NDArray, level: int = 7) -> npt.NDArray:
        """
        Extract the long-term trend from a time-series signal using wavelet analysis.

        :param signal: A numpy array representing the input time-series signal.
        :param level: The maximum decomposition level to use (default: 8).

        :return: A numpy array representing the long-term trend of the time-series signal.
        """

        # Perform the wavelet decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)

        # Zero all detail coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = np.zeros_like(coeffs[i])

        # Reconstruct the long-term trend using only the approximation coefficients
        long_term_data = pywt.waverec(coeffs, self.wavelet)

        return long_term_data

    def decomposition(self, signal: npt.NDArray, level: int = 7) -> List[npt.NDArray]:
        """
        Decompose a time-series signal into different signals at each decomposition level using DWT.

        :param signal:
            A numpy array representing the input time-series signal.

        :param level:
            An integer specifying the number of decomposition levels (default: 7).

        :return:
            A list of numpy arrays representing the decomposed signals at each level.
        """

        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)

        # Reconstruct each level separately
        reconstructed_signals = []
        for i in range(level + 1):
            coeff_copy = [np.zeros_like(coeff) for coeff in coeffs]
            coeff_copy[i] = coeffs[i]
            reconstructed_signal = pywt.waverec(coeff_copy, self.wavelet)
            reconstructed_signals.append(reconstructed_signal)

        return reconstructed_signals


def ema(data: np.ndarray, n: int) -> np.ndarray:
    # Calculate the weighting multiplier
    multiplier = 2 / (n + 1)

    # Initialize the EMA array with the same length as data, first value of EMA is
    # the first value of data
    np_ema = np.zeros_like(data)
    np_ema[0] = data[0]

    # Calculate EMA for each point
    for i in range(1, len(data)):
        np_ema[i] = (data[i] * multiplier) + (np_ema[i - 1] * (1 - multiplier))

    return np_ema


def quadratic_fit_series(series: pd.Series | pd.DataFrame) -> pd.Series:
    x = np.arange(len(series))
    y = series.values
    coeffs = np.polyfit(x, y, 2)
    y_fit = np.polyval(coeffs, x)

    return pd.Series(y_fit, index=series.index)
