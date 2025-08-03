# -*- coding: utf-8 -*-
import numpy as np
from enum import Enum

import pandas as pd
import numpy.typing as npt
from typing import List


class FilterType(Enum):
    GAUSSIAN = 'gaussian'
    AVERAGE = 'average'
    CONVOLVE = 'convolve'


class ConvolutionFilter:
    __name__ = 'ConvolutionFilter'

    def __init__(self, adim: int, length: int, penal: float = 1., ftype: str = "full"):
        """
        Initialize the ConvolutionFilter class.

        :param adim:        The dimension of the input data.
        :param length:      The length of the filter.
        :param penal:       The penalization factor for the filter weights (default=1.0).
        """

        dist = length / 2
        np_arg_idx = np.arange(0, adim)

        self.length = int(2. * (length // 2) + 1)
        self.adim = adim

        self.list_search = []
        self.list_np_wg = []
        self.list_sum_wg = []

        for adim_i in range(self.adim):
            if ftype == "full":
                np_dist_i = np.abs(np_arg_idx - adim_i)
                np_idx_i = np.where(np_dist_i <= dist)[0]
                np_dist_nrm_i = np_dist_i[np_idx_i]
                np_wg_i = np.abs((np_dist_nrm_i - dist) / dist) ** penal

            elif ftype == "forward":
                # FIXME
                np_dist_i = np_arg_idx - adim_i
                np_idx_i = np.where(np_dist_i <= dist)[0]
                np_dist_nrm_i = np_dist_i[np_idx_i]
                np_wg_i = np.abs((np_dist_nrm_i - dist) / dist) ** penal

            elif ftype == "backward":
                np_idx_i = np.where(np_arg_idx <= adim_i)[0][:self.length]
                np_dist_nrm_i = adim_i - np_arg_idx[np_idx_i]
                np_wg_i = np.abs((np_dist_nrm_i - length) / length)

            else:
                raise ValueError("Invalid filter type!")

            self.list_search.append(np_idx_i)
            self.list_np_wg.append(np_wg_i)
            self.list_sum_wg.append(np_wg_i.sum())

    def __call__(self, data: npt.NDArray) -> npt.NDArray:
        """
        Apply the convolution filter to the input data.

        :param data: The input data as a NumPy array.

        :return: The filtered data as a NumPy array.
        """

        assert (self.adim == data.shape[0]), 'Array dim is wrong!'

        np_data_ou = np.zeros(self.adim, dtype=float)

        for adim_i in range(self.adim):
            np_data_i = data[self.list_search[adim_i]]
            np_data_ou[adim_i] = np.dot(np_data_i, self.list_np_wg[adim_i]) / self.list_sum_wg[adim_i]

        return np_data_ou

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
