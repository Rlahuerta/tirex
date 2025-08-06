# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import numpy.typing as npt

from typing import Tuple, List, Dict, Any, Optional
from scipy.fft import rfft, ifft, fft
from scipy.fftpack import fftshift, ifftshift
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution, Bounds

from tirex.utils.filters import FilterType, WaveletFilter


def meyer_beta_function(x):
    """
        function used in the construction of Meyer's wavelet
    """

    if isinstance(x, float):
        if x < 0.:
            bm = 0.

        elif x > 1.:
            bm = 1.

        else:
            bm = (x**4) * (35. - 84. * x + 70. * (x**2) - 20. * (x**3))

        return bm

    elif isinstance(x, np.ndarray):
        np_upp_idx = x > 1.
        np_idx = (x >= 0.) * (x <= 1.)
        np_x = x[np_idx]

        np_bm = np.zeros_like(x)
        np_bm[np_idx] = (np_x**4) * (35. - 84. * np_x + 70. * (np_x**2) - 20. * (np_x**3))

        if np_upp_idx.sum() >= 1:
            np_bm[np_upp_idx] = 1.

        return np_bm

    else:
        raise NotImplementedError


def meyer_wavelet(gamma: float, nbcs: int, lb: float, ub: float) -> npt.NDArray:
    """
    Constructs the Fourier transform of the Meyer wavelet on the band [lb, ub].

    Input parameters:

        gamma:                  Transition ratio, controlling the width of the transition band.
        nbcs:                   Number of points in the frequency domain (size of the FFT).
        lb:                     lower boundary of the frequency band.
        ub:                     upper boundary of the frequency band.

    Output:
        ymw:                     Fourier transform of the wavelet on the band [lb, ub]
    """

    # Check if lb and ub are equal or lb >= ub
    if lb >= ub:
        # Return zeros
        return np.zeros(nbcs)

    mi = nbcs // 2
    np_ymw = np.zeros(nbcs)

    # Create a frequency vector np_lin from 0 to 2π
    np_lin = np.linspace(0, 2 * np.pi - 2 * np.pi / nbcs, num=nbcs)

    # Shift the zero-frequency component to the center using fftshift.
    np_lin_shift = fftshift(np_lin)
    np_lin_shift[0:mi] = -2 * np.pi + np_lin_shift[0:mi]
    np_aw = np.abs(np_lin_shift)

    # Add epsilon to prevent division by zero
    epsilon = 1.e-8
    gamma = max(gamma, epsilon)  # Ensure gamma is not zero
    adjusted_lb = max(lb, epsilon)
    adjusted_ub = max(ub, epsilon)

    # Check for lb >= ub
    if adjusted_lb >= adjusted_ub:
        # Return zeros if the frequency band is invalid
        return np.zeros(nbcs)

    a_lb = 1. / (2. * gamma * adjusted_lb)
    a_ub = 1. / (2. * gamma * adjusted_ub)

    pb_lb = (1. + gamma) * lb
    mb_lb = (1. - gamma) * lb

    pb_ub = (1. + gamma) * ub
    mb_ub = (1. - gamma) * ub

    for k, aw_k in enumerate(np_aw):
        # case 1, for frequencies within the passband, set the value to 1.
        if pb_lb <= aw_k <= mb_ub:
            np_ymw[k] = 1.

        # case 2, for frequencies in the upper transition band, compute the cosine function.
        elif mb_ub <= aw_k <= pb_ub:
            np_ymw[k] = np.cos(np.pi * meyer_beta_function(a_ub * (aw_k - mb_ub)) / 2)

        # case 3, for frequencies in the lower transition band, compute the sine function.
        elif mb_lb <= aw_k <= pb_lb:
            np_ymw[k] = np.sin(np.pi * meyer_beta_function(a_lb * (aw_k - mb_lb)) / 2)

    np_ymw_shift = ifftshift(np_ymw)

    return np_ymw_shift


def local_max(np_fft: npt.NDArray, nsignal: int) -> npt.NDArray:
    """
    Segments the FFT signal into a maximum of N supports by taking the middle point between the N largest local maxima.

    Note: the detected boundaries are given in terms of indices

    Inputs:
        np_fft :            FFT Array Signal, The absolute value of the Fourier spectrum of the signal.
        nsignal :           maximal number of bands, The maximum number of supports (modes) to detect.

    Outputs:
        bound:              np.ndarray of detected boundaries
    """

    nsignal = min(nsignal - 1, np_fft.size)
    np_loc_min = max(np_fft) * np.ones_like(np_fft)

    np_bcs = np.zeros(nsignal)
    np_loc_max = np.zeros_like(np_fft)

    for i in np.arange(1, np_fft.size - 1):
        if np_fft[i - 1] < np_fft[i] and np_fft[i] > np_fft[i + 1]:
            np_loc_max[i] = np_fft[i]

        if np_fft[i - 1] > np_fft[i] and np_fft[i] <= np_fft[i + 1]:
            np_loc_min[i] = np_fft[i]

    # keep the N-th the highest maxima
    max_indices = np.argpartition(np_loc_max, -nsignal)[-nsignal:]
    np_max_idx = np.sort(np_loc_max.argsort()[::-1][:nsignal])

    # Middle point between consecutive maxima
    for i in range(nsignal):
        if i == 0:
            a_i = np_max_idx[0] // 2  # Start from half of the first peak index
        else:
            a_i = (np_max_idx[i - 1] + np_max_idx[i]) // 2
        np_bcs[i] = a_i

    # Ensure boundaries are greater than zero
    np_bcs = np_bcs[np_bcs > 0]

    return np_bcs


def local_max_min(np_fft: npt.NDArray, nbands: int, fm=0) -> npt.NDArray:
    """
    Segments the FFT signal by detecting the lowest local minima between the N largest local maxima.

     bound = local_max_min(np_fft, N, fm)

    This function segments np_fft signal into a maximum of N supports by detecting the lowest local minima between the
    N largest local maxima. If the input fm is provided then the local maxima are computed on np_fft and the local
    minima on fm otherwise, both are computed on np_fft (this is useful if you want to compute the maxima on a
    regularized version of your signal while detecting the "true" minima).

    Note: the detected boundaries are given in terms of indices

    Inputs:
        np_fft:             FFT Array Signal
        nbands:             The maximum number of supports (modes) to detect.
        fm:                 Optionally, another function to compute minima on (defaults to np_fft).

    Outputs:
        bound:                  np.ndarray of detected boundaries
    """

    np_loc_max = np.zeros_like(np_fft)

    if isinstance(fm, int):
        f2 = np_fft
    else:
        f2 = fm

    loc_min = max(f2) * np.ones_like(f2)

    # detect local minima and maxima
    for i in np.arange(1, np_fft.size - 1):
        if (np_fft[i - 1] < np_fft[i]) and (np_fft[i] > np_fft[i + 1]):
            np_loc_max[i] = np_fft[i]

        if (f2[i - 1] > f2[i]) and (f2[i] < f2[i + 1]):
            loc_min[i] = f2[i]

    # keep the N-th the highest maxima and their index
    np_bcs = np.zeros(nbands, dtype=int)

    if nbands != -1:
        nbands = nbands - 1
        # keep the N-th the highest maxima
        i_max = np.sort(np_loc_max.argsort()[::-1][:nbands])

        # detect the lowest minima between two consecutive maxima
        for i in range(nbands):
            if i == 0:
                a = 1
            else:
                a = i_max[i - 1]

            l_min_i = np.sort(loc_min[a:i_max[i]])
            ind = np.argsort(loc_min[a:i_max[i]])
            tmpp = l_min_i[0]
            n = 0

            if n < len(l_min_i):
                n = 1
                while (n < len(l_min_i)) and (tmpp == l_min_i[n]):
                    n = n + 1

            np_bcs[i] = a + ind[n // 2] - 1
    else:
        k = 0
        for i in range(loc_min):
            if loc_min[i] < max(f2):
                np_bcs[k] = i - 1
                k = k + 1

    return np_bcs


class EmpiricalWaveletTransform:
    def __init__(self,
                 nsupport: List = None,
                 gamma: float = 1.0,
                 log_nrm: bool = True,
                 completion: bool = False,
                 mbcs: float = 5.,
                 mode: str = "antireflect",
                 ftype: FilterType = FilterType.WAVELET,
                 detection: str = "locmax",
                 ):
        """
        Initialize the Empirical Wavelet Transform (EWT) object with the given parameters.

        Parameters:
            nsupport (list):
                A list containing the minimum and maximum number of supports (modes or signal components) to consider
                during the boundary search. Default is [4, 160].

            gamma (float):
                A regularization parameter used in the construction of Meyer wavelets. Default is 1.0.

            log_nrm (bool):
                If True, the logarithm of the Fourier spectrum will be normalized. Default is True.

            completion (bool):
                If True, the boundaries vector will be completed to get a total of NT boundaries by equally splitting
                the last band (the highest frequencies). Default is False.

            mbcs (int):
                The number of consecutive boundary candidates to consider in the boundary detection. Default is 5.

            mode (str):
                The boundary mode to apply. The default value is "antireflect".

            ftype (FilterType):
                An enumeration representing the type of filter to use for signal regularization.
                Default is FilterType.WAVELET.

            detection (str):
                The method to use for boundary detection. Default is "locmax", which detects the mid-point between two
                consecutive local maxima computed on the regularized spectrum.

        """

        if nsupport is None:
            nsupport = [10, 400]

        self.nsupport = nsupport
        self.gamma = gamma
        self.log_nrm = log_nrm
        self.completion = completion

        self.sigma = 3                          # standard deviation of the above Gaussian filter
        self.ftype = ftype                      # Regularization of the signal

        # Normalization mode for Fourier Transform, where norm="ortho", both directions are scaled by 1/sqrt(n)
        self.wtype = 'ortho'
        self.detection = detection

        # [1 -> inf]
        self.bcs_mirror = mbcs
        self.mode = mode

    def filter(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Apply the selected filter type to the input signal for regularization.

        Parameters:
            signal (npt.NDArray): A 1D NumPy array representing the input signal to be filtered.

        Returns:
            npt.NDArray: A 1D NumPy array representing the filtered signal.

        Note:
        The method uses the filter type specified during the initialization of the Empirical Wavelet Transform (EWT)
        object.

        Available filter types are Gaussian, Average, Convolve, and Wavelet. If an unsupported filter type is provided,
        the original signal will be returned without any filtering.
        """

        if signal.shape[0] >= 10000:
            length_in = (signal.shape[0] // 1000) + 1
        else:
            length_in = 3

        if self.ftype.value is FilterType.GAUSSIAN.value:
            return gaussian_filter1d(signal, self.sigma)

        elif self.ftype.value is FilterType.AVERAGE.value:
            np_kernel = np.ones(length_in) / length_in
            return np.convolve(signal, np_kernel, mode='same')

        elif self.ftype.value is FilterType.CONVOLVE.value:
            np_lin = np.linspace(0., 1., num=length_in // 2 + 1) ** 20.
            np_kernel = np.concatenate([np_lin, np.flip(np_lin[:-1])], axis=0)
            np_kernel /= np_kernel.sum()
            return np.convolve(signal, np_kernel, mode='same')

        elif self.ftype.value is FilterType.WAVELET.value:
            wavelet_filter = WaveletFilter(mode=self.mode)
            return wavelet_filter.denoise(signal)

        else:
            return signal

    def boundary(self, fft_signal: npt.NDArray, ncomp: int) -> npt.NDArray:
        """
        Segments the signal into supports using the specified boundary detection method.

        This function segments the signal into a certain amount of supports by  using different technics:
         - middle point between consecutive local maxima (default),
         - lowest minima between consecutive local maxima (locmaxmin),
         - lowest minima between consecutive local maxima of original spectrum (locmaxminf),

         Regularized version of the spectrum can be obtained by the following methods:
         - Gaussian filtering (its parameters are filter of width length and standard deviation sigma)
         - Average filtering (its parameters are filter of width length)
         - Convolve filtering (its parameters are filter of width length)

         Note: the detected boundaries are given in terms of indices

         Inputs:
           fft_signal:          1-D discrete Fourier Transform Signal
           ncomp:               Maximum number of supports (modes or signal components)

         Outputs:
           boundaries:      Array with detected boundaries
        """

        # Check if fft_signal is all zeros
        if np.all(fft_signal == 0):
            # Return default or empty boundaries
            return np.array([])

        # Mid-point between two consecutive local maxima computed on the regularized spectrum
        if self.detection == "locmax":
            np_bcs = local_max(fft_signal, ncomp)

        # extract the lowest local minima between two selected local maxima
        elif self.detection == "locmaxmin":
            np_bcs = local_max_min(fft_signal, ncomp)

        else:
            raise NotImplementedError

        # Normalization of boundary array
        np_bcs_nrm = np_bcs * np.pi / round(fft_signal.size)

        if self.completion is True and np_bcs.shape[0] < ncomp - 1:
            np_bcs_nrm = self._bc(np_bcs_nrm, ncomp - 1)

        return np_bcs_nrm

    @staticmethod
    def _bc(boundaries: npt.NDArray, nbcs: int) -> npt.NDArray:
        """
        This function permits to complete the boundaries vector to get a total of NT boundaries by equally splitting the
        last band (the highest frequencies)

        Inputs:
          -boundaries:              the boundaries vector you want to complete
          -NT:                      the total number of boundaries wanted

        Output:
          -boundaries: the completed boundaries vector

        """

        eq_bcs = nbcs - boundaries.shape[0]
        delta_w = (np.pi - boundaries[-1]) / (eq_bcs + 1)

        for k in range(eq_bcs):
            boundaries = np.append(boundaries, boundaries[-1] + delta_w)

        return boundaries

    def wavelet(self, bcs: npt.NDArray, nbcs: int):
        """
        Compute the wavelets for the given boundary conditions and number of boundary conditions using Meyer wavelets.

        Parameters:
            bcs (npt.NDArray): A 1D NumPy array representing the boundary conditions.
            nbcs (int): The number of boundary conditions.

        Returns:
            np_wvl (np.ndarray):
                A 2D NumPy array with dimensions (nbcs, npic + 1), where npic is the number of boundary conditions given
                in the input bcs. Each column represents a Meyer wavelet corresponding to the respective boundary
                condition.

        Note:
            This method computes the Meyer wavelets for the given boundary conditions and number of boundary conditions.
            The wavelets are used for the Empirical Wavelet Transform (EWT) framework to decompose and reconstruct
            signals.
        """

        if bcs.size == 0:
            # Return an array of zeros
            return np.zeros((nbcs, 1))

        npic = bcs.shape[0]
        mi = nbcs // 2

        # Ensure gamma is not zero
        gamma = max(self.gamma, 1e-8)

        np_bcs = np.copy(bcs)
        np_yms = np.zeros(nbcs, dtype=float)
        np_wvl = np.zeros([nbcs, npic + 1], dtype=float)
        np_lin = np.linspace(0., 2 * np.pi - 2 * np.pi / nbcs, num=nbcs)

        np_w = fftshift(np_lin)
        np_w[0:mi] = -2 * np.pi + np_w[0:mi]
        np_aw = np.abs(np_w)

        # Adjust gamma if necessary
        for k in range(npic - 1):
            denominator = np_bcs[k + 1] + np_bcs[k]
            denominator = max(denominator, 1e-8)  # Prevent division by zero
            r_k = (np_bcs[k + 1] - np_bcs[k]) / denominator
            if r_k < gamma:
                gamma = r_k

        denominator_r_nrm = (np.pi + np_bcs[npic - 1])
        denominator_r_nrm = max(denominator_r_nrm, 1e-8)  # Prevent division by zero
        r_nrm = (np.pi - np_bcs[npic - 1]) / denominator_r_nrm
        if r_nrm < gamma:
            gamma = r_nrm

        # this ensure that gamma is chosen as strictly less than the min
        gamma *= (1. - 1. / nbcs)

        ############################################################
        # Handle zero boundary
        epsilon = 1e-8
        adjusted_bcs0 = np_bcs[0] if np_bcs[0] != 0 else epsilon

        a_lb = 1. / (2 * gamma * adjusted_bcs0)
        pb_lb = (1. + gamma) * np_bcs[0]
        mb_lb = (1. - gamma) * np_bcs[0]

        np_aw_mb_lb_up_idx = np.where(np_aw <= mb_lb)[0]

        np_aw_mb_lb_lw_idx = np.where(np_aw >= mb_lb)[0]
        np_aw_pb_lb_up_idx = np.where(np_aw <= pb_lb)[0]

        # case 3: (1 - gamma) * wt <= abs(w) <= (1 + gamma) * wt
        np_slc = np.intersect1d(np_aw_mb_lb_lw_idx, np_aw_pb_lb_up_idx)

        np_yms[np_aw_mb_lb_up_idx] = 1.

        # Avoid invalid operations if `a_lb` is infinite
        np_yms[np_slc] = np.cos(np.pi * meyer_beta_function(a_lb * (np_aw[np_slc] - mb_lb)) / 2)

        # Lower Bound
        np_wvl[:, 0] = ifftshift(np_yms)

        # generate rest of the wavelets
        for k in range(npic - 1):
            lb = np_bcs[k]
            ub = np_bcs[k + 1]
            if lb >= ub:
                continue  # Skip invalid bands
            np_wvl[:, k + 1] = meyer_wavelet(gamma, nbcs, lb, ub)

        # Upper Bound
        lb = np_bcs[-1]
        ub = np.pi
        if lb < ub:
            np_wvl[:, npic] = meyer_wavelet(gamma, nbcs, lb, ub)
        else:
            np_wvl[:, npic] = np.zeros(nbcs)

        return np_wvl

    def signal_bcs(self, signal: npt.NDArray) -> npt.NDArray:
        """
        Extends the input signal using the mirror method for boundary handling.

        This method creates an extended version of the input signal by mirroring a specified number of samples from
        each end. This can help reduce edge artifacts when performing filtering or wavelet decomposition on the signal.

        Parameters:
        ----------
        signal : npt.NDArray
            The input signal to be extended.

        Returns:
        -------
        npt.NDArray
            The extended signal with mirrored boundaries.
        """

        nsz = int(signal.shape[0] // self.bcs_mirror + 1)

        if self.mode == "symmetric":
            left_padding = np.flip(signal[1:nsz])
            right_padding = np.flip(signal[-(nsz - 1):])
            list_signal_pad = [left_padding, signal, right_padding]

        elif self.mode == "antireflect":
            left_padding = signal[1:1 + nsz][::-1]
            right_padding = signal[-nsz - 1:-1][::-1]
            list_signal_pad = [left_padding[1:nsz], signal, right_padding[-(nsz - 1):]]

        else:
            raise NotImplementedError

        return np.concatenate(list_signal_pad)

    def _signal_prep(self, signal: npt.NDArray) -> (npt.NDArray, npt.NDArray, npt.NDArray):
        """
        Prepare the input signal for the Empirical Wavelet Transform by filtering and computing the real FFT.

        Parameters:
            signal (npt.NDArray): A 1D NumPy array representing the input signal to be prepared.

            Returns:
            np_signal_ft (npt.NDArray): A 1D NumPy array representing the filtered input signal.
            np_rfft (npt.NDArray): A 1D NumPy array representing the real FFT of the filtered input signal.
            np_afft (npt.NDArray): A 1D NumPy array representing the absolute values of the positive half of the real FFT.

            Note:
            This method preprocesses the input signal by filtering it and then computing the real Fast Fourier Transform
            (FFT). The filtered signal, the real FFT, and the absolute values of the positive half of the real FFT are
            returned for further processing in the Empirical Wavelet Transform (EWT) framework.
        """

        np_signal_ft = self.filter(signal)
        np_rfft = rfft(np_signal_ft, norm=self.wtype)
        nsz_fft = np_rfft.size // 2
        np_afft = np.absolute(np_rfft[nsz_fft:])

        return np_signal_ft, np_rfft, np_afft

    def _signal_decomp(self,
                       np_signal_ft: npt.NDArray,
                       np_afft: npt.NDArray,
                       nsignals: int,
                       mbcs: float = None,
                       ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Perform the Empirical Wavelet Transform (EWT) on the filtered signal to extract sub-bands.

        Parameters:
            np_signal_ft (npt.NDArray):
                A 1D NumPy array representing the filtered input signal.
            np_afft (npt.NDArray):
                A 1D NumPy array representing the absolute values of the positive half of the real FFT.
            nsignals (int):
                The number of sub-bands to extract from the signal.
            mbcs (int):
                The number of boundary symmetric size. Default is 5 in __init__.

        Returns:
            np_ewt_res (npt.NDArray):
                A 2D NumPy array representing the extracted sub-bands.
            np_mwvlt (npt.NDArray):
                A 2D NumPy array representing the Meyer wavelets.
            np_bcs (npt.NDArray):
                A 1D NumPy array representing the detected boundaries in the Fourier domain.

        Note:
            This method decomposes the filtered input signal into sub-bands using the Empirical Wavelet Transform (EWT)
            algorithm. The Meyer wavelets are computed based on the detected boundaries, and then the sub-bands are
            extracted by filtering the signal using these wavelets.
        """
        if mbcs is not None:
            self.bcs_mirror = mbcs

        np_bcs = self.boundary(np_afft, nsignals)
        np_fft_flp = fft(self.signal_bcs(np_signal_ft))
        np_mwvlt = self.wavelet(np_bcs, np_fft_flp.shape[0])

        if np_bcs.size == 0 or np_mwvlt.size == 0:
            # Return zeros or handle accordingly
            np_ewt_res = np.zeros((np_signal_ft.size, 1))
            return np_ewt_res, np_mwvlt, np_bcs

        # filter the signal to extract each sub-band
        np_ewt = np.zeros(np_mwvlt.shape, dtype=float)

        for k in range(np_mwvlt.shape[1]):
            np_ewt[:, k] = np.real(ifft(np.conjugate(np_mwvlt[:, k]) * np_fft_flp))

        # Recover the signal proper shape/ size
        nsz_bcs = int(np_signal_ft.shape[0] // self.bcs_mirror + 1)
        np_ewt_res = np_ewt[nsz_bcs - 1:-(nsz_bcs - 1), :]

        return np_ewt_res, np_mwvlt, np_bcs

    def __call__(self,
                 signal: np.ndarray,
                 nsignals: int,
                 mbcs: float = None,
                 ) -> (npt.NDArray, npt.NDArray, npt.NDArray):
        """
        Perform the Empirical Wavelet Transform of np_fft over N scales. See also the documentation of ewt_bcs_detect
        for more details about the available methods and their parameters.

        signal (npt.NDArray):
            np_signal:      1D input signal
        nsignals (int):
            The number of sub-bands to extract from the signal.
        mbcs (int):
            The number of boundary symmetric size. Default is 5.

        Returns:
            np_ewt_res (npt.NDArray):
                A 2D NumPy array representing the extracted sub-bands.
            np_mwvlt (npt.NDArray):
                A 2D NumPy array representing the Meyer wavelets.
            np_bcs (npt.NDArray):
                A 1D NumPy array representing the detected boundaries in the Fourier domain.
        """

        np_signal_ft, _, np_afft = self._signal_prep(signal)

        return self._signal_decomp(np_signal_ft, np_afft, nsignals, mbcs=mbcs)

    def run(self, signal: npt.NDArray, mbcs: float = None, opt: bool = False) -> (Dict[str, npt.NDArray], int):
        """
        Perform Empirical Wavelet Transform (EWT) on the input signal and return the best decomposition results
        based on the minimum norm of the difference between the original signal and the sum of the extracted
        sub-bands. The method iteratively searches for the optimal number of sub-bands to decompose the signal into.

        Args:
            signal (npt.NDArray): The input signal to be decomposed.
            mbcs
            opt (bool):

        Returns:
            tuple: A tuple containing two elements:
                - dict_output (Dict[str, Any]): A dictionary with the best decomposition results.
                  The keys are "ewt" (the extracted sub-bands), "wvlt" (the wavelet coefficients),
                  "boundaries" (the frequency boundaries), "N" (optimal number of sub-bands), and "diff"
                  (the difference between the original signal and the sum of the sub-bands).
                - optimal_sub_bands (int): The optimal number of sub-bands used for the best decomposition.
        """

        if np.all(signal == 0):
            # Return zeros or default values
            np_signal_ft = np.zeros_like(signal)
            result = {
                'ewt': np.zeros((signal.size, 1)),
                'wvlt': np.zeros((1, 1)),
                'boundaries': np.array([]),
                'N': 0,
                'diff': np.zeros_like(signal)
            }
            return np_signal_ft, result, 0

        np_signal_ft, _, np_afft = self._signal_prep(signal)

        dt_spt = (np.diff(self.nsupport)).item() - 1
        dt_spt = dt_spt // 4

        dict_output = dict()
        dict_anrm = dict()
        key_res = ["ewt", "wvlt", "boundaries"]

        stt_spt = self.nsupport[0]
        end_spt = self.nsupport[-1]
        np_nsupport_k = np.arange(stt_spt, end_spt, dt_spt)

        if not opt:
            for _ in range(8):
                for n_i in np_nsupport_k:
                    if dict_output.get(n_i) is None:
                        res_i = self._signal_decomp(np_signal_ft, np_afft, n_i, mbcs=mbcs)
                        out_i = {key_j: res_i[j] for j, key_j in enumerate(key_res)}
                        out_i["N"] = n_i
                        out_i["diff"] = signal - res_i[0].sum(axis=1)
                        dict_output[n_i] = out_i

                        dict_anrm[n_i] = np.linalg.norm(np.abs(signal - res_i[0].sum(axis=1)))

                list_keys_k = list(dict_anrm.keys())
                list_values_k = list(dict_anrm.values())

                np_argsort_k = np.argsort(list_values_k)

                arg_spt_k = np.sort([list_keys_k[np_argsort_k[0]], list_keys_k[np_argsort_k[1]]])
                dt_spt = dt_spt // 4

                if dt_spt > 0:
                    np_nsupport_k = np.arange(*arg_spt_k, dt_spt)
                else:
                    break

            opt_idx = np.argmin(list_values_k)

            return np_signal_ft, dict_output[list_keys_k[opt_idx]], int(list_keys_k[opt_idx])

        else:
            xi_ini = np.array([200., 2.])
            scp_bounds = Bounds([5., 1.1], [500., 50.])

            def residum(xi: npt.NDArray) -> float:
                res = self._signal_decomp(np_signal_ft, np_afft, int(np.round(xi[0], 0)), mbcs=np.round(xi[1], 2))

                try:  # pragma: no cover
                    res_nrm = np.linalg.norm(np.abs(signal - res[0].sum(axis=1)))
                    eqv_res = np.log(res_nrm) + np.log(xi[1])

                    print(f"  => Residuum Value: {eqv_res}, Nrm Res: {res_nrm}, Params: {xi}")

                except Exception as m_err:
                    print(f' Warning or Error: {m_err} ')
                    eqv_res = 200.

                return eqv_res

            opt_res = differential_evolution(residum, scp_bounds, maxiter=10, polish=False, x0=xi_ini, disp=True)

            nsupport = int(np.round(opt_res.x[0], 0))
            self.bcs_mirror = np.round(opt_res.x[1], 2)

            res_fnl = self._signal_decomp(np_signal_ft, np_afft, nsupport, self.bcs_mirror)

            out_i = {key_j: res_fnl[j] for j, key_j in enumerate(key_res)}
            out_i["N"] = nsupport
            out_i["diff"] = signal - res_fnl[0].sum(axis=1)

            return np_signal_ft, out_i, nsupport


def ewt_decomp(signal_data: pd.Series,
               ftype: FilterType = FilterType.WAVELET,
               mbcs: int = 5,
               ) -> (pd.DataFrame, np.ndarray):

    np_signal = signal_data.values
    np_date = signal_data.index.values.astype('datetime64[s]')

    ewt_pro = EmpiricalWaveletTransform(ftype=ftype, mbcs=mbcs, gamma=1., completion=True)
    ewt_opt, nsign_opt = ewt_pro.run(np_signal, opt=True)
    # ewty_opt, nsign_opt = ewt_pro.run(np_signal, opt=False)

    np_ewt = ewt_opt["ewt"]
    np_ft = ewt_pro.filter(np_signal)

    ##########################################################################
    # frequency domain (spectrum)
    dict_ewt = {'ft': np_ft.copy()}

    list_arg_max = []

    for signal_i in range(np_ewt.shape[1]):
        list_arg_max.append((np.abs(rfft(np_ewt[:, signal_i]))).argmax())

    ##########################################################################
    # Frequency by pick
    run_time = (np_date[-1] - np_date[0]).astype('int') / 60

    np_time_linspc = np.linspace(0., run_time, num=np_signal.shape[0])

    list_freq = []

    for i in range(np_ewt.shape[1]):
        fpeaks_upp_i, _ = find_peaks(np_ewt[:, i])
        fpeaks_lwr_i, _ = find_peaks(-np_ewt[:, i])

        fpeaks_i = np.concatenate([fpeaks_upp_i, fpeaks_lwr_i])
        fpeaks_i.sort()
        np_peaks_i = fpeaks_i[-6:]

        list_freq.append(1. / (2. * (np.diff(np_time_linspc[np_peaks_i])).mean()))

    np_arg_sort = np.flip(np.argsort(list_freq))

    for i, idx_i in enumerate(np_arg_sort):
        pwd_i = int(10 + i * 10)
        dict_ewt[pwd_i] = np_ewt[:, idx_i]

    pd_ewt = pd.DataFrame(dict_ewt, index=signal_data.index)
    pd_ewt.columns = pd_ewt.columns.map(str)

    np_period = 1. / (np.array(list_freq))[np_arg_sort]

    return pd_ewt, np_period


def ewt_filter(ewt: np.ndarray, convolve_filter: np.ndarray, nsize: int = 6) -> np.ndarray:
    list_residuum = []
    list_residuum_l2 = []

    for sig_i in range(nsize):
        np_ewt_cv_i = ewt[:, :sig_i + 1].sum(axis=1)
        res_i = convolve_filter - np_ewt_cv_i

        list_residuum.append(res_i)
        list_residuum_l2.append(np.linalg.norm(res_i))

    return np.array(list_residuum_l2)
