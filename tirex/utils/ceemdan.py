import logging
import numpy as np
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
from scipy.interpolate import interp1d, Akima1DInterpolator, CubicHermiteSpline, CubicSpline, PchipInterpolator

FindExtremaOutput = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def TDMAsolver(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm to solve tridiagonal linear systems with
    non-periodic BC.

    | b0  c0                 | | . |     | . |
    | a1  b1  c1             | | . |     | . |
    |     a2  b2  c2         | | x |  =  | d |
    |         ..........     | | . |     | . |
    |             an  bn  cn | | . |     | . |
    """
    n = len(b)
    if n == 0:
        return np.array([])

    cp = np.zeros(n, dtype=d.dtype)  # Ensure cp and dp have same dtype as d for precision
    dp = np.zeros(n, dtype=d.dtype)

    # Forward elimination
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if denom == 0:  # Avoid division by zero
            denom = np.finfo(d.dtype).eps  # Add small epsilon or handle as singular
        if i < n - 1:  # c is defined up to n-2 for cp
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # Backward substitution
    x = np.zeros(n, dtype=d.dtype)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def filt6(f, alpha):
    """
    6th Order compact filter (non-periodic BC).

    References:
    -----------
    Lele, S. K. - Compact finite difference schemes with spectral-like
    resolution. Journal of Computational Physics 103 (1992) 16-42

    Visbal, M. R. and Gaitonde, D. V. - On the use of higher-order finite-
    difference schemes on curvilinear and deforming meshes. Journal of
    Computational Physics 181 (2002) 155-185
    """
    Ca = (11.0 + 10.0 * alpha) / 16.0
    Cb = (15.0 + 34.0 * alpha) / 32.0
    Cc = (-3.0 + 6.0 * alpha) / 16.0
    Cd = (1.0 - 2.0 * alpha) / 32.0

    n = len(f)

    rhs = np.zeros(n)

    rhs[3:-3] = (
        Cd * 0.5 * (f[6:] + f[:-6]) + Cc * 0.5 * (f[5:-1] + f[1:-5]) + Cb * 0.5 * (f[4:-2] + f[2:-4]) + Ca * f[3:-3]
    )

    # Non-periodic BC:
    rhs[0] = (15.0 / 16.0) * f[0] + (4.0 * f[1] - 6.0 * f[2] + 4.0 * f[3] - f[4]) / 16.0

    rhs[1] = (3.0 / 4.0) * f[1] + (f[0] + 6.0 * f[2] - 4.0 * f[3] + f[4]) / 16.0

    rhs[2] = (5.0 / 8.0) * f[2] + (-f[0] + 4.0 * f[1] + 4.0 * f[3] - f[4]) / 16.0

    rhs[-1] = (15.0 / 16.0) * f[-1] + (4.0 * f[-2] - 6.0 * f[-3] + 4.0 * f[-4] - f[-5]) / 16.0

    rhs[-2] = (3.0 / 4.0) * f[-2] + (f[-1] + 6.0 * f[-3] - 4.0 * f[-4] + f[-5]) / 16.0

    rhs[-3] = (5.0 / 8.0) * f[-3] + (-f[-1] + 4.0 * f[-2] + 4.0 * f[-4] - f[-5]) / 16.0

    Da = alpha * np.ones(n)
    Db = np.ones(n)
    Dc = alpha * np.ones(n)

    # 1st point
    Dc[0] = 0.0
    # 2nd point
    Da[1] = Dc[1] = 0.0
    # 3rd point
    Da[2] = Dc[2] = 0.0
    # last point
    Da[-1] = 0.0
    # 2nd from last
    Da[-2] = Dc[-2] = 0.0
    # 3rd from last
    Da[-3] = Dc[-3] = 0.0

    return TDMAsolver(Da, Db, Dc, rhs)


def pade6(vec, h):
    """
    6th Order compact finite difference scheme (non-periodic BC).

    Lele, S. K. - Compact finite difference schemes with spectral-like
    resolution. Journal of Computational Physics 103 (1992) 16-42
    """
    n = len(vec)
    rhs = np.zeros(n)

    a = 14.0 / 18.0
    b = 1.0 / 36.0

    rhs[2:-2] = (vec[3:-1] - vec[1:-3]) * (a / h) + (vec[4:] - vec[0:-4]) * (b / h)

    # boundaries:
    rhs[0] = (
        (-197.0 / 60.0) * vec[0]
        + (-5.0 / 12.0) * vec[1]
        + 5.0 * vec[2]
        + (-5.0 / 3.0) * vec[3]
        + (5.0 / 12.0) * vec[4]
        + (-1.0 / 20.0) * vec[5]
    ) / h

    rhs[1] = (
        (-20.0 / 33.0) * vec[0]
        + (-35.0 / 132.0) * vec[1]
        + (34.0 / 33.0) * vec[2]
        + (-7.0 / 33.0) * vec[3]
        + (2.0 / 33.0) * vec[4]
        + (-1.0 / 132.0) * vec[5]
    ) / h

    rhs[-1] = (
        (197.0 / 60.0) * vec[-1]
        + (5.0 / 12.0) * vec[-2]
        + (-5.0) * vec[-3]
        + (5.0 / 3.0) * vec[-4]
        + (-5.0 / 12.0) * vec[-5]
        + (1.0 / 20.0) * vec[-6]
    ) / h

    rhs[-2] = (
        (20.0 / 33.0) * vec[-1]
        + (35.0 / 132.0) * vec[-2]
        + (-34.0 / 33.0) * vec[-3]
        + (7.0 / 33.0) * vec[-4]
        + (-2.0 / 33.0) * vec[-5]
        + (1.0 / 132.0) * vec[-6]
    ) / h

    alpha1 = 5.0  # j = 1 and n
    alpha2 = 2.0 / 11  # j = 2 and n-1
    alpha = 1.0 / 3.0

    Db = np.ones(n)
    Da = alpha * np.ones(n)
    Dc = alpha * np.ones(n)

    # boundaries:
    Da[1] = alpha2
    Da[-1] = alpha1
    Da[-2] = alpha2
    Dc[0] = alpha1
    Dc[1] = alpha2
    Dc[-2] = alpha2

    return TDMAsolver(Da, Db, Dc, rhs)


def cubic_spline_3pts(x, y, T):
    """
    Apparently scipy.interpolate.interp1d does not support
    cubic spline for less than 4 points.
    """
    x0, x1, x2 = x
    y0, y1, y2 = y

    x1x0, x2x1 = x1 - x0, x2 - x1
    y1y0, y2y1 = y1 - y0, y2 - y1
    _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

    m11, m12, m13 = 2 * _x1x0, _x1x0, 0
    m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
    m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

    v1 = 3 * y1y0 * _x1x0 * _x1x0
    v3 = 3 * y2y1 * _x2x1 * _x2x1
    v2 = v1 + v3

    M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v = np.array([v1, v2, v3]).T
    k = np.linalg.inv(M).dot(v)

    a1 = k[0] * x1x0 - y1y0
    b1 = -k[1] * x1x0 + y1y0
    a2 = k[1] * x2x1 - y2y1
    b2 = -k[2] * x2x1 + y2y1

    t = T[np.r_[T >= x0] & np.r_[T <= x2]]
    t1 = (T[np.r_[T >= x0] & np.r_[T < x1]] - x0) / x1x0
    t2 = (T[np.r_[T >= x1] & np.r_[T <= x2]] - x1) / x2x1
    t11, t22 = 1.0 - t1, 1.0 - t2

    q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
    q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
    q = np.append(q1, q2)

    return t, q


def akima(X, Y, x):
    spl = Akima1DInterpolator(X, Y)
    return spl(x)


def cubic_hermite(X, Y, x):
    dydx = np.gradient(Y, X)
    spl = CubicHermiteSpline(X, Y, dydx)
    return spl(x)


def cubic(X, Y, x):
    spl = CubicSpline(X, Y)
    return spl(x)


def pchip(X, Y, x):
    spl = PchipInterpolator(X, Y)
    return spl(x)


def get_timeline(range_max: int, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Returns timeline array for requirements.

    Parameters
    ----------
    range_max : int
        Largest value in range. Assume `range(range_max)`. Commonly that's length of the signal.
    dtype : np.dtype
        Minimal definition type. Returned timeline will have dtype that's the same or with higher byte size.

    """
    timeline = np.arange(0, range_max, dtype=dtype)
    if timeline[-1] != range_max - 1:
        inclusive_dtype = smallest_inclusive_dtype(timeline.dtype, range_max)
        timeline = np.arange(0, range_max, dtype=inclusive_dtype)
    return timeline


def smallest_inclusive_dtype(ref_dtype: np.dtype, ref_value) -> np.dtype:
    """Returns a numpy dtype with the same base as reference dtype (ref_dtype)
    but with the range that includes reference value (ref_value).

    Parameters
    ----------
    ref_dtype : dtype
         Reference dtype. Used to select the base, i.e. int or float, for returned type.
    ref_value : value
        A value which needs to be included in returned dtype. Value will be typically int or float.

    """
    # Integer path
    if np.issubdtype(ref_dtype, np.integer):
        for dtype in [np.uint16, np.uint32, np.uint64]:
            if ref_value < np.iinfo(dtype).max:
                return dtype
        max_val = np.iinfo(np.uint32).max
        raise ValueError("Requested too large integer range. Exceeds max( uint64 ) == '{}.".format(max_val))

    # Float path
    if np.issubdtype(ref_dtype, np.floating):
        for dtype in [np.float16, np.float32, np.float64]:
            if ref_value < np.finfo(dtype).max:
                return dtype
        max_val = np.finfo(np.float64).max
        raise ValueError("Requested too large integer range. Exceeds max( float64 ) == '{}.".format(max_val))

    raise ValueError("Unsupported dtype '{}'. Only intX and floatX are supported.".format(ref_dtype))


class EMD:
    """
    .. _EMD:

    **Empirical Mode Decomposition**

    Method of decomposing signal into Intrinsic Mode Functions (IMFs)
    based on algorithm presented in Huang et al. [Huang1998]_.

    Algorithm was validated with Rilling et al. [Rilling2003]_ Matlab's version from 3.2007.

    Threshold which control the goodness of the decomposition:
        * `std_thr` --- Test for the proto-IMF how variance changes between siftings.
        * `svar_thr` -- Test for the proto-IMF how energy changes between siftings.
        * `total_power_thr` --- Test for the whole decomp how much of energy is solved.
        * `range_thr` --- Test for the whole decomp whether the difference is tiny.


    References
    ----------
    .. [Huang1998] N. E. Huang et al., "The empirical mode decomposition and the
        Hilbert spectrum for non-linear and non stationary time series
        analysis", Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
    .. [Rilling2003] G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode
        Decomposition and its algorithms", IEEE-EURASIP Workshop on
        Nonlinear Signal and Image Processing NSIP-03, Grado (I), June 2003

    Examples
    --------
    >>> import numpy as np
    >>> T = np.linspace(0, 1, 100)
    >>> S = np.sin(2*2*np.pi*T)
    >>> emd = EMD(extrema_detection='parabol')
    >>> IMFs = emd.emd(S)
    >>> IMFs.shape
    (1, 100)
    """

    logger = logging.getLogger(__name__)

    def __init__(self, spline_kind: str = "cubic", nbsym: int = 2, **kwargs):
        """Initiate *EMD* instance.

        Configuration, such as threshold values, can be passed as kwargs (keyword arguments).

        Parameters
        ----------
        FIXE : int (default: 0)
        FIXE_H : int (default: 0)
        MAX_ITERATION : int (default 1000)
            Maximum number of iterations per single sifting in EMD.
        energy_ratio_thr : float (default: 0.2)
            Threshold value on energy ratio per IMF check.
        std_thr float : (default 0.2)
            Threshold value on standard deviation per IMF check.
        svar_thr float : (default 0.001)
            Threshold value on scaled variance per IMF check.
        total_power_thr : float (default 0.005)
            Threshold value on total power per EMD decomposition.
        range_thr : float (default 0.001)
            Threshold for amplitude range (after scaling) per EMD decomposition.
        extrema_detection : str (default 'simple')
            Method used to finding extrema.
        DTYPE : np.dtype (default np.float64)
            Data type used.

        Examples
        --------
        >>> emd = EMD(std_thr=0.01, range_thr=0.05)
        """

        # Declare constants
        self.energy_ratio_thr = float(kwargs.get("energy_ratio_thr", 0.2))
        self.std_thr = float(kwargs.get("std_thr", 0.2))
        self.svar_thr = float(kwargs.get("svar_thr", 0.001))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.005))
        self.range_thr = float(kwargs.get("range_thr", 0.001))

        self.nbsym = int(kwargs.get("nbsym", nbsym))
        self.scale_factor = float(kwargs.get("scale_factor", 1.0))

        self.spline_kind = spline_kind
        self.extrema_detection = kwargs.get("extrema_detection", "simple")
        assert self.extrema_detection in ("simple", "parabol"), "Only 'simple' and 'parabol' values supported"
        self.DTYPE = kwargs.get("DTYPE", np.float64)
        self.FIXE = int(kwargs.get("FIXE", 0))
        self.FIXE_H = int(kwargs.get("FIXE_H", 0))
        self.MAX_ITERATION = int(kwargs.get("MAX_ITERATION", 1000))
        self.imfs: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None

    def __call__(self,
                 S: np.ndarray,
                 T: Optional[np.ndarray] = None,
                 max_imf: int = -1,
                 ) -> np.ndarray:
        return self.emd(S, T=T, max_imf=max_imf)

    def extract_max_min_spline(self,
                               T: np.ndarray,
                               S: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts top and bottom envelopes based on the signal,
        which are constructed based on maxima and minima, respectively.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        S : numpy array
            Input data S(T).

        Returns
        -------
        max_spline : numpy array
            Spline spanned on S maxima.
        min_spline : numpy array
            Spline spanned on S minima.
        max_extrema : numpy array
            Points indicating local maxima.
        min_extrema : numpy array
            Points indicating local minima.
        """

        # Get indexes of extrema
        ext_res = self.find_extrema(T, S)
        max_pos, max_val = ext_res[0], ext_res[1]
        min_pos, min_val = ext_res[2], ext_res[3]

        if len(max_pos) + len(min_pos) < 3:
            return [-1] * 4  # TODO: Fix this. Doesn't match the signature.

        #########################################
        # Extrapolation of signal (over boundaries)
        max_extrema, min_extrema = self.prepare_points(T, S, max_pos, max_val, min_pos, min_val)

        _, max_spline = self.spline_points(T, max_extrema)
        _, min_spline = self.spline_points(T, min_extrema)

        return max_spline, min_spline, max_extrema, min_extrema

    def prepare_points(
        self,
        T: np.ndarray,
        S: np.ndarray,
        max_pos: np.ndarray,
        max_val: np.ndarray,
        min_pos: np.ndarray,
        min_val: np.ndarray,
    ):
        """
        Performs extrapolation on edges by adding extra extrema, also known
        as mirroring signal. The number of added points depends on *nbsym*
        variable.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        S : numpy array
            Input signal.
        max_pos : iterable
            Sorted time positions of maxima.
        max_val : iterable
            Signal values at max_pos positions.
        min_pos : iterable
            Sorted time positions of minima.
        min_val : iterable
            Signal values at min_pos positions.

        Returns
        -------
        max_extrema : numpy array (2 rows)
            Position (1st row) and values (2nd row) of minima.
        min_extrema : numpy array (2 rows)
            Position (1st row) and values (2nd row) of maxima.
        """
        if self.extrema_detection == "parabol":
            return self._prepare_points_parabol(T, S, max_pos, max_val, min_pos, min_val)
        elif self.extrema_detection == "simple":
            return self._prepare_points_simple(T, S, max_pos, max_val, min_pos, min_val)
        else:
            msg = "Incorrect extrema detection type. Please try: 'simple' or 'parabol'."
            raise ValueError(msg)

    def _prepare_points_parabol(self, T, S, max_pos, max_val, min_pos, min_val) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs mirroring on signal which extrema do not necessarily
        belong on the position array.

        See :meth:`EMD.prepare_points`.
        """

        # Need at least two extrema to perform mirroring
        max_extrema = np.zeros((2, len(max_pos)), dtype=self.DTYPE)
        min_extrema = np.zeros((2, len(min_pos)), dtype=self.DTYPE)

        max_extrema[0], min_extrema[0] = max_pos, min_pos
        max_extrema[1], min_extrema[1] = max_val, min_val

        # Local variables
        nbsym = self.nbsym
        end_min, end_max = len(min_pos), len(max_pos)

        ####################################
        # Left bound
        d_pos = max_pos[0] - min_pos[0]
        left_ext_max_type = d_pos < 0  # True -> max, else min

        # Left extremum is maximum
        if left_ext_max_type:
            if (S[0] > min_val[0]) and (np.abs(d_pos) > (max_pos[0] - T[0])):
                # mirror signal to first extrema
                expand_left_max_pos = 2 * max_pos[0] - max_pos[1 : nbsym + 1]
                expand_left_min_pos = 2 * max_pos[0] - min_pos[0:nbsym]
                expand_left_max_val = max_val[1 : nbsym + 1]
                expand_left_min_val = min_val[0:nbsym]
            else:
                # mirror signal to beginning
                expand_left_max_pos = 2 * T[0] - max_pos[0:nbsym]
                expand_left_min_pos = 2 * T[0] - np.append(T[0], min_pos[0 : nbsym - 1])
                expand_left_max_val = max_val[0:nbsym]
                expand_left_min_val = np.append(S[0], min_val[0 : nbsym - 1])

        # Left extremum is minimum
        else:
            if (S[0] < max_val[0]) and (np.abs(d_pos) > (min_pos[0] - T[0])):
                # mirror signal to first extrema
                expand_left_max_pos = 2 * min_pos[0] - max_pos[0:nbsym]
                expand_left_min_pos = 2 * min_pos[0] - min_pos[1 : nbsym + 1]
                expand_left_max_val = max_val[0:nbsym]
                expand_left_min_val = min_val[1 : nbsym + 1]
            else:
                # mirror signal to beginning
                expand_left_max_pos = 2 * T[0] - np.append(T[0], max_pos[0 : nbsym - 1])
                expand_left_min_pos = 2 * T[0] - min_pos[0:nbsym]
                expand_left_max_val = np.append(S[0], max_val[0 : nbsym - 1])
                expand_left_min_val = min_val[0:nbsym]

        if not expand_left_min_pos.shape:
            expand_left_min_pos, expand_left_min_val = min_pos, min_val

        if not expand_left_max_pos.shape:
            expand_left_max_pos, expand_left_max_val = max_pos, max_val

        expand_left_min = np.vstack((expand_left_min_pos[::-1], expand_left_min_val[::-1]))
        expand_left_max = np.vstack((expand_left_max_pos[::-1], expand_left_max_val[::-1]))

        ####################################
        # Right bound
        d_pos = max_pos[-1] - min_pos[-1]
        right_ext_max_type = d_pos > 0

        # Right extremum is maximum
        if not right_ext_max_type:
            if (S[-1] < max_val[-1]) and (np.abs(d_pos) > (T[-1] - min_pos[-1])):
                # mirror signal to last extrema
                idx_max = max(0, end_max - nbsym)
                idx_min = max(0, end_min - nbsym - 1)
                expand_right_max_pos = 2 * min_pos[-1] - max_pos[idx_max:]
                expand_right_min_pos = 2 * min_pos[-1] - min_pos[idx_min:-1]
                expand_right_max_val = max_val[idx_max:]
                expand_right_min_val = min_val[idx_min:-1]
            else:
                # mirror signal to end
                idx_max = max(0, end_max - nbsym + 1)
                idx_min = max(0, end_min - nbsym)
                expand_right_max_pos = 2 * T[-1] - np.append(max_pos[idx_max:], T[-1])
                expand_right_min_pos = 2 * T[-1] - min_pos[idx_min:]
                expand_right_max_val = np.append(max_val[idx_max:], S[-1])
                expand_right_min_val = min_val[idx_min:]

        # Right extremum is minimum
        else:
            if (S[-1] > min_val[-1]) and len(max_pos) > 1 and (np.abs(d_pos) > (T[-1] - max_pos[-1])):
                # mirror signal to last extremum
                idx_max = max(0, end_max - nbsym - 1)
                idx_min = max(0, end_min - nbsym)
                expand_right_max_pos = 2 * max_pos[-1] - max_pos[idx_max:-1]
                expand_right_min_pos = 2 * max_pos[-1] - min_pos[idx_min:]
                expand_right_max_val = max_val[idx_max:-1]
                expand_right_min_val = min_val[idx_min:]
            else:
                # mirror signal to end
                idx_max = max(0, end_max - nbsym)
                idx_min = max(0, end_min - nbsym + 1)
                expand_right_max_pos = 2 * T[-1] - max_pos[idx_max:]
                expand_right_min_pos = 2 * T[-1] - np.append(min_pos[idx_min:], T[-1])
                expand_right_max_val = max_val[idx_max:]
                expand_right_min_val = np.append(min_val[idx_min:], S[-1])

        if not expand_right_min_pos.shape:
            expand_right_min_pos, expand_right_min_val = min_pos, min_val
        if not expand_right_max_pos.shape:
            expand_right_max_pos, expand_right_max_val = max_pos, max_val

        expand_right_min = np.vstack((expand_right_min_pos[::-1], expand_right_min_val[::-1]))
        expand_right_max = np.vstack((expand_right_max_pos[::-1], expand_right_max_val[::-1]))

        max_extrema = np.hstack((expand_left_max, max_extrema, expand_right_max))
        min_extrema = np.hstack((expand_left_min, min_extrema, expand_right_min))

        return max_extrema, min_extrema

    def _prepare_points_simple(
        self,
        T: np.ndarray,
        S: np.ndarray,
        max_pos: np.ndarray,
        max_val: Optional[np.ndarray],
        min_pos: np.ndarray,
        min_val: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs mirroring on signal which extrema can be indexed on
        the position array.

        See :meth:`EMD.prepare_points`.
        """

        # Find indexes of pass
        ind_min = min_pos.astype(int)
        ind_max = max_pos.astype(int)

        # Local variables
        nbsym = self.nbsym
        end_min, end_max = len(min_pos), len(max_pos)

        ####################################
        # Left bound - mirror nbsym points to the left
        if ind_max[0] < ind_min[0]:
            if S[0] > S[ind_min[0]]:
                lmax = ind_max[1 : min(end_max, nbsym + 1)][::-1]
                lmin = ind_min[0 : min(end_min, nbsym + 0)][::-1]
                lsym = ind_max[0]
            else:
                lmax = ind_max[0 : min(end_max, nbsym)][::-1]
                lmin = np.append(ind_min[0 : min(end_min, nbsym - 1)][::-1], 0)
                lsym = 0
        else:
            if S[0] < S[ind_max[0]]:
                lmax = ind_max[0 : min(end_max, nbsym + 0)][::-1]
                lmin = ind_min[1 : min(end_min, nbsym + 1)][::-1]
                lsym = ind_min[0]
            else:
                lmax = np.append(ind_max[0 : min(end_max, nbsym - 1)][::-1], 0)
                lmin = ind_min[0 : min(end_min, nbsym)][::-1]
                lsym = 0

        ####################################
        # Right bound - mirror nbsym points to the right
        if ind_max[-1] < ind_min[-1]:
            if S[-1] < S[ind_max[-1]]:
                rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
                rmin = ind_min[max(end_min - nbsym - 1, 0) : -1][::-1]
                rsym = ind_min[-1]
            else:
                rmax = np.append(ind_max[max(end_max - nbsym + 1, 0) :], len(S) - 1)[::-1]
                rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
                rsym = len(S) - 1
        else:
            if S[-1] > S[ind_min[-1]]:
                rmax = ind_max[max(end_max - nbsym - 1, 0) : -1][::-1]
                rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
                rsym = ind_max[-1]
            else:
                rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
                rmin = np.append(ind_min[max(end_min - nbsym + 1, 0) :], len(S) - 1)[::-1]
                rsym = len(S) - 1

        # In case any array missing
        if lmin.size == 0:
            lmin = ind_min
        if rmin.size == 0:
            rmin = ind_min
        if lmax.size == 0:
            lmax = ind_max
        if rmax.size == 0:
            rmax = ind_max

        # Mirror points
        tlmin = 2 * T[lsym] - T[lmin]
        tlmax = 2 * T[lsym] - T[lmax]
        trmin = 2 * T[rsym] - T[rmin]
        trmax = 2 * T[rsym] - T[rmax]

        # If mirrored points are not outside passed time range.
        if tlmin[0] > T[0] or tlmax[0] > T[0]:
            if lsym == ind_max[0]:
                lmax = ind_max[0 : min(end_max, nbsym)][::-1]
            else:
                lmin = ind_min[0 : min(end_min, nbsym)][::-1]

            if lsym == 0:
                raise Exception("Left edge BUG")

            lsym = 0
            tlmin = 2 * T[lsym] - T[lmin]
            tlmax = 2 * T[lsym] - T[lmax]

        if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
            if rsym == ind_max[-1]:
                rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            else:
                rmin = ind_min[max(end_min - nbsym, 0) :][::-1]

            if rsym == len(S) - 1:
                raise Exception("Right edge BUG")

            rsym = len(S) - 1
            trmin = 2 * T[rsym] - T[rmin]
            trmax = 2 * T[rsym] - T[rmax]

        zlmax = S[lmax]
        zlmin = S[lmin]
        zrmax = S[rmax]
        zrmin = S[rmin]

        tmin = np.append(tlmin, np.append(T[ind_min], trmin))
        tmax = np.append(tlmax, np.append(T[ind_max], trmax))
        zmin = np.append(zlmin, np.append(S[ind_min], zrmin))
        zmax = np.append(zlmax, np.append(S[ind_max], zrmax))

        max_extrema = np.array([tmax, zmax])
        min_extrema = np.array([tmin, zmin])

        # Make double sure, that each extremum is significant
        max_dup_idx = np.where(max_extrema[0, 1:] == max_extrema[0, :-1])
        max_extrema = np.delete(max_extrema, max_dup_idx, axis=1)
        min_dup_idx = np.where(min_extrema[0, 1:] == min_extrema[0, :-1])
        min_extrema = np.delete(min_extrema, min_dup_idx, axis=1)

        return max_extrema, min_extrema

    def spline_points(self, T: np.ndarray, extrema: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs spline over given points.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        extrema : numpy array
            Position (1st row) and values (2nd row) of points.

        Returns
        -------
        T : numpy array
            Position array (same as input).
        spline : numpy array
            Spline array over given positions T.
        """

        kind = self.spline_kind.lower()
        t = T[np.r_[T >= extrema[0, 0]] & np.r_[T <= extrema[0, -1]]]

        if kind == "akima":
            return t, akima(extrema[0], extrema[1], t)

        elif kind == "cubic":
            if extrema.shape[1] > 3:
                return t, cubic(extrema[0], extrema[1], t)
            else:
                return cubic_spline_3pts(extrema[0], extrema[1], t)

        elif kind == "pchip":
            return t, pchip(extrema[0], extrema[1], t)

        elif kind == "cubic_hermite":
            return t, cubic_hermite(extrema[0], extrema[1], t)

        elif kind in ["slinear", "quadratic", "linear"]:
            return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)

        else:
            raise ValueError("No such interpolation method!")

    @staticmethod
    def _not_duplicate(S: np.ndarray) -> np.ndarray:
        """
        Returns indices for not repeating values, where there is no extremum.

        Example
        -------
        >>> S = [0, 1, 1, 1, 2, 3]
        >>> idx = self._not_duplicate(S)
        [0, 1, 3, 4, 5]
        """
        dup = np.r_[S[1:-1] == S[0:-2]] & np.r_[S[1:-1] == S[2:]]
        not_dup_idx = np.arange(1, len(S) - 1)[~dup]

        idx = np.empty(len(not_dup_idx) + 2, dtype=np.int64)
        idx[0] = 0
        idx[-1] = len(S) - 1
        idx[1:-1] = not_dup_idx

        return idx

    def find_extrema(self, T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
        """
        Returns extrema (minima and maxima) for given signal S.
        Detection and definition of the extrema depends on
        ``extrema_detection`` variable, set on initiation of EMD.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        S : numpy array
            Input data S(T).

        Returns
        -------
        local_max_pos : numpy array
            Position of local maxima.
        local_max_val : numpy array
            Values of local maxima.
        local_min_pos : numpy array
            Position of local minima.
        local_min_val : numpy array
            Values of local minima.
        """
        if self.extrema_detection == "parabol":
            return self._find_extrema_parabol(T, S)
        elif self.extrema_detection == "simple":
            return self._find_extrema_simple(T, S)
        else:
            raise ValueError("Incorrect extrema detection type. Please try: 'simple' or 'parabol'.")

    def _find_extrema_parabol(self, T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
        """
        Performs parabolic estimation of extremum, i.e. an extremum is a peak of parabolic spanned on 3 consecutive points,
        where the mid-point is the closest.

        See :meth:`EMD.find_extrema()`.
        """
        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1 * S2 < 0)[0]
        if np.any(S == 0):
            indz = np.nonzero(S == 0)[0]
            if np.any(np.diff(indz) == 1):
                zer = S == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0] - 1
                indz = np.round((debz + finz) / 2.0)

            indzer = np.sort(np.append(indzer, indz))

        dt = float(T[1] - T[0])
        # scale = 2.0 * dt * dt

        idx = self._not_duplicate(S)
        T = T[idx]
        S = S[idx]

        # p - previous
        # 0 - current
        # n - next
        Tp, T0, Tn = T[:-2], T[1:-1], T[2:]
        Sp, S0, Sn = S[:-2], S[1:-1], S[2:]
        # a = Sn + Sp - 2*S0
        # b = 2*(Tn+Tp)*S0 - ((Tn+T0)*Sp+(T0+Tp)*Sn)
        # c = Sp*T0*Tn -2*Tp*S0*Tn + Tp*T0*Sn
        TnTp, T0Tn, TpT0 = Tn - Tp, T0 - Tn, Tp - T0
        scale = Tp * Tn * Tn + Tp * Tp * T0 + T0 * T0 * Tn - Tp * Tp * Tn - Tp * T0 * T0 - T0 * Tn * Tn

        a = T0Tn * Sp + TnTp * S0 + TpT0 * Sn
        b = (S0 - Sn) * Tp**2 + (Sn - Sp) * T0**2 + (Sp - S0) * Tn**2
        c = T0 * Tn * T0Tn * Sp + Tn * Tp * TnTp * S0 + Tp * T0 * TpT0 * Sn

        a = a / scale
        b = b / scale
        c = c / scale
        a[a == 0] = 1e-14
        tVertex = -0.5 * b / a
        idx = np.r_[tVertex < T0 + 0.5 * (Tn - T0)] & np.r_[tVertex >= T0 - 0.5 * (T0 - Tp)]

        a, b, c = a[idx], b[idx], c[idx]
        tVertex = tVertex[idx]
        sVertex = a * tVertex * tVertex + b * tVertex + c

        local_max_pos, local_max_val = tVertex[a < 0], sVertex[a < 0]
        local_min_pos, local_min_val = tVertex[a > 0], sVertex[a > 0]

        return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer

    @staticmethod
    def _find_extrema_simple(T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
        """
        Performs extrema detection, where extremum is defined as a point,
        that is above/below its neighbours.

        See :meth:`EMD.find_extrema`.
        """

        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1 * S2 < 0)[0]
        if np.any(S == 0):
            indz = np.nonzero(S == 0)[0]
            if np.any(np.diff(indz) == 1):
                zer = S == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0] - 1
                indz = np.round((debz + finz) / 2.0)

            indzer = np.sort(np.append(indzer, indz))

        # Finds local extrema
        d = np.diff(S)
        d1, d2 = d[:-1], d[1:]
        indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
        indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1

        # When two or more points have the same value
        if np.any(d == 0):
            imax, imin = [], []

            bad = d == 0
            dd = np.diff(np.append(np.append(0, bad), 0))
            debs = np.nonzero(dd == 1)[0]
            fins = np.nonzero(dd == -1)[0]
            if len(debs) > 0 and debs[0] == 1:
                if len(debs) > 1:
                    debs, fins = debs[1:], fins[1:]
                else:
                    debs, fins = [], []

            if len(debs) > 0:
                if fins[-1] == len(S) - 1:
                    if len(debs) > 1:
                        debs, fins = debs[:-1], fins[:-1]
                    else:
                        debs, fins = [], []

            lc = len(debs)
            if lc > 0:
                for k in range(lc):
                    if d[debs[k] - 1] > 0:
                        if d[fins[k]] < 0:
                            imax.append(np.round((fins[k] + debs[k]) / 2.0))
                    else:
                        if d[fins[k]] > 0:
                            imin.append(np.round((fins[k] + debs[k]) / 2.0))

            if len(imax) > 0:
                indmax = indmax.tolist()
                for x in imax:
                    indmax.append(int(x))
                indmax.sort()

            if len(imin) > 0:
                indmin = indmin.tolist()
                for x in imin:
                    indmin.append(int(x))
                indmin.sort()

        local_max_pos = T[indmax]
        local_max_val = S[indmax]
        local_min_pos = T[indmin]
        local_min_val = S[indmin]

        return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer

    def end_condition(self, S: np.ndarray, IMF: np.ndarray) -> bool:
        """Tests for end condition of whole EMD. The procedure will stop if:

        * Absolute amplitude (max - min) is below *range_thr* threshold, or
        * Metric L1 (mean absolute difference) is below *total_power_thr* threshold.

        Parameters
        ----------
        S : numpy array
            Original signal on which EMD was performed.
        IMF : numpy 2D array
            Set of IMFs where each row is IMF. Their order is not important.

        Returns
        -------
        end : bool
            Whether sifting is finished.
        """
        # When to stop EMD
        tmp = S - np.sum(IMF, axis=0)

        if np.max(tmp) - np.min(tmp) < self.range_thr:
            self.logger.debug("FINISHED -- RANGE")
            return True

        if np.sum(np.abs(tmp)) < self.total_power_thr:
            self.logger.debug("FINISHED -- SUM POWER")
            return True

        return False

    def check_imf(
        self,
        imf_new: np.ndarray,
        imf_old: np.ndarray,
        eMax: np.ndarray,
        eMin: np.ndarray,
    ) -> bool:
        """
        Huang criteria for **IMF** (similar to Cauchy convergence test).
        Signal is an IMF if consecutive siftings do not affect signal
        in a significant manner.
        """
        # local max are >0 and local min are <0
        if np.any(eMax[1] < 0) or np.any(eMin[1] > 0):
            return False

        # Convergence
        if np.sum(imf_new**2) < 1e-10:
            return False

        # Precompute values
        imf_diff = imf_new - imf_old
        imf_diff_sqrd_sum = np.sum(imf_diff * imf_diff)

        # Scaled variance test
        svar = imf_diff_sqrd_sum / (max(imf_old) - min(imf_old))
        if svar < self.svar_thr:
            self.logger.debug("Scaled variance -- PASSED")
            return True

        # Standard deviation test
        std = np.sum((imf_diff / imf_new) ** 2)
        if std < self.std_thr:
            self.logger.debug("Standard deviation -- PASSED")
            return True

        energy_ratio = imf_diff_sqrd_sum / np.sum(imf_old * imf_old)
        if energy_ratio < self.energy_ratio_thr:
            self.logger.debug("Energy ratio -- PASSED")
            return True

        return False

    @staticmethod
    def _common_dtype(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Casts inputs (x, y) into a common numpy DTYPE."""
        dtype = np.result_type(x.dtype, y.dtype)
        if x.dtype != dtype:
            x = x.astype(dtype)
        if y.dtype != dtype:
            y = y.astype(dtype)
        return x, y

    @staticmethod
    def _normalize_time(t: np.ndarray) -> np.ndarray:
        """
        Normalize time array so that it doesn't explode on tiny values.
        Returned array starts with 0 and the smallest increase is by 1.
        """
        d = np.diff(t)
        assert np.all(d != 0), "All time domain values needs to be unique"
        return (t - t[0]) / np.min(d)

    def emd(self,
            S: np.ndarray,
            T: Optional[np.ndarray] = None,
            max_imf: int = -1,
            ) -> np.ndarray:
        """
        Performs Empirical Mode Decomposition on signal S.
        The decomposition is limited to *max_imf* imfs.
        Returns IMF functions and residue in a single numpy array format.

        Parameters
        ----------
        S : numpy array,
            Input signal.
        T : numpy array, (default: None)
            Position or time array. If None is passed or self.extrema_detection == "simple",
            then numpy range is created.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.

        Returns
        -------
        IMFs and residue : numpy array
            A numpy array which cointains both the IMFs and residual, if any, appended as
            the last slice.
        """
        if T is not None and len(S) != len(T):
            raise ValueError(f"Time series have different sizes: len(S) -> {len(S)} != {len(T)} <- len(T)")

        if T is None or self.extrema_detection == "simple":
            T = get_timeline(len(S), S.dtype)

        # T_orig_dtype = T.dtype
        T = self._normalize_time(T)  # Normalize T for stability

        S_orig_dtype = S.dtype
        S, T = self._common_dtype(S, T)
        self.DTYPE = S.dtype
        N = len(S)

        residue = S.astype(self.DTYPE)
        imf = np.zeros(N, dtype=self.DTYPE)
        imf_old = np.nan  # Use np.nan for clearer uninitialized state

        imfNo = 0
        IMF_list: List[np.ndarray] = []  # Use a list to append IMFs

        while imfNo != max_imf:
            self.logger.debug(f"IMF -- {imfNo}")

            current_signal_for_sifting = residue.copy()  # Sift on the current residue
            imf = current_signal_for_sifting.copy()

            mean_env = np.zeros(N, dtype=self.DTYPE)  # mean envelope

            # Sifting process for the current IMF
            for k_sift in range(self.MAX_ITERATION):
                ext_res = self.find_extrema(T, imf)
                max_pos, max_val, min_pos, min_val, indzer = ext_res

                num_extrema = len(max_pos) + len(min_pos)

                if num_extrema < 3:  # Not enough extrema to form envelopes
                    self.logger.debug(f"Not enough extrema ({num_extrema}) to continue sifting for IMF {imfNo}.")
                    if imfNo == 0 and num_extrema < 1:  # No imfs if no extrema in original signal
                        self.imfs = np.empty((0, N), dtype=self.DTYPE)
                        self.residue = S.copy().astype(S_orig_dtype)
                        return self.residue.reshape(1, N) if self.residue.ndim == 1 else self.residue
                    break  # Break sifting loop

                max_env, min_env, eMax, eMin = self.extract_max_min_spline(T, imf)

                # Handle cases where spline fitting might fail at edges
                if not isinstance(max_env, np.ndarray) or not isinstance(min_env, np.ndarray):
                    self.logger.warning("Spline extraction failed, stopping sifting for this IMF.")
                    break

                mean_env[:] = 0.5 * (max_env + min_env)

                imf_old = imf.copy()
                imf -= mean_env

                # Check IMF stopping criteria (Huang criteria)
                if self.FIXE > 0 and k_sift + 1 >= self.FIXE:
                    break
                if self.FIXE_H > 0:  # Simplified for now, original logic complex
                    num_zero_crossings = len(indzer)
                    if abs(num_extrema - num_zero_crossings) <= 1:
                        if k_sift + 1 >= self.FIXE_H: break  # TODO: Original FIXE_H logic is more involved with n_h counter

                if k_sift > 0 and self.check_imf(imf, imf_old, eMax,
                                                 eMin):  # eMax, eMin from previous iteration might be more stable
                    break
            else:  # max_iterations reached
                self.logger.info(f"Max sifting iterations ({self.MAX_ITERATION}) reached for IMF {imfNo}.")

            IMF_list.append(imf.copy())
            imfNo += 1

            residue -= imf  # Update residue for next IMF extraction

            # Check overall EMD stopping criteria
            # 1. Max IMF count reached
            if max_imf != -1 and imfNo >= max_imf:
                break
            # 2. Residue is monotonic or has too few extrema to continue
            ext_res_residue = self.find_extrema(T, residue)
            if len(ext_res_residue[0]) + len(ext_res_residue[2]) < 3:
                self.logger.debug("Residue is monotonic or has too few extrema.")
                break
            # 3. Residue's power or range is too small (already part of self.end_condition)
            # Simplified: check if sum(abs(residue)) is small relative to original signal power
            if np.sum(np.abs(residue)) < self.total_power_thr * np.sum(np.abs(S)):  # Compare with original S power
                self.logger.debug("Residue power too small.")
                break

        self.imfs = np.array(IMF_list, dtype=S_orig_dtype) if IMF_list else np.empty((0, N), dtype=S_orig_dtype)
        self.residue = residue.astype(S_orig_dtype)

        # Combine IMFs and residue for output
        if not np.allclose(self.residue, 0,
                           atol=self.range_thr * np.max(np.abs(S)) if np.max(np.abs(S)) > 0 else self.range_thr):
            # Only add residue if it's significant
            output_IMFs = np.vstack((self.imfs, self.residue))
        else:
            output_IMFs = self.imfs

        return output_IMFs if output_IMFs.size > 0 else np.array([S.astype(S_orig_dtype)])

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal.

        Returns
        -------
        imfs : np.ndarray
            Obtained IMFs
        residue : np.ndarray
            Residue.

        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.imfs, self.residue

    def get_imfs_and_trend(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and trend from recently analysed signal.
        Note that this may differ from the `get_imfs_and_residue` as the trend isn't
        necessarily the residue. Residue is a point-wise difference between input signal
        and all obtained components, whereas trend is the slowest component (can be zero).

        Returns
        -------
        imfs : np.ndarray
            Obtained IMFs
        trend : np.ndarray
            The main trend.

        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")

        imfs, residue = self.get_imfs_and_residue()
        if np.allclose(residue, 0):
            return imfs[:-1].copy(), imfs[-1].copy()
        else:
            return imfs, residue


class EEMD:
    """
    **Ensemble Empirical Mode Decomposition**

    Ensemble empirical mode decomposition (EEMD) [Wu2009]_
    is noise-assisted technique, which is meant to be more robust
    than simple Empirical Mode Decomposition (EMD). The robustness is
    checked by performing many decompositions on signals slightly
    perturbed from their initial position. In the grand average over
    all IMF results the noise will cancel each other out and the result
    is pure decomposition.

    Parameters
    ----------
    trials : int (default: 100)
        Number of trials or EMD performance with added noise.
    noise_width : float (default: 0.05)
        Standard deviation of Gaussian noise (:math:`\hat\sigma`).
        It's relative to absolute amplitude of the signal, i.e.
        :math:`\hat\sigma = \sigma\cdot|\max(S)-\min(S)|`, where
        :math:`\sigma` is noise_width.
    ext_EMD : EMD (default: None)
        One can pass EMD object defined outside, which will be
        used to compute IMF decompositions in each trial. If none
        is passed then EMD with default options is used.
    parallel : bool (default: False)
        Flag whether to use multiprocessing in EEMD execution.
        Since each EMD(s+noise) is independent this should improve execution
        speed considerably.
        *Note* that it's disabled by default because it's the most common
        problem when EEMD takes too long time to finish.
        If you set the flag to True, make also sure to set `processes` to
        some reasonable value.
    processes : int or None (optional)
        Number of processes harness when executing in parallel mode.
        The value should be between 1 and max that depends on your hardware.
    separate_trends : bool (default: False)
        Flag whether to isolate trends from each EMD decomposition into a separate component.
        If `true`, the resulting EEMD will contain ensemble only from IMFs and
        the mean residue will be stacked as the last element.

    References
    ----------
    .. [Wu2009] Z. Wu and N. E. Huang, "Ensemble empirical mode decomposition:
        A noise-assisted data analysis method", Advances in Adaptive
        Data Analysis, Vol. 1, No. 1 (2009) 1-41.
    """

    logger = logging.getLogger(__name__)

    noise_kinds_all = ["normal", "uniform"]

    def __init__(self,
                 trials: int = 100,
                 noise_width: float = 0.05,
                 ext_EMD: Optional[EMD] = None,
                 parallel: bool = False,
                 processes: Optional[int] = None,
                 **kwargs,
                 ):

        # Ensemble constants
        self._S: Optional[np.ndarray] = None
        self._T: Optional[np.ndarray] = None
        self._N: Optional[int] = None
        self._scale: Optional[float] = None
        self.max_imf: Optional[int] = None

        self.trials = trials
        self.noise_width = noise_width
        self.separate_trends = bool(kwargs.get("separate_trends", False))

        self.random = np.random.RandomState(kwargs.get("seed")) # Allow seed to be passed
        self.noise_kind = kwargs.get("noise_kind", "normal")
        self.parallel = parallel
        self.processes = processes
        if self.processes is not None and not self.parallel:
            self.logger.warning("Passed value for process has no effect when `parallel` is False.")

        self.EMD = ext_EMD if ext_EMD is not None else EMD(**kwargs)
        self.E_IMF: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None
        self._all_imfs: Dict[int, List[np.ndarray]] = {}

    def __call__(self,
                 S: np.ndarray,
                 T: Optional[np.ndarray] = None,
                 max_imf: int = -1,
                 progress: bool = False,
                 ) -> np.ndarray:

        return self.eemd(S, T=T, max_imf=max_imf, progress=progress)

    def __getstate__(self) -> Dict:
        self_dict = self.__dict__.copy()
        return self_dict

    def generate_noise(self, scale: float, size: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Generate noise with specified parameters.
        Currently supported distributions are:

        * *normal* with std equal scale.
        * *uniform* with range [-scale/2, scale/2].

        Parameters
        ----------
        scale : float
            Width for the distribution.
        size : int
            Number of generated samples.

        Returns
        -------
        noise : numpy array
            Noise sampled from selected distribution.
        """
        if self.noise_kind == "normal":
            noise = self.random.normal(loc=0, scale=scale, size=size)
        elif self.noise_kind == "uniform":
            noise = self.random.uniform(low=-scale / 2, high=scale / 2, size=size)
        else:
            raise ValueError(
                "Unsupported noise kind. Please assigned `noise_kind` to be one of these: {0}".format(
                    str(self.noise_kinds_all)
                )
            )
        return noise

    def noise_seed(self, seed: int) -> None:
        """Set seed for noise generation."""
        self.random.seed(seed)

    def eemd(self,
             S: np.ndarray,
             T: Optional[np.ndarray] = None,
             max_imf: int = -1,
             progress: bool = False,
             ) -> np.ndarray:
        """
        Performs EEMD on provided signal.

        For a large number of iterations defined by `trials` attr
        the method performs :py:meth:`emd` on a signal with added white noise.

        Parameters
        ----------
        S : numpy array,
            Input signal on which EEMD is performed.
        T : numpy array or None, (default: None)
            If none passed samples are numerated.
        max_imf : int, (default: -1)
            Defines up to how many IMFs each decomposition should
            be performed. By default, (negative value) it decomposes
            all IMFs.
        progress: bool

        Returns
        -------
        eIMF : numpy array
            Set of ensemble IMFs produced from input signal. In general,
            these do not have to be, and most likely will not be, same as IMFs
            produced using EMD.
        """
        if T is None:
            T = get_timeline(len(S), S.dtype)

        scale = self.noise_width * np.abs(np.max(S) - np.min(S))
        self._S = S
        self._T = T
        self._N = len(S)
        self._scale = scale
        self.max_imf = max_imf

        # For trial number of iterations perform EMD on a signal
        # with added white noise
        if self.parallel:
            pool = Pool(processes=self.processes)
            map_pool = pool.map
        else:
            map_pool = map

        all_IMFs = map_pool(self._trial_update, range(self.trials))

        if self.parallel:
            pool.close()

        self._all_imfs = defaultdict(list)
        it = iter if not progress else lambda x: tqdm(x, desc="EEMD", total=self.trials)
        for imfs, trend in it(all_IMFs):
            # A bit of explanation here.
            # If the `trend` is not None, that means it was intentionally separated in the decomp process.
            # This might due to `separate_trends` flag which means that trends are summed up separately
            # and treated as the last component. Since `proto_eimfs` is a dict, that `-1` is treated literally
            # and **not** as the *last position*. We can then use that `-1` to always add it as the last pos
            # in the actual eIMF, which indicates the trend.
            if trend is not None:
                self._all_imfs[-1].append(trend)

            for imf_num, imf in enumerate(imfs):
                self._all_imfs[imf_num].append(imf)

        # Convert default dict back to dict and explicitly rename `-1` position to be {the last value} for consistency.
        self._all_imfs = dict(self._all_imfs)
        if -1 in self._all_imfs:
            self._all_imfs[len(self._all_imfs)] = self._all_imfs.pop(-1)

        for imf_num in self._all_imfs.keys():
            self._all_imfs[imf_num] = np.array(self._all_imfs[imf_num])

        self.E_IMF = self.ensemble_mean()
        self.residue = S - np.sum(self.E_IMF, axis=0)

        return self.E_IMF

    def _trial_update(self, trial) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """A single trial evaluation, i.e. EMD(signal + noise).

        *Note*: Although `trial` argument isn't used it's needed for the (multiprocessing) map method.
        """
        noise = self.generate_noise(self._scale, self._N)
        imfs = self.emd(self._S + noise, self._T, self.max_imf)
        trend = None
        if self.separate_trends:
            imfs, trend = self.EMD.get_imfs_and_trend()

        return imfs, trend

    def emd(self, S: np.ndarray, T: np.ndarray, max_imf: int = -1) -> np.ndarray:
        """Vanilla EMD method.

        Provides emd evaluation from provided EMD class.
        For reference please see :class:`PyEMD.EMD`.
        """
        return self.EMD.emd(S, T, max_imf=max_imf)

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal.

        Returns
        -------
        (imfs, residue) : (np.ndarray, np.ndarray)
            Tuple that contains all imfs and a residue (if any).

        """
        if self.E_IMF is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.E_IMF, self.residue

    @property
    def all_imfs(self):
        """A dictionary with all computed imfs per given order."""
        return self._all_imfs

    def ensemble_count(self) -> List[int]:
        """Count of imfs observed for given order, e.g. 1st proto-imf, in the whole ensemble."""
        return [len(imfs) for imfs in self._all_imfs.values()]

    def ensemble_mean(self) -> np.ndarray:
        """Pointwise mean over computed ensemble. Same as the output of `eemd()` method."""
        return np.array([imfs.mean(axis=0) for imfs in self._all_imfs.values()])

    def ensemble_std(self) -> np.ndarray:
        """Pointwise standard deviation over computed ensemble."""
        return np.array([imfs.std(axis=0) for imfs in self._all_imfs.values()])


class CEEMDAN:
    """
    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN).

    This class implements the CEEMDAN algorithm for signal decomposition.

    Parameters
    ----------
    trials : int, optional
        Number of trials for the ensemble, default is 100.
    epsilon : float, optional
        Scale factor for the added noise, default is 0.005.
    ext_emd : EMD or None, optional
        An instance of the EMD class to use for decomposition. If None, a default EMD instance is created.
    parallel : bool, optional
        Whether to use parallel processing, default is False.
    processes : int or None, optional
        Number of processes to use if parallel is True. Defaults to the number of CPUs.
    noise_kind : str, optional
        Type of noise to add ('normal' or 'uniform'), default is 'normal'.
    range_thr : float, optional
        Threshold for the range of the residue to stop decomposition, default is 0.01.
    total_power_thr : float, optional
        Threshold for the total power of the residue to stop decomposition, default is 0.05.
    seed : int or None, optional
        Random seed for noise generation.

    Attributes
    ----------
    C_IMF : np.ndarray
        Array containing the extracted IMFs.
    residue : np.ndarray
        Residue after decomposition.

    References
    ----------

    [Torres2011] M.E. Torres, M.A. Colominas, G. Schlotthauer, P. Flandrin
        A complete ensemble empirical mode decomposition with adaptive noise.
        Acoustics, Speech and Signal Processing (ICASSP), 2011, pp. 4144--4147

    [Colominas2014] M.A. Colominas, G. Schlotthauer, M.E. Torres,
        Improved complete ensemble EMD: A suitable tool for biomedical signal
        processing, In Biomed. Sig. Proc. and Control, V. 14, 2014, pp. 19--29
    """

    logger = logging.getLogger(__name__)

    noise_kinds_all = ["normal", "uniform"]

    def __init__(self,
                 trials: int = 200,
                 epsilon: float = 0.005,
                 ext_emd: EMD = None,
                 parallel: bool = False,
                 seed: Optional[int] = None,
                 **kwargs,
                 ):

        self.all_noises = None
        self.all_noise_emd = None
        self.trials = trials
        self.epsilon = epsilon
        self.range_thr = float(kwargs.get("range_thr", 0.01))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.05))

        self.beta_progress = bool(kwargs.get("beta_progress", True))  # Scale noise by std
        self.random = np.random.RandomState(seed=kwargs.get("seed"))
        self.noise_kind = kwargs.get("noise_kind", "normal")
        self.noise_scale = float(kwargs.get("noise_scale", 1.0))

        self._max_imf = int(kwargs.get("max_imf", 100))
        self.parallel = parallel
        self.processes = kwargs.get("processes")  # Optional[int]

        if self.processes is not None and not self.parallel:
            self.logger.warning("Passed value for process has no effect when `parallel` is False.")

        self.all_noise_emd = []

        if ext_emd is None:
            self.EMD = EEMD(**kwargs)
        else:
            self.EMD = ext_emd

        # Internal variables
        self.C_IMF: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None
        self.seed = seed
        self.noise_seed(seed)

    def __call__(self,
                 S: np.ndarray,
                 T: Optional[np.ndarray] = None,
                 max_imf: int = -1,
                 progress: bool = False,
                 ) -> np.ndarray:

        return self.ceemdan(S, T=T, max_imf=max_imf, progress=progress)

    def __getstate__(self) -> Dict:
        self_dict = self.__dict__.copy()
        if "pool" in self_dict:
            del self_dict["pool"]
        return self_dict

    def _generate_noise(self,
                        size: Union[int, Sequence[int]],
                        ) -> np.ndarray:
        """
        Generate noise for all trials.

        * *normal* with std equal scale.
        * *uniform* with range [-scale/2, scale/2].

        return np.array(local_means)

        scale : float
            Width for the distribution.
        size : int or shape
            Shape of the noise that is added. In case of `int` an array of that len is generated.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the IMFs and the residue.
        """

        if self.noise_kind == "normal":
            noise = self.random.normal(loc=0., scale=1., size=size)
        elif self.noise_kind == "uniform":
            noise = self.random.uniform(low=-0.5, high=0.5, size=size)
        else:
            raise ValueError(
                "Unsupported noise kind. Please assigned `noise_kind` to be one of these: {0}".format(
                    str(self.noise_kinds_all)
                )
            )

        return noise

    def noise_seed(self, seed: int = None) -> None:
        """Set seed for noise generation."""

        if seed is None:
            self.random.seed(self.seed)
        else:
            self.random.seed(seed)

    def ceemdan(self,
                S: np.ndarray,
                T: Optional[np.ndarray] = None,
                max_imf: int = -1,
                progress: bool = False,
                ) -> np.ndarray:
        """Perform CEEMDAN decomposition.

        Parameters
        ----------
        S : numpy array
            Original signal on which CEEMDAN is to perform.
        T : Optional(numpy array) (default: None)
            Time (x) values for the signal. If not passed, i.e. `T = None`, then assumes equidistant values.
        max_imf : int (default: -1)
            Maximum number of components to extract.
        progress : bool (default: False)
            Whether to print out '.' every 1s to indicate progress.

        Returns
        -------
        components : np.ndarray
            CEEMDAN components.
        """
        if np.all(S == 0):
            # Return an array of zeros with shape (1, len(S))
            self.C_IMF = np.zeros((1, len(S)))
            self.residue = np.zeros_like(S)
            return self.C_IMF

        if T is not None and len(S) != len(T):
            raise ValueError("Time series have different sizes: len(input Signal) -> {} != {} <- len(T)".format(len(S), len(T)))

        scale_s = np.std(S)
        if scale_s == 0.:
            scale_s = 1  # Avoid division by zero

        # ######################################################################################
        # main code
        S = S / scale_s
        res_i = np.zeros_like(S)

        # Define all noise
        self.all_noises = self.noise_scale * self._generate_noise((self.trials, S.size))

        # Decompose all noise and remember 1st's std
        self.logger.debug("Decomposing all noises")
        self.all_noise_emd = self._decompose_noise()

        list_cimfs = []

        self.logger.debug("Starting CEEMDAN")
        total = (max_imf - 1) if max_imf != -1 else None
        it = iter if not progress else lambda x: tqdm(x, desc="cIMF decomposition", total=total)

        for i in it(range(self._max_imf)):

            if i == 0:
                # Create first IMF
                last_imf_i = self._eemd(S, T, max_imf=1, progress=progress)[0]
                list_cimfs.append(last_imf_i)
                res_i = S - last_imf_i

            else:
                imf_i = len(list_cimfs)
                beta_i = self.epsilon * np.std(res_i)
                np_mean_i = np.zeros_like(S)

                for trial_k in range(self.trials):
                    # Skip if noise[trial] didn't have k'th mode
                    noise_imf_k = self.all_noise_emd[trial_k]
                    res_k = res_i.copy()
                    if len(noise_imf_k) > imf_i:
                        res_k += beta_i * noise_imf_k[imf_i]

                    # Extract local mean, which is at 2nd position
                    np_imfs_k = self.emd(res_k, T, max_imf=1)
                    np_mean_i += np_imfs_k[-1] / self.trials

                last_imf_i = res_i - np_mean_i
                list_cimfs.append(last_imf_i)
                res_i = np_mean_i.copy()

            # Check end condition in the beginning because we've already have 1 IMF
            if self._end_condition(S, np.array(list_cimfs), max_imf):
                self.logger.debug("End Condition - Pass")
                break

        res = S - np.sum(list_cimfs, axis=0)
        list_cimfs.append(res)
        np_cimfs = np.array(list_cimfs) * scale_s

        # Empty all IMFs noise
        self.all_noise_emd = None

        self.C_IMF = np_cimfs
        self.residue = S * scale_s - np.sum(self.C_IMF, axis=0)

        return self.C_IMF

    def _end_condition(self,
                       S: np.ndarray,
                       cIMFs: np.ndarray,
                       max_imf: int,
                       ) -> bool:
        """
        Test for end condition of CEEMDAN.

        Procedure stops if:

        * number of components reach provided `max_imf`, or
        * last component is close to being pure noise (range or power), or
        * set of provided components reconstructs sufficiently input.

        Parameters
        ----------
        S : numpy array
            Original signal on which CEEMDAN was performed.
        cIMFs : List of numpy 1D array
            Set of cIMFs where each row is cIMF.
        max_imf : int
            The maximum number of imfs to extract.

        Returns
        -------
        end : bool
            Whether to stop CEEMDAN.
        """
        imf_no = cIMFs.shape[0]

        # Check if hit maximum number of cIMFs
        if 0 < max_imf <= imf_no:
            return True

        # Compute EMD on residue
        res = S - np.sum(cIMFs, axis=0)
        _test_imf = self.emd(res, None, max_imf=1)

        # Check if residue is IMF or no extrema
        if _test_imf.shape[0] == 1:
            self.logger.debug("Not enough extrema")
            return True

        # Check for range threshold
        if np.max(res) - np.min(res) < self.range_thr:
            self.logger.debug("FINISHED -- RANGE")
            return True

        # Check for power threshold
        if np.sum(np.abs(res)) < self.total_power_thr:
            self.logger.debug("FINISHED -- SUM POWER")
            return True

        return False

    def _decompose_noise(self) -> List[np.ndarray]:
        if self.parallel:
            pool = Pool(processes=self.processes)
            all_noise_emd = pool.map(self.emd, self.all_noises)
            pool.close()
        else:
            all_noise_emd = [self.emd(noise, max_imf=-1) for noise in self.all_noises]

        # Normalize w/ respect to 1st IMF's std
        if self.beta_progress:
            all_stds = [np.std(imfs[0]) for imfs in all_noise_emd]
            all_noise_emd = [imfs / imfs_std for (imfs, imfs_std) in zip(all_noise_emd, all_stds)]

        return all_noise_emd

    def _eemd(self,
              S: np.ndarray,
              T: Optional[np.ndarray] = None,
              max_imf: int = -1,
              progress: bool = True,
              ) -> np.ndarray:

        if T is None:
            T = np.arange(len(S), dtype=S.dtype)

        self._S = S
        self._T = T
        self._N = N = len(S)
        self.max_imf = max_imf

        # For trial number of iterations perform EMD on a signal with added white noise
        if self.parallel:
            pool = Pool(processes=self.processes)
            map_pool = pool.imap_unordered
        else:  # Not parallel
            map_pool = map

        self.E_IMF = np.zeros((1, N))
        it = iter if not progress else lambda x: tqdm(x, desc="Decomposing noise", total=self.trials)

        for IMFs in it(map_pool(self._trial_update, range(self.trials))):
            if self.E_IMF.shape[0] < IMFs.shape[0]:
                num_new_layers = IMFs.shape[0] - self.E_IMF.shape[0]
                self.E_IMF = np.vstack((self.E_IMF, np.zeros(shape=(num_new_layers, N))))
            self.E_IMF[: IMFs.shape[0]] += IMFs

        if self.parallel:
            pool.close()

        return self.E_IMF / self.trials

    def _trial_update(self,
                      trial: int,
                      ) -> np.ndarray:
        """A single trial evaluation, i.e. EMD(signal + noise)."""

        # Generate noise
        noise = self.epsilon * self.all_noise_emd[trial][0]
        return self.emd(self._S + noise, self._T, self.max_imf)

    def emd(self,
            S: np.ndarray,
            T: Optional[np.ndarray] = None,
            max_imf: int = -1,
            ) -> np.ndarray:
        """Vanilla EMD method.

        Provides emd evaluation from provided EMD class.
        For reference please see :class:`PyEMD.EMD`.
        """
        return self.EMD.emd(S, T, max_imf=max_imf)

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal.
        :return: (imfs, residue)
        """
        if self.C_IMF is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.C_IMF, self.residue


class ICEEMDAN:
    """
    Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (ICEEMDAN).
    This implementation is based on the principles described by Colominas et al. (2014).
    It aims to provide a more robust decomposition by adaptively adding noise derived
    from previous stages of the decomposition.

    Parameters
    ----------
    trials : int, optional
        Number of noise realizations (ensemble size), default is 100.
    epsilon : float, optional
        Coefficient for the amplitude of the added noise (related to std of residue), default is 0.005.
    ext_EMD_eemd : EEMD or None, optional
        An instance of the EEMD class to use for the initial EEMD step and noise decomposition.
        If None, a default EEMD instance (which itself uses a default EMD) is created.
    ext_EMD_direct : EMD or None, optional
        An instance of the EMD class for direct EMD calls on residues + noise IMFs.
        If None, a default EMD instance is created.
    parallel : bool, optional
        Whether to use parallel processing for noise decomposition and EEMD steps, default is False.
    seed : int or None, optional
        Random seed for noise generation for reproducibility.
    noise_kind : str, optional
        Type of noise to add ('normal' or 'uniform'), default is 'normal'.
    noise_scale_factor : float, optional
        Factor to scale the generated unit noise before EMD decomposition. Default is 1.0.
    max_imf_noise_decomp : int, optional
        Maximum IMFs to extract from noise realizations. Default is -1 (all).

    Attributes
    ----------
    C_IMF : np.ndarray
        Array containing the extracted ICEEMDAN components (IMFs).
    residue : np.ndarray
        Residue after decomposition.
    """

    logger = logging.getLogger(__name__)
    noise_kinds_all = ["normal", "uniform"]

    def __init__(self,
                 trials: int = 100,  # Typical value for ICEEMDAN
                 epsilon: float = 0.2,  # Typical value for epsilon_k
                 ext_EMD_eemd: Optional[EEMD] = None,  # For decomposing noise and initial EEMD step
                 ext_EMD_direct: Optional[EMD] = None,  # For EMD on residue + Ek(noise)
                 parallel: bool = False,
                 seed: Optional[int] = None,
                 **kwargs,
                 ):

        self.trials = trials
        self.epsilon = epsilon  # This is beta_k in some notations
        self.range_thr = float(kwargs.get("range_thr", 0.01))  # From EMD
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.05))  # From EMD

        self.random_state = np.random.RandomState(seed)
        self.noise_kind = kwargs.get("noise_kind", "normal")
        # Scale factor for the initial std=1 noise before it's decomposed
        self.noise_scale_factor = float(kwargs.get("noise_scale_factor", 1.0))

        self._max_imf_components = int(kwargs.get("max_imf_components", 100))  # Max IMFs for the signal
        self.parallel = parallel
        self.processes = kwargs.get("processes")

        # EMD instance for EEMD-like steps (decomposing noise, initial EEMD for IMF1)
        emd_config_for_eemd = {k: v for k, v in kwargs.items() if k not in ['trials', 'epsilon', 'seed']}
        self.EMD_for_eemd = ext_EMD_eemd if ext_EMD_eemd is not None else EEMD(trials=self.trials,
                                                                               noise_width=self.epsilon,
                                                                               **emd_config_for_eemd)
        self.EMD_for_eemd.noise_seed(seed)

        # EMD instance for direct sifting of (residue + beta_k * Ek(noise))
        self.EMD_direct = ext_EMD_direct if ext_EMD_direct is not None else EMD(**kwargs)

        # Internal variables
        self.C_IMF: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None
        self.decomposed_noise_imfs: List[np.ndarray] = []  # Stores Ek(w_i) for each trial

    def __call__(self,
                 S: np.ndarray,
                 T: Optional[np.ndarray] = None,
                 max_imf: int = -1,
                 ) -> np.ndarray:
        return self.iceemdan(S, T=T, max_imf=max_imf)

    def __getstate__(self) -> Dict:
        self_dict = self.__dict__.copy()
        if "pool" in self_dict:
            del self_dict["pool"]
        return self_dict

    def _generate_unit_noise(self, size: Union[int, Sequence[int]]) -> np.ndarray:
        """Generates noise with unit variance and zero mean."""
        if self.noise_kind == "normal":
            return self.random_state.normal(loc=0., scale=1., size=size)
        elif self.noise_kind == "uniform":  # Uniform needs scaling to approx unit variance
            noise = self.random_state.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=size)
            return noise
        else:
            raise ValueError(f"Unsupported noise kind: {self.noise_kind}")

    def _get_operator_emd1(self,
                           signal_ensemble: np.ndarray,
                           T_values: Optional[np.ndarray],
                           ) -> np.ndarray:
        """
        Computes M(signal_ensemble) = mean(EMD1(signal_ensemble_trial_i))
        where EMD1 extracts the first IMF from each trial.
        This is equivalent to taking the mean of the first modes of an EEMD-like decomposition.
        """
        if self.parallel:
            # Each trial (row in signal_ensemble) is EMDed to get its first IMF
            def emd_first_mode(s_trial):
                # Ensure EMD_direct does not run in parallel itself if this is already in a pool
                # Assuming EMD_direct is configured for serial execution here
                imfs_trial = self.EMD_direct.emd(s_trial, T_values, max_imf=1)
                return imfs_trial[0] if imfs_trial.shape[0] > 0 else np.zeros_like(s_trial)

            with Pool(processes=self.processes) as pool:
                first_modes_ensemble = pool.map(emd_first_mode, signal_ensemble)
        else:
            first_modes_ensemble = []
            for s_trial in signal_ensemble:
                imfs_trial = self.EMD_direct.emd(s_trial, T_values, max_imf=1)
                first_modes_ensemble.append(imfs_trial[0] if imfs_trial.shape[0] > 0 else np.zeros_like(s_trial))

        return np.mean(np.array(first_modes_ensemble), axis=0)

    def iceemdan(self,
                 S: np.ndarray,
                 T: Optional[np.ndarray] = None,
                 max_imf: int = -1,
                 ) -> np.ndarray:
        """
        Performs Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (ICEEMDAN).

        Parameters
        ----------
        S : numpy array
            Input signal.
        T : numpy array, optional
            Time array corresponding to S. If None, assumes equidistant samples.
        max_imf : int, optional
            Maximum number of IMFs to extract. If -1, extracts until stopping criteria are met.
            Defaults to -1.

        Returns
        -------
        np.ndarray
            Array of extracted IMFs, with the residue as the last row if significant.
        """
        self.logger.info("Starting ICEEMDAN decomposition.")
        N = S.shape[0]
        if T is None:
            T = np.arange(N, dtype=S.dtype)

        # --- Step 1: Initializations ---
        original_std = np.std(S)
        if original_std == 0:  # Handle zero or constant signal
            self.C_IMF = np.zeros((1, N), dtype=S.dtype)
            self.residue = S.copy()
            return np.vstack((self.C_IMF, self.residue))

        # Normalize signal S
        s_normalized = S / original_std

        k = 0  # IMF index
        current_residue = s_normalized.copy()
        extracted_imfs_list: List[np.ndarray] = []

        # --- Step 2: Generate and decompose noise realizations E_k(w^(i)) ---
        self.logger.info(f"Generating and decomposing {self.trials} noise realizations...")
        white_noise_realizations = self._generate_unit_noise(size=(self.trials, N)) * self.noise_scale_factor

        # Decompose each noise realization w^(i) to get E_k(w^(i))
        # This is a list of arrays, where each array contains IMFs for one noise trial
        self.decomposed_noise_imfs = []
        if self.parallel:
            with Pool(processes=self.processes) as pool:
                self.decomposed_noise_imfs = pool.map(
                    lambda noise_trial: self.EMD_direct.emd(noise_trial, T, max_imf=-1), white_noise_realizations)
        else:
            for i in tqdm(range(self.trials), desc="Decomposing noise realizations"):
                self.decomposed_noise_imfs.append(self.EMD_direct.emd(white_noise_realizations[i], T, max_imf=-1))

        # --- Step 3: Main ICEEMDAN loop ---
        while k < (self._max_imf_components if max_imf == -1 else max_imf):
            self.logger.info(f"Extracting IMF {k + 1}...")

            # --- Step 3a: Calculate < M(r_k + beta_k * E_{k+1}(w^(i))) > ---
            # For k=0, beta_0 * E_1(w^(i)) is added. For k>0, beta_k * E_{k+1}(w^(i))
            beta_k_coeff = self.epsilon * np.std(current_residue) if k > 0 else self.epsilon  # beta_0 in paper is just epsilon0

            ensemble_for_mean_operator: List[np.ndarray] = []
            for i_trial in range(self.trials):
                # Get E_{k+1}(w^(i))
                # If fewer than k+1 IMFs for this noise trial, use zeros or skip
                if self.decomposed_noise_imfs[i_trial].shape[0] > k:
                    ek_plus_1_noise = self.decomposed_noise_imfs[i_trial][k, :]
                else:  # Not enough noise IMFs for this trial, use zeros or skip
                    ek_plus_1_noise = np.zeros(N, dtype=s_normalized.dtype)

                signal_plus_scaled_noise_imf = current_residue + beta_k_coeff * ek_plus_1_noise
                ensemble_for_mean_operator.append(signal_plus_scaled_noise_imf)

            current_imf = self._get_operator_emd1(np.array(ensemble_for_mean_operator), T)

            extracted_imfs_list.append(current_imf)
            current_residue -= current_imf
            k += 1

            # --- Step 3b: Check stopping criteria ---
            # (Simplified: check if residue is monotonic or has too few extrema)
            ext_res_residue = self.EMD_direct.find_extrema(T, current_residue)
            num_extrema_residue = len(ext_res_residue[0]) + len(ext_res_residue[2])
            if num_extrema_residue < 3:
                self.logger.info("Residue is monotonic or has too few extrema. Stopping.")
                break
            # Add other EMD stopping conditions if needed (range_thr, total_power_thr) on current_residue
            if (np.max(current_residue) - np.min(current_residue)) * original_std < self.EMD_direct.range_thr:
                self.logger.info("Residue range too small. Stopping.")
                break
            if np.sum(np.abs(current_residue * original_std)) < self.EMD_direct.total_power_thr:
                self.logger.info("Residue power too small. Stopping.")
                break

        # --- Step 4: Finalize ---
        if extracted_imfs_list:
            self.C_IMF = np.array(extracted_imfs_list) * original_std
            self.residue = current_residue * original_std
            output_imfs = np.vstack((self.C_IMF, self.residue))
        else:  # Should not happen if S is not zero
            self.C_IMF = np.empty((0, N), dtype=S.dtype)
            self.residue = S.copy()
            output_imfs = self.residue.reshape(1, N)

        self.logger.info(f"ICEEMDAN decomposition finished. Extracted {self.C_IMF.shape[0]} IMFs.")
        return output_imfs

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.C_IMF is None or self.residue is None:
            raise ValueError("No IMF found. Please, run ICEEMDAN method first.")
        return self.C_IMF, self.residue
