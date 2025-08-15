import numpy as np
from scipy.linalg import svd


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    sl, sk = matrix.shape
    # empty matrix yields no diagonals
    if sl == 0 or sk == 0:
        return np.empty(0)
    nv = sl + sk - 1
    series = np.zeros(nv)
    for i in range(nv):
        values = []
        for j in range(max(0, i + 1 - sk), min(i + 1, sl)):
            values.append(matrix[j, i - j])
        series[i] = np.mean(values)
    return series


def ssa(input_signal: np.ndarray,
        nsignal: int,
        wlen: int = 20,
        ) -> np.ndarray:
    """
    Perform Singular Spectrum Analysis (SSA) decomposition on a signal.

    Args:
        input_signal (numpy array): Time series to decompose.
        nsignal (int): Number of components to extract.
        wlen (int): Window length for embedding.

    Returns:
        pd.DataFrame: Reconstructed components as columns.
    """
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array.")

    if input_signal.ndim != 1:
        raise ValueError("Input signal must be a 1-dimensional array.")

    # Step 1: Embedding
    k = input_signal.size - wlen + 1
    trajectory_matrix = np.zeros((wlen, k))
    for i in range(k):
        trajectory_matrix[:, i] = input_signal[i:i + wlen]

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
