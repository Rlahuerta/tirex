import numpy as np


def cauchy_fval(residuum: np.ndarray, c: float, **kwargs) -> float:
    """
    Compute the Cauchy loss function value.

    Args:
        residuum (np.ndarray): Residual vector.
        c (float): Scaling parameter.

    Returns:
        float: Function value.
    """

    residuum_in = residuum.copy()

    if len(residuum_in.shape) == 1:
        residuum_in = np.expand_dims(residuum_in, axis=0)

    list_fvals = []
    for i in range(residuum_in.shape[0]):
        residuum2_i = (residuum_in[i, :] / c) ** 2
        fval_i = 0.5 * c ** 2 * np.sum(np.log1p(residuum2_i))
        list_fvals.append(fval_i)

    return sum(list_fvals)