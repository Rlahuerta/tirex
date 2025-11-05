import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def apply_minmax_inverse_scaler(scaler: MinMaxScaler, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Apply an existing MinMaxScaler to a DataFrame or Series.

    Args:
        scaler (MinMaxScaler): Fitted MinMaxScaler object.
        data (pd.DataFrame or pd.Series): Data to rescale.

    Returns:
        pd.DataFrame or pd.Series: Rescaled data.
    """

    if scaler is not None:
        if isinstance(data, pd.Series):
            if data.shape[0] > 0:
                scaled = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
                return pd.Series(scaled, index=data.index, name=data.name)
            else:
                return pd.Series([])
        elif isinstance(data, pd.DataFrame):
            if data.shape[0] > 0:
                scaled = scaler.inverse_transform(data.values)
                return pd.DataFrame(scaled, index=data.index, columns=data.columns)
            else:
                return pd.DataFrame([])
        else:
            raise TypeError("Input must be a pandas DataFrame or Series.")
    else:
        return data
