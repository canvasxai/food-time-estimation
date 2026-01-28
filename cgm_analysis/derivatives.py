"""
Derivative calculation functions for CGM signal processing.
"""

import pandas as pd
import numpy as np


def apply_smoothing(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Apply rolling average smoothing to reduce sensor noise.

    Args:
        df: DataFrame with 'value' column containing raw CGM readings
        window: Rolling window size (default 5)

    Returns:
        Series with smoothed glucose values
    """
    return df["value"].rolling(window=window, min_periods=1, center=True).mean()


def calculate_first_derivative(df: pd.DataFrame) -> pd.Series:
    """
    Calculate dG/dt at each point using central difference.

    Formula: dG_dt[t] = (smoothed_glucose[t+1] - smoothed_glucose[t-1]) / (time[t+1] - time[t-1])
    Units: mg/dL per minute

    Args:
        df: DataFrame with 'smoothed' and 'datetime_ist' columns

    Returns:
        Series with first derivative values
    """
    smoothed = df["smoothed"].values
    times = df["datetime_ist"].values

    dG_dt = np.zeros(len(df))

    for t in range(1, len(df) - 1):
        time_diff = (times[t + 1] - times[t - 1]) / np.timedelta64(1, 'm')  # Convert to minutes
        if time_diff > 0:
            dG_dt[t] = (smoothed[t + 1] - smoothed[t - 1]) / time_diff

    # Handle edges
    if len(df) > 1:
        time_diff = (times[1] - times[0]) / np.timedelta64(1, 'm')
        if time_diff > 0:
            dG_dt[0] = (smoothed[1] - smoothed[0]) / time_diff

        time_diff = (times[-1] - times[-2]) / np.timedelta64(1, 'm')
        if time_diff > 0:
            dG_dt[-1] = (smoothed[-1] - smoothed[-2]) / time_diff

    return pd.Series(dG_dt, index=df.index)


def calculate_second_derivative(df: pd.DataFrame) -> pd.Series:
    """
    Calculate d²G/dt² at each point using central difference on first derivative.

    Formula: d2G_dt2[t] = (dG_dt[t+1] - dG_dt[t-1]) / (time[t+1] - time[t-1])
    Units: mg/dL per minute²

    Args:
        df: DataFrame with 'dG_dt' and 'datetime_ist' columns

    Returns:
        Series with second derivative values
    """
    dG_dt = df["dG_dt"].values
    times = df["datetime_ist"].values

    d2G_dt2 = np.zeros(len(df))

    for t in range(1, len(df) - 1):
        time_diff = (times[t + 1] - times[t - 1]) / np.timedelta64(1, 'm')
        if time_diff > 0:
            d2G_dt2[t] = (dG_dt[t + 1] - dG_dt[t - 1]) / time_diff

    return pd.Series(d2G_dt2, index=df.index)
