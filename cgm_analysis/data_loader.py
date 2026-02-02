"""
Data loading functions for CGM and meal data.
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta
import pytz

# Constants
USER_DATA_PATH = Path("user_data")
IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC


def get_available_users() -> list:
    """Get list of available user IDs from user_data folder."""
    users = [d.name for d in USER_DATA_PATH.iterdir() if d.is_dir()]
    return sorted(users)


def load_cgm_data(user_id: str) -> pd.DataFrame:
    """Load CGM data for a user and convert timestamps to IST."""
    cgm_path = USER_DATA_PATH / user_id / "cgm.csv"
    df = pd.read_csv(cgm_path)

    # Convert milliseconds to datetime in IST
    df["datetime"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)
    df["datetime_ist"] = df["datetime"].dt.tz_convert(IST)
    df["date"] = df["datetime_ist"].dt.date
    df["time"] = df["datetime_ist"].dt.time

    return df


def load_meals_data(user_id: str) -> pd.DataFrame:
    """Load meals data for a user and convert timestamps to IST."""
    meals_path = USER_DATA_PATH / user_id / "meals.csv"
    df = pd.read_csv(meals_path)

    # Convert timestamp to datetime - handle both milliseconds and string formats
    # Check if the column is numeric (includes numpy.int64, numpy.float64, etc.)
    if pd.api.types.is_numeric_dtype(df["meal_timestamp"]):
        # Milliseconds format (numeric)
        df["datetime"] = pd.to_datetime(df["meal_timestamp"], unit="ms", utc=True)
    else:
        # String format - check if it's numeric string or datetime string
        first_val = df["meal_timestamp"].iloc[0]
        if isinstance(first_val, str) and first_val.isdigit():
            # Numeric string (milliseconds)
            df["datetime"] = pd.to_datetime(df["meal_timestamp"].astype(int), unit="ms", utc=True)
        else:
            # String datetime format (e.g., "2024-10-25 22:12:50+05:30")
            df["datetime"] = pd.to_datetime(df["meal_timestamp"], utc=True)

    df["datetime_ist"] = df["datetime"].dt.tz_convert(IST)
    df["date"] = df["datetime_ist"].dt.date
    df["time"] = df["datetime_ist"].dt.strftime("%H:%M:%S")

    # Parse derived_meal_time if present
    if "derived_meal_time" in df.columns:
        df["derived_meal_time_ist"] = pd.to_datetime(
            df["derived_meal_time"], format="mixed", utc=True
        ).dt.tz_convert(IST)

    return df


def get_available_dates(cgm_df: pd.DataFrame) -> list:
    """Get list of dates that have CGM data."""
    return sorted(cgm_df["date"].unique())


def load_dietary_response_data(user_id: str) -> pd.DataFrame:
    """
    Load dietary response data for a user.

    This data contains meal entries with CGM response metrics.
    Timestamps are already in IST format (YYYY-MM-DD HH:MM:SS).
    """
    dietary_path = USER_DATA_PATH / user_id / "dietary_response.csv"

    if not dietary_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(dietary_path)

    # Parse meal_timestamp (already in IST format)
    df["datetime_ist"] = pd.to_datetime(df["meal_timestamp"])
    # Localize to IST timezone
    df["datetime_ist"] = df["datetime_ist"].dt.tz_localize(IST)
    df["date"] = df["datetime_ist"].dt.date
    df["time"] = df["datetime_ist"].dt.strftime("%H:%M:%S")

    return df


def filter_data_for_date(df: pd.DataFrame, selected_date, extend_hours: int = 2) -> pd.DataFrame:
    """
    Filter dataframe for a specific date, optionally extending into the next day.

    Args:
        df: DataFrame with 'date' and 'datetime_ist' columns
        selected_date: The date to filter for
        extend_hours: Hours to extend into the next day (default 2)

    Returns:
        Filtered DataFrame sorted by datetime
    """
    # Get data for the selected date
    selected_df = df[df["date"] == selected_date].copy()

    if extend_hours > 0:
        # Calculate the next day
        next_date = selected_date + timedelta(days=1)

        # Get data from next day up to extend_hours
        next_day_df = df[df["date"] == next_date].copy()

        if len(next_day_df) > 0:
            # Filter next day data to only include hours before extend_hours
            next_day_df = next_day_df[
                next_day_df["datetime_ist"].dt.hour < extend_hours
            ]

            # Combine the dataframes
            selected_df = pd.concat([selected_df, next_day_df], ignore_index=True)

    return selected_df.sort_values("datetime_ist").reset_index(drop=True)
