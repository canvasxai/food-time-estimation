"""
CGM Visualization and Meal Detection App
A Streamlit application for visualizing CGM data and detecting meal times using derivative analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import pytz

# Constants
USER_DATA_PATH = Path("user_data")
IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MealEvent:
    """Represents a detected meal event."""
    event_type: str  # "MEAL_CLEAN", "MEAL_STACKED", "SNACK_HIDDEN", "PEAK"
    detected_at: datetime  # When the signal was detected in CGM data
    estimated_meal_time: Optional[datetime]  # Back-calculated actual eating time
    confidence: float  # 0.0 - 1.0 based on signal strength
    dG_dt_at_detection: float  # First derivative value
    d2G_dt2_at_detection: float  # Second derivative value
    glucose_at_detection: float  # Glucose value at detection point


@dataclass
class MealMatch:
    """Represents a match between a logged meal and a detected CGM event."""
    meal_name: str
    meal_time: datetime
    meal_slot: str
    event_type: str
    event_time: datetime
    peak_time: Optional[datetime]  # Time of the associated peak (for physio score)
    time_offset_minutes: float  # meal_time - event_time in minutes
    s_time: float  # Time proximity score
    s_slot: float  # Meal slot window score
    s_physio: float  # Physiological plausibility score
    s_size: float  # Meal size match score
    composite_score: float  # Weighted composite score
    # Meal nutrients for reference
    carbs: float
    protein: float
    fat: float
    fiber: float


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_available_users():
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

    # Convert milliseconds to datetime in IST
    df["datetime"] = pd.to_datetime(df["meal_timestamp"], unit="ms", utc=True)
    df["datetime_ist"] = df["datetime"].dt.tz_convert(IST)
    df["date"] = df["datetime_ist"].dt.date
    df["time"] = df["datetime_ist"].dt.strftime("%H:%M:%S")

    return df


def get_available_dates(cgm_df: pd.DataFrame) -> list:
    """Get list of dates that have CGM data."""
    return sorted(cgm_df["date"].unique())


def filter_data_for_date(df: pd.DataFrame, selected_date, extend_hours: int = 2) -> pd.DataFrame:
    """
    Filter dataframe for a specific date, optionally extending into the next day.
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


# =============================================================================
# MEAL MATCHING ALGORITHM
# =============================================================================

# Scoring weights
W_TIME = 0.2   # Time proximity weight (reduced - user-logged times are often inaccurate)
W_SLOT = 0.4   # Meal slot window weight (new - major factor based on meal slot time windows)
W_PHYSIO = 0.2 # Physiological plausibility weight
W_SIZE = 0.2   # Meal size match weight

# Minimum composite score threshold for a valid match (70%)
MIN_MATCH_THRESHOLD = 0.70

# Meal slot time windows (hour ranges)
# Format: {meal_slot: (start_hour, end_hour)}
MEAL_SLOT_WINDOWS = {
    "Breakfast": (7, 12),
    "Pre Breakfast": (7, 12),
    "Morning Snack": (7, 12),
    "Mid Morning Snack": (7, 12),
    "Lunch": (12, 17),
    "Afternoon Snack": (12, 17),
    "Post Lunch Snack": (12, 17),
    "Dinner": (17, 24),
    "Evening Snack": (17, 24),
    "Late Night Snack": (17, 24),
}


def calculate_time_proximity_score(meal_time: datetime, detected_time: datetime) -> float:
    """
    Calculate time proximity score (S_time).

    Measures how close the logged meal time is to the detected event time.
    Sweet spot is when Δt ≈ 30 min (meal logged ~30 min after detection starts).

    Args:
        meal_time: When the meal was logged
        detected_time: When the CGM event was detected

    Returns:
        Score between 0.0 and 1.0
    """
    delta_t = (meal_time - detected_time).total_seconds() / 60  # in minutes

    if -30 <= delta_t <= 90:
        # Within expected window: peak score at delta_t = 30
        score = 1 - ((abs(delta_t - 30)) / 120) ** 2
    else:
        # Outside expected window: decay based on distance
        score = max(0, 1 - (abs(delta_t) / 180) ** 2)

    return max(0.0, min(1.0, score))


def calculate_expected_time_to_peak(fiber: float, protein: float, fat: float) -> float:
    """
    Calculate expected time to peak based on meal composition.

    Formula: TTP_expected = 30 + 3·fiber + 1.5·protein + 2·fat (minutes)
    Clamped to [30, 180] minutes.
    """
    ttp = 30 + 3 * fiber + 1.5 * protein + 2 * fat
    return max(30, min(180, ttp))


def calculate_physio_score(meal_time: datetime, peak_time: Optional[datetime],
                           fiber: float, protein: float, fat: float) -> float:
    """
    Calculate physiological plausibility score (S_physio).

    Checks if the time-to-peak matches what we'd expect given meal composition.

    Args:
        meal_time: When the meal was logged
        peak_time: When the glucose peak occurred
        fiber, protein, fat: Meal nutrients in grams

    Returns:
        Score between 0.0 and 1.0
    """
    if peak_time is None:
        return 0.5  # Neutral score if no peak available

    ttp_expected = calculate_expected_time_to_peak(fiber, protein, fat)
    ttp_actual = (peak_time - meal_time).total_seconds() / 60  # in minutes

    error = abs(ttp_actual - ttp_expected)
    score = max(0, 1 - (error / 120) ** 2)

    return max(0.0, min(1.0, score))


def calculate_glycemic_load_proxy(carbs: float, fiber: float, protein: float, fat: float) -> float:
    """
    Calculate Glycemic Load Proxy.

    Formula:
        net_carbs = max(0, carbs - 0.5·fiber)
        dampening = 1 / (1 + 0.02·protein + 0.01·fat)
        GL_proxy = net_carbs × dampening
    """
    net_carbs = max(0, carbs - 0.5 * fiber)
    dampening = 1 / (1 + 0.02 * protein + 0.01 * fat)
    return net_carbs * dampening


def calculate_size_score(carbs: float, fiber: float, protein: float, fat: float,
                         event_type: str) -> float:
    """
    Calculate meal size match score (S_size).

    Uses Glycemic Load Proxy to match meal magnitude to event type.

    | Event Type         | GL_proxy > 30 | GL_proxy 15-30 | GL_proxy < 15 |
    |--------------------|---------------|----------------|---------------|
    | est_meal           | 1.0           | 0.7            | 0.4           |
    | est_secondary_meal | 0.4           | 0.7            | 1.0           |

    Args:
        carbs, fiber, protein, fat: Meal nutrients in grams
        event_type: Type of detected event

    Returns:
        Score between 0.0 and 1.0
    """
    gl_proxy = calculate_glycemic_load_proxy(carbs, fiber, protein, fat)

    # Normalize event type (handle variations)
    event_type_lower = event_type.lower()
    is_main_meal = "meal_start" in event_type_lower or event_type_lower == "est_meal"

    if is_main_meal:
        # Main meal event
        if gl_proxy > 30:
            return 1.0
        elif gl_proxy >= 15:
            return 0.7
        else:
            return 0.4
    else:
        # Secondary/snack event
        if gl_proxy < 15:
            return 1.0
        elif gl_proxy <= 30:
            return 0.7
        else:
            return 0.4


def calculate_slot_score(meal_slot: str, event_time: datetime) -> float:
    """
    Calculate meal slot window score (S_slot).

    Checks if the detected event falls within the expected time window for the meal slot.
    This is a major factor since user-logged meal times are often inaccurate,
    but the meal slot (Breakfast, Lunch, Dinner) is usually correct.

    Time windows:
    - Breakfast/Morning Snack: 7:00-12:00
    - Lunch/Afternoon Snack: 12:00-17:00
    - Dinner/Evening Snack: 17:00-24:00

    Args:
        meal_slot: The meal slot name (e.g., "Breakfast", "Lunch", "Dinner")
        event_time: When the CGM event was detected

    Returns:
        Score between 0.0 and 1.0
        - 1.0 if event is within the expected window
        - Decays based on how far outside the window the event is
    """
    # Get the expected window for this meal slot
    # Default to full day if slot not found
    window = MEAL_SLOT_WINDOWS.get(meal_slot, (0, 24))
    start_hour, end_hour = window

    event_hour = event_time.hour + event_time.minute / 60.0

    # Handle midnight wraparound for evening meals (17-24 means 17:00 to 23:59)
    if end_hour == 24:
        end_hour = 24.0  # 24:00 = midnight

    # Check if event is within the window
    if start_hour <= event_hour < end_hour:
        # Within window - perfect score
        # Optionally, give slightly higher score to events closer to window center
        window_center = (start_hour + end_hour) / 2
        distance_from_center = abs(event_hour - window_center)
        window_half_width = (end_hour - start_hour) / 2
        # Score ranges from 0.9 (at edges) to 1.0 (at center)
        score = 0.9 + 0.1 * (1 - distance_from_center / window_half_width)
        return score
    else:
        # Outside window - calculate penalty based on distance
        if event_hour < start_hour:
            distance = start_hour - event_hour
        else:
            distance = event_hour - end_hour

        # Decay: score drops to 0 when 4+ hours outside window
        score = max(0.0, 1.0 - (distance / 4.0))
        return score


def is_meal_time_in_slot(meal_time: datetime, meal_slot: str) -> bool:
    """
    Check if the logged meal time falls within the expected time window for the meal slot.

    If a user logs "Breakfast" at 1:30 AM, that's clearly outside the breakfast window,
    indicating the logged time is unreliable.
    """
    window = MEAL_SLOT_WINDOWS.get(meal_slot, (0, 24))
    start_hour, end_hour = window

    meal_hour = meal_time.hour + meal_time.minute / 60.0

    # Handle midnight wraparound
    if end_hour == 24:
        end_hour = 24.0

    return start_hour <= meal_hour < end_hour


def compute_composite_score(s_time: float, s_slot: float, s_physio: float, s_size: float,
                            meal_time_in_slot: bool = True) -> float:
    """
    Compute weighted composite matching score.

    If the logged meal time is within its expected slot:
        S_composite = 0.2·S_time + 0.4·S_slot + 0.2·S_physio + 0.2·S_size

    If the logged meal time is OUTSIDE its expected slot (e.g., Breakfast logged at 1:30 AM):
        - Ignore S_time and S_slot completely (logged time is unreliable)
        - Only use physio and size to match:
        S_composite = 0.5·S_physio + 0.5·S_size

    Args:
        s_time: Time proximity score
        s_slot: Meal slot window score
        s_physio: Physiological plausibility score
        s_size: Meal size match score
        meal_time_in_slot: Whether the logged meal time falls within its expected slot
    """
    if meal_time_in_slot:
        # Normal scoring - logged time appears trustworthy
        return W_TIME * s_time + W_SLOT * s_slot + W_PHYSIO * s_physio + W_SIZE * s_size
    else:
        # Logged time is outside the slot window - can't trust it
        # Ignore S_time and S_slot, only use physio and size
        return 0.5 * s_physio + 0.5 * s_size


def find_associated_peak(event: MealEvent, events: List[MealEvent]) -> Optional[datetime]:
    """
    Find the peak event associated with a meal start event.

    Looks for the next PEAK event within a reasonable time window (30-180 min).
    """
    if event.event_type not in ["MEAL_START", "SECONDARY_MEAL"]:
        return None

    # Find peaks that occur after this meal start
    peaks = [e for e in events if e.event_type == "PEAK" and e.detected_at > event.detected_at]

    if not peaks:
        return None

    # Find the closest peak within 180 minutes
    for peak in sorted(peaks, key=lambda p: p.detected_at):
        time_diff = (peak.detected_at - event.detected_at).total_seconds() / 60
        if 10 <= time_diff <= 180:  # Peak should be 10-180 min after meal start
            return peak.detected_at

    return None


def resolve_nearby_events(primary_events: List[MealEvent], secondary_events: List[MealEvent],
                          meal_time: datetime, proximity_minutes: float = 45) -> List[MealEvent]:
    """
    Resolve conflicts between primary and secondary meals that are close together.

    If a primary and secondary meal are within proximity_minutes of each other,
    return only the one closer to the logged meal time.

    Args:
        primary_events: List of MEAL_START events
        secondary_events: List of SECONDARY_MEAL events
        meal_time: The logged meal time to compare against
        proximity_minutes: Time window to consider events as "nearby" (default 45 min)

    Returns:
        List of resolved events (duplicates removed based on proximity rule)
    """
    all_events = primary_events + secondary_events

    if len(all_events) <= 1:
        return all_events

    # Sort by time
    all_events_sorted = sorted(all_events, key=lambda e: e.estimated_meal_time or e.detected_at)

    resolved = []
    skip_indices = set()

    for i, event in enumerate(all_events_sorted):
        if i in skip_indices:
            continue

        event_time = event.estimated_meal_time or event.detected_at

        # Look for nearby events of different type
        nearby_different_type = []
        for j, other_event in enumerate(all_events_sorted):
            if i == j or j in skip_indices:
                continue

            other_time = other_event.estimated_meal_time or other_event.detected_at
            time_diff = abs((event_time - other_time).total_seconds() / 60)

            # Check if within proximity window and different type
            if time_diff <= proximity_minutes and event.event_type != other_event.event_type:
                nearby_different_type.append((j, other_event, time_diff))

        if nearby_different_type:
            # There are nearby events of different type - pick the one closer to meal_time
            candidates = [(i, event)] + [(j, e) for j, e, _ in nearby_different_type]

            # Find the one closest to the logged meal time
            best_idx, best_event = min(
                candidates,
                key=lambda x: abs((meal_time - (x[1].estimated_meal_time or x[1].detected_at)).total_seconds())
            )

            resolved.append(best_event)

            # Mark all candidates as processed
            for idx, _, _ in nearby_different_type:
                skip_indices.add(idx)
            skip_indices.add(i)
        else:
            # No nearby events of different type - keep this one
            resolved.append(event)

    return resolved


def match_meals_to_events(meals_df: pd.DataFrame, events: List[MealEvent],
                          cgm_df: pd.DataFrame = None,
                          thresholds: "SimplifiedThresholds" = None) -> tuple:
    """
    Match logged meals to detected CGM events using greedy assignment.

    Includes both primary (MEAL_START) and secondary (SECONDARY_MEAL) events.
    When primary and secondary events are within 30-45 minutes of each other,
    the algorithm picks whichever is closer to the logged meal time.

    Algorithm:
    1. Detect secondary meal events from CGM data
    2. For each logged meal, resolve nearby primary/secondary events
    3. Compute all pairwise scores (meals × resolved events)
    4. Sort by composite score (descending)
    5. Iterate: assign highest-scoring pair if neither meal nor event is already assigned
    6. Continue until all meals are assigned (or no valid matches remain)

    Args:
        meals_df: DataFrame with meal data (must have datetime_ist, meal_name, etc.)
        events: List of detected MealEvent objects (from detect_meal_events_simplified)
        cgm_df: CGM DataFrame for detecting secondary meals
        thresholds: Detection thresholds

    Returns:
        Tuple of (list of MealMatch objects, list of unmatched event indices, list of matchable events)
    """
    if len(meals_df) == 0 or len(events) == 0:
        return [], list(range(len(events))), []

    # Get primary meal events (MEAL_START)
    primary_events = [e for e in events if e.event_type == "MEAL_START"]

    # Detect secondary meal events if CGM data is provided
    secondary_events = []
    if cgm_df is not None and thresholds is not None:
        secondary_events = detect_secondary_meal_events(cgm_df, thresholds)

    # Combine all matchable events
    all_matchable_events = primary_events + secondary_events

    if len(all_matchable_events) == 0:
        return [], [], []

    # Compute all pairwise scores
    scores = []
    for meal_idx, meal_row in meals_df.iterrows():
        meal_time = meal_row["datetime_ist"]
        carbs = meal_row.get("carbohydrates", 0) or 0
        protein = meal_row.get("protein", 0) or 0
        fat = meal_row.get("fat", 0) or 0
        fiber = meal_row.get("fibre", 0) or 0

        # Resolve nearby events for this specific meal
        resolved_events = resolve_nearby_events(primary_events, secondary_events, meal_time)

        for event in resolved_events:
            # Find the index in all_matchable_events
            event_idx = next(
                (i for i, e in enumerate(all_matchable_events)
                 if e.detected_at == event.detected_at and e.event_type == event.event_type),
                None
            )

            if event_idx is None:
                continue

            # Find associated peak for physio score
            peak_time = find_associated_peak(event, events)

            # Get event time for scoring
            event_time = event.estimated_meal_time or event.detected_at

            # Get meal slot for slot scoring
            meal_slot = meal_row.get("meal_slot", "Unknown")

            # Check if the logged meal time is within the expected slot window
            meal_time_in_slot = is_meal_time_in_slot(meal_time, meal_slot)

            # If meal time is outside its slot, only consider events within the slot window
            if not meal_time_in_slot:
                # Check if event is within the expected slot window
                event_hour = event_time.hour + event_time.minute / 60.0
                slot_window = MEAL_SLOT_WINDOWS.get(meal_slot, (0, 24))
                slot_start, slot_end = slot_window
                if slot_end == 24:
                    slot_end = 24.0

                # Skip events outside the meal slot window
                if not (slot_start <= event_hour < slot_end):
                    continue

            # Calculate individual scores
            s_time = calculate_time_proximity_score(meal_time, event_time)
            s_slot = calculate_slot_score(meal_slot, event_time)
            s_physio = calculate_physio_score(meal_time, peak_time, fiber, protein, fat)
            s_size = calculate_size_score(carbs, fiber, protein, fat, event.event_type)

            # If meal time is outside its slot, set S_time and S_slot to 0 (unreliable)
            if not meal_time_in_slot:
                s_time = 0.0
                s_slot = 0.0

            composite = compute_composite_score(s_time, s_slot, s_physio, s_size, meal_time_in_slot)

            scores.append({
                "meal_idx": meal_idx,
                "event_idx": event_idx,
                "meal_row": meal_row,
                "event": event,
                "peak_time": peak_time,
                "s_time": s_time,
                "s_slot": s_slot,
                "s_physio": s_physio,
                "s_size": s_size,
                "composite": composite
            })

    # Sort by composite score descending
    scores.sort(key=lambda x: x["composite"], reverse=True)

    # Greedy assignment
    assigned_meals = set()
    assigned_events = set()
    matches = []

    for score_entry in scores:
        meal_idx = score_entry["meal_idx"]
        event_idx = score_entry["event_idx"]

        if meal_idx in assigned_meals or event_idx in assigned_events:
            continue

        # Skip matches below the minimum threshold (70%)
        if score_entry["composite"] < MIN_MATCH_THRESHOLD:
            continue

        # Create match
        meal_row = score_entry["meal_row"]
        event = score_entry["event"]
        meal_time = meal_row["datetime_ist"]
        event_time = event.estimated_meal_time or event.detected_at

        match = MealMatch(
            meal_name=meal_row.get("meal_name", "Unknown"),
            meal_time=meal_time,
            meal_slot=meal_row.get("meal_slot", "Unknown"),
            event_type=event.event_type,
            event_time=event_time,
            peak_time=score_entry["peak_time"],
            time_offset_minutes=(meal_time - event_time).total_seconds() / 60,
            s_time=score_entry["s_time"],
            s_slot=score_entry["s_slot"],
            s_physio=score_entry["s_physio"],
            s_size=score_entry["s_size"],
            composite_score=score_entry["composite"],
            carbs=meal_row.get("carbohydrates", 0) or 0,
            protein=meal_row.get("protein", 0) or 0,
            fat=meal_row.get("fat", 0) or 0,
            fiber=meal_row.get("fibre", 0) or 0
        )

        matches.append(match)
        assigned_meals.add(meal_idx)
        assigned_events.add(event_idx)

    # Find unmatched events
    unmatched_event_indices = [i for i in range(len(all_matchable_events)) if i not in assigned_events]

    return matches, unmatched_event_indices, all_matchable_events


# =============================================================================
# DERIVATIVE-BASED MEAL DETECTION ALGORITHM
# =============================================================================

def apply_smoothing(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Apply rolling average smoothing to reduce sensor noise."""
    return df["value"].rolling(window=window, min_periods=1, center=True).mean()


def calculate_first_derivative(df: pd.DataFrame) -> pd.Series:
    """
    Calculate dG/dt at each point using central difference.
    dG_dt[t] = (smoothed_glucose[t+1] - smoothed_glucose[t-1]) / (time[t+1] - time[t-1])
    Units: mg/dL per minute
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
    d2G_dt2[t] = (dG_dt[t+1] - dG_dt[t-1]) / (time[t+1] - time[t-1])
    Units: mg/dL per minute²
    """
    dG_dt = df["dG_dt"].values
    times = df["datetime_ist"].values

    d2G_dt2 = np.zeros(len(df))

    for t in range(1, len(df) - 1):
        time_diff = (times[t + 1] - times[t - 1]) / np.timedelta64(1, 'm')
        if time_diff > 0:
            d2G_dt2[t] = (dG_dt[t + 1] - dG_dt[t - 1]) / time_diff

    return pd.Series(d2G_dt2, index=df.index)


# =============================================================================
# SIMPLIFIED MEAL DETECTION (ZERO-CROSSING METHOD)
# =============================================================================

@dataclass
class SimplifiedThresholds:
    """Configuration thresholds for simplified zero-crossing detection."""
    smoothing_window: int = 5  # Rolling average window size
    min_derivative_magnitude: float = 0.01  # Minimum |dG/dt| to consider a crossing significant
    start_hour: int = 7  # Only detect events after this hour (24h format)
    meal_absorption_lag: int = 0  # minutes - time from eating to detectable rise


def detect_secondary_meal_events(cgm_df: pd.DataFrame, thresholds: SimplifiedThresholds) -> List[MealEvent]:
    """
    Detect secondary meal events based on d²G/dt² maxima.

    A secondary meal is detected when d²G/dt² has a local maximum while
    dG/dt is positive (glucose is rising).

    Args:
        cgm_df: DataFrame with CGM data
        thresholds: Detection thresholds

    Returns:
        List of MealEvent objects with event_type="SECONDARY_MEAL"
    """
    if len(cgm_df) < 5:
        return []

    df = cgm_df.sort_values("datetime_ist").copy().reset_index(drop=True)

    # Determine primary date for filtering
    primary_date = df["datetime_ist"].dt.date.mode().iloc[0]

    # Apply smoothing and calculate derivatives
    df["smoothed"] = apply_smoothing(df, window=thresholds.smoothing_window)
    df["dG_dt"] = calculate_first_derivative(df)
    df["d2G_dt2"] = calculate_second_derivative(df)

    d2G_dt2_values = df["d2G_dt2"].values
    dG_dt_values = df["dG_dt"].values

    events = []

    for i in range(1, len(d2G_dt2_values) - 1):
        # Check for local maximum in d²G/dt²
        if (d2G_dt2_values[i] > d2G_dt2_values[i - 1] and
            d2G_dt2_values[i] > d2G_dt2_values[i + 1] and
            d2G_dt2_values[i] > 0 and  # Only positive maxima (acceleration peaks)
            dG_dt_values[i] > 0):  # Only when glucose is rising

            curr_datetime = df.iloc[i]["datetime_ist"]
            curr_date = curr_datetime.date()
            curr_hour = curr_datetime.hour

            # Skip if before start hour ON THE PRIMARY DATE
            if curr_date == primary_date and curr_hour < thresholds.start_hour:
                continue

            # Create secondary meal event
            # Estimated meal time is slightly before detection (less lag than primary)
            estimated_meal_time = curr_datetime - timedelta(minutes=thresholds.meal_absorption_lag // 2)

            # Confidence based on d²G/dt² magnitude
            confidence = min(0.5 + d2G_dt2_values[i] * 10, 1.0)

            events.append(MealEvent(
                event_type="SECONDARY_MEAL",
                detected_at=curr_datetime,
                estimated_meal_time=estimated_meal_time,
                confidence=confidence,
                dG_dt_at_detection=dG_dt_values[i],
                d2G_dt2_at_detection=d2G_dt2_values[i],
                glucose_at_detection=df.iloc[i]["smoothed"]
            ))

    return events


def interpolate_zero_crossing(t_prev: datetime, t_curr: datetime,
                               val_prev: float, val_curr: float,
                               glucose_prev: float, glucose_curr: float) -> tuple:
    """
    Linear interpolation to find exact zero-crossing point.

    Args:
        t_prev, t_curr: Timestamps before and after crossing
        val_prev, val_curr: dG/dt values before and after crossing
        glucose_prev, glucose_curr: Glucose values before and after crossing

    Returns:
        (interpolated_time, interpolated_glucose) at the zero-crossing
    """
    # Linear interpolation factor: find where the line crosses zero
    # val_prev + alpha * (val_curr - val_prev) = 0
    # alpha = -val_prev / (val_curr - val_prev)
    if val_curr == val_prev:
        alpha = 0.5
    else:
        alpha = -val_prev / (val_curr - val_prev)

    # Clamp alpha to [0, 1] for safety
    alpha = max(0.0, min(1.0, alpha))

    # Interpolate timestamp
    time_diff = (t_curr - t_prev).total_seconds()
    interpolated_time = t_prev + timedelta(seconds=time_diff * alpha)

    # Interpolate glucose value
    interpolated_glucose = glucose_prev + alpha * (glucose_curr - glucose_prev)

    return interpolated_time, interpolated_glucose


def detect_meal_events_simplified(cgm_df: pd.DataFrame, thresholds: SimplifiedThresholds = None) -> List[MealEvent]:
    """
    Simplified meal detection using zero-crossings of the first derivative.

    Logic:
    - MEAL_START: When dG/dt crosses from negative to positive (local minimum)
    - PEAK: When dG/dt crosses from positive to negative (local maximum)

    Both events use linear interpolation to find the exact zero-crossing point,
    ensuring the detected time coincides precisely with where dG/dt = 0.

    Filters:
    - Only consider events after the specified start hour (default 7 AM)
    - Extended midnight hours (e.g., 0-2 AM next day) are always included
    - Ignore crossings where |dG/dt| is below the minimum threshold (noise filter)

    Args:
        cgm_df: DataFrame with CGM data for a single day
        thresholds: Configuration thresholds (uses defaults if None)

    Returns:
        List of detected MealEvent objects
    """
    if thresholds is None:
        thresholds = SimplifiedThresholds()

    if len(cgm_df) < 5:
        return []

    # Sort by time
    df = cgm_df.sort_values("datetime_ist").copy().reset_index(drop=True)

    # Determine the primary date (most common date in the data)
    primary_date = df["datetime_ist"].dt.date.mode().iloc[0]

    # Apply smoothing
    df["smoothed"] = apply_smoothing(df, window=thresholds.smoothing_window)

    # Calculate first derivative
    df["dG_dt"] = calculate_first_derivative(df)

    events = []

    for t in range(1, len(df)):
        timestamp_curr = df.iloc[t]["datetime_ist"]
        timestamp_prev = df.iloc[t - 1]["datetime_ist"]
        glucose_curr = df.iloc[t]["smoothed"]
        glucose_prev = df.iloc[t - 1]["smoothed"]
        dG_dt_curr = df.iloc[t]["dG_dt"]
        dG_dt_prev = df.iloc[t - 1]["dG_dt"]

        # Filter 1: Only consider events after start hour
        # But allow extended midnight hours (next day data)
        curr_date = timestamp_curr.date()
        curr_hour = timestamp_curr.hour

        # Skip if before start hour ON THE PRIMARY DATE
        # Extended hours (next day, early morning) are always included
        if curr_date == primary_date and curr_hour < thresholds.start_hour:
            continue

        # Filter 2: Check if the derivative values around the crossing are significant
        # We need a clear transition, not just hovering around zero
        magnitude_before = abs(dG_dt_prev)
        magnitude_after = abs(dG_dt_curr)

        # At least one side of the crossing should have significant magnitude
        has_significant_crossing = (magnitude_before >= thresholds.min_derivative_magnitude or
                                    magnitude_after >= thresholds.min_derivative_magnitude)

        if not has_significant_crossing:
            continue

        # Check for negative to positive crossing (MEAL_START)
        if dG_dt_prev < 0 and dG_dt_curr > 0:
            # Interpolate to find exact zero-crossing point
            crossing_time, crossing_glucose = interpolate_zero_crossing(
                timestamp_prev, timestamp_curr,
                dG_dt_prev, dG_dt_curr,
                glucose_prev, glucose_curr
            )

            # estimated_meal_time is when the person likely ate (before absorption lag)
            estimated_meal_time = crossing_time - timedelta(minutes=thresholds.meal_absorption_lag)

            # Confidence based on the magnitude of the transition
            transition_magnitude = abs(dG_dt_curr - dG_dt_prev)
            confidence = min(0.5 + (transition_magnitude / 2.0) * 0.5, 1.0)

            events.append(MealEvent(
                event_type="MEAL_START",
                detected_at=crossing_time,
                estimated_meal_time=estimated_meal_time,
                confidence=confidence,
                dG_dt_at_detection=0.0,  # At the crossing, dG/dt = 0 by definition
                d2G_dt2_at_detection=0.0,  # Not used in simplified method
                glucose_at_detection=crossing_glucose
            ))

        # Check for positive to negative crossing (PEAK)
        elif dG_dt_prev > 0 and dG_dt_curr < 0:
            # Interpolate to find exact zero-crossing point (the true peak)
            crossing_time, crossing_glucose = interpolate_zero_crossing(
                timestamp_prev, timestamp_curr,
                dG_dt_prev, dG_dt_curr,
                glucose_prev, glucose_curr
            )

            # Confidence based on the magnitude of the transition
            transition_magnitude = abs(dG_dt_curr - dG_dt_prev)
            confidence = min(0.5 + (transition_magnitude / 2.0) * 0.5, 1.0)

            events.append(MealEvent(
                event_type="PEAK",
                detected_at=crossing_time,  # Exact interpolated zero-crossing time
                estimated_meal_time=None,  # Peaks don't have an estimated meal time
                confidence=confidence,
                dG_dt_at_detection=0.0,  # At the crossing, dG/dt = 0 by definition
                d2G_dt2_at_detection=0.0,  # Not used in simplified method
                glucose_at_detection=crossing_glucose  # Interpolated glucose at the peak
            ))

    return events


def create_simplified_derivative_plot(cgm_df: pd.DataFrame, events: List[MealEvent],
                                       thresholds: SimplifiedThresholds,
                                       matches: List[MealMatch] = None) -> go.Figure:
    """
    Create a plot for the simplified detection method showing:
    1. Raw and smoothed glucose curves
    2. First derivative (dG/dt) with zero line and threshold bands
    3. Second derivative (d²G/dt²)
    4. Detected events marked
    5. Matched meal events annotated (if matches provided)
    """
    df = cgm_df.sort_values("datetime_ist").copy().reset_index(drop=True)

    # Apply preprocessing
    df["smoothed"] = apply_smoothing(df, window=thresholds.smoothing_window)
    df["dG_dt"] = calculate_first_derivative(df)
    df["d2G_dt2"] = calculate_second_derivative(df)

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "CGM Glucose (Raw & Smoothed)",
            "First Derivative (dG/dt) - Zero Crossings Indicate Events",
            "Second Derivative (d²G/dt²) - Acceleration"
        ),
        row_heights=[0.45, 0.30, 0.25]
    )

    # Plot 1: Raw glucose
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["value"],
            mode="lines+markers",
            name="CGM (raw)",
            line=dict(color="#2E86AB", width=2),
            marker=dict(size=5),
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose (raw):</b> %{y:.1f} mg/dL<extra></extra>"
        ),
        row=1, col=1
    )

    # Plot 1: Smoothed glucose
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["smoothed"],
            mode="lines",
            name="CGM (smoothed)",
            line=dict(color="#E94F37", width=1.5, dash="dot"),
            opacity=0.7,
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose (smoothed):</b> %{y:.1f} mg/dL<extra></extra>"
        ),
        row=1, col=1
    )

    # Find and plot second derivative maxima on glucose plot
    # A local maximum in d²G/dt² occurs when it's greater than both neighbors
    d2G_dt2_values = df["d2G_dt2"].values
    dG_dt_values = df["dG_dt"].values

    # Determine primary date for filtering
    primary_date = df["datetime_ist"].dt.date.mode().iloc[0]

    maxima_indices = []
    for i in range(1, len(d2G_dt2_values) - 1):
        if (d2G_dt2_values[i] > d2G_dt2_values[i - 1] and
            d2G_dt2_values[i] > d2G_dt2_values[i + 1] and
            d2G_dt2_values[i] > 0 and  # Only positive maxima (acceleration peaks)
            dG_dt_values[i] > 0):  # Only when glucose is rising (dG/dt positive)
            # Filter by start hour, but allow extended midnight hours (next day)
            curr_datetime = df.iloc[i]["datetime_ist"]
            curr_date = curr_datetime.date()
            curr_hour = curr_datetime.hour

            # Skip if before start hour ON THE PRIMARY DATE
            # Extended hours (next day, early morning) are always included
            if curr_date == primary_date and curr_hour < thresholds.start_hour:
                continue
            maxima_indices.append(i)

    if maxima_indices:
        maxima_times = df.iloc[maxima_indices]["datetime_ist"]
        maxima_glucose = df.iloc[maxima_indices]["smoothed"]
        maxima_d2G = df.iloc[maxima_indices]["d2G_dt2"]

        # Add markers on glucose plot (row 1)
        fig.add_trace(
            go.Scatter(
                x=maxima_times,
                y=maxima_glucose,
                mode="markers",
                name="Est. Secondary Meal",
                marker=dict(
                    size=12,
                    color="#27AE60",  # Green to match meal markers
                    symbol="diamond",
                    line=dict(color="white", width=1)
                ),
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<br><b>d²G/dt²:</b> %{customdata:.4f}<extra>Est. Secondary Meal</extra>",
                customdata=maxima_d2G
            ),
            row=1, col=1
        )

        # Add vertical lines for each secondary meal estimate
        for idx in maxima_indices:
            sec_meal_time = df.iloc[idx]["datetime_ist"]
            sec_meal_time_ms = int(sec_meal_time.timestamp() * 1000)
            sec_meal_glucose = df.iloc[idx]["smoothed"]

            for row in [1, 2, 3]:
                fig.add_vline(
                    x=sec_meal_time_ms,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="#27AE60",
                    opacity=0.6,
                    row=row, col=1
                )

            # Add annotation
            fig.add_annotation(
                x=sec_meal_time_ms,
                y=sec_meal_glucose + 8,
                text=f"Est. Sec. Meal<br>{sec_meal_time.strftime('%H:%M')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#27AE60",
                font=dict(size=9, color="#27AE60"),
                row=1, col=1
            )

        # Also add markers on the d²G/dt² plot (row 3) for clarity
        fig.add_trace(
            go.Scatter(
                x=maxima_times,
                y=maxima_d2G,
                mode="markers",
                name="Est. Secondary Meal",
                marker=dict(
                    size=10,
                    color="#27AE60",
                    symbol="diamond",
                    line=dict(color="white", width=1)
                ),
                showlegend=False,
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>d²G/dt²:</b> %{y:.4f} mg/dL/min²<extra>Est. Secondary Meal</extra>"
            ),
            row=3, col=1
        )

    # Plot 2: First derivative
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["dG_dt"],
            mode="lines",
            name="dG/dt",
            line=dict(color="#28A745", width=2),
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>dG/dt:</b> %{y:.3f} mg/dL/min<extra></extra>"
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2, row=2, col=1)

    # Add threshold bands (noise zone)
    fig.add_hline(y=thresholds.min_derivative_magnitude, line_dash="dash",
                  line_color="gray", line_width=1, row=2, col=1,
                  annotation_text=f"+{thresholds.min_derivative_magnitude}")
    fig.add_hline(y=-thresholds.min_derivative_magnitude, line_dash="dash",
                  line_color="gray", line_width=1, row=2, col=1,
                  annotation_text=f"-{thresholds.min_derivative_magnitude}")

    # Add shaded "noise zone" around zero
    # Find x range for the shaded region
    x_min = df["datetime_ist"].min()
    x_max = df["datetime_ist"].max()

    fig.add_shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=-thresholds.min_derivative_magnitude,
        y1=thresholds.min_derivative_magnitude,
        fillcolor="lightgray",
        opacity=0.3,
        line_width=0,
        row=2, col=1
    )

    # Plot 3: Second derivative
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["d2G_dt2"],
            mode="lines",
            name="d²G/dt²",
            line=dict(color="#FF6B35", width=2),
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>d²G/dt²:</b> %{y:.4f} mg/dL/min²<extra></extra>"
        ),
        row=3, col=1
    )

    # Add zero line to second derivative
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=3, col=1)

    # Add start hour vertical line
    start_datetime = df["datetime_ist"].iloc[0].replace(hour=thresholds.start_hour, minute=0, second=0)
    if start_datetime >= df["datetime_ist"].min() and start_datetime <= df["datetime_ist"].max():
        start_time_ms = int(start_datetime.timestamp() * 1000)
        for row in [1, 2, 3]:
            fig.add_vline(
                x=start_time_ms,
                line_width=2,
                line_dash="dashdot",
                line_color="purple",
                opacity=0.5,
                row=row, col=1
            )
        fig.add_annotation(
            x=start_time_ms,
            y=df["value"].max() + 10,
            text=f"Detection starts ({thresholds.start_hour}:00)",
            showarrow=False,
            font=dict(size=10, color="purple"),
            row=1, col=1
        )

    # Add event markers
    event_colors = {
        "MEAL_START": "#2ECC71",  # Green for meal start
        "PEAK": "#3498DB"  # Blue for peak
    }

    y_max = df["value"].max()
    annotation_offset = 0

    for event in events:
        color = event_colors.get(event.event_type, "#333333")
        event_time_ms = int(event.detected_at.timestamp() * 1000)

        # Add vertical line for detection time
        for row in [1, 2, 3]:
            fig.add_vline(
                x=event_time_ms,
                line_width=2,
                line_dash="solid" if event.event_type == "MEAL_START" else "dot",
                line_color=color,
                opacity=0.7,
                row=row, col=1
            )

        if event.event_type == "MEAL_START" and event.estimated_meal_time:
            # Add estimated meal time marker
            est_meal_time_ms = int(event.estimated_meal_time.timestamp() * 1000)

            for row in [1, 2, 3]:
                fig.add_vline(
                    x=est_meal_time_ms,
                    line_width=2,
                    line_dash="solid",
                    line_color="#27AE60",
                    opacity=0.8,
                    row=row, col=1
                )

            # Annotation for estimated meal time
            fig.add_annotation(
                x=est_meal_time_ms,
                y=y_max + 15 + annotation_offset,
                text=f"🍽️ Est. Meal<br>{event.estimated_meal_time.strftime('%H:%M')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#27AE60",
                font=dict(size=10, color="#27AE60", weight="bold"),
                row=1, col=1
            )
            annotation_offset += 25

        elif event.event_type == "PEAK":
            # Annotation for peak
            fig.add_annotation(
                x=event_time_ms,
                y=event.glucose_at_detection + 10,
                text=f"📈 Peak<br>{event.detected_at.strftime('%H:%M')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                font=dict(size=9, color=color),
                row=1, col=1
            )

    # Add matched meal annotations if matches are provided
    if matches:
        for match in matches:
            # Find the glucose value at the matched event time for positioning
            event_time = match.event_time
            event_time_ms = int(event_time.timestamp() * 1000)
            meal_time_ms = int(match.meal_time.timestamp() * 1000)

            # Find closest glucose value in the dataframe
            time_diffs = abs(df["datetime_ist"] - event_time)
            closest_idx = time_diffs.idxmin()
            glucose_at_event = df.loc[closest_idx, "smoothed"]

            # Determine marker color based on match quality
            if match.composite_score >= 0.85:
                match_color = "#2ECC71"  # Green - excellent match
            elif match.composite_score >= 0.75:
                match_color = "#F39C12"  # Orange - good match
            else:
                match_color = "#E74C3C"  # Red - marginal match

            # Format event type for display
            if match.event_type == "MEAL_START":
                event_type_display = "Est. Meal"
            elif match.event_type == "SECONDARY_MEAL":
                event_type_display = "Est. Sec. Meal"
            else:
                event_type_display = match.event_type

            # Add a marker for the matched meal on the glucose plot
            fig.add_trace(
                go.Scatter(
                    x=[event_time],
                    y=[glucose_at_event],
                    mode="markers",
                    name=f"Matched: {match.meal_slot}",
                    marker=dict(
                        size=16,
                        color=match_color,
                        symbol="star",
                        line=dict(color="white", width=2)
                    ),
                    hovertemplate=(
                        f"<b>MATCHED EVENT</b><br>"
                        f"<b>Meal:</b> {match.meal_slot}<br>"
                        f"<b>Meal Name:</b> {match.meal_name[:25]}{'...' if len(match.meal_name) > 25 else ''}<br>"
                        f"<b>Meal Time:</b> {match.meal_time.strftime('%H:%M')}<br>"
                        f"<b>Event Type:</b> {event_type_display}<br>"
                        f"<b>Event Time:</b> {match.event_time.strftime('%H:%M')}<br>"
                        f"<b>Time Offset:</b> {match.time_offset_minutes:+.0f} min<br>"
                        f"<b>---Scores---</b><br>"
                        f"<b>S_time:</b> {match.s_time:.2f}<br>"
                        f"<b>S_slot:</b> {match.s_slot:.2f}<br>"
                        f"<b>S_physio:</b> {match.s_physio:.2f}<br>"
                        f"<b>S_size:</b> {match.s_size:.2f}<br>"
                        f"<b>Composite:</b> {match.composite_score:.1%}<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )

            # Add annotation with match info
            fig.add_annotation(
                x=event_time_ms,
                y=glucose_at_event - 15,
                text=f"✓ {match.meal_slot}<br>{match.composite_score:.0%}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=match_color,
                font=dict(size=9, color=match_color, weight="bold"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=match_color,
                borderwidth=1,
                row=1, col=1
            )

            # Add a connecting line from meal time to event time on the glucose plot
            # Find glucose at meal time
            meal_time_diffs = abs(df["datetime_ist"] - match.meal_time)
            meal_closest_idx = meal_time_diffs.idxmin()
            glucose_at_meal = df.loc[meal_closest_idx, "smoothed"]

            # Draw a dashed line connecting meal time to event time
            fig.add_trace(
                go.Scatter(
                    x=[match.meal_time, event_time],
                    y=[glucose_at_meal, glucose_at_event],
                    mode="lines",
                    line=dict(color=match_color, width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=1, col=1
            )

            # Add a small marker at meal time
            fig.add_trace(
                go.Scatter(
                    x=[match.meal_time],
                    y=[glucose_at_meal],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=match_color,
                        symbol="circle",
                        line=dict(color="white", width=1)
                    ),
                    hovertemplate=(
                        f"<b>LOGGED MEAL</b><br>"
                        f"<b>{match.meal_slot}:</b> {match.meal_name[:30]}<br>"
                        f"<b>Time:</b> {match.meal_time.strftime('%H:%M')}<br>"
                        f"<b>Carbs:</b> {match.carbs:.0f}g | <b>Protein:</b> {match.protein:.0f}g<br>"
                        f"<b>Fat:</b> {match.fat:.0f}g | <b>Fiber:</b> {match.fiber:.0f}g<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )

    # Update layout
    fig.update_layout(
        height=850,
        showlegend=True,
        hovermode="x unified",
        title="Simplified Zero-Crossing Meal Detection"
    )

    fig.update_xaxes(tickformat="%H:%M", dtick=3600000 * 2, row=3, col=1)
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
    fig.update_yaxes(title_text="dG/dt (mg/dL/min)", row=2, col=1)
    fig.update_yaxes(title_text="d²G/dt² (mg/dL/min²)", row=3, col=1)

    return fig


def create_simple_cgm_plot(cgm_df: pd.DataFrame) -> go.Figure:
    """Create a simple CGM plot without meal detection."""
    fig = go.Figure()

    # CGM line
    fig.add_trace(go.Scatter(
        x=cgm_df["datetime_ist"],
        y=cgm_df["value"],
        mode="lines+markers",
        name="CGM Glucose",
        line=dict(color="#2E86AB", width=2),
        marker=dict(size=4),
        hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<extra></extra>"
    ))

    # Get y-axis range
    y_min = max(0, cgm_df["value"].min() - 20)
    y_max = cgm_df["value"].max() + 30

    # Reference lines for glucose ranges
    fig.add_shape(type="line", x0=0, x1=1, y0=70, y1=70,
                  line=dict(color="orange", width=1, dash="dot"),
                  xref="paper", yref="y")
    fig.add_shape(type="line", x0=0, x1=1, y0=140, y1=140,
                  line=dict(color="orange", width=1, dash="dot"),
                  xref="paper", yref="y")
    fig.add_annotation(x=0, y=70, text="Low (70)", showarrow=False,
                       xref="paper", font=dict(size=10, color="orange"), xanchor="right", xshift=-5)
    fig.add_annotation(x=0, y=140, text="High (140)", showarrow=False,
                       xref="paper", font=dict(size=10, color="orange"), xanchor="right", xshift=-5)

    fig.update_layout(
        title="CGM Data - 24 Hour View",
        xaxis_title="Time (IST)",
        yaxis_title="Glucose (mg/dL)",
        hovermode="x unified",
        height=500,
        showlegend=True,
        xaxis=dict(
            tickformat="%H:%M",
            dtick=3600000 * 2,
        ),
        yaxis=dict(
            range=[y_min, y_max]
        )
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="CGM Meal Detection",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 CGM Visualization & Meal Detection")
    st.markdown("Visualize CGM data and detect meal times using derivative analysis.")

    # Sidebar for controls
    st.sidebar.header("Controls")

    # User selection
    users = get_available_users()
    selected_user = st.sidebar.selectbox("Select User", users)

    # Load data for selected user
    cgm_df = load_cgm_data(selected_user)
    meals_df = load_meals_data(selected_user)

    # Date selection
    available_dates = get_available_dates(cgm_df)
    selected_date = st.sidebar.selectbox(
        "Select Date",
        available_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d (%A)")
    )

    # Extend view past midnight
    extend_hours = st.sidebar.slider(
        "Extend past midnight (hours)",
        min_value=0, max_value=6, value=2,
        help="Show CGM data extending into the next day to see dinner effects"
    )

    # Filter data for selected date
    day_cgm = filter_data_for_date(cgm_df, selected_date, extend_hours=extend_hours)
    day_meals = filter_data_for_date(meals_df, selected_date, extend_hours=extend_hours)

    # Detection method selection
    st.sidebar.header("Meal Detection")
    enable_detection = st.sidebar.checkbox("Enable Meal Detection", value=True)

    # Main content area
    if enable_detection:
        # Simplified zero-crossing method
        st.sidebar.header("Detection Parameters")

        smoothing_window = st.sidebar.slider(
            "Smoothing Window",
            min_value=3, max_value=9, value=5, step=2,
            help="Rolling average window size (must be odd)"
        )

        min_derivative = st.sidebar.slider(
            "Min Derivative Magnitude",
            min_value=0.01, max_value=1.0, value=0.01, step=0.01,
            help="Minimum |dG/dt| to consider a zero-crossing significant. Higher = fewer events (filters noise)."
        )

        start_hour = st.sidebar.slider(
            "Start Hour",
            min_value=5, max_value=10, value=7, step=1,
            help="Only detect events after this hour (24h format)"
        )

        absorption_lag = st.sidebar.slider(
            "Absorption Lag (min)",
            min_value=0, max_value=20, value=0, step=1,
            help="Time from eating to detectable glucose rise"
        )

        simplified_thresholds = SimplifiedThresholds(
            smoothing_window=smoothing_window,
            min_derivative_magnitude=min_derivative,
            start_hour=start_hour,
            meal_absorption_lag=absorption_lag
        )

        if len(day_cgm) > 0:
            # Detect events using simplified method
            events = detect_meal_events_simplified(day_cgm, simplified_thresholds)

            # Perform matching first so we can show matches on the plot
            matches_for_plot = []
            if len(day_meals) > 0 and events:
                matches_for_plot, _, _ = match_meals_to_events(
                    day_meals, events, cgm_df=day_cgm, thresholds=simplified_thresholds
                )

            # Create and show plot (with matches if available)
            fig = create_simplified_derivative_plot(day_cgm, events, simplified_thresholds, matches=matches_for_plot)
            st.plotly_chart(fig, use_container_width=True)

            # Show detected events
            if events:
                meal_events = [e for e in events if e.event_type == "MEAL_START"]
                peak_events = [e for e in events if e.event_type == "PEAK"]

                if meal_events:
                    st.subheader("🍽️ Estimated Meal Times")
                    st.markdown("*Detected when dG/dt crosses from negative to positive (local minimum):*")

                    meal_cols = st.columns(min(len(meal_events), 4))
                    for i, event in enumerate(meal_events):
                        with meal_cols[i % 4]:
                            hour = event.estimated_meal_time.hour
                            if 5 <= hour < 11:
                                meal_label = "Breakfast"
                            elif 11 <= hour < 15:
                                meal_label = "Lunch"
                            elif 15 <= hour < 18:
                                meal_label = "Snack"
                            else:
                                meal_label = "Dinner"

                            st.metric(
                                label=f"{meal_label}",
                                value=event.estimated_meal_time.strftime("%H:%M"),
                                delta=f"{event.confidence:.0%} confidence"
                            )

                if peak_events:
                    st.subheader("📈 Post-Prandial Peaks")
                    st.markdown("*Detected when dG/dt crosses from positive to negative (local maximum):*")

                    peak_cols = st.columns(min(len(peak_events), 4))
                    for i, event in enumerate(peak_events):
                        with peak_cols[i % 4]:
                            st.metric(
                                label=f"Peak @ {event.glucose_at_detection:.0f} mg/dL",
                                value=event.detected_at.strftime("%H:%M"),
                                delta=f"{event.confidence:.0%} confidence"
                            )

                # Show all events in detailed table
                with st.expander(f"📊 All Detected Events ({len(events)} total)", expanded=False):
                    events_data = []
                    for event in events:
                        row = {
                            "Type": event.event_type,
                            "Detected At": event.detected_at.strftime("%H:%M:%S"),
                            "Est. Meal Time": event.estimated_meal_time.strftime("%H:%M:%S") if event.estimated_meal_time else "N/A",
                            "Confidence": f"{event.confidence:.0%}",
                            "dG/dt": f"{event.dG_dt_at_detection:.3f}",
                            "Glucose": f"{event.glucose_at_detection:.1f}"
                        }
                        events_data.append(row)

                    events_df = pd.DataFrame(events_data)
                    st.dataframe(events_df, use_container_width=True, hide_index=True)

                    st.markdown(f"""
                    **Event Types:**
                    - 🟢 **MEAL_START**: dG/dt crosses from negative to positive (local minimum = meal onset)
                    - 🔵 **PEAK**: dG/dt crosses from positive to negative (local maximum = post-prandial peak)

                    **Chart Legend:**
                    - 🟢 **Solid green lines**: Estimated meal times (when you likely ate)
                    - 🔵 **Dotted blue lines**: Peak times (post-prandial peak)
                    - 🟣 **Dash-dot purple line**: Detection start time ({start_hour}:00)
                    - ⬜ **Gray shaded zone**: Noise threshold (crossings here are ignored)
                    """)

                # =============================================================
                # MEAL MATCHING ANALYSIS SECTION
                # =============================================================
                if len(day_meals) > 0:
                    st.divider()
                    st.subheader("🔗 Meal-to-CGM Event Matching Analysis")
                    st.markdown("*Matching logged meals to detected CGM events using multi-factor scoring.*")

                    # Perform matching (pass cgm_df and thresholds for secondary meal detection)
                    matches, unmatched_indices, meal_events_for_matching = match_meals_to_events(
                        day_meals, events, cgm_df=day_cgm, thresholds=simplified_thresholds
                    )

                    if matches:
                        # Sort matches by meal time for display
                        matches_sorted = sorted(matches, key=lambda m: m.meal_time)

                        # Create match results table
                        match_data = []
                        for match in matches_sorted:
                            # Format event type for display
                            if match.event_type == "MEAL_START":
                                event_type_display = "Est. Meal"
                            elif match.event_type == "SECONDARY_MEAL":
                                event_type_display = "Est. Secondary Meal"
                            else:
                                event_type_display = match.event_type

                            match_data.append({
                                "Meal Slot": match.meal_slot,
                                "Meal Name": match.meal_name[:30] + "..." if len(match.meal_name) > 30 else match.meal_name,
                                "Meal Time": match.meal_time.strftime("%H:%M"),
                                "Event Type": event_type_display,
                                "Event Time": match.event_time.strftime("%H:%M"),
                                "Time Offset": f"{match.time_offset_minutes:+.0f} min",
                                "S_time": f"{match.s_time:.2f}",
                                "S_slot": f"{match.s_slot:.2f}",
                                "S_physio": f"{match.s_physio:.2f}",
                                "S_size": f"{match.s_size:.2f}",
                                "Composite": f"{match.composite_score:.1%}"
                            })

                        match_df = pd.DataFrame(match_data)

                        # Style the dataframe
                        st.dataframe(
                            match_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Composite": st.column_config.TextColumn(
                                    "Composite Score",
                                    help="Weighted score: 20% time + 40% slot + 20% physio + 20% size"
                                ),
                                "S_time": st.column_config.TextColumn(
                                    "S_time",
                                    help="Time proximity score (0-1)"
                                ),
                                "S_slot": st.column_config.TextColumn(
                                    "S_slot",
                                    help="Meal slot window score (0-1) - checks if event is in expected time window"
                                ),
                                "S_physio": st.column_config.TextColumn(
                                    "S_physio",
                                    help="Physiological plausibility score (0-1)"
                                ),
                                "S_size": st.column_config.TextColumn(
                                    "S_size",
                                    help="Meal size match score (0-1)"
                                ),
                                "Time Offset": st.column_config.TextColumn(
                                    "Time Offset",
                                    help="Meal time - Event time (positive = meal logged after detection)"
                                )
                            }
                        )

                        # Summary statistics
                        avg_composite = sum(m.composite_score for m in matches) / len(matches)
                        avg_offset = sum(m.time_offset_minutes for m in matches) / len(matches)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Match Confidence", f"{avg_composite:.1%}")
                        with col2:
                            st.metric("Average Time Offset", f"{avg_offset:+.0f} min")
                        with col3:
                            st.metric("Meals Matched", f"{len(matches)}/{len(day_meals)}")

                        # Scoring explanation expander
                        with st.expander("📐 Scoring Formula Details"):
                            st.markdown("""
                            **Composite Score Formula:**

                            *If logged meal time is within its expected slot window:*
                            ```
                            S_composite = 0.2·S_time + 0.4·S_slot + 0.2·S_physio + 0.2·S_size
                            ```

                            *If logged meal time is OUTSIDE its slot window (e.g., Breakfast at 1:30 AM):*
                            ```
                            S_time = 0, S_slot = 0  (logged time can't be trusted)
                            Only match events within the expected slot window
                            S_composite = 0.5·S_physio + 0.5·S_size
                            ```

                            **S_time (Time Proximity) - 20% weight (or 0% if time unreliable):**
                            - Measures how close logged meal time is to detected event time
                            - Set to 0 if logged time is outside the meal slot window

                            **S_slot (Meal Slot Window) - 40% weight (or 0% if time unreliable):**
                            - Checks if event falls within expected time window
                            - Breakfast/Morning Snack: 7:00-12:00
                            - Lunch/Afternoon Snack: 12:00-17:00
                            - Dinner/Evening Snack: 17:00-24:00
                            - When logged time is unreliable, we filter to only consider events in this window

                            **S_physio (Physiological Plausibility) - 20% or 50% weight:**
                            - Checks if time-to-peak matches expected based on meal composition
                            - Expected TTP: `30 + 3·fiber + 1.5·protein + 2·fat` minutes
                            - Weight increases to 50% when logged time is unreliable

                            **S_size (Meal Size Match) - 20% or 50% weight:**
                            - Uses Glycemic Load Proxy: `net_carbs × dampening`
                            - High GL meals should match "Est. Meal" events
                            - Low GL snacks should match "Est. Secondary Meal" events
                            - Weight increases to 50% when logged time is unreliable
                            """)

                    # Unmatched Events Section
                    if unmatched_indices:
                        st.divider()
                        st.subheader("❓ Unmatched CGM Events")
                        st.markdown("*These detected events didn't match any logged meal - possible unlogged meals or noise.*")

                        unmatched_data = []
                        for idx in unmatched_indices:
                            event = meal_events_for_matching[idx]
                            # Format event type for display
                            if event.event_type == "MEAL_START":
                                event_type_display = "Est. Meal"
                            elif event.event_type == "SECONDARY_MEAL":
                                event_type_display = "Est. Secondary Meal"
                            else:
                                event_type_display = event.event_type

                            unmatched_data.append({
                                "Event Type": event_type_display,
                                "Detected Time": event.detected_at.strftime("%H:%M"),
                                "Est. Meal Time": event.estimated_meal_time.strftime("%H:%M") if event.estimated_meal_time else "N/A",
                                "Glucose": f"{event.glucose_at_detection:.0f} mg/dL",
                                "Confidence": f"{event.confidence:.0%}"
                            })

                        unmatched_df = pd.DataFrame(unmatched_data)
                        st.dataframe(unmatched_df, use_container_width=True, hide_index=True)

                        st.info("💡 These events may represent unlogged snacks or meals. Consider reviewing your meal log for this day.")

            else:
                st.info("No meal events detected. Try lowering the 'Min Derivative Magnitude' to be more sensitive.")
        else:
            st.warning(f"No CGM data available for {selected_date}")

    else:
        # No detection - just show CGM plot
        col1, col2 = st.columns([3, 1])

        with col1:
            if len(day_cgm) > 0:
                fig = create_simple_cgm_plot(day_cgm)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No CGM data available for {selected_date}")

        with col2:
            # Summary stats
            st.subheader("📈 Day Summary")
            if len(day_cgm) > 0:
                st.metric("Average Glucose", f"{day_cgm['value'].mean():.1f} mg/dL")
                st.metric("Min", f"{day_cgm['value'].min():.1f} mg/dL")
                st.metric("Max", f"{day_cgm['value'].max():.1f} mg/dL")
                st.metric("Readings", len(day_cgm))

    # Meals table
    st.subheader("🍽️ Logged Meals for Selected Date")

    if len(day_meals) > 0:
        # Prepare display dataframe
        display_meals = day_meals[["time", "meal_slot", "meal_name", "calories",
                                    "protein", "carbohydrates", "fat", "fibre"]].copy()
        display_meals.columns = ["Time (IST)", "Meal Slot", "Meal Name", "Calories",
                                 "Protein (g)", "Carbs (g)", "Fat (g)", "Fibre (g)"]

        # Format numeric columns
        for col in ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Fibre (g)"]:
            display_meals[col] = display_meals[col].round(1)

        st.dataframe(display_meals, use_container_width=True, hide_index=True)
    else:
        st.info(f"No meals logged for {selected_date}")

    # User info in expander
    with st.expander("📋 User Information"):
        import json
        static_path = USER_DATA_PATH / selected_user / "static_data.json"
        with open(static_path) as f:
            user_info = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Age:** {user_info.get('age', 'N/A')}")
            st.write(f"**Gender:** {user_info.get('gender', 'N/A')}")
        with col2:
            st.write(f"**Weight:** {user_info.get('weight', 'N/A')} kg")
            st.write(f"**Diabetes Type:** {user_info.get('diabetes_type', 'N/A')}")
        with col3:
            st.write(f"**Cuisine Preference:** {', '.join(user_info.get('cuisine_pref', []))}")


if __name__ == "__main__":
    main()
