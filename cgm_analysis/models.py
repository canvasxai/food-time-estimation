"""
Data classes for CGM meal detection and matching.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class MealEvent:
    """Represents a detected meal event."""
    event_type: str  # "MEAL_START", "SECONDARY_MEAL", "PEAK"
    detected_at: datetime  # When the signal was detected in CGM data
    estimated_meal_time: Optional[datetime]  # Back-calculated actual eating time
    confidence: float  # 0.0 - 1.0 based on signal strength
    dG_dt_at_detection: float  # First derivative value
    d2G_dt2_at_detection: float  # Second derivative value
    glucose_at_detection: float  # Glucose value at detection point


@dataclass
class CompositeMealEvent:
    """
    Represents a composite meal event where multiple meals are stacked.

    A composite event occurs when a MEAL_START is followed by SECONDARY_MEAL(s)
    without an intervening PEAK - meaning glucose is still rising when the
    secondary meal occurs. In this case:
    - We can't reliably calculate time-to-peak for individual meals (S_physio unreliable)
    - We can't attribute glucose rise to individual meals (S_size unreliable)
    - We CAN validate the combined glycemic load against total glucose rise

    Multiple logged meals can be matched to a single composite event.
    """
    event_id: str  # Unique identifier for this composite event
    start_time: datetime  # Time of the initial MEAL_START event
    end_time: datetime  # Time of the last event before the peak (or last secondary)
    peak_time: Optional[datetime]  # Time of the eventual peak (if found)
    primary_event: MealEvent  # The initial MEAL_START event
    secondary_events: List[MealEvent]  # List of SECONDARY_MEAL events in this composite
    glucose_at_start: float  # Glucose level at the start
    glucose_at_peak: Optional[float]  # Glucose level at the peak
    total_glucose_rise: Optional[float]  # peak - start glucose (if peak found)
    is_stacked: bool  # True if there are secondary events (stacked meals)


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
    # Composite event reference (if matched to a composite)
    composite_event_id: Optional[str] = None
    is_stacked_meal: bool = False


@dataclass
class SimplifiedThresholds:
    """Configuration thresholds for simplified zero-crossing detection."""
    smoothing_window: int = 5  # Rolling average window size
    min_derivative_magnitude: float = 0.01  # Minimum |dG/dt| to consider a crossing significant
    start_hour: int = 7  # Only detect events after this hour (24h format)
    meal_absorption_lag: int = 0  # minutes - time from eating to detectable rise
