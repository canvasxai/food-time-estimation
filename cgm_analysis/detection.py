"""
Meal detection algorithms using derivative analysis.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from .models import MealEvent, CompositeMealEvent, SimplifiedThresholds
from .derivatives import apply_smoothing, calculate_first_derivative, calculate_second_derivative


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
        # Check for local maximum in d²G/dt² while glucose is rising
        # A meal event requires:
        # 1. d²G/dt² is at a local maximum (acceleration peak)
        # 2. d²G/dt² > 0 (positive acceleration)
        # 3. dG/dt > 0 (glucose is increasing)
        # When dG/dt <= 0 (glucose falling), it's a "change event", not a meal event
        if (d2G_dt2_values[i] > d2G_dt2_values[i - 1] and
            d2G_dt2_values[i] > d2G_dt2_values[i + 1] and
            d2G_dt2_values[i] > 0 and  # Positive acceleration (d²G/dt² > 0)
            dG_dt_values[i] > 0):  # Glucose must be rising (dG/dt > 0)

            # Check start_hour filter against the ORIGINAL detection time (index i)
            detection_datetime = df.iloc[i]["datetime_ist"]
            detection_date = detection_datetime.date()
            detection_hour = detection_datetime.hour

            # Skip if before start hour ON THE PRIMARY DATE
            if detection_date == primary_date and detection_hour < thresholds.start_hour:
                continue

            # Use the PREVIOUS reading (i-1) for the event time
            event_idx = i - 1
            curr_datetime = df.iloc[event_idx]["datetime_ist"]

            # Estimated meal time is slightly before detection (less lag than primary)
            estimated_meal_time = curr_datetime - timedelta(minutes=thresholds.meal_absorption_lag // 2)

            # Confidence based on d²G/dt² magnitude
            confidence = min(0.5 + d2G_dt2_values[i] * 10, 1.0)

            events.append(MealEvent(
                event_type="SECONDARY_MEAL",
                detected_at=curr_datetime,
                estimated_meal_time=estimated_meal_time,
                confidence=confidence,
                dG_dt_at_detection=dG_dt_values[event_idx],
                d2G_dt2_at_detection=d2G_dt2_values[i],
                glucose_at_detection=df.iloc[event_idx]["smoothed"]
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
    if val_curr == val_prev:
        alpha = 0.5
    else:
        alpha = -val_prev / (val_curr - val_prev)

    alpha = max(0.0, min(1.0, alpha))

    time_diff = (t_curr - t_prev).total_seconds()
    interpolated_time = t_prev + timedelta(seconds=time_diff * alpha)
    interpolated_glucose = glucose_prev + alpha * (glucose_curr - glucose_prev)

    return interpolated_time, interpolated_glucose


def detect_meal_events_simplified(cgm_df: pd.DataFrame, thresholds: SimplifiedThresholds = None) -> List[MealEvent]:
    """
    Simplified meal detection using zero-crossings of the first derivative.

    Logic:
    - MEAL_START: When dG/dt crosses from negative to positive (local minimum)
    - PEAK: When dG/dt crosses from positive to negative (local maximum)

    Both events use linear interpolation to find the exact zero-crossing point.

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

    df = cgm_df.sort_values("datetime_ist").copy().reset_index(drop=True)
    primary_date = df["datetime_ist"].dt.date.mode().iloc[0]

    df["smoothed"] = apply_smoothing(df, window=thresholds.smoothing_window)
    df["dG_dt"] = calculate_first_derivative(df)

    events = []

    for t in range(1, len(df)):
        timestamp_curr = df.iloc[t]["datetime_ist"]
        timestamp_prev = df.iloc[t - 1]["datetime_ist"]
        glucose_curr = df.iloc[t]["smoothed"]
        glucose_prev = df.iloc[t - 1]["smoothed"]
        dG_dt_curr = df.iloc[t]["dG_dt"]
        dG_dt_prev = df.iloc[t - 1]["dG_dt"]

        curr_date = timestamp_curr.date()
        curr_hour = timestamp_curr.hour

        if curr_date == primary_date and curr_hour < thresholds.start_hour:
            continue

        magnitude_before = abs(dG_dt_prev)
        magnitude_after = abs(dG_dt_curr)

        has_significant_crossing = (magnitude_before >= thresholds.min_derivative_magnitude or
                                    magnitude_after >= thresholds.min_derivative_magnitude)

        if not has_significant_crossing:
            continue

        # Check for negative to positive crossing (MEAL_START)
        if dG_dt_prev < 0 and dG_dt_curr > 0:
            crossing_time, crossing_glucose = interpolate_zero_crossing(
                timestamp_prev, timestamp_curr,
                dG_dt_prev, dG_dt_curr,
                glucose_prev, glucose_curr
            )

            estimated_meal_time = crossing_time - timedelta(minutes=thresholds.meal_absorption_lag)
            transition_magnitude = abs(dG_dt_curr - dG_dt_prev)
            confidence = min(0.5 + (transition_magnitude / 2.0) * 0.5, 1.0)

            events.append(MealEvent(
                event_type="MEAL_START",
                detected_at=crossing_time,
                estimated_meal_time=estimated_meal_time,
                confidence=confidence,
                dG_dt_at_detection=0.0,
                d2G_dt2_at_detection=0.0,
                glucose_at_detection=crossing_glucose
            ))

        # Check for positive to negative crossing (PEAK)
        elif dG_dt_prev > 0 and dG_dt_curr < 0:
            crossing_time, crossing_glucose = interpolate_zero_crossing(
                timestamp_prev, timestamp_curr,
                dG_dt_prev, dG_dt_curr,
                glucose_prev, glucose_curr
            )

            transition_magnitude = abs(dG_dt_curr - dG_dt_prev)
            confidence = min(0.5 + (transition_magnitude / 2.0) * 0.5, 1.0)

            events.append(MealEvent(
                event_type="PEAK",
                detected_at=crossing_time,
                estimated_meal_time=None,
                confidence=confidence,
                dG_dt_at_detection=0.0,
                d2G_dt2_at_detection=0.0,
                glucose_at_detection=crossing_glucose
            ))

    # Also detect "soft peaks" - local maxima where dG/dt approaches zero but doesn't cross
    df["d2G_dt2"] = calculate_second_derivative(df)
    d2G_dt2_values = df["d2G_dt2"].values
    dG_dt_array = df["dG_dt"].values
    smoothed_values = df["smoothed"].values

    for t in range(2, len(df) - 2):
        timestamp_curr = df.iloc[t]["datetime_ist"]
        curr_date = timestamp_curr.date()
        curr_hour = timestamp_curr.hour

        if curr_date == primary_date and curr_hour < thresholds.start_hour:
            continue

        dG_dt_curr = dG_dt_array[t]
        d2G_dt2_curr = d2G_dt2_values[t]
        glucose_curr = smoothed_values[t]

        is_near_zero = abs(dG_dt_curr) < thresholds.min_derivative_magnitude * 2
        is_concave_down = d2G_dt2_curr < -0.001
        is_local_max = (glucose_curr > smoothed_values[t-1] and
                        glucose_curr > smoothed_values[t+1] and
                        glucose_curr > smoothed_values[t-2] and
                        glucose_curr > smoothed_values[t+2])

        if is_near_zero and is_concave_down and is_local_max:
            nearby_peak = False
            for existing_event in events:
                if existing_event.event_type == "PEAK":
                    time_diff = abs((timestamp_curr - existing_event.detected_at).total_seconds() / 60)
                    if time_diff < 30:
                        nearby_peak = True
                        break

            if not nearby_peak:
                confidence = min(0.4 + abs(d2G_dt2_curr) * 5, 0.8)

                events.append(MealEvent(
                    event_type="PEAK",
                    detected_at=timestamp_curr,
                    estimated_meal_time=None,
                    confidence=confidence,
                    dG_dt_at_detection=dG_dt_curr,
                    d2G_dt2_at_detection=d2G_dt2_curr,
                    glucose_at_detection=glucose_curr
                ))

    return events


def find_associated_peak(event: MealEvent, events: List[MealEvent]) -> Optional[datetime]:
    """
    Find the peak event associated with a meal start event.

    Looks for the next PEAK event within a reasonable time window (10-180 min).
    """
    if event.event_type not in ["MEAL_START", "SECONDARY_MEAL"]:
        return None

    peaks = [e for e in events if e.event_type == "PEAK" and e.detected_at > event.detected_at]

    if not peaks:
        return None

    for peak in sorted(peaks, key=lambda p: p.detected_at):
        time_diff = (peak.detected_at - event.detected_at).total_seconds() / 60
        if 10 <= time_diff <= 180:
            return peak.detected_at

    return None


def detect_composite_events(events: List[MealEvent], cgm_df: pd.DataFrame) -> List[CompositeMealEvent]:
    """
    Detect composite meal events by grouping stacked meals.

    A composite event is formed when:
    1. A MEAL_START is followed by one or more SECONDARY_MEAL events
    2. WITHOUT an intervening PEAK between them
    3. This indicates glucose was still rising when secondary meals occurred (stacked)

    Clean events (MEAL_START → PEAK → MEAL_START) are also returned as non-stacked composites.

    Args:
        events: List of all detected MealEvent objects
        cgm_df: CGM DataFrame for glucose values

    Returns:
        List of CompositeMealEvent objects
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.detected_at)

    meal_starts = [e for e in sorted_events if e.event_type == "MEAL_START"]
    secondary_meals = [e for e in sorted_events if e.event_type == "SECONDARY_MEAL"]
    peaks = [e for e in sorted_events if e.event_type == "PEAK"]

    composite_events = []
    event_counter = 0

    for i, meal_start in enumerate(meal_starts):
        event_counter += 1
        event_id = f"CE_{event_counter}"

        # Find the next peak after this meal start
        next_peak = None
        for peak in peaks:
            if peak.detected_at > meal_start.detected_at:
                next_peak = peak
                break

        # Find the next meal start (for boundary detection)
        next_meal_start = None
        if i + 1 < len(meal_starts):
            next_meal_start = meal_starts[i + 1]

        # Determine the boundary for this composite event
        if next_peak and (not next_meal_start or next_peak.detected_at < next_meal_start.detected_at):
            boundary_time = next_peak.detected_at
            has_peak = True
        elif next_meal_start:
            boundary_time = next_meal_start.detected_at
            has_peak = False
            next_peak = None
        else:
            boundary_time = cgm_df["datetime_ist"].max() if len(cgm_df) > 0 else meal_start.detected_at + timedelta(hours=3)
            has_peak = False

        # Find secondary meals between meal_start and boundary
        stacked_secondary = []
        for sec in secondary_meals:
            if meal_start.detected_at < sec.detected_at < boundary_time:
                peak_between = any(
                    meal_start.detected_at < p.detected_at < sec.detected_at
                    for p in peaks
                )
                if not peak_between:
                    stacked_secondary.append(sec)

        # Determine end time
        if stacked_secondary:
            end_time = max(sec.detected_at for sec in stacked_secondary)
        else:
            end_time = meal_start.detected_at

        # Get glucose values
        glucose_at_start = meal_start.glucose_at_detection
        glucose_at_peak = next_peak.glucose_at_detection if next_peak else None
        total_glucose_rise = glucose_at_peak - glucose_at_start if glucose_at_peak else None

        composite = CompositeMealEvent(
            event_id=event_id,
            start_time=meal_start.estimated_meal_time or meal_start.detected_at,
            end_time=end_time,
            peak_time=next_peak.detected_at if next_peak else None,
            primary_event=meal_start,
            secondary_events=stacked_secondary,
            glucose_at_start=glucose_at_start,
            glucose_at_peak=glucose_at_peak,
            total_glucose_rise=total_glucose_rise,
            is_stacked=len(stacked_secondary) > 0
        )

        composite_events.append(composite)

    # Also create composites for standalone secondary events
    grouped_secondary_ids = set()
    for comp in composite_events:
        for sec in comp.secondary_events:
            grouped_secondary_ids.add(id(sec))

    for sec in secondary_meals:
        if id(sec) not in grouped_secondary_ids:
            event_counter += 1
            event_id = f"CE_{event_counter}"

            next_peak = None
            for peak in peaks:
                if peak.detected_at > sec.detected_at:
                    next_peak = peak
                    break

            glucose_at_peak = None
            total_glucose_rise = None
            if next_peak:
                glucose_at_peak = next_peak.glucose_at_detection
                total_glucose_rise = glucose_at_peak - sec.glucose_at_detection

            composite = CompositeMealEvent(
                event_id=event_id,
                start_time=sec.estimated_meal_time or sec.detected_at,
                end_time=sec.detected_at,
                peak_time=next_peak.detected_at if next_peak else None,
                primary_event=sec,
                secondary_events=[],
                glucose_at_start=sec.glucose_at_detection,
                glucose_at_peak=glucose_at_peak,
                total_glucose_rise=total_glucose_rise,
                is_stacked=False
            )

            composite_events.append(composite)

    return sorted(composite_events, key=lambda c: c.start_time)
