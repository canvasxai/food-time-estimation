"""
Meal detection algorithms using derivative analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from scipy.signal import find_peaks

from .models import MealEvent, CompositeMealEvent, SimplifiedThresholds
from .derivatives import apply_smoothing, calculate_first_derivative, calculate_second_derivative


def detect_secondary_meal_events(
    cgm_df: pd.DataFrame,
    thresholds: SimplifiedThresholds,
    return_filtered: bool = False,
    primary_events: List[MealEvent] = None
) -> List[MealEvent]:
    """
    Detect secondary meal events based on d²G/dt² maxima.

    A secondary meal is detected when d²G/dt² has a local maximum while
    dG/dt is above the configured threshold. By default (threshold=0), this
    requires glucose to be rising. A negative threshold allows detection
    when glucose is nearly flat or slightly declining.

    Args:
        cgm_df: DataFrame with CGM data
        thresholds: Detection thresholds
        return_filtered: If True, returns tuple of (passing_events, filtered_events, merged_events)
        primary_events: List of primary MEAL_START events for merge detection

    Returns:
        List of MealEvent objects with event_type="SECONDARY_MEAL"
        If return_filtered=True, returns tuple of (passing_events, filtered_events, merged_events)
    """
    if len(cgm_df) < 5:
        return ([], []) if return_filtered else []

    df = cgm_df.sort_values("datetime_ist").copy().reset_index(drop=True)

    # Determine primary date for filtering
    primary_date = df["datetime_ist"].dt.date.mode().iloc[0]

    # Apply smoothing and calculate derivatives
    df["smoothed"] = apply_smoothing(df, window=thresholds.smoothing_window)
    df["dG_dt"] = calculate_first_derivative(df)
    df["d2G_dt2"] = calculate_second_derivative(df)

    d2G_dt2_values = df["d2G_dt2"].values
    dG_dt_values = df["dG_dt"].values

    passing_events = []
    filtered_events = []

    # Use scipy find_peaks to detect local maxima in d²G/dt²
    # height=0 ensures we only get positive peaks (d²G/dt² > 0)
    peak_indices, _ = find_peaks(d2G_dt2_values, height=0)

    for i in peak_indices:
        # Check start_hour filter against the detection time
        detection_datetime = df.iloc[i]["datetime_ist"]
        detection_date = detection_datetime.date()
        detection_hour = detection_datetime.hour

        # Skip if before start hour ON THE PRIMARY DATE
        if detection_date == primary_date and detection_hour < thresholds.start_hour:
            continue

        # Estimated meal time is slightly before detection
        estimated_meal_time = detection_datetime - timedelta(minutes=thresholds.meal_absorption_lag // 2)

        # Confidence based on d²G/dt² magnitude
        confidence = min(0.5 + d2G_dt2_values[i] * 10, 1.0)

        event = MealEvent(
            event_type="SECONDARY_MEAL",
            detected_at=detection_datetime,
            estimated_meal_time=estimated_meal_time,
            confidence=confidence,
            dG_dt_at_detection=dG_dt_values[i],
            d2G_dt2_at_detection=d2G_dt2_values[i],
            glucose_at_detection=df.iloc[i]["smoothed"]
        )

        # Filter: dG/dt must be above threshold
        if dG_dt_values[i] > thresholds.secondary_meal_dg_dt_threshold:
            passing_events.append(event)
        elif return_filtered:
            event.event_type = "SECONDARY_MEAL_FILTERED"
            filtered_events.append(event)

    if return_filtered:
        # Determine which passing events would be merged with primary events
        merged_events = []
        remaining_passing = []

        if primary_events:
            for event in passing_events:
                is_merged = False
                for primary in primary_events:
                    if primary.event_type == "MEAL_START":
                        time_diff = abs((event.detected_at - primary.detected_at).total_seconds() / 60)
                        if time_diff <= thresholds.event_merge_gap_minutes:
                            # This event would be merged with a primary event
                            event.event_type = "SECONDARY_MEAL_MERGED"
                            merged_events.append(event)
                            is_merged = True
                            break
                if not is_merged:
                    remaining_passing.append(event)
        else:
            remaining_passing = passing_events

        return remaining_passing, filtered_events, merged_events

    return passing_events


def merge_nearby_meal_events(events: List[MealEvent], min_gap_minutes: int = 30) -> List[MealEvent]:
    """
    Merge meal events (MEAL_START and SECONDARY_MEAL) that are within min_gap_minutes of each other.

    When events are close together, keep the MEAL_START if present, otherwise keep the earlier event.
    PEAK events are not merged.

    Args:
        events: List of MealEvent objects
        min_gap_minutes: Minimum time gap between events to keep them separate

    Returns:
        List of MealEvent objects with nearby meal events merged
    """
    if not events:
        return events

    # Separate meal events from peak events
    meal_events = [e for e in events if e.event_type in ("MEAL_START", "SECONDARY_MEAL")]
    peak_events = [e for e in events if e.event_type == "PEAK"]

    if not meal_events:
        return events

    # Sort meal events by time
    meal_events.sort(key=lambda e: e.detected_at)

    merged = []
    i = 0

    while i < len(meal_events):
        current = meal_events[i]

        # Find all events within min_gap_minutes of current
        group = [current]
        j = i + 1
        while j < len(meal_events):
            time_diff = abs((meal_events[j].detected_at - current.detected_at).total_seconds() / 60)
            if time_diff <= min_gap_minutes:
                group.append(meal_events[j])
                j += 1
            else:
                break

        # From the group, prefer MEAL_START over SECONDARY_MEAL
        meal_starts = [e for e in group if e.event_type == "MEAL_START"]
        if meal_starts:
            # Keep the MEAL_START (use the first one if multiple)
            merged.append(meal_starts[0])
        else:
            # Keep the earliest SECONDARY_MEAL
            merged.append(group[0])

        i = j

    # Combine merged meal events with peak events
    return merged + peak_events


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

    # Detect PEAK events using scipy find_peaks on raw glucose values
    # This finds local maxima in the raw glucose signal
    raw_glucose = df["value"].values

    # Use find_peaks to detect local maxima in raw glucose
    # prominence: minimum prominence of peaks (how much they stand out)
    # distance: minimum samples between peaks
    peak_indices, peak_properties = find_peaks(
        raw_glucose,
        prominence=thresholds.peak_prominence,
        distance=thresholds.peak_distance
    )

    for i, idx in enumerate(peak_indices):
        # Get timestamp directly from DataFrame to preserve timezone
        peak_timestamp = df.iloc[idx]["datetime_ist"]
        peak_date = peak_timestamp.date()
        peak_hour = peak_timestamp.hour

        # Apply start_hour filter
        if peak_date == primary_date and peak_hour < thresholds.start_hour:
            continue

        peak_glucose = raw_glucose[idx]

        # Calculate confidence based on prominence (how much peak stands out)
        prominence = peak_properties["prominences"][i]
        confidence = min(0.5 + (prominence / 50.0) * 0.5, 1.0)

        # Get derivative values at peak for reference
        dG_dt_at_peak = df.iloc[idx]["dG_dt"] if idx < len(df) else 0.0

        events.append(MealEvent(
            event_type="PEAK",
            detected_at=peak_timestamp,
            estimated_meal_time=None,
            confidence=confidence,
            dG_dt_at_detection=dG_dt_at_peak,
            d2G_dt2_at_detection=0.0,
            glucose_at_detection=peak_glucose
        ))

    # Detect secondary meal events and merge with primary events
    secondary_events = detect_secondary_meal_events(cgm_df, thresholds)
    all_events = events + secondary_events

    # Merge nearby meal events (MEAL_START and SECONDARY_MEAL within configured gap)
    merged_events = merge_nearby_meal_events(all_events, min_gap_minutes=thresholds.event_merge_gap_minutes)

    return merged_events


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

    Classification:
    - CLEAN: Has a well-defined peak within 180 minutes with no intervening events
    - COMPOSITE: Has other events between the meal event and peak (stacked meals)

    A composite event is formed when:
    1. A MEAL_START is followed by one or more SECONDARY_MEAL events
    2. WITHOUT an intervening PEAK between them
    3. This indicates glucose was still rising when secondary meals occurred (stacked)

    For composite events, we track:
    - Overall delta G (from event to eventual peak)
    - Delta G to next event (from current glucose to glucose at next event)

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

        # Calculate time to peak in minutes
        time_to_peak_minutes = None
        if next_peak:
            time_to_peak_minutes = (next_peak.detected_at - meal_start.detected_at).total_seconds() / 60

        # Determine if event is clean:
        # - Has a peak within 180 minutes
        # - No intervening events (no stacked secondary meals)
        is_clean = (
            next_peak is not None and
            len(stacked_secondary) == 0 and
            time_to_peak_minutes is not None and
            time_to_peak_minutes <= 180
        )

        # For composite events, calculate delta G to next event
        delta_g_to_next_event = None
        next_event_time = None
        if stacked_secondary:
            # Get the first secondary event after meal_start
            first_secondary = min(stacked_secondary, key=lambda s: s.detected_at)
            delta_g_to_next_event = first_secondary.glucose_at_detection - glucose_at_start
            next_event_time = first_secondary.detected_at

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
            is_stacked=len(stacked_secondary) > 0,
            is_clean=is_clean,
            time_to_peak_minutes=time_to_peak_minutes,
            delta_g_to_next_event=delta_g_to_next_event,
            next_event_time=next_event_time
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
            time_to_peak_minutes = None
            if next_peak:
                glucose_at_peak = next_peak.glucose_at_detection
                total_glucose_rise = glucose_at_peak - sec.glucose_at_detection
                time_to_peak_minutes = (next_peak.detected_at - sec.detected_at).total_seconds() / 60

            # Standalone secondary events are clean if peak within 180 minutes
            is_clean = (
                next_peak is not None and
                time_to_peak_minutes is not None and
                time_to_peak_minutes <= 180
            )

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
                is_stacked=False,
                is_clean=is_clean,
                time_to_peak_minutes=time_to_peak_minutes,
                delta_g_to_next_event=None,
                next_event_time=None
            )

            composite_events.append(composite)

    return sorted(composite_events, key=lambda c: c.start_time)
