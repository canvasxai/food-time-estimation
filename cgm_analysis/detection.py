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


def detect_composite_events(events: List[MealEvent], cgm_df: pd.DataFrame,
                            merge_gap_minutes: int = 30,
                            max_peak_window_minutes: int = 180) -> List[CompositeMealEvent]:
    """
    Detect composite meal events by analyzing meal-to-peak relationships.

    Classification:
    - CLEAN: Meal event followed by peak within 180 minutes, no intervening meal events
             Also includes merged meals (events within merge_gap_minutes are treated as one)
    - COMPOSITE: Meal event followed by another distinct meal event, then a peak within 180 minutes
      ALL meal events that share the same peak are labeled as composite
    - NO_PEAK: Meal event where no peak is found within 180 minutes

    Algorithm:
    1. Collect all meal events (MEAL_START + SECONDARY_MEAL) sorted by time
    2. For each meal event, determine if it goes directly to a peak or to another meal first
    3. If the next meal is within merge_gap_minutes, treat as merged (clean)
    4. If multiple distinct meals share the same peak, ALL of them are marked as composite
    5. If no peak is found within max_peak_window_minutes, mark as no_peak

    Args:
        events: List of all detected MealEvent objects
        cgm_df: CGM DataFrame for glucose values
        merge_gap_minutes: Events within this gap are considered merged (clean)
        max_peak_window_minutes: Maximum time window to look for a peak (default 180)

    Returns:
        List of CompositeMealEvent objects
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.detected_at)

    # Collect all meal events (both MEAL_START and SECONDARY_MEAL)
    meal_events = [e for e in sorted_events if e.event_type in ("MEAL_START", "SECONDARY_MEAL")]
    peaks = [e for e in sorted_events if e.event_type == "PEAK"]

    if not meal_events:
        return []

    # === STEP 1: Build segments for each meal event ===
    # For each meal event, find:
    # - The next meal event (if any)
    # - The next peak after this event
    # - Whether segment goes to_next_meal or to_peak
    # - Whether the next meal is within merge gap (would be merged)

    event_segments = []

    for i, event in enumerate(meal_events):
        event_time = event.estimated_meal_time or event.detected_at
        event_glucose = event.glucose_at_detection

        # Find next meal event (if any)
        next_meal = meal_events[i + 1] if i + 1 < len(meal_events) else None

        # Find next peak after this event WITHIN the max_peak_window_minutes
        next_peak = None
        next_peak_within_window = None
        for peak in peaks:
            if peak.detected_at > event.detected_at:
                next_peak = peak  # First peak after event (any time)
                time_to_this_peak = (peak.detected_at - event.detected_at).total_seconds() / 60
                if time_to_this_peak <= max_peak_window_minutes:
                    next_peak_within_window = peak
                break

        # Check if this event is merged with previous or next event
        # An event is "merged" if it's within merge_gap of an adjacent event
        is_merged_with_next = False
        is_merged_with_prev = False

        if next_meal:
            time_to_next_meal = (next_meal.detected_at - event.detected_at).total_seconds() / 60
            is_merged_with_next = time_to_next_meal <= merge_gap_minutes

        prev_meal = meal_events[i - 1] if i > 0 else None
        if prev_meal:
            time_from_prev_meal = (event.detected_at - prev_meal.detected_at).total_seconds() / 60
            is_merged_with_prev = time_from_prev_meal <= merge_gap_minutes

        # Event is part of a merge group if merged with either neighbor
        is_part_of_merge_group = is_merged_with_next or is_merged_with_prev

        # Determine segment type
        # Key change: use next_peak_within_window to check if peak is within 180 min
        segment_type = "unknown"
        segment_end = None
        segment_end_glucose = None

        if next_peak_within_window is None and not is_part_of_merge_group:
            # No peak within 180 minutes and not part of a merge group
            segment_type = "no_peak"
            segment_end = None
            segment_end_glucose = None
        elif next_meal and next_peak_within_window:
            if next_meal.detected_at < next_peak_within_window.detected_at:
                # Next meal comes before peak
                if is_merged_with_next:
                    # Merged with next - treat as going to peak (clean)
                    segment_type = "to_peak_merged"
                    segment_end = next_peak_within_window.detected_at
                    segment_end_glucose = next_peak_within_window.glucose_at_detection
                else:
                    # Distinct meals - composite
                    segment_type = "to_next_meal"
                    segment_end = next_meal.detected_at
                    segment_end_glucose = next_meal.glucose_at_detection
            else:
                # Peak comes before next meal - segment goes to peak
                # If merged with previous, mark as merged
                segment_type = "to_peak_merged" if is_merged_with_prev else "to_peak"
                segment_end = next_peak_within_window.detected_at
                segment_end_glucose = next_peak_within_window.glucose_at_detection
        elif next_peak_within_window:
            # If merged with previous, mark as merged
            segment_type = "to_peak_merged" if is_merged_with_prev else "to_peak"
            segment_end = next_peak_within_window.detected_at
            segment_end_glucose = next_peak_within_window.glucose_at_detection
        elif next_meal:
            # No peak within window, but there's a next meal
            if is_merged_with_next:
                # Check if the merged group eventually has a peak within window
                # For now, mark as merged and let the next event determine
                segment_type = "to_peak_merged"
            else:
                # Check if there's a peak after the next meal within the window from THIS event
                # If yes, it's composite; if no, it's no_peak
                peak_after_next_meal = None
                for peak in peaks:
                    if peak.detected_at > next_meal.detected_at:
                        time_from_event = (peak.detected_at - event.detected_at).total_seconds() / 60
                        if time_from_event <= max_peak_window_minutes:
                            peak_after_next_meal = peak
                        break

                if peak_after_next_meal:
                    segment_type = "to_next_meal"
                    segment_end = next_meal.detected_at
                    segment_end_glucose = next_meal.glucose_at_detection
                else:
                    segment_type = "no_peak"
                    segment_end = None
                    segment_end_glucose = None
        else:
            # No next meal and no peak within window
            segment_type = "no_peak"
            segment_end = None
            segment_end_glucose = None

        # Calculate delta G
        delta_g = segment_end_glucose - event_glucose if segment_end_glucose and event_glucose else 0

        # Calculate time to peak (use the peak within window if available)
        time_to_peak = None
        if next_peak_within_window:
            time_to_peak = (next_peak_within_window.detected_at - event.detected_at).total_seconds() / 60

        event_segments.append({
            "event": event,
            "event_time": event_time,
            "event_glucose": event_glucose,
            "segment_end": segment_end,
            "segment_end_glucose": segment_end_glucose,
            "segment_type": segment_type,
            "delta_g": delta_g,
            "next_peak": next_peak,  # Keep the actual next peak (for reference)
            "next_peak_within_window": next_peak_within_window,  # Peak within 180 min
            "time_to_peak": time_to_peak,
            "is_composite": False,  # Will be determined in next step
            "is_no_peak": segment_type == "no_peak"
        })

    # === STEP 2: Determine which events are COMPOSITE ===
    # An event is COMPOSITE if:
    # - Its segment_type is "to_next_meal" (there's another distinct meal before its peak), OR
    # - It shares the same peak with a previous meal event that was "to_next_meal"
    #
    # Events with segment_type "to_peak" or "to_peak_merged" are CLEAN
    # Events with segment_type "no_peak" are NO_PEAK (no peak within 180 min)
    #
    # We iterate forward and propagate composite status

    for i in range(len(event_segments)):
        seg = event_segments[i]

        if seg["segment_type"] == "no_peak":
            # Already marked as no_peak, skip
            continue
        elif seg["segment_type"] == "to_next_meal":
            # This event goes to another distinct meal before peak - it's composite
            seg["is_composite"] = True
        elif seg["segment_type"] in ("to_peak", "to_peak_merged"):
            # Check if any previous event shares the same peak AND was composite
            if i > 0:
                prev_seg = event_segments[i - 1]
                if (prev_seg["next_peak_within_window"] == seg["next_peak_within_window"] and
                    prev_seg["next_peak_within_window"] is not None and
                    prev_seg["segment_type"] == "to_next_meal"):
                    # Previous event was heading to a distinct meal, and we share the same peak
                    # Both are composite
                    seg["is_composite"] = True
                    # Also mark all previous events that share this peak as composite
                    for j in range(i - 1, -1, -1):
                        if (event_segments[j]["next_peak_within_window"] == seg["next_peak_within_window"] and
                            event_segments[j]["next_peak_within_window"] is not None):
                            event_segments[j]["is_composite"] = True
                        else:
                            break

    # === STEP 3: Create CompositeMealEvent objects ===
    composite_events = []

    for i, seg in enumerate(event_segments):
        event = seg["event"]
        next_peak_within_window = seg["next_peak_within_window"]
        is_composite = seg["is_composite"]
        is_no_peak = seg["is_no_peak"]

        # Determine classification
        if is_no_peak:
            classification = "no_peak"
            is_clean = False
        elif is_composite:
            classification = "composite"
            is_clean = False
        else:
            classification = "clean"
            is_clean = True

        # Calculate delta G to next event (only for composite events going to next meal)
        delta_g_to_next_event = None
        next_event_time = None
        if seg["segment_type"] == "to_next_meal":
            delta_g_to_next_event = seg["delta_g"]
            next_event_time = seg["segment_end"]

        composite = CompositeMealEvent(
            event_id=f"E_{i}",
            start_time=seg["event_time"],
            end_time=seg["event_time"],
            peak_time=next_peak_within_window.detected_at if next_peak_within_window else None,
            primary_event=event,
            secondary_events=[],  # Not tracking secondary events separately anymore
            glucose_at_start=seg["event_glucose"],
            glucose_at_peak=next_peak_within_window.glucose_at_detection if next_peak_within_window else None,
            total_glucose_rise=seg["delta_g"] if seg["segment_type"] in ("to_peak", "to_peak_merged") else None,
            is_stacked=is_composite,  # True if multiple distinct meals share this peak
            is_clean=is_clean,
            is_no_peak=is_no_peak,
            classification=classification,
            time_to_peak_minutes=seg["time_to_peak"],
            delta_g_to_next_event=delta_g_to_next_event,
            next_event_time=next_event_time
        )

        composite_events.append(composite)

    return sorted(composite_events, key=lambda c: c.start_time)
