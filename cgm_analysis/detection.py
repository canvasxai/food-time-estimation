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

        # Skip if before start hour or after end hour ON THE PRIMARY DATE
        if detection_date == primary_date:
            if detection_hour < thresholds.start_hour:
                continue
            if thresholds.end_hour < 24 and detection_hour >= thresholds.end_hour:
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

        # Apply start_hour and end_hour filters on the primary date
        if curr_date == primary_date:
            if curr_hour < thresholds.start_hour:
                continue
            if thresholds.end_hour < 24 and curr_hour >= thresholds.end_hour:
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

        # Apply start_hour and end_hour filters
        if peak_date == primary_date:
            if peak_hour < thresholds.start_hour:
                continue
            if thresholds.end_hour < 24 and peak_hour >= thresholds.end_hour:
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


def detect_composite_events(events: List[MealEvent], cgm_df: pd.DataFrame = None,
                            merge_gap_minutes: int = 30,
                            max_peak_window_minutes: int = 180,
                            return_discarded: bool = False):
    """
    Detect and classify composite meal events by analyzing meal-to-peak relationships.

    THIS IS THE SINGLE SOURCE OF TRUTH FOR EVENT CLASSIFICATION.
    All classification (clean/composite/no_peak) happens here.

    Classification:
    - CLEAN: Meal event followed by peak within 180 minutes, no intervening meal events
             Also includes merged meals (events within merge_gap_minutes are treated as one)
    - COMPOSITE: Meal event followed by another distinct meal event, then a peak within 180 minutes
      ALL meal events that share the same peak are labeled as composite
    - NO_PEAK: Meal event where no peak is found within 180 minutes

    Algorithm:
    1. Collect all meal events (MEAL_START + SECONDARY_MEAL) sorted by time
    2. Build merge groups (events within merge_gap_minutes)
    3. For each merge group, determine if it goes directly to a peak or to another meal first
    4. If multiple distinct meal groups share the same peak, ALL of them are marked as composite
    5. If no peak is found within max_peak_window_minutes, mark as no_peak
    6. For merged groups, use the mean time as the event time

    Args:
        events: List of all detected MealEvent objects
        cgm_df: CGM DataFrame for glucose values (optional, not currently used)
        merge_gap_minutes: Events within this gap are considered merged (clean)
        max_peak_window_minutes: Maximum time window to look for a peak (default 180)
        return_discarded: If True, also return list of discarded events (insufficient ΔG)

    Returns:
        List of CompositeMealEvent objects with classification already set
        If return_discarded=True, returns tuple of (composite_events, discarded_events)
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
    event_segments = []

    for i, event in enumerate(meal_events):
        event_time = event.estimated_meal_time or event.detected_at
        event_glucose = event.glucose_at_detection

        # Find next meal event (if any)
        next_meal = meal_events[i + 1] if i + 1 < len(meal_events) else None

        # Find next peak after this event
        next_peak = None
        next_peak_within_window = None
        for peak in peaks:
            if peak.detected_at > event.detected_at:
                next_peak = peak
                time_to_this_peak = (peak.detected_at - event.detected_at).total_seconds() / 60
                if time_to_this_peak <= max_peak_window_minutes:
                    next_peak_within_window = peak
                break

        # Determine segment type and end point
        segment_type = "unknown"
        segment_end = None
        segment_end_glucose = None

        if next_meal and next_peak_within_window:
            if next_meal.detected_at < next_peak_within_window.detected_at:
                segment_type = "to_next_meal"
                segment_end = next_meal.detected_at
                segment_end_glucose = next_meal.glucose_at_detection
            else:
                segment_type = "to_peak"
                segment_end = next_peak_within_window.detected_at
                segment_end_glucose = next_peak_within_window.glucose_at_detection
        elif next_peak_within_window:
            segment_type = "to_peak"
            segment_end = next_peak_within_window.detected_at
            segment_end_glucose = next_peak_within_window.glucose_at_detection
        elif next_meal:
            # Check if there's a peak after the next meal within window
            peak_after_next_meal = None
            for peak in peaks:
                if peak.detected_at > next_meal.detected_at:
                    time_from_event = (peak.detected_at - event.detected_at).total_seconds() / 60
                    if time_from_event <= max_peak_window_minutes:
                        peak_after_next_meal = peak
                        next_peak_within_window = peak  # Update for later use
                    break

            if peak_after_next_meal:
                segment_type = "to_next_meal"
                segment_end = next_meal.detected_at
                segment_end_glucose = next_meal.glucose_at_detection
            else:
                segment_type = "no_peak"
        else:
            segment_type = "no_peak"

        # Calculate delta G and time to peak
        delta_g = segment_end_glucose - event_glucose if segment_end_glucose and event_glucose else 0
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
            "next_peak": next_peak,
            "next_peak_within_window": next_peak_within_window,
            "time_to_peak": time_to_peak
        })

    # === STEP 2: Build merge groups ===
    # Group events that are within merge_gap_minutes of each other
    merge_groups = []
    used_indices = set()

    for i, seg in enumerate(event_segments):
        if i in used_indices:
            continue

        group = [i]
        used_indices.add(i)

        # Find all events that should be merged with this one
        j = i + 1
        while j < len(event_segments):
            prev_seg = event_segments[group[-1]]
            curr_seg = event_segments[j]
            time_diff = (curr_seg["event_time"] - prev_seg["event_time"]).total_seconds() / 60

            if time_diff <= merge_gap_minutes:
                group.append(j)
                used_indices.add(j)
                j += 1
            else:
                break

        merge_groups.append(group)

    # === STEP 3: Build group info for each merge group ===
    group_info = []
    for group_idx, group in enumerate(merge_groups):
        first_seg = event_segments[group[0]]
        last_seg = event_segments[group[-1]]

        # Calculate mean time for merged events
        if len(group) > 1:
            group_times = [event_segments[idx]["event_time"] for idx in group]
            mean_timestamp = sum(t.timestamp() for t in group_times) / len(group_times)
            start_time = datetime.fromtimestamp(mean_timestamp, tz=group_times[0].tzinfo)
        else:
            start_time = first_seg["event_time"]

        # Use the first event's glucose as start
        start_glucose = first_seg["event_glucose"]
        end_time = last_seg["event_time"]

        # Peak is the next peak after the last event in the group
        next_peak = last_seg["next_peak_within_window"]

        # Calculate time to peak from merged event time
        time_to_peak = None
        if next_peak:
            time_to_peak = (next_peak.detected_at - start_time).total_seconds() / 60

        # Calculate total glucose rise from start to peak
        total_rise = None
        if next_peak and start_glucose is not None:
            total_rise = next_peak.glucose_at_detection - start_glucose

        # Determine if this group goes to another distinct meal before peak
        segment_type = last_seg["segment_type"]
        goes_to_next_meal = segment_type == "to_next_meal"

        # Check if merged - if merged with next event, treat as going to peak
        if len(group) > 1 or (group[-1] + 1 < len(event_segments)):
            # Check if the "next meal" is actually within merge gap
            if goes_to_next_meal and group[-1] + 1 < len(event_segments):
                next_seg = event_segments[group[-1] + 1]
                time_to_next = (next_seg["event_time"] - last_seg["event_time"]).total_seconds() / 60
                if time_to_next <= merge_gap_minutes:
                    # Next event would be merged, so this is not truly going to next meal
                    goes_to_next_meal = False

        # Delta G to next event (always calculate for filtering, not just when goes_to_next_meal)
        delta_g_to_next = None
        next_event_time = None
        if goes_to_next_meal:
            delta_g_to_next = last_seg["delta_g"]
            next_event_time = last_seg["segment_end"]
        elif group[-1] + 1 < len(event_segments):
            # Even if not "goes_to_next_meal", calculate ΔG to next event for filtering
            next_seg = event_segments[group[-1] + 1]
            delta_g_to_next = next_seg["event_glucose"] - start_glucose
            next_event_time = next_seg["event_time"]

        group_info.append({
            "group_idx": group_idx,
            "indices": group,
            "start_time": start_time,
            "end_time": end_time,
            "start_glucose": start_glucose,
            "next_peak": next_peak,
            "time_to_peak": time_to_peak,
            "total_rise": total_rise,
            "goes_to_next_meal": goes_to_next_meal,
            "delta_g_to_next": delta_g_to_next,
            "next_event_time": next_event_time,
            "primary_event": first_seg["event"],
            "secondary_events": [event_segments[idx]["event"] for idx in group[1:]],
            "is_composite": False,
            "is_no_peak": next_peak is None
        })

    # === STEP 3.5: Filter out events with insufficient ΔG to next event ===
    # When glucose is flat, dG/dt hovers near zero causing spurious zero-crossings.
    # Discard events where ΔG to the next MEAL event is < 5 mg/dL (noise threshold).
    #
    # IMPORTANT: If an event goes directly to a peak (not to another meal), keep it!
    # Only discard if: goes_to_next_meal=True AND ΔG to next meal < threshold
    MIN_DELTA_G_THRESHOLD = 5.0  # mg/dL

    filtered_group_info = []
    discarded_group_info = []  # Track discarded events for visualization
    i = 0
    while i < len(group_info):
        info = group_info[i]

        # If this event goes directly to a peak (not to another meal), keep it
        # Only apply the ΔG filter when event goes to another meal first
        if info["goes_to_next_meal"]:
            # Check ΔG to next meal event
            has_next_group = (i + 1) < len(group_info)
            if has_next_group:
                next_group_info = group_info[i + 1]
                if info["start_glucose"] is not None and next_group_info["start_glucose"] is not None:
                    delta_g_to_next_group = next_group_info["start_glucose"] - info["start_glucose"]

                    if abs(delta_g_to_next_group) < MIN_DELTA_G_THRESHOLD:
                        # Discard this event - insufficient glucose change to next meal event (noise)
                        info["is_discarded"] = True
                        info["discard_reason"] = f"ΔG to next meal ({delta_g_to_next_group:.1f}) < {MIN_DELTA_G_THRESHOLD}"
                        discarded_group_info.append(info)
                        i += 1
                        continue

        # For events that go directly to peak, only discard if total rise is very small
        if not info["goes_to_next_meal"] and not info["is_no_peak"]:
            if info["total_rise"] is not None:
                if abs(info["total_rise"]) < MIN_DELTA_G_THRESHOLD:
                    # Very small rise to peak - likely noise
                    info["is_discarded"] = True
                    info["discard_reason"] = f"Total rise to peak ({info['total_rise']:.1f}) < {MIN_DELTA_G_THRESHOLD}"
                    discarded_group_info.append(info)
                    i += 1
                    continue

        # This event is valid - keep it
        info["is_discarded"] = False
        filtered_group_info.append(info)
        i += 1

    # Re-index the filtered groups
    for new_idx, info in enumerate(filtered_group_info):
        info["group_idx"] = new_idx

    group_info = filtered_group_info

    # === STEP 4: Determine composite status for each group ===
    # All groups that share the same peak AND have a distinct meal in between are composite
    for i, info in enumerate(group_info):
        if info["goes_to_next_meal"]:
            info["is_composite"] = True

            # Also mark the next group(s) that share the same peak as composite
            if info["next_peak"]:
                for j in range(i + 1, len(group_info)):
                    next_info = group_info[j]
                    if next_info["next_peak"] == info["next_peak"]:
                        next_info["is_composite"] = True
                    else:
                        break

    # === STEP 5: Create CompositeMealEvent objects ===
    composite_events = []
    for info in group_info:
        is_clean = not info["is_composite"] and not info["is_no_peak"]
        classification = "clean" if is_clean else ("no_peak" if info["is_no_peak"] else "composite")

        composite = CompositeMealEvent(
            event_id=f"E_{info['group_idx']}",
            start_time=info["start_time"],
            end_time=info["end_time"],
            peak_time=info["next_peak"].detected_at if info["next_peak"] else None,
            primary_event=info["primary_event"],
            secondary_events=info["secondary_events"],
            glucose_at_start=info["start_glucose"],
            glucose_at_peak=info["next_peak"].glucose_at_detection if info["next_peak"] else None,
            total_glucose_rise=info["total_rise"],
            is_stacked=info["is_composite"],
            is_clean=is_clean,
            is_no_peak=info["is_no_peak"],
            classification=classification,
            time_to_peak_minutes=info["time_to_peak"],
            delta_g_to_next_event=info["delta_g_to_next"],
            next_event_time=info["next_event_time"]
        )
        composite_events.append(composite)

    sorted_composite_events = sorted(composite_events, key=lambda c: c.start_time)

    if return_discarded:
        # Extract the primary events from discarded groups for visualization
        discarded_events = [info["primary_event"] for info in discarded_group_info]
        return sorted_composite_events, discarded_events

    return sorted_composite_events
