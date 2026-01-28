"""
Meal-to-CGM event matching algorithms.

Simplified approach:
1. Collect all meal events (MEAL_START and SECONDARY_MEAL) as a flat list
2. For each event, calculate ΔG to the next event (or peak)
3. Match meals to events using:
   - Slot ordering (Pre-breakfast < Breakfast < Lunch, etc.)
   - Carbs ranking ↔ ΔG ranking within overlapping time windows
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

from .models import MealEvent, CompositeMealEvent, MealMatch, SimplifiedThresholds
from .scoring import (
    MEAL_SLOT_WINDOWS, MIN_MATCH_THRESHOLD,
    get_slot_priority, get_slot_temporal_index
)
from .detection import detect_secondary_meal_events


def get_glucose_at_time(cgm_df: pd.DataFrame, target_time: datetime) -> Optional[float]:
    """Get glucose value at or near a specific time."""
    if cgm_df is None or len(cgm_df) == 0:
        return None

    df = cgm_df.sort_values("datetime_ist")

    # Find closest reading
    time_diffs = abs((df["datetime_ist"] - target_time).dt.total_seconds())
    closest_idx = time_diffs.idxmin()

    # Only use if within 10 minutes
    if time_diffs[closest_idx] <= 600:
        return df.loc[closest_idx, "value"]
    return None


def calculate_peak_offset_penalty(time_to_peak_minutes: float) -> float:
    """
    Calculate a penalty for events where the peak is too far from the event.

    - 0-120 min: No penalty (1.0)
    - 120-300 min: Gradual penalty (1.0 → 0.3)
    - >300 min: Severe penalty (0.3)
    """
    if time_to_peak_minutes is None:
        return 0.7

    if time_to_peak_minutes <= 120:
        return 1.0

    excess_minutes = time_to_peak_minutes - 120
    penalty = max(0.3, 1.0 - (excess_minutes / 30) * 0.1)
    return penalty


def match_meals_to_events(meals_df: pd.DataFrame, events: List[MealEvent],
                          cgm_df: pd.DataFrame = None,
                          thresholds: SimplifiedThresholds = None,
                          debug: bool = False) -> tuple:
    """
    Match logged meals to detected CGM events using simplified segment-based matching.

    Algorithm:
    1. Collect all meal events (MEAL_START + SECONDARY_MEAL) sorted by time
    2. Find peaks to determine event boundaries
    3. Calculate ΔG for each segment (event → next event or peak)
    4. Match meals to events using:
       - Slot temporal ordering (earlier slots → earlier events)
       - Carbs ↔ ΔG ranking (high carb meals → high ΔG events)

    Args:
        meals_df: DataFrame with meal data
        events: List of detected MealEvent objects
        cgm_df: CGM DataFrame for glucose lookups
        thresholds: Detection thresholds
        debug: Print debug information

    Returns:
        Tuple of (matches, unmatched_event_indices, all_events_as_composites, validation_results)
    """
    if len(meals_df) == 0 or len(events) == 0:
        return [], [], [], {}

    # === STEP 1: Collect all meal events ===
    # Get events from the passed list
    meal_events = [e for e in events if e.event_type in ("MEAL_START", "SECONDARY_MEAL")]
    peak_events = [e for e in events if e.event_type == "PEAK"]

    # Also detect secondary meals fresh (in case some were merged)
    if cgm_df is not None and thresholds is not None:
        fresh_secondary = detect_secondary_meal_events(cgm_df, thresholds)
        existing_times = {e.detected_at for e in meal_events}
        for sec in fresh_secondary:
            if sec.detected_at not in existing_times:
                meal_events.append(sec)

    # Sort all meal events by time
    meal_events = sorted(meal_events, key=lambda e: e.detected_at)
    peak_events = sorted(peak_events, key=lambda e: e.detected_at)

    if debug:
        print(f"[DEBUG] Meal events: {len(meal_events)}")
        for e in meal_events:
            print(f"  - {e.detected_at.strftime('%H:%M')} ({e.event_type}) glucose={e.glucose_at_detection:.1f}")
        print(f"[DEBUG] Peak events: {len(peak_events)}")
        for p in peak_events:
            print(f"  - {p.detected_at.strftime('%H:%M')} glucose={p.glucose_at_detection:.1f}")

    if len(meal_events) == 0:
        return [], [], [], {}

    # === STEP 2: Calculate ΔG segments ===
    # For each meal event, find the next event (meal or peak) and calculate ΔG
    # Events within merge_gap are considered merged (clean), not composite
    merge_gap_minutes = thresholds.event_merge_gap_minutes if thresholds else 30
    event_segments = []

    for i, event in enumerate(meal_events):
        event_time = event.estimated_meal_time or event.detected_at
        event_glucose = event.glucose_at_detection

        # Find next meal event (if any)
        next_meal = meal_events[i + 1] if i + 1 < len(meal_events) else None

        # Find next peak after this event
        next_peak = None
        for peak in peak_events:
            if peak.detected_at > event.detected_at:
                next_peak = peak
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

        # Determine the "end point" for this segment
        # It's either the next meal event or the peak, whichever comes first
        # Merged meals are treated as going to peak (clean)
        segment_end = None
        segment_end_glucose = None
        segment_type = "unknown"

        if next_meal and next_peak:
            if next_meal.detected_at < next_peak.detected_at:
                # Next meal comes before peak
                if is_merged_with_next:
                    # Merged with next - treat as going to peak (clean)
                    segment_type = "to_peak_merged"
                    segment_end = next_peak.detected_at
                    segment_end_glucose = next_peak.glucose_at_detection
                else:
                    # Distinct meals - composite
                    segment_type = "to_next_meal"
                    segment_end = next_meal.detected_at
                    segment_end_glucose = next_meal.glucose_at_detection
            else:
                # Peak comes before next meal
                segment_end = next_peak.detected_at
                segment_end_glucose = next_peak.glucose_at_detection
                # If merged with previous, mark as merged
                segment_type = "to_peak_merged" if is_merged_with_prev else "to_peak"
        elif next_peak:
            segment_end = next_peak.detected_at
            segment_end_glucose = next_peak.glucose_at_detection
            # If merged with previous, mark as merged
            segment_type = "to_peak_merged" if is_merged_with_prev else "to_peak"
        elif next_meal:
            if is_merged_with_next:
                segment_type = "to_peak_merged"  # Will be clean even without peak
            else:
                segment_type = "to_next_meal"
            segment_end = next_meal.detected_at
            segment_end_glucose = next_meal.glucose_at_detection

        # Calculate ΔG for this segment
        if segment_end_glucose is not None and event_glucose is not None:
            delta_g = segment_end_glucose - event_glucose
        else:
            delta_g = 0

        # Calculate time to peak (for peak offset penalty)
        time_to_peak = None
        if next_peak:
            time_to_peak = (next_peak.detected_at - event.detected_at).total_seconds() / 60

        event_segments.append({
            "event": event,
            "event_time": event_time,
            "event_glucose": event_glucose,
            "segment_end": segment_end,
            "segment_end_glucose": segment_end_glucose,
            "segment_type": segment_type,
            "delta_g": delta_g,
            "next_peak": next_peak,
            "time_to_peak": time_to_peak
        })

    if debug:
        print(f"[DEBUG] Event segments:")
        for seg in event_segments:
            print(f"  - {seg['event_time'].strftime('%H:%M')}: ΔG={seg['delta_g']:.1f} ({seg['segment_type']}), "
                  f"T→Peak={seg['time_to_peak']:.0f}min" if seg['time_to_peak'] else "no peak")

    # === STEP 3: Prepare meals for matching ===
    primary_date = meals_df["date"].mode().iloc[0] if len(meals_df) > 0 else None

    meal_list = []
    for meal_idx, meal_row in meals_df.iterrows():
        carbs = meal_row.get("carbohydrates", 0) or 0
        meal_slot = meal_row.get("meal_slot", "Unknown")
        slot_temporal_idx = get_slot_temporal_index(meal_slot)

        meal_list.append({
            "meal_idx": meal_idx,
            "meal_row": meal_row,
            "carbs": carbs,
            "meal_slot": meal_slot,
            "slot_temporal_idx": slot_temporal_idx,
            "logged_time": meal_row["datetime_ist"]
        })

    # Sort meals by slot temporal order, then by logged time
    meal_list = sorted(meal_list, key=lambda m: (m["slot_temporal_idx"], m["logged_time"]))

    if debug:
        print(f"[DEBUG] Meals sorted by slot order:")
        for m in meal_list:
            print(f"  - {m['meal_slot']} ({m['slot_temporal_idx']}): {m['carbs']:.0f}g carbs, "
                  f"logged {m['logged_time'].strftime('%H:%M')}")

    # === STEP 4: Match meals to events ===
    # Strategy:
    # 1. Group meals by slot
    # 2. For each slot, find events in the slot's time window
    # 3. Match by carbs ranking ↔ ΔG ranking
    # 4. Apply peak offset penalty

    matches = []
    assigned_meals = set()
    assigned_events = set()

    # Get slot windows
    slot_windows = {}
    for meal in meal_list:
        slot = meal["meal_slot"]
        if slot not in slot_windows:
            slot_windows[slot] = MEAL_SLOT_WINDOWS.get(slot, (0, 24))

    # Process slots in temporal order
    slots_in_order = sorted(slot_windows.keys(), key=lambda s: get_slot_temporal_index(s))

    for slot in slots_in_order:
        slot_start, slot_end = slot_windows[slot]

        # Get unassigned meals in this slot
        slot_meals = [m for m in meal_list
                      if m["meal_slot"] == slot and m["meal_idx"] not in assigned_meals]

        if not slot_meals:
            continue

        # Get unassigned events in this slot's time window
        slot_events = []
        for i, seg in enumerate(event_segments):
            if i in assigned_events:
                continue

            event_time = seg["event_time"]
            event_hour = event_time.hour + event_time.minute / 60.0

            # Handle next-day events
            event_date = event_time.date() if hasattr(event_time, 'date') else event_time.to_pydatetime().date()
            is_next_day = primary_date is not None and event_date > primary_date
            if is_next_day and event_hour < 6:
                event_hour += 24

            if slot_start <= event_hour < slot_end:
                slot_events.append((i, seg))

        if not slot_events:
            continue

        if debug:
            print(f"[DEBUG] Matching slot '{slot}': {len(slot_meals)} meals, {len(slot_events)} events")

        # Sort meals by carbs (descending)
        slot_meals_sorted = sorted(slot_meals, key=lambda m: m["carbs"], reverse=True)

        # Sort events by ΔG (descending), with peak offset penalty
        def event_score(item):
            idx, seg = item
            delta_g = seg["delta_g"]
            peak_penalty = calculate_peak_offset_penalty(seg["time_to_peak"])
            return delta_g * peak_penalty

        slot_events_sorted = sorted(slot_events, key=event_score, reverse=True)

        # Match in order: highest carbs → highest ΔG
        for meal in slot_meals_sorted:
            if not slot_events_sorted:
                break

            # Take the best available event
            event_idx, seg = slot_events_sorted.pop(0)

            event = seg["event"]
            event_time = seg["event_time"]
            delta_g = seg["delta_g"]
            next_peak = seg["next_peak"]
            time_to_peak = seg["time_to_peak"]

            meal_row = meal["meal_row"]
            logged_time = meal["logged_time"]

            # Calculate time offset
            time_offset = (logged_time - event_time).total_seconds() / 60

            # Create match
            match = MealMatch(
                meal_name=meal_row.get("meal_name", "Unknown"),
                meal_time=logged_time,
                meal_slot=meal["meal_slot"],
                event_type=event.event_type,
                event_time=event_time,
                peak_time=next_peak.detected_at if next_peak else None,
                time_offset_minutes=time_offset,
                s_time=0.5,
                s_slot=0.5,
                s_physio=0.5,
                s_size=0.5,
                composite_score=calculate_peak_offset_penalty(time_to_peak),
                carbs=meal["carbs"],
                protein=meal_row.get("protein", 0) or 0,
                fat=meal_row.get("fat", 0) or 0,
                fiber=meal_row.get("fibre", 0) or 0,
                composite_event_id=f"E_{event_idx}",
                is_stacked_meal=False,
                glucose_at_start=seg["event_glucose"],
                glucose_at_peak=next_peak.glucose_at_detection if next_peak else None,
                glucose_rise=delta_g,
                is_clean_event=(seg["segment_type"] in ("to_peak", "to_peak_merged")),
                time_to_peak_minutes=time_to_peak,
                delta_g_to_next_event=delta_g if seg["segment_type"] == "to_next_meal" else None
            )

            matches.append(match)
            assigned_meals.add(meal["meal_idx"])
            assigned_events.add(event_idx)

            if debug:
                print(f"  Matched: {meal['meal_slot']} {meal['carbs']:.0f}g → "
                      f"{event_time.strftime('%H:%M')} (ΔG={delta_g:.1f})")

    # Create dummy composite events for compatibility with existing code
    composite_events = []
    for i, seg in enumerate(event_segments):
        event = seg["event"]
        next_peak = seg["next_peak"]

        composite = CompositeMealEvent(
            event_id=f"E_{i}",
            start_time=seg["event_time"],
            end_time=seg["event_time"],
            peak_time=next_peak.detected_at if next_peak else None,
            primary_event=event,
            secondary_events=[],
            glucose_at_start=seg["event_glucose"],
            glucose_at_peak=next_peak.glucose_at_detection if next_peak else None,
            total_glucose_rise=seg["delta_g"],
            is_stacked=(seg["segment_type"] == "to_next_meal"),
            is_clean=(seg["segment_type"] in ("to_peak", "to_peak_merged")),
            time_to_peak_minutes=seg["time_to_peak"],
            delta_g_to_next_event=seg["delta_g"] if seg["segment_type"] == "to_next_meal" else None,
            next_event_time=seg["segment_end"]
        )
        composite_events.append(composite)

    unmatched_indices = [i for i in range(len(event_segments)) if i not in assigned_events]

    return matches, unmatched_indices, composite_events, {}
