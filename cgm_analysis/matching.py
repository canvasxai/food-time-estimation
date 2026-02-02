"""
Meal-to-CGM event matching algorithms.

Simplified approach:
1. Get classified composite events from detection.py (single source of truth)
2. Match meals to events using:
   - Slot ordering (Pre-breakfast < Breakfast < Lunch, etc.)
   - Carbs ranking ↔ ΔG ranking within overlapping time windows

Note: Classification (clean/composite/no_peak) happens in detection.py, not here.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

from .models import MealEvent, CompositeMealEvent, MealMatch, SimplifiedThresholds
from .scoring import (
    MEAL_SLOT_WINDOWS, MIN_MATCH_THRESHOLD,
    get_slot_priority, get_slot_temporal_index
)
from .detection import detect_secondary_meal_events, detect_composite_events


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
    Match logged meals to detected CGM events.

    This function:
    1. Gets classified composite events from detect_composite_events() (single source of truth)
    2. Matches meals to events using slot ordering and carbs ↔ ΔG ranking

    Note: Classification (clean/composite/no_peak) happens in detection.py, not here.

    Args:
        meals_df: DataFrame with meal data
        events: List of detected MealEvent objects
        cgm_df: CGM DataFrame for glucose lookups
        thresholds: Detection thresholds
        debug: Print debug information

    Returns:
        Tuple of (matches, unmatched_event_indices, composite_events, validation_results)
    """
    if len(meals_df) == 0 or len(events) == 0:
        return [], [], [], {}

    # === STEP 1: Get classified composite events from detection.py ===
    # This is the SINGLE SOURCE OF TRUTH for classification
    merge_gap_minutes = thresholds.event_merge_gap_minutes if thresholds else 30
    composite_events = detect_composite_events(
        events,
        cgm_df=cgm_df,
        merge_gap_minutes=merge_gap_minutes,
        max_peak_window_minutes=180
    )

    if debug:
        print(f"[DEBUG] Composite events from detection.py: {len(composite_events)}")
        for comp in composite_events:
            print(f"  - {comp.event_id}: {comp.start_time.strftime('%H:%M')} ({comp.classification})")

    if len(composite_events) == 0:
        return [], [], [], {}

    # === STEP 2: Prepare meals for matching ===
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

    # === STEP 3: Match meals to composite events ===
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

        # Get unassigned composite events in this slot's time window
        slot_events = []
        for comp in composite_events:
            if comp.event_id in assigned_events:
                continue

            event_time = comp.start_time
            event_hour = event_time.hour + event_time.minute / 60.0

            # Handle next-day events
            event_date = event_time.date() if hasattr(event_time, 'date') else event_time.to_pydatetime().date()
            is_next_day = primary_date is not None and event_date > primary_date
            if is_next_day and event_hour < 6:
                event_hour += 24

            if slot_start <= event_hour < slot_end:
                slot_events.append(comp)

        if not slot_events:
            continue

        if debug:
            print(f"[DEBUG] Matching slot '{slot}': {len(slot_meals)} meals, {len(slot_events)} events")

        # Sort meals by carbs (descending)
        slot_meals_sorted = sorted(slot_meals, key=lambda m: m["carbs"], reverse=True)

        # Sort events by ΔG (descending), with peak offset penalty
        def event_score(comp):
            delta_g = comp.total_glucose_rise if comp.total_glucose_rise else 0
            peak_penalty = calculate_peak_offset_penalty(comp.time_to_peak_minutes)
            return delta_g * peak_penalty

        slot_events_sorted = sorted(slot_events, key=event_score, reverse=True)

        # Match in order: highest carbs → highest ΔG
        for meal in slot_meals_sorted:
            if not slot_events_sorted:
                break

            logged_time = meal["logged_time"]

            # Find the best available event with timing constraint:
            # CGM event time cannot be more than 60 min AFTER logged time
            #
            # Example valid cases:
            # - CGM event 10:30, logged 14:00 -> diff = -210 min -> VALID (person logged late)
            # - CGM event 10:30, logged 10:00 -> diff = +30 min -> VALID (person logged, then ate)
            # - CGM event 11:30, logged 10:00 -> diff = +90 min -> INVALID (CGM event too late)
            #
            # No constraint on how early CGM event can be (people log hours after eating)
            comp = None
            for candidate in slot_events_sorted:
                # time_diff > 0 means CGM event is AFTER logged time
                time_diff_minutes = (candidate.start_time - logged_time).total_seconds() / 60
                if time_diff_minutes <= 60:  # CGM event is not more than 60 min after logged time
                    comp = candidate
                    slot_events_sorted.remove(candidate)
                    break

            if comp is None:
                if debug:
                    print(f"  No valid event for {meal['meal_slot']} logged at {logged_time.strftime('%H:%M')} "
                          f"(all CGM events occur more than 60 min after logged time)")
                continue

            meal_row = meal["meal_row"]

            # Calculate time offset
            time_offset = (logged_time - comp.start_time).total_seconds() / 60

            # Create match (classification comes from composite event)
            match = MealMatch(
                meal_name=meal_row.get("meal_name", "Unknown"),
                meal_time=logged_time,
                meal_slot=meal["meal_slot"],
                event_type=comp.primary_event.event_type,
                event_time=comp.start_time,
                peak_time=comp.peak_time,
                time_offset_minutes=time_offset,
                s_time=0.5,
                s_slot=0.5,
                s_physio=0.5,
                s_size=0.5,
                composite_score=calculate_peak_offset_penalty(comp.time_to_peak_minutes),
                carbs=meal["carbs"],
                protein=meal_row.get("protein", 0) or 0,
                fat=meal_row.get("fat", 0) or 0,
                fiber=meal_row.get("fibre", 0) or 0,
                composite_event_id=comp.event_id,
                is_stacked_meal=comp.is_stacked,
                glucose_at_start=comp.glucose_at_start,
                glucose_at_peak=comp.glucose_at_peak,
                glucose_rise=comp.total_glucose_rise if comp.total_glucose_rise else 0,
                is_clean_event=comp.is_clean,
                time_to_peak_minutes=comp.time_to_peak_minutes,
                delta_g_to_next_event=comp.delta_g_to_next_event
            )

            matches.append(match)
            assigned_meals.add(meal["meal_idx"])
            assigned_events.add(comp.event_id)

            if debug:
                print(f"  Matched: {meal['meal_slot']} {meal['carbs']:.0f}g → "
                      f"{comp.start_time.strftime('%H:%M')} (ΔG={comp.total_glucose_rise or 0:.1f}, {comp.classification})")

    # Collect unmatched event indices
    unmatched_indices = [i for i, comp in enumerate(composite_events) if comp.event_id not in assigned_events]

    return matches, unmatched_indices, composite_events, {}
