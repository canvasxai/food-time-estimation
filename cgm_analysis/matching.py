"""
Meal-to-CGM event matching algorithms.
"""

import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any

from .models import MealEvent, CompositeMealEvent, MealMatch, SimplifiedThresholds
from .scoring import (
    MEAL_SLOT_WINDOWS, MIN_MATCH_THRESHOLD,
    calculate_glycemic_load_proxy, compute_relative_carbs_rise_score,
    validate_carbs_rise_match, get_slot_priority
)
from .detection import (
    detect_secondary_meal_events, detect_composite_events, find_associated_peak
)


def calculate_expected_glucose_rise(total_carbs: float, total_fiber: float,
                                     total_protein: float, total_fat: float) -> float:
    """
    Calculate expected glucose rise based on combined meal nutrients.

    Uses the Glycemic Load Proxy to estimate expected glucose rise.
    A rough approximation: GL_proxy of 30 ~ 50 mg/dL rise for typical person.

    Args:
        total_carbs, total_fiber, total_protein, total_fat: Combined nutrients

    Returns:
        Expected glucose rise in mg/dL
    """
    gl_proxy = calculate_glycemic_load_proxy(total_carbs, total_fiber, total_protein, total_fat)
    expected_rise = 20 + (gl_proxy * 1.5)
    return min(expected_rise, 150)


def validate_composite_match(composite: CompositeMealEvent, matched_meals: List[dict]) -> dict:
    """
    Validate if the combined glycemic load of matched meals explains the glucose rise.

    Args:
        composite: The composite meal event
        matched_meals: List of meal data dicts matched to this composite

    Returns:
        dict with validation results
    """
    if not matched_meals:
        return {
            "is_valid": False,
            "combined_gl_proxy": 0,
            "expected_rise": 0,
            "actual_rise": composite.total_glucose_rise,
            "confidence": 0.0
        }

    total_carbs = sum(m.get("carbohydrates", 0) or 0 for m in matched_meals)
    total_fiber = sum(m.get("fibre", 0) or 0 for m in matched_meals)
    total_protein = sum(m.get("protein", 0) or 0 for m in matched_meals)
    total_fat = sum(m.get("fat", 0) or 0 for m in matched_meals)

    combined_gl_proxy = calculate_glycemic_load_proxy(total_carbs, total_fiber, total_protein, total_fat)
    expected_rise = calculate_expected_glucose_rise(total_carbs, total_fiber, total_protein, total_fat)

    actual_rise = composite.total_glucose_rise

    if actual_rise is None:
        confidence = 0.5
        is_valid = True
    else:
        error_ratio = abs(actual_rise - expected_rise) / max(expected_rise, 20)

        if error_ratio <= 0.3:
            confidence = 1.0 - error_ratio
            is_valid = True
        elif error_ratio <= 0.5:
            confidence = 0.7 - (error_ratio - 0.3)
            is_valid = True
        else:
            confidence = max(0.3, 0.5 - (error_ratio - 0.5) * 0.5)
            is_valid = False

    return {
        "is_valid": is_valid,
        "combined_gl_proxy": combined_gl_proxy,
        "expected_rise": expected_rise,
        "actual_rise": actual_rise,
        "confidence": confidence
    }


def resolve_nearby_events(primary_events: List[MealEvent], secondary_events: List[MealEvent],
                          meal_time: datetime, proximity_minutes: float = 45) -> List[MealEvent]:
    """
    Resolve conflicts between primary and secondary meals that are close together.

    If a primary and secondary meal are within proximity_minutes of each other,
    return only the one closer to the logged meal time.
    """
    all_events = primary_events + secondary_events

    if len(all_events) <= 1:
        return all_events

    all_events_sorted = sorted(all_events, key=lambda e: e.estimated_meal_time or e.detected_at)

    resolved = []
    skip_indices = set()

    for i, event in enumerate(all_events_sorted):
        if i in skip_indices:
            continue

        event_time = event.estimated_meal_time or event.detected_at

        nearby_different_type = []
        for j, other_event in enumerate(all_events_sorted):
            if i == j or j in skip_indices:
                continue

            other_time = other_event.estimated_meal_time or other_event.detected_at
            time_diff = abs((event_time - other_time).total_seconds() / 60)

            if time_diff <= proximity_minutes and event.event_type != other_event.event_type:
                nearby_different_type.append((j, other_event, time_diff))

        if nearby_different_type:
            candidates = [(i, event)] + [(j, e) for j, e, _ in nearby_different_type]

            best_idx, best_event = min(
                candidates,
                key=lambda x: abs((meal_time - (x[1].estimated_meal_time or x[1].detected_at)).total_seconds())
            )

            resolved.append(best_event)

            for idx, _, _ in nearby_different_type:
                skip_indices.add(idx)
            skip_indices.add(i)
        else:
            resolved.append(event)

    return resolved


def match_meals_to_events(meals_df: pd.DataFrame, events: List[MealEvent],
                          cgm_df: pd.DataFrame = None,
                          thresholds: SimplifiedThresholds = None) -> tuple:
    """
    Match logged meals to detected CGM events using slot-group-aware matching.

    This algorithm:
    1. Detects composite events (grouping MEAL_START + SECONDARY_MEALs without intervening peaks)
    2. Processes overlapping slot groups together for better matching in ambiguous time windows
    3. Uses glucose rise magnitude to differentiate meals within slots
    4. For composite events, uses linear interpolation to attribute rise to individual meals
    5. Validates GL proxy against actual rise - leaves unmatched if significantly mismatched
    6. Maintains flexibility through composite scoring while using slot priority as tiebreaker

    Args:
        meals_df: DataFrame with meal data (must have datetime_ist, meal_name, etc.)
        events: List of detected MealEvent objects (from detect_meal_events_simplified)
        cgm_df: CGM DataFrame for detecting secondary meals and composite events
        thresholds: Detection thresholds

    Returns:
        Tuple of (list of MealMatch objects, list of unmatched composite indices,
                  list of composite events, validation results)
    """
    if len(meals_df) == 0 or len(events) == 0:
        return [], [], [], {}

    # Detect secondary meal events if CGM data is provided
    secondary_events = []
    if cgm_df is not None and thresholds is not None:
        secondary_events = detect_secondary_meal_events(cgm_df, thresholds)

    # Add secondary events to the full events list for composite detection
    all_events = events + secondary_events

    # Detect composite events
    if cgm_df is not None:
        composite_events = detect_composite_events(all_events, cgm_df)
    else:
        # Fallback: create simple composites from MEAL_START events
        composite_events = []
        for i, event in enumerate(e for e in events if e.event_type == "MEAL_START"):
            peak_time = find_associated_peak(event, events)
            composite_events.append(CompositeMealEvent(
                event_id=f"CE_{i+1}",
                start_time=event.estimated_meal_time or event.detected_at,
                end_time=event.detected_at,
                peak_time=peak_time,
                primary_event=event,
                secondary_events=[],
                glucose_at_start=event.glucose_at_detection,
                glucose_at_peak=None,
                total_glucose_rise=None,
                is_stacked=False
            ))

    if len(composite_events) == 0:
        return [], [], [], {}

    # Determine the primary date
    primary_date = meals_df["date"].mode().iloc[0] if len(meals_df) > 0 else None

    # Group meals and events by slot window
    slot_groups = {}

    for meal_idx, meal_row in meals_df.iterrows():
        carbs = meal_row.get("carbohydrates", 0) or 0
        meal_slot = meal_row.get("meal_slot", "Unknown")
        slot_window = MEAL_SLOT_WINDOWS.get(meal_slot, (0, 24))

        if meal_slot not in slot_groups:
            slot_groups[meal_slot] = {"meals": [], "events": [], "slot_window": slot_window}

        slot_groups[meal_slot]["meals"].append({
            "meal_idx": meal_idx,
            "meal_row": meal_row,
            "carbs": carbs,
            "meal_slot": meal_slot
        })

    # Collect events for each slot
    for comp_idx, composite in enumerate(composite_events):
        event_time = composite.start_time
        event_date = event_time.date() if hasattr(event_time, 'date') else event_time.to_pydatetime().date()
        is_next_day = primary_date is not None and event_date > primary_date

        event_hour = event_time.hour + event_time.minute / 60.0
        if is_next_day and event_hour < 6:
            event_hour += 24

        attributed_rise = composite.total_glucose_rise
        if attributed_rise is None or attributed_rise <= 0:
            if composite.glucose_at_peak and composite.glucose_at_start:
                attributed_rise = composite.glucose_at_peak - composite.glucose_at_start
            else:
                attributed_rise = 0

        for meal_slot, group in slot_groups.items():
            slot_start, slot_end = group["slot_window"]
            if slot_start <= event_hour < slot_end:
                group["events"].append({
                    "comp_idx": comp_idx,
                    "composite": composite,
                    "attributed_rise": attributed_rise,
                    "event_time": event_time
                })

    # Compute scores using relative ranking within each slot
    scores = []

    for meal_slot, group in slot_groups.items():
        meals = group["meals"]
        events_in_slot = group["events"]

        if not meals or not events_in_slot:
            continue

        meals_sorted = sorted(meals, key=lambda m: m["carbs"], reverse=True)
        meal_carbs_rank = {m["meal_idx"]: rank for rank, m in enumerate(meals_sorted)}

        events_sorted = sorted(events_in_slot, key=lambda e: e["attributed_rise"], reverse=True)
        event_rise_rank = {e["comp_idx"]: rank for rank, e in enumerate(events_sorted)}

        total_meals = len(meals)
        total_events = len(events_in_slot)

        for meal_info in meals:
            meal_idx = meal_info["meal_idx"]
            carbs = meal_info["carbs"]
            m_rank = meal_carbs_rank[meal_idx]

            slot_priority = get_slot_priority(meal_slot)

            for event_info in events_in_slot:
                comp_idx = event_info["comp_idx"]
                composite = event_info["composite"]
                attributed_rise = event_info["attributed_rise"]
                e_rank = event_rise_rank[comp_idx]

                carbs_rise_score = compute_relative_carbs_rise_score(
                    carbs, m_rank, total_meals,
                    attributed_rise, e_rank, total_events
                )

                composite_score = carbs_rise_score

                scores.append({
                    "meal_idx": meal_idx,
                    "comp_idx": comp_idx,
                    "meal_row": meal_info["meal_row"],
                    "composite": composite,
                    "carbs_rise_score": carbs_rise_score,
                    "composite_score": composite_score,
                    "is_stacked": composite.is_stacked,
                    "slot_priority": slot_priority,
                    "attributed_rise": attributed_rise,
                    "carbs": carbs
                })

    scores.sort(key=lambda x: (x["composite_score"], -x["slot_priority"]), reverse=True)

    # Greedy assignment with carbs/rise validation
    assigned_meals = set()
    assigned_composites_for_clean = set()
    composite_meal_assignments = {}
    matches = []
    unmatched_due_to_carbs_mismatch = []

    for score_entry in scores:
        meal_idx = score_entry["meal_idx"]
        comp_idx = score_entry["comp_idx"]

        if meal_idx in assigned_meals:
            continue

        composite = score_entry["composite"]

        threshold = MIN_MATCH_THRESHOLD if not composite.is_stacked else MIN_MATCH_THRESHOLD - 0.1

        if score_entry["composite_score"] < threshold:
            continue

        if not composite.is_stacked and comp_idx in assigned_composites_for_clean:
            continue

        actual_rise = score_entry.get("attributed_rise")
        carbs = score_entry["carbs"]

        if actual_rise is not None:
            if not validate_carbs_rise_match(carbs, actual_rise):
                unmatched_due_to_carbs_mismatch.append({
                    "meal_idx": meal_idx,
                    "meal_name": score_entry["meal_row"].get("meal_name", "Unknown"),
                    "carbs": carbs,
                    "actual_rise": actual_rise,
                    "reason": "Extreme carbs/rise mismatch"
                })
                continue

        meal_row = score_entry["meal_row"]
        event_time = composite.start_time
        logged_meal_time = meal_row["datetime_ist"]  # When user logged the meal

        matched_event_type = composite.primary_event.event_type

        # Calculate time offset: logged_meal_time - event_time (positive = logged after event)
        time_offset = (logged_meal_time - event_time).total_seconds() / 60

        match = MealMatch(
            meal_name=meal_row.get("meal_name", "Unknown"),
            meal_time=logged_meal_time,  # User's logged meal time
            meal_slot=meal_row.get("meal_slot", "Unknown"),
            event_type=matched_event_type,
            event_time=event_time,  # CGM detected event time
            peak_time=composite.peak_time,
            time_offset_minutes=time_offset,
            s_time=0.5,
            s_slot=0.5,
            s_physio=0.5,
            s_size=score_entry["carbs_rise_score"],
            composite_score=score_entry["composite_score"],
            carbs=meal_row.get("carbohydrates", 0) or 0,
            protein=meal_row.get("protein", 0) or 0,
            fat=meal_row.get("fat", 0) or 0,
            fiber=meal_row.get("fibre", 0) or 0,
            composite_event_id=composite.event_id,
            is_stacked_meal=composite.is_stacked
        )

        matches.append(match)
        assigned_meals.add(meal_idx)

        if comp_idx not in composite_meal_assignments:
            composite_meal_assignments[comp_idx] = []
        composite_meal_assignments[comp_idx].append(meal_row.to_dict())

        if not composite.is_stacked:
            assigned_composites_for_clean.add(comp_idx)

    # Validate composite matches
    validation_results = {}
    for comp_idx, meal_dicts in composite_meal_assignments.items():
        composite = composite_events[comp_idx]
        if composite.is_stacked:
            validation = validate_composite_match(composite, meal_dicts)
            validation_results[composite.event_id] = validation

    if unmatched_due_to_carbs_mismatch:
        validation_results["_carbs_rise_mismatches"] = unmatched_due_to_carbs_mismatch

    unmatched_composite_indices = [i for i in range(len(composite_events)) if i not in composite_meal_assignments]

    return matches, unmatched_composite_indices, composite_events, validation_results
