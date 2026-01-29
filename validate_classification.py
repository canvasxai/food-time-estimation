"""
Validation script for event classification.

This script loads CGM data for a specific user and date, runs the detection
algorithm, and outputs a detailed report of all events with their classifications.
"""

import sys
from datetime import datetime, date
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from cgm_analysis import (
    load_cgm_data,
    load_meals_data,
    filter_data_for_date,
    SimplifiedThresholds,
    detect_meal_events_simplified,
    detect_composite_events,
)


def investigate_thresholds(user_id: str, target_date: date):
    """
    Investigate how different peak_prominence thresholds affect NO_PEAK events.
    """
    print(f"\n{'='*80}")
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*80}\n")

    # Load data
    cgm_df = load_cgm_data(user_id)
    day_cgm = filter_data_for_date(cgm_df, target_date, extend_hours=2)

    # Test different peak_prominence values
    prominence_values = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]

    print(f"{'Prominence':<12} {'Clean':<8} {'Composite':<10} {'No Peak':<10} {'Total':<8}")
    print("-" * 50)

    results = []
    for prominence in prominence_values:
        thresholds = SimplifiedThresholds(peak_prominence=prominence)
        events = detect_meal_events_simplified(day_cgm, thresholds)
        composite_events = detect_composite_events(
            events, day_cgm,
            merge_gap_minutes=thresholds.event_merge_gap_minutes
        )

        clean = sum(1 for e in composite_events if e.classification == "clean")
        composite = sum(1 for e in composite_events if e.classification == "composite")
        no_peak = sum(1 for e in composite_events if e.classification == "no_peak")
        total = len(composite_events)

        print(f"{prominence:<12.1f} {clean:<8} {composite:<10} {no_peak:<10} {total:<8}")
        results.append({
            "prominence": prominence,
            "clean": clean,
            "composite": composite,
            "no_peak": no_peak,
            "total": total,
            "events": composite_events
        })

    # Find which NO_PEAK events become CLEAN with lower prominence
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS: How NO_PEAK events change with lower prominence")
    print(f"{'='*80}\n")

    # Compare default (5.0) with lower (2.0) prominence
    default_result = next(r for r in results if r["prominence"] == 5.0)
    low_result = next(r for r in results if r["prominence"] == 2.0)

    # Get NO_PEAK events from default
    no_peak_events_default = [e for e in default_result["events"] if e.classification == "no_peak"]

    print(f"With prominence=5.0 (default): {len(no_peak_events_default)} NO_PEAK events")
    for e in no_peak_events_default:
        print(f"  - {e.event_id} at {e.start_time.strftime('%H:%M:%S')} "
              f"(type: {e.primary_event.event_type})")

    # Check what happens to these times with lower prominence
    print(f"\nWith prominence=2.0:")
    low_events_by_time = {e.start_time.strftime('%H:%M:%S'): e for e in low_result["events"]}

    for e in no_peak_events_default:
        time_key = e.start_time.strftime('%H:%M:%S')
        if time_key in low_events_by_time:
            low_e = low_events_by_time[time_key]
            change = "→ STILL NO_PEAK" if low_e.classification == "no_peak" else f"→ NOW {low_e.classification.upper()}"
            peak_info = f" (peak at {low_e.peak_time.strftime('%H:%M')})" if low_e.peak_time else ""
            print(f"  - {e.event_id} at {time_key}: {change}{peak_info}")

    return results


def validate_user_date(user_id: str, target_date: date, output_file: str = None):
    """
    Validate event classification for a specific user and date.

    Args:
        user_id: The user ID to analyze
        target_date: The date to analyze
        output_file: Optional output file path. If None, prints to stdout.
    """
    # Load data
    print(f"Loading data for user {user_id}...")
    cgm_df = load_cgm_data(user_id)
    meals_df = load_meals_data(user_id)

    # Filter for the target date (extend 2 hours into next day)
    day_cgm = filter_data_for_date(cgm_df, target_date, extend_hours=2)
    day_meals = filter_data_for_date(meals_df, target_date, extend_hours=2)

    if len(day_cgm) == 0:
        print(f"No CGM data found for {target_date}")
        return

    print(f"Found {len(day_cgm)} CGM readings and {len(day_meals)} meals")

    # Use default thresholds
    thresholds = SimplifiedThresholds()

    # Detect events
    events = detect_meal_events_simplified(day_cgm, thresholds)

    # Get composite events with classification
    composite_events = detect_composite_events(
        events,
        day_cgm,
        merge_gap_minutes=thresholds.event_merge_gap_minutes
    )

    # Build the report
    lines = []
    lines.append("=" * 80)
    lines.append(f"EVENT CLASSIFICATION VALIDATION REPORT")
    lines.append(f"User: {user_id}")
    lines.append(f"Date: {target_date.strftime('%Y-%m-%d (%A)')}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    total_events = len(composite_events)
    clean_count = sum(1 for e in composite_events if e.classification == "clean")
    composite_count = sum(1 for e in composite_events if e.classification == "composite")
    no_peak_count = sum(1 for e in composite_events if e.classification == "no_peak")
    unknown_count = sum(1 for e in composite_events if e.classification not in ("clean", "composite", "no_peak"))

    lines.append(f"Total Events Detected: {total_events}")
    lines.append(f"  - Clean:     {clean_count}")
    lines.append(f"  - Composite: {composite_count}")
    lines.append(f"  - No Peak:   {no_peak_count}")
    if unknown_count > 0:
        lines.append(f"  - Unknown:   {unknown_count} (ERROR - should be 0)")
    lines.append("")

    # Validation check
    if unknown_count > 0:
        lines.append("⚠️  VALIDATION FAILED: Some events have unknown classification!")
    else:
        lines.append("✅ VALIDATION PASSED: All events are classified as clean, composite, or no_peak")
    lines.append("")

    # Raw events detected
    lines.append("RAW EVENTS DETECTED")
    lines.append("-" * 40)
    meal_starts = [e for e in events if e.event_type == "MEAL_START"]
    secondary_meals = [e for e in events if e.event_type == "SECONDARY_MEAL"]
    peaks = [e for e in events if e.event_type == "PEAK"]

    lines.append(f"MEAL_START events: {len(meal_starts)}")
    for e in meal_starts:
        lines.append(f"  - {e.detected_at.strftime('%H:%M:%S')} (glucose: {e.glucose_at_detection:.1f} mg/dL)")

    lines.append(f"SECONDARY_MEAL events: {len(secondary_meals)}")
    for e in secondary_meals:
        lines.append(f"  - {e.detected_at.strftime('%H:%M:%S')} (glucose: {e.glucose_at_detection:.1f} mg/dL)")

    lines.append(f"PEAK events: {len(peaks)}")
    for e in peaks:
        lines.append(f"  - {e.detected_at.strftime('%H:%M:%S')} (glucose: {e.glucose_at_detection:.1f} mg/dL)")
    lines.append("")

    # Detailed event classification
    lines.append("DETAILED EVENT CLASSIFICATION")
    lines.append("-" * 40)

    for i, comp in enumerate(composite_events):
        lines.append("")
        lines.append(f"Event {comp.event_id}: {comp.classification.upper()}")
        lines.append(f"  Primary Event Type: {comp.primary_event.event_type}")
        lines.append(f"  Start Time: {comp.start_time.strftime('%H:%M:%S')}")
        lines.append(f"  Glucose at Start: {comp.glucose_at_start:.1f} mg/dL")

        if comp.peak_time:
            lines.append(f"  Peak Time: {comp.peak_time.strftime('%H:%M:%S')}")
            lines.append(f"  Glucose at Peak: {comp.glucose_at_peak:.1f} mg/dL")
            lines.append(f"  Time to Peak: {comp.time_to_peak_minutes:.0f} minutes")
            if comp.total_glucose_rise is not None:
                lines.append(f"  Total Glucose Rise: {comp.total_glucose_rise:.1f} mg/dL")
        else:
            lines.append(f"  Peak Time: None (no peak within 180 min)")

        if comp.classification == "composite":
            if comp.next_event_time:
                lines.append(f"  Next Event Time: {comp.next_event_time.strftime('%H:%M:%S')}")
            if comp.delta_g_to_next_event is not None:
                lines.append(f"  Delta G to Next Event: {comp.delta_g_to_next_event:.1f} mg/dL")

        # Classification reason
        if comp.classification == "clean":
            lines.append(f"  Reason: Peak within 180 min, no intervening meal events")
        elif comp.classification == "composite":
            lines.append(f"  Reason: Another meal event occurs before the peak")
        elif comp.classification == "no_peak":
            lines.append(f"  Reason: No peak found within 180 minutes")
        else:
            lines.append(f"  Reason: UNKNOWN - Classification error")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Output the report
    report = "\n".join(lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report written to: {output_file}")
    else:
        print(report)

    return composite_events


def validate_all_dates(user_id: str, peak_prominence: float = 2.0, output_file: str = None):
    """
    Validate event classification for all available dates for a user.

    Args:
        user_id: The user ID to analyze
        peak_prominence: Peak prominence threshold to use
        output_file: Optional output file path. If None, prints to stdout.
    """
    # Load data
    print(f"Loading data for user {user_id}...")
    cgm_df = load_cgm_data(user_id)
    meals_df = load_meals_data(user_id)

    # Get all available dates
    from cgm_analysis import get_available_dates
    available_dates = get_available_dates(cgm_df)

    print(f"Found {len(available_dates)} dates with CGM data")

    # Use custom thresholds with specified peak_prominence
    thresholds = SimplifiedThresholds(peak_prominence=peak_prominence)

    lines = []
    lines.append("=" * 80)
    lines.append(f"EVENT CLASSIFICATION VALIDATION REPORT - ALL DATES")
    lines.append(f"User: {user_id}")
    lines.append(f"Peak Prominence: {peak_prominence}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Summary counters
    total_clean = 0
    total_composite = 0
    total_no_peak = 0
    total_unknown = 0
    total_events = 0

    # Process each date
    for target_date in available_dates:
        # Filter for the target date
        day_cgm = filter_data_for_date(cgm_df, target_date, extend_hours=2)
        day_meals = filter_data_for_date(meals_df, target_date, extend_hours=2)

        if len(day_cgm) == 0:
            continue

        # Detect events
        events = detect_meal_events_simplified(day_cgm, thresholds)

        # Get composite events with classification
        composite_events = detect_composite_events(
            events,
            day_cgm,
            merge_gap_minutes=thresholds.event_merge_gap_minutes
        )

        # Count classifications
        clean_count = sum(1 for e in composite_events if e.classification == "clean")
        composite_count = sum(1 for e in composite_events if e.classification == "composite")
        no_peak_count = sum(1 for e in composite_events if e.classification == "no_peak")
        unknown_count = sum(1 for e in composite_events if e.classification not in ("clean", "composite", "no_peak"))

        total_clean += clean_count
        total_composite += composite_count
        total_no_peak += no_peak_count
        total_unknown += unknown_count
        total_events += len(composite_events)

        # Date header
        lines.append("-" * 80)
        lines.append(f"DATE: {target_date.strftime('%Y-%m-%d (%A)')}")
        lines.append(f"CGM Readings: {len(day_cgm)} | Meals Logged: {len(day_meals)}")
        lines.append(f"Events: {len(composite_events)} (Clean: {clean_count}, Composite: {composite_count}, No Peak: {no_peak_count})")
        lines.append("-" * 80)

        if unknown_count > 0:
            lines.append(f"⚠️  WARNING: {unknown_count} events with unknown classification!")

        # List each event
        for comp in composite_events:
            event_type = comp.primary_event.event_type
            classification = comp.classification.upper()

            if comp.peak_time:
                peak_info = f"Peak: {comp.peak_time.strftime('%H:%M')} ({comp.time_to_peak_minutes:.0f}min, +{comp.total_glucose_rise:.0f}mg/dL)" if comp.total_glucose_rise else f"Peak: {comp.peak_time.strftime('%H:%M')}"
            else:
                peak_info = "No peak within 180min"

            lines.append(f"  {comp.event_id} | {classification:<10} | {event_type:<14} | {comp.start_time.strftime('%H:%M:%S')} | {peak_info}")

        lines.append("")

    # Overall summary
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Dates Analyzed: {len(available_dates)}")
    lines.append(f"Total Events: {total_events}")
    lines.append(f"  - Clean:     {total_clean} ({100*total_clean/total_events:.1f}%)" if total_events > 0 else "  - Clean:     0")
    lines.append(f"  - Composite: {total_composite} ({100*total_composite/total_events:.1f}%)" if total_events > 0 else "  - Composite: 0")
    lines.append(f"  - No Peak:   {total_no_peak} ({100*total_no_peak/total_events:.1f}%)" if total_events > 0 else "  - No Peak:   0")
    if total_unknown > 0:
        lines.append(f"  - Unknown:   {total_unknown} (ERROR)")
    lines.append("")

    if total_unknown > 0:
        lines.append("⚠️  VALIDATION FAILED: Some events have unknown classification!")
    else:
        lines.append("✅ VALIDATION PASSED: All events are classified as clean, composite, or no_peak")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Output the report
    report = "\n".join(lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report written to: {output_file}")
    else:
        print(report)

    return {
        "total_events": total_events,
        "clean": total_clean,
        "composite": total_composite,
        "no_peak": total_no_peak,
        "unknown": total_unknown
    }


if __name__ == "__main__":
    # Validate for user 100001878 - all dates with peak_prominence=2.0
    user_id = "100001878"
    peak_prominence = 2.0
    output_file = f"validation_output_{user_id}_all_dates_prominence{peak_prominence}.txt"

    validate_all_dates(user_id, peak_prominence=peak_prominence, output_file=output_file)
