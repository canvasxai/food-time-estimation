"""
Tests for event classification validation.

This test ensures that the classification logic produces consistent results
for known test data. If classification counts change, the test fails and
reports what changed.
"""

import sys
from pathlib import Path
from datetime import date

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cgm_analysis import (
    load_cgm_data,
    load_meals_data,
    filter_data_for_date,
    get_available_dates,
    SimplifiedThresholds,
    detect_meal_events_simplified,
    detect_composite_events,
)


# Expected counts for user 100001878 with peak_prominence=2.0
EXPECTED_RESULTS = {
    "user_id": "100001878",
    "peak_prominence": 2.0,
    "total_events": 104,
    "clean": 66,
    "composite": 31,
    "no_peak": 7,
    # Per-date breakdown for detailed validation
    "per_date": {
        "2024-10-23": {"clean": 3, "composite": 5, "no_peak": 1, "total": 9},
        "2024-10-24": {"clean": 4, "composite": 2, "no_peak": 0, "total": 6},
        "2024-10-25": {"clean": 6, "composite": 0, "no_peak": 1, "total": 7},
        "2024-10-26": {"clean": 4, "composite": 4, "no_peak": 2, "total": 10},
        "2024-10-27": {"clean": 6, "composite": 0, "no_peak": 0, "total": 6},
        "2024-10-28": {"clean": 3, "composite": 6, "no_peak": 0, "total": 9},
        "2024-10-29": {"clean": 5, "composite": 2, "no_peak": 0, "total": 7},
        "2024-10-30": {"clean": 5, "composite": 2, "no_peak": 0, "total": 7},
        "2024-10-31": {"clean": 4, "composite": 2, "no_peak": 0, "total": 6},
        "2024-11-01": {"clean": 6, "composite": 0, "no_peak": 0, "total": 6},
        "2024-11-02": {"clean": 7, "composite": 2, "no_peak": 0, "total": 9},
        "2024-11-03": {"clean": 4, "composite": 2, "no_peak": 1, "total": 7},
        "2024-11-04": {"clean": 6, "composite": 0, "no_peak": 0, "total": 6},
        "2024-11-05": {"clean": 3, "composite": 4, "no_peak": 0, "total": 7},
        "2024-11-06": {"clean": 0, "composite": 0, "no_peak": 2, "total": 2},
    }
}


def run_classification_test():
    """
    Run classification test for user 100001878 with peak_prominence=2.0.

    Returns:
        tuple: (passed: bool, message: str)
    """
    user_id = EXPECTED_RESULTS["user_id"]
    peak_prominence = EXPECTED_RESULTS["peak_prominence"]

    # Check if user data exists
    user_data_path = Path(__file__).parent.parent / "user_data" / user_id
    if not user_data_path.exists():
        return False, f"User data not found: {user_data_path}"

    # Load data
    cgm_df = load_cgm_data(user_id)
    meals_df = load_meals_data(user_id)
    available_dates = get_available_dates(cgm_df)

    # Use custom thresholds
    thresholds = SimplifiedThresholds(peak_prominence=peak_prominence)

    # Track results
    total_clean = 0
    total_composite = 0
    total_no_peak = 0
    total_events = 0
    failures = []

    # Process each date
    for target_date in available_dates:
        day_cgm = filter_data_for_date(cgm_df, target_date, extend_hours=2)

        if len(day_cgm) == 0:
            continue

        # Detect events
        events = detect_meal_events_simplified(day_cgm, thresholds)
        composite_events = detect_composite_events(
            events,
            day_cgm,
            merge_gap_minutes=thresholds.event_merge_gap_minutes
        )

        # Count classifications
        clean = sum(1 for e in composite_events if e.classification == "clean")
        composite = sum(1 for e in composite_events if e.classification == "composite")
        no_peak = sum(1 for e in composite_events if e.classification == "no_peak")
        unknown = sum(1 for e in composite_events if e.classification not in ("clean", "composite", "no_peak"))

        total_clean += clean
        total_composite += composite
        total_no_peak += no_peak
        total_events += len(composite_events)

        # Check for unknown classifications
        if unknown > 0:
            failures.append(f"{target_date}: {unknown} events with UNKNOWN classification!")

        # Check against expected per-date counts
        date_key = target_date.strftime("%Y-%m-%d")
        if date_key in EXPECTED_RESULTS["per_date"]:
            expected = EXPECTED_RESULTS["per_date"][date_key]
            actual = {"clean": clean, "composite": composite, "no_peak": no_peak, "total": len(composite_events)}

            if actual != expected:
                failures.append(
                    f"{date_key}: Expected {expected}, got {actual}"
                )

    # Check overall totals
    if total_events != EXPECTED_RESULTS["total_events"]:
        failures.append(
            f"Total events: Expected {EXPECTED_RESULTS['total_events']}, got {total_events}"
        )
    if total_clean != EXPECTED_RESULTS["clean"]:
        failures.append(
            f"Total clean: Expected {EXPECTED_RESULTS['clean']}, got {total_clean}"
        )
    if total_composite != EXPECTED_RESULTS["composite"]:
        failures.append(
            f"Total composite: Expected {EXPECTED_RESULTS['composite']}, got {total_composite}"
        )
    if total_no_peak != EXPECTED_RESULTS["no_peak"]:
        failures.append(
            f"Total no_peak: Expected {EXPECTED_RESULTS['no_peak']}, got {total_no_peak}"
        )

    if failures:
        message = "CLASSIFICATION TEST FAILED:\n" + "\n".join(f"  - {f}" for f in failures)
        return False, message

    message = (
        f"CLASSIFICATION TEST PASSED\n"
        f"  User: {user_id}\n"
        f"  Peak Prominence: {peak_prominence}\n"
        f"  Total Events: {total_events}\n"
        f"  Clean: {total_clean} ({100*total_clean/total_events:.1f}%)\n"
        f"  Composite: {total_composite} ({100*total_composite/total_events:.1f}%)\n"
        f"  No Peak: {total_no_peak} ({100*total_no_peak/total_events:.1f}%)"
    )
    return True, message


def test_classification_counts():
    """pytest-compatible test function."""
    passed, message = run_classification_test()
    print(message)
    assert passed, message


if __name__ == "__main__":
    passed, message = run_classification_test()
    print(message)
    sys.exit(0 if passed else 1)
