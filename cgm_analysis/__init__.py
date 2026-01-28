"""
CGM Analysis Package

A modular package for CGM (Continuous Glucose Monitor) meal detection and matching.

Modules:
    models: Data classes for meal events, matches, and configuration
    data_loader: Functions for loading CGM and meal data
    derivatives: Derivative calculation functions for signal processing
    scoring: Scoring functions for meal-event matching
    detection: Meal detection algorithms using derivative analysis
    matching: Meal-to-CGM event matching algorithms
    visualization: Plotting functions for CGM data and events
"""

from .models import (
    MealEvent,
    CompositeMealEvent,
    MealMatch,
    SimplifiedThresholds
)

from .data_loader import (
    get_available_users,
    load_cgm_data,
    load_meals_data,
    get_available_dates,
    filter_data_for_date,
    USER_DATA_PATH,
    IST,
    UTC
)

from .derivatives import (
    apply_smoothing,
    calculate_first_derivative,
    calculate_second_derivative
)

from .scoring import (
    calculate_time_proximity_score,
    calculate_expected_time_to_peak,
    calculate_physio_score,
    calculate_glycemic_load_proxy,
    calculate_size_score,
    calculate_slot_score,
    is_meal_time_in_slot,
    compute_composite_score,
    compute_relative_carbs_rise_score,
    compute_carbs_rise_score,
    get_ordered_points_in_composite,
    compute_composite_rise_score,
    validate_carbs_rise_match,
    get_slot_group_for_meal,
    get_slot_priority,
    MEAL_SLOT_WINDOWS,
    OVERLAPPING_SLOT_GROUPS,
    MIN_MATCH_THRESHOLD
)

from .detection import (
    detect_secondary_meal_events,
    interpolate_zero_crossing,
    detect_meal_events_simplified,
    find_associated_peak,
    detect_composite_events
)

from .matching import (
    calculate_expected_glucose_rise,
    validate_composite_match,
    resolve_nearby_events,
    match_meals_to_events
)

from .visualization import (
    create_simplified_derivative_plot,
    create_simple_cgm_plot
)

__all__ = [
    # Models
    'MealEvent',
    'CompositeMealEvent',
    'MealMatch',
    'SimplifiedThresholds',
    # Data loading
    'get_available_users',
    'load_cgm_data',
    'load_meals_data',
    'get_available_dates',
    'filter_data_for_date',
    'USER_DATA_PATH',
    'IST',
    'UTC',
    # Derivatives
    'apply_smoothing',
    'calculate_first_derivative',
    'calculate_second_derivative',
    # Scoring
    'calculate_time_proximity_score',
    'calculate_expected_time_to_peak',
    'calculate_physio_score',
    'calculate_glycemic_load_proxy',
    'calculate_size_score',
    'calculate_slot_score',
    'is_meal_time_in_slot',
    'compute_composite_score',
    'compute_relative_carbs_rise_score',
    'compute_carbs_rise_score',
    'get_ordered_points_in_composite',
    'compute_composite_rise_score',
    'validate_carbs_rise_match',
    'get_slot_group_for_meal',
    'get_slot_priority',
    'MEAL_SLOT_WINDOWS',
    'OVERLAPPING_SLOT_GROUPS',
    'MIN_MATCH_THRESHOLD',
    # Detection
    'detect_secondary_meal_events',
    'interpolate_zero_crossing',
    'detect_meal_events_simplified',
    'find_associated_peak',
    'detect_composite_events',
    # Matching
    'calculate_expected_glucose_rise',
    'validate_composite_match',
    'resolve_nearby_events',
    'match_meals_to_events',
    # Visualization
    'create_simplified_derivative_plot',
    'create_simple_cgm_plot',
]
