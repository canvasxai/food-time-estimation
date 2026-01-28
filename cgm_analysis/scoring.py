"""
Scoring functions for meal-to-CGM event matching.
"""

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import MealEvent, CompositeMealEvent

# Scoring weights
W_TIME = 0.2   # Time proximity weight (reduced - user-logged times are often inaccurate)
W_SLOT = 0.4   # Meal slot window weight (new - major factor based on meal slot time windows)
W_PHYSIO = 0.2 # Physiological plausibility weight
W_SIZE = 0.2   # Meal size match weight

# Minimum composite score threshold for a valid match (70%)
MIN_MATCH_THRESHOLD = 0.70

# Meal slot time windows (hour ranges)
# Format: {meal_slot: (start_hour, end_hour)}
# Note: end_hour > 24 indicates wraparound into next day (e.g., 26 = 2 AM next day)
MEAL_SLOT_WINDOWS = {
    "Pre Breakfast": (5, 9),      # Early morning, before main breakfast
    "Breakfast": (6, 12),          # Main breakfast window
    "Morning Snack": (9, 12),      # Mid-morning
    "Mid Morning Snack": (9, 12),  # Mid-morning
    "Lunch": (11, 16),             # Lunch window (some eat early)
    "Early Afternoon Snack": (12, 17),  # After lunch snack
    "Afternoon Snack": (14, 18),   # Afternoon
    "Post Lunch Snack": (13, 17),  # After lunch
    "Evening Snack": (17, 21),     # Early evening
    "Dinner": (18, 26),            # Dinner window (extends to 2 AM for late effects)
    "Late Night Snack": (21, 26),  # Late night (extends to 2 AM next day)
}

# Overlapping slot groups for improved matching
# Slots within each group share overlapping time windows and are processed together
# Priority order determines which slot gets first consideration for ambiguous events
OVERLAPPING_SLOT_GROUPS = [
    {
        "name": "early_morning",
        "slots": ["Pre Breakfast", "Breakfast", "Morning Snack", "Mid Morning Snack"],
        "window": (5, 12),  # Union of all slot windows in this group
        "priority_order": ["Pre Breakfast", "Breakfast", "Morning Snack", "Mid Morning Snack"]
    },
    {
        "name": "midday",
        "slots": ["Lunch", "Early Afternoon Snack", "Post Lunch Snack"],
        "window": (11, 17),
        "priority_order": ["Lunch", "Post Lunch Snack", "Early Afternoon Snack"]
    },
    {
        "name": "evening",
        "slots": ["Afternoon Snack", "Evening Snack", "Dinner", "Late Night Snack"],
        "window": (14, 26),  # Extends past midnight
        "priority_order": ["Afternoon Snack", "Evening Snack", "Dinner", "Late Night Snack"]
    }
]

# Slot priority mapping (lower number = higher priority within overlapping groups)
SLOT_PRIORITY = {slot: idx for group in OVERLAPPING_SLOT_GROUPS
                 for idx, slot in enumerate(group["priority_order"])}

# Temporal ordering of meal slots (earlier slots must match earlier events)
# This defines the natural order of meals throughout the day
SLOT_TEMPORAL_ORDER = [
    "Pre Breakfast",
    "Breakfast",
    "Morning Snack",
    "Mid Morning Snack",
    "Lunch",
    "Early Afternoon Snack",
    "Post Lunch Snack",
    "Afternoon Snack",
    "Evening Snack",
    "Dinner",
    "Late Night Snack",
]

# Create mapping from slot name to temporal index
SLOT_TEMPORAL_INDEX = {slot: idx for idx, slot in enumerate(SLOT_TEMPORAL_ORDER)}

# GL proxy validation tolerance (50% = actual can be 50%-150% of expected)
GL_PROXY_TOLERANCE = 0.5


def calculate_time_proximity_score(meal_time: datetime, detected_time: datetime) -> float:
    """
    Calculate time proximity score (S_time).

    Measures how close the logged meal time is to the detected event time.
    Sweet spot is when Δt ≈ 30 min (meal logged ~30 min after detection starts).

    Args:
        meal_time: When the meal was logged
        detected_time: When the CGM event was detected

    Returns:
        Score between 0.0 and 1.0
    """
    delta_t = (meal_time - detected_time).total_seconds() / 60  # in minutes

    if -30 <= delta_t <= 90:
        # Within expected window: peak score at delta_t = 30
        score = 1 - ((abs(delta_t - 30)) / 120) ** 2
    else:
        # Outside expected window: decay based on distance
        score = max(0, 1 - (abs(delta_t) / 180) ** 2)

    return max(0.0, min(1.0, score))


def calculate_expected_time_to_peak(fiber: float, protein: float, fat: float) -> float:
    """
    Calculate expected time to peak based on meal composition.

    Formula: TTP_expected = 30 + 3·fiber + 1.5·protein + 2·fat (minutes)
    Clamped to [30, 180] minutes.
    """
    ttp = 30 + 3 * fiber + 1.5 * protein + 2 * fat
    return max(30, min(180, ttp))


def calculate_physio_score(meal_time: datetime, peak_time: Optional[datetime],
                           fiber: float, protein: float, fat: float) -> float:
    """
    Calculate physiological plausibility score (S_physio).

    Checks if the time-to-peak matches what we'd expect given meal composition.

    Args:
        meal_time: When the meal was logged
        peak_time: When the glucose peak occurred
        fiber, protein, fat: Meal nutrients in grams

    Returns:
        Score between 0.0 and 1.0
    """
    if peak_time is None:
        return 0.5  # Neutral score if no peak available

    ttp_expected = calculate_expected_time_to_peak(fiber, protein, fat)
    ttp_actual = (peak_time - meal_time).total_seconds() / 60  # in minutes

    error = abs(ttp_actual - ttp_expected)
    score = max(0, 1 - (error / 120) ** 2)

    return max(0.0, min(1.0, score))


def calculate_glycemic_load_proxy(carbs: float, fiber: float, protein: float, fat: float) -> float:
    """
    Calculate Glycemic Load Proxy.

    Formula:
        net_carbs = max(0, carbs - 0.5·fiber)
        dampening = 1 / (1 + 0.02·protein + 0.01·fat)
        GL_proxy = net_carbs × dampening
    """
    net_carbs = max(0, carbs - 0.5 * fiber)
    dampening = 1 / (1 + 0.02 * protein + 0.01 * fat)
    return net_carbs * dampening


def calculate_size_score(carbs: float, fiber: float, protein: float, fat: float,
                         event_type: str) -> float:
    """
    Calculate meal size match score (S_size).

    Uses Glycemic Load Proxy to match meal magnitude to event type.

    | Event Type         | GL_proxy > 30 | GL_proxy 15-30 | GL_proxy < 15 |
    |--------------------|---------------|----------------|---------------|
    | est_meal           | 1.0           | 0.7            | 0.4           |
    | est_secondary_meal | 0.4           | 0.7            | 1.0           |

    Args:
        carbs, fiber, protein, fat: Meal nutrients in grams
        event_type: Type of detected event

    Returns:
        Score between 0.0 and 1.0
    """
    gl_proxy = calculate_glycemic_load_proxy(carbs, fiber, protein, fat)

    # Normalize event type (handle variations)
    event_type_lower = event_type.lower()
    is_main_meal = "meal_start" in event_type_lower or event_type_lower == "est_meal"

    if is_main_meal:
        # Main meal event
        if gl_proxy > 30:
            return 1.0
        elif gl_proxy >= 15:
            return 0.7
        else:
            return 0.4
    else:
        # Secondary/snack event
        if gl_proxy < 15:
            return 1.0
        elif gl_proxy <= 30:
            return 0.7
        else:
            return 0.4


def calculate_slot_score(meal_slot: str, event_time: datetime, is_next_day: bool = False) -> float:
    """
    Calculate meal slot window score (S_slot).

    Checks if the detected event falls within the expected time window for the meal slot.
    This is a major factor since user-logged meal times are often inaccurate,
    but the meal slot (Breakfast, Lunch, Dinner) is usually correct.

    Args:
        meal_slot: The meal slot name (e.g., "Breakfast", "Lunch", "Dinner")
        event_time: When the CGM event was detected
        is_next_day: If True, event is from extended hours (next day after midnight)

    Returns:
        Score between 0.0 and 1.0
    """
    # Get the expected window for this meal slot
    window = MEAL_SLOT_WINDOWS.get(meal_slot, (0, 24))
    start_hour, end_hour = window

    event_hour = event_time.hour + event_time.minute / 60.0

    # If this event is from next day's extended hours (e.g., 0-2 AM),
    # convert to 24+ hour format for comparison with windows that extend past midnight
    if is_next_day and event_hour < 6:  # Early morning hours of next day
        event_hour += 24  # e.g., 0:30 AM becomes 24.5

    # Check if event is within the window
    if start_hour <= event_hour < end_hour:
        # Within window - perfect score
        window_center = (start_hour + end_hour) / 2
        if end_hour > 24:
            window_center = min(window_center, 23)
        distance_from_center = abs(event_hour - window_center)
        window_half_width = (end_hour - start_hour) / 2
        score = 0.9 + 0.1 * (1 - min(distance_from_center / window_half_width, 1.0))
        return score
    else:
        # Outside window - calculate penalty based on distance
        if event_hour < start_hour:
            distance = start_hour - event_hour
        else:
            distance = event_hour - end_hour

        # Decay: score drops to 0 when 4+ hours outside window
        score = max(0.0, 1.0 - (distance / 4.0))
        return score


def is_meal_time_in_slot(meal_time: datetime, meal_slot: str) -> bool:
    """
    Check if the logged meal time falls within the expected time window for the meal slot.

    For late-night slots with windows extending past midnight,
    early morning hours (0-2 AM) are considered valid.
    """
    window = MEAL_SLOT_WINDOWS.get(meal_slot, (0, 24))
    start_hour, end_hour = window

    meal_hour = meal_time.hour + meal_time.minute / 60.0

    if end_hour > 24 and meal_hour < 6:
        meal_hour += 24

    return start_hour <= meal_hour < end_hour


def compute_composite_score(s_time: float, s_slot: float, s_physio: float, s_size: float,
                            meal_time_in_slot: bool = True) -> float:
    """
    Compute weighted composite matching score.

    If the logged meal time is within its expected slot:
        S_composite = 0.2·S_time + 0.4·S_slot + 0.2·S_physio + 0.2·S_size

    If the logged meal time is OUTSIDE its expected slot:
        S_composite = 0.5·S_physio + 0.5·S_size
    """
    if meal_time_in_slot:
        return W_TIME * s_time + W_SLOT * s_slot + W_PHYSIO * s_physio + W_SIZE * s_size
    else:
        return 0.5 * s_physio + 0.5 * s_size


def compute_relative_carbs_rise_score(meal_carbs: float, meal_carbs_rank: int, total_meals: int,
                                        event_rise: float, event_rise_rank: int, total_events: int) -> float:
    """
    Calculate score based on RELATIVE ranking of carbs and glucose rise.

    This avoids assuming any fixed carb-to-rise ratio since glycemic response
    varies greatly between individuals.

    Args:
        meal_carbs: Carbohydrate content in grams (for reference)
        meal_carbs_rank: Rank of this meal by carbs (0 = highest carbs)
        total_meals: Total number of meals being ranked
        event_rise: Glucose rise in mg/dL (for reference)
        event_rise_rank: Rank of this event by rise (0 = highest rise)
        total_events: Total number of events being ranked

    Returns:
        Score between 0.0 and 1.0
    """
    if total_meals <= 1 or total_events <= 1:
        return 0.8

    if event_rise is None or event_rise <= 0:
        return 0.5

    meal_normalized = meal_carbs_rank / (total_meals - 1) if total_meals > 1 else 0
    event_normalized = event_rise_rank / (total_events - 1) if total_events > 1 else 0

    rank_diff = abs(meal_normalized - event_normalized)
    score = 1.0 - (rank_diff * 0.8)

    return max(0.2, min(1.0, score))


def compute_carbs_rise_score(carbs: float, actual_rise: float) -> float:
    """
    Simple fallback score when relative ranking is not available.

    Uses a loose heuristic - higher carbs should generally match higher rises,
    but with very generous tolerance.
    """
    if actual_rise is None or actual_rise <= 0:
        return 0.5

    if carbs <= 5:
        if actual_rise < 30:
            return 0.8
        elif actual_rise < 50:
            return 0.6
        else:
            return 0.4

    if carbs >= 50:
        if actual_rise >= 40:
            return 0.8
        elif actual_rise >= 20:
            return 0.6
        else:
            return 0.4

    return 0.7


def get_ordered_points_in_composite(composite: "CompositeMealEvent") -> List[dict]:
    """
    Get all detection points in a composite event, ordered by time.

    Returns list of dicts with time and glucose for each point:
    - Primary event (MEAL_START or SECONDARY_MEAL)
    - All secondary events
    - Peak (if available)
    """
    points = []

    # Add primary event
    primary = composite.primary_event
    points.append({
        'time': primary.estimated_meal_time or primary.detected_at,
        'glucose': primary.glucose_at_detection,
        'event_type': primary.event_type,
        'event': primary
    })

    # Add secondary events
    for sec in composite.secondary_events:
        points.append({
            'time': sec.estimated_meal_time or sec.detected_at,
            'glucose': sec.glucose_at_detection,
            'event_type': sec.event_type,
            'event': sec
        })

    # Sort by time
    points.sort(key=lambda p: p['time'])

    # Add peak at the end if available
    if composite.peak_time and composite.glucose_at_peak:
        points.append({
            'time': composite.peak_time,
            'glucose': composite.glucose_at_peak,
            'event_type': 'PEAK',
            'event': None
        })

    return points


def compute_composite_rise_score(carbs: float, event: "MealEvent",
                                  composite: "CompositeMealEvent") -> float:
    """
    For composite events (stacked meals), estimate how much of the total rise
    this specific event contributed using linear interpolation.
    """
    if composite.total_glucose_rise is None or composite.total_glucose_rise <= 0:
        return 0.5

    points = get_ordered_points_in_composite(composite)

    if len(points) < 2:
        return 0.5

    event_time = event.estimated_meal_time or event.detected_at
    event_idx = None

    for idx, point in enumerate(points):
        if point['event'] is event:
            event_idx = idx
            break
        if point['event_type'] != 'PEAK':
            time_diff = abs((point['time'] - event_time).total_seconds())
            if time_diff < 300:
                event_idx = idx
                break

    if event_idx is None:
        return 0.5

    glucose_at_event = points[event_idx]['glucose']

    if event_idx < len(points) - 1:
        glucose_at_next = points[event_idx + 1]['glucose']
    else:
        glucose_at_next = composite.glucose_at_peak or glucose_at_event

    attributed_rise = max(0, glucose_at_next - glucose_at_event)

    return compute_carbs_rise_score(carbs, attributed_rise)


def validate_carbs_rise_match(carbs: float, actual_rise: float,
                               tolerance: float = None) -> bool:
    """
    Very permissive validation - only filter out extreme mismatches.
    """
    if actual_rise is None:
        return True

    if carbs <= 2 and actual_rise > 60:
        return False

    if carbs >= 80 and actual_rise < 10:
        return False

    return True


def get_slot_group_for_meal(meal_slot: str) -> Optional[dict]:
    """
    Find the overlapping slot group that contains this meal slot.
    """
    for group in OVERLAPPING_SLOT_GROUPS:
        if meal_slot in group["slots"]:
            return group
    return None


def get_slot_priority(meal_slot: str) -> int:
    """
    Get the priority of a meal slot within its overlapping group.
    Lower number = higher priority.
    """
    return SLOT_PRIORITY.get(meal_slot, 99)


def get_slot_temporal_index(meal_slot: str) -> int:
    """
    Get the temporal index of a meal slot.
    Lower index = earlier in the day.
    Used to enforce that meals in earlier slots must match earlier events.
    """
    return SLOT_TEMPORAL_INDEX.get(meal_slot, 99)
