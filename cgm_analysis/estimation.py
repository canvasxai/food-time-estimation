"""
Estimation utilities for comparing clean meals and CGM events.

NOTE:
This module currently provides minimal placeholder implementations so that
the Streamlit app and the ``cgm_analysis`` package can be imported without
errors in all environments (including Streamlit Cloud).

The API surface matches what ``cgm_analysis.__init__`` expects, but most
functions intentionally raise ``NotImplementedError`` as they are not yet
used by the app. This preserves future extensibility without changing any
current behaviour of the deployed application.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class CleanMealRecord:
    """
    Represents a "clean" logged meal suitable for template-based estimation.

    This is intentionally minimal for now – fields can be expanded later
    when the estimation workflow is implemented.
    """

    user_id: str
    meal_time: datetime
    meal_name: str
    carbs: float
    protein: float
    fat: float
    fibre: float


@dataclass
class CleanCGMEvent:
    """
    Represents a "clean" CGM event (meal→peak) suitable for template building.
    """

    user_id: str
    start_time: datetime
    peak_time: Optional[datetime]
    glucose_at_start: float
    glucose_at_peak: Optional[float]
    total_glucose_rise: Optional[float]


@dataclass
class SimilarityWeights:
    """
    Weighting configuration for similarity calculations between two curves.
    """

    time_weight: float = 0.25
    magnitude_weight: float = 0.5
    shape_weight: float = 0.25


def _not_implemented(name: str) -> None:
    """
    Helper to provide a consistent, explicit placeholder failure mode.

    Using ``NotImplementedError`` keeps current functionality unchanged:
    previously, importing this module failed entirely; now it imports
    cleanly, and any *future* callers will receive a clear error message.
    """

    raise NotImplementedError(
        f"`cgm_analysis.estimation.{name}` is a placeholder and has not "
        f"been fully implemented yet."
    )


def extract_cgm_window(*args: Any, **kwargs: Any) -> Any:
    _not_implemented("extract_cgm_window")


def collect_clean_meals(*args: Any, **kwargs: Any) -> List[CleanMealRecord]:
    _not_implemented("collect_clean_meals")


def collect_clean_cgm_events(*args: Any, **kwargs: Any) -> List[CleanCGMEvent]:
    _not_implemented("collect_clean_cgm_events")


def calculate_similarity_score(*args: Any, **kwargs: Any) -> float:
    _not_implemented("calculate_similarity_score")


def calculate_shape_similarity_score(*args: Any, **kwargs: Any) -> float:
    _not_implemented("calculate_shape_similarity_score")


def find_most_similar_meal(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
    _not_implemented("find_most_similar_meal")


def find_most_similar_cgm_event(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
    _not_implemented("find_most_similar_cgm_event")


def plot_meal_comparison(*args: Any, **kwargs: Any) -> Any:
    _not_implemented("plot_meal_comparison")


def plot_cgm_event_comparison(*args: Any, **kwargs: Any) -> Any:
    _not_implemented("plot_cgm_event_comparison")


def normalize_cgm_curve(*args: Any, **kwargs: Any) -> Any:
    _not_implemented("normalize_cgm_curve")


def add_starting_point(*args: Any, **kwargs: Any) -> Any:
    _not_implemented("add_starting_point")


def run_estimation_analysis(*args: Any, **kwargs: Any) -> Any:
    _not_implemented("run_estimation_analysis")


def list_clean_meals(*args: Any, **kwargs: Any) -> List[CleanMealRecord]:
    _not_implemented("list_clean_meals")


def list_clean_cgm_events(*args: Any, **kwargs: Any) -> List[CleanCGMEvent]:
    _not_implemented("list_clean_cgm_events")
