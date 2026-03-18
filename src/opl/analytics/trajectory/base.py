"""Base class and shared utilities for trajectory prediction approaches."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from opl.analytics.features import extract_features
from opl.core.models import Lifter


def _default_trajectory_curve() -> list[tuple[int, float]]:
    return []


@dataclass
class TrajectoryPrediction:
    """Predicted future performance for a lifter."""

    next_total_kg: float | None = None
    next_squat_kg: float | None = None
    next_bench_kg: float | None = None
    next_deadlift_kg: float | None = None
    confidence_interval: tuple[float, float] | None = None
    percentile: float | None = None
    trajectory_curve: list[tuple[int, float]] = field(default_factory=_default_trajectory_curve)
    target_date: datetime.date | None = None
    target_bodyweight_kg: float | None = None


class BaseTrajectoryModel(ABC):
    """Abstract base for all trajectory prediction approaches."""

    name: str = ""
    display_name: str = ""
    description: str = ""

    @abstractmethod
    def train(self, lifters: list[Lifter], min_entries: int = 3) -> dict[str, float]:
        """Train the model from a list of lifters.

        Returns:
            Dictionary with R² scores for each target.
        """

    @abstractmethod
    def predict(
        self,
        lifter: Lifter,
        target_date: datetime.date | None = None,
        target_bodyweight_kg: float | None = None,
    ) -> TrajectoryPrediction:
        """Predict performance for a lifter at a specific time and bodyweight."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the trained model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a trained model from disk."""

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained or loaded."""
        return self._is_trained

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_is_trained"):
            cls._is_trained = False  # type: ignore[attr-defined]


# --- Shared feature helpers ---

_FEATURE_KEYS = [
    "career_length_days",
    "competition_count",
    "competition_frequency",
    "best_total_kg",
    "best_squat_kg",
    "best_bench_kg",
    "best_deadlift_kg",
    "latest_total_kg",
    "latest_bodyweight_kg",
    "total_progression_rate",
    "squat_to_total_ratio",
    "bench_to_total_ratio",
    "deadlift_to_total_ratio",
    "age_at_latest",
    "age_at_first",
    "weight_class_numeric",
    "days_since_last_comp",
    "recent_avg_total_kg",
    "recent_progression_rate",
    "total_std_kg",
    "meets_since_peak",
]


def features_to_array(feat_dict: Mapping[str, object]) -> list[float]:
    """Convert a feature dict to a numeric list for ML input."""
    row: list[float] = []
    for key in _FEATURE_KEYS:
        val = feat_dict.get(key)
        if val is None:
            row.append(float("nan"))
        elif isinstance(val, bool):
            row.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            row.append(float(val))
        else:
            row.append(float("nan"))
    return row


def build_feature_row(
    feat_dict: Mapping[str, object],
    days_to_target: float,
    target_bodyweight_kg: float | None,
) -> list[float]:
    """Build a full feature row: history features + time offset + target bodyweight."""
    row = features_to_array(feat_dict)
    row.append(float(days_to_target))
    row.append(float(target_bodyweight_kg) if target_bodyweight_kg is not None else float("nan"))
    return row


def build_training_data(
    lifters: list[Lifter], min_entries: int = 3
) -> tuple[list[list[float]], list[float], list[float], list[float], list[float]]:
    """Build training data from lifters using the leave-future-out scheme.

    For each lifter with enough entries, generates multiple training samples:
    one per entry pair, where each sample includes history features plus
    the time offset and target bodyweight as input features.

    Returns:
        Tuple of (x_rows, y_total, y_squat, y_bench, y_deadlift).
    """
    x_rows: list[list[float]] = []
    y_total: list[float] = []
    y_squat: list[float] = []
    y_bench: list[float] = []
    y_deadlift: list[float] = []

    for lifter in lifters:
        history = lifter.history()
        if len(history) < min_entries:
            continue

        for i in range(1, len(history)):
            target_entry = history[i]
            if target_entry.total_kg is None:
                continue

            sub_lifter = Lifter(name=lifter.name, entries=history[:i])
            try:
                features = extract_features(sub_lifter)
            except (ValueError, ZeroDivisionError):
                continue

            days_to_target = (target_entry.date - history[i - 1].date).days
            target_bw = target_entry.bodyweight_kg

            row = build_feature_row(features.to_dict(), days_to_target, target_bw)
            x_rows.append(row)
            y_total.append(target_entry.total_kg)
            y_squat.append(target_entry.best3_squat_kg or 0.0)
            y_bench.append(target_entry.best3_bench_kg or 0.0)
            y_deadlift.append(target_entry.best3_deadlift_kg or 0.0)

    return x_rows, y_total, y_squat, y_bench, y_deadlift


def resolve_prediction_context(
    lifter: Lifter,
    target_date: datetime.date | None,
    target_bodyweight_kg: float | None,
) -> tuple[float, datetime.date, float | None]:
    """Resolve target date and bodyweight into days_to_target, resolved_date, resolved_bw.

    Returns:
        Tuple of (days_to_target, resolved_date, resolved_bw).
    """
    features = extract_features(lifter)
    history = lifter.history()
    latest = history[-1]

    if target_date is None:
        if features.competition_count > 1 and features.career_length_days > 0:
            avg_gap = features.career_length_days / (features.competition_count - 1)
        else:
            avg_gap = 180
        days_to_target = avg_gap
        resolved_date = latest.date + datetime.timedelta(days=int(avg_gap))
    else:
        days_to_target = float((target_date - latest.date).days)
        resolved_date = target_date

    resolved_bw = target_bodyweight_kg if target_bodyweight_kg is not None else latest.bodyweight_kg

    return days_to_target, resolved_date, resolved_bw


def project_trajectory_with_model(
    lifter: Lifter,
    model: BaseTrajectoryModel,
    target_bodyweight_kg: float | None,
    months: int = 12,
    points: int = 6,
) -> list[tuple[int, float]]:
    """Project a trajectory curve using the trained model at multiple future time points."""
    import numpy as np

    features = extract_features(lifter)
    feat_dict = features.to_dict()

    curve: list[tuple[int, float]] = []
    interval = months // points
    for i in range(1, points + 1):
        month = i * interval
        days = int(month * 30.44)
        row = build_feature_row(feat_dict, days, target_bodyweight_kg)
        x = np.array([row], dtype=np.float64)
        predicted_total = float(model._predict_total(x))  # type: ignore[attr-defined]
        curve.append((month, round(predicted_total, 1)))

    return curve


def project_trajectory_linear(
    lifter: Lifter, predicted_next: float, months: int = 12, points: int = 6
) -> list[tuple[int, float]]:
    """Project a simple linear trajectory curve based on historical progression."""
    history = lifter.history()
    totals = [(e.date, e.total_kg) for e in history if e.total_kg is not None]

    if len(totals) < 2:
        interval = months // points
        return [(i * interval, predicted_next) for i in range(1, points + 1)]

    first_date, first_total = totals[0]
    last_date, last_total = totals[-1]

    days = (last_date - first_date).days
    if days <= 0:
        interval = months // points
        return [(i * interval, predicted_next) for i in range(1, points + 1)]

    monthly_rate = (last_total - first_total) / (days / 30.44)

    curve: list[tuple[int, float]] = []
    interval = months // points
    for i in range(1, points + 1):
        month = i * interval
        projected = predicted_next + (monthly_rate * month)
        curve.append((month, round(projected, 1)))

    return curve
