"""Trajectory prediction for powerlifting performance using ML."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from opl.analytics.features import extract_features
from opl.core.models import Lifter

if TYPE_CHECKING:
    from sklearn.ensemble import HistGradientBoostingRegressor

_has_sklearn = False
try:
    import sklearn  # noqa: F401  # pyright: ignore[reportUnusedImport]

    _has_sklearn = True
except ImportError:
    pass

_MODEL_VERSION = 2


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


class TrajectoryModel:
    """ML model for predicting lifter trajectories.

    Predicts performance at a specific point in time and bodyweight.
    The model learns how lifters progress over time and how bodyweight
    affects their lifts, enabling predictions like "what will this lifter
    total at 93kg bodyweight, 6 months from now?"
    """

    def __init__(self) -> None:
        if not _has_sklearn:
            raise ImportError(
                "scikit-learn is required for trajectory prediction. "
                "Install with: pip install opl-py[analytics]"
            )
        self._total_model: HistGradientBoostingRegressor | None = None
        self._squat_model: HistGradientBoostingRegressor | None = None
        self._bench_model: HistGradientBoostingRegressor | None = None
        self._deadlift_model: HistGradientBoostingRegressor | None = None
        self._is_trained = False

    def train(self, lifters: list[Lifter], min_entries: int = 3) -> dict[str, float]:
        """Train the trajectory model from a list of lifters.

        For each lifter with enough entries, generates multiple training samples:
        one per entry pair, where each sample includes history features plus
        the time offset and target bodyweight as input features.

        Args:
            lifters: List of Lifter objects with competition histories.
            min_entries: Minimum number of competition entries required.

        Returns:
            Dictionary with R² scores for each target.
        """
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        x_rows: list[list[float]] = []
        y_total: list[float] = []
        y_squat: list[float] = []
        y_bench: list[float] = []
        y_deadlift: list[float] = []

        for lifter in lifters:
            history = lifter.history()
            if len(history) < min_entries:
                continue

            # Generate one sample per entry pair: history[:i] -> predict entry[i]
            for i in range(1, len(history)):
                target_entry = history[i]
                if target_entry.total_kg is None:
                    continue

                sub_lifter = Lifter(name=lifter.name, entries=history[:i])
                try:
                    features = extract_features(sub_lifter)
                except (ValueError, ZeroDivisionError):
                    continue

                # Time offset: days from latest entry in sub-history to target
                days_to_target = (target_entry.date - history[i - 1].date).days
                target_bw = target_entry.bodyweight_kg

                row = _build_feature_row(features.to_dict(), days_to_target, target_bw)
                x_rows.append(row)
                y_total.append(target_entry.total_kg)
                y_squat.append(target_entry.best3_squat_kg or 0.0)
                y_bench.append(target_entry.best3_bench_kg or 0.0)
                y_deadlift.append(target_entry.best3_deadlift_kg or 0.0)

        if len(x_rows) < 10:
            raise ValueError(
                f"Not enough training data: got {len(x_rows)} samples, need at least 10. "
                "Ensure lifters have 3+ competition entries."
            )

        x = np.array(x_rows, dtype=np.float64)
        scores: dict[str, float] = {}

        for name, y_values, attr in [
            ("total", y_total, "_total_model"),
            ("squat", y_squat, "_squat_model"),
            ("bench", y_bench, "_bench_model"),
            ("deadlift", y_deadlift, "_deadlift_model"),
        ]:
            y = np.array(y_values, dtype=np.float64)
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            model = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(x_train, y_train)
            scores[name] = round(model.score(x_test, y_test), 4)
            setattr(self, attr, model)

        self._is_trained = True
        return scores

    def predict(
        self,
        lifter: Lifter,
        target_date: datetime.date | None = None,
        target_bodyweight_kg: float | None = None,
    ) -> TrajectoryPrediction:
        """Predict performance for a lifter at a specific time and bodyweight.

        Args:
            lifter: A Lifter object with competition history.
            target_date: Date to predict for. Defaults to projecting forward
                by the lifter's average inter-competition gap.
            target_bodyweight_kg: Bodyweight to predict at. Defaults to the
                lifter's most recent competition bodyweight.

        Returns:
            TrajectoryPrediction with predicted lifts at the target conditions.
        """
        import numpy as np

        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        features = extract_features(lifter)
        history = lifter.history()
        latest = history[-1]

        # Resolve target date → days_to_target
        if target_date is None:
            # Default: average inter-competition gap
            if features.competition_count > 1 and features.career_length_days > 0:
                avg_gap = features.career_length_days / (features.competition_count - 1)
            else:
                avg_gap = 180  # default 6 months
            days_to_target = avg_gap
            resolved_date = latest.date + datetime.timedelta(days=int(avg_gap))
        else:
            days_to_target = (target_date - latest.date).days
            resolved_date = target_date

        # Resolve target bodyweight
        if target_bodyweight_kg is None:
            resolved_bw = latest.bodyweight_kg
        else:
            resolved_bw = target_bodyweight_kg

        row = _build_feature_row(features.to_dict(), days_to_target, resolved_bw)
        x = np.array([row], dtype=np.float64)

        next_total = float(self._total_model.predict(x)[0])  # type: ignore[union-attr]
        next_squat = float(self._squat_model.predict(x)[0])  # type: ignore[union-attr]
        next_bench = float(self._bench_model.predict(x)[0])  # type: ignore[union-attr]
        next_deadlift = float(self._deadlift_model.predict(x)[0])  # type: ignore[union-attr]

        # Simple confidence interval: ±5% of predicted total
        margin = next_total * 0.05
        ci = (round(next_total - margin, 1), round(next_total + margin, 1))

        # Generate trajectory curve using the model at multiple future time points
        trajectory = _project_trajectory_with_model(
            lifter, self, resolved_bw or latest.bodyweight_kg, months=12, points=6
        )

        return TrajectoryPrediction(
            next_total_kg=round(next_total, 1),
            next_squat_kg=round(next_squat, 1),
            next_bench_kg=round(next_bench, 1),
            next_deadlift_kg=round(next_deadlift, 1),
            confidence_interval=ci,
            trajectory_curve=trajectory,
            target_date=resolved_date,
            target_bodyweight_kg=resolved_bw,
        )

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        import joblib  # type: ignore[import-untyped]

        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        joblib.dump(
            {
                "version": _MODEL_VERSION,
                "total": self._total_model,
                "squat": self._squat_model,
                "bench": self._bench_model,
                "deadlift": self._deadlift_model,
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        import joblib  # type: ignore[import-untyped]

        data = joblib.load(path)
        version = data.get("version", 1)
        if version < _MODEL_VERSION:
            raise ValueError(
                f"Model file is version {version}, but version {_MODEL_VERSION} is required. "
                "Please retrain with the current version."
            )
        self._total_model = data["total"]
        self._squat_model = data["squat"]
        self._bench_model = data["bench"]
        self._deadlift_model = data["deadlift"]
        self._is_trained = True


def predict_trajectory(
    lifter: Lifter,
    model: TrajectoryModel | None = None,
    target_date: datetime.date | None = None,
    target_bodyweight_kg: float | None = None,
) -> TrajectoryPrediction:
    """Convenience function to predict a lifter's trajectory.

    Args:
        lifter: A Lifter object with competition history.
        model: A trained TrajectoryModel. Required.
        target_date: Date to predict for. Defaults to next expected competition.
        target_bodyweight_kg: Bodyweight to predict at. Defaults to latest.

    If no model is provided, raises an error instructing the user to train one first.
    """
    if model is None:
        raise ValueError(
            "A trained TrajectoryModel is required. "
            "Create one with TrajectoryModel() and call .train() first."
        )
    return model.predict(lifter, target_date=target_date, target_bodyweight_kg=target_bodyweight_kg)


# --- Internal helpers ---

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
]


def _features_to_array(feat_dict: Mapping[str, object]) -> list[float]:
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


def _build_feature_row(
    feat_dict: Mapping[str, object],
    days_to_target: float,
    target_bodyweight_kg: float | None,
) -> list[float]:
    """Build a full feature row: history features + time offset + target bodyweight."""
    row = _features_to_array(feat_dict)
    row.append(float(days_to_target))
    row.append(float(target_bodyweight_kg) if target_bodyweight_kg is not None else float("nan"))
    return row


def _project_trajectory(
    lifter: Lifter, predicted_next: float, months: int = 12, points: int = 6
) -> list[tuple[int, float]]:
    """Project a simple linear trajectory curve based on historical progression."""
    history = lifter.history()
    totals = [(e.date, e.total_kg) for e in history if e.total_kg is not None]

    if len(totals) < 2:
        # Can't calculate a rate, just return flat projection
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


def _project_trajectory_with_model(
    lifter: Lifter,
    model: TrajectoryModel,
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
        row = _build_feature_row(feat_dict, days, target_bodyweight_kg)
        x = np.array([row], dtype=np.float64)
        predicted_total = float(model._total_model.predict(x)[0])  # type: ignore[union-attr]
        curve.append((month, round(predicted_total, 1)))

    return curve
