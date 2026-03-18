"""Trajectory prediction for powerlifting performance using ML."""

from __future__ import annotations

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


class TrajectoryModel:
    """ML model for predicting lifter trajectories."""

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

        Args:
            lifters: List of Lifter objects with competition histories.
            min_entries: Minimum number of competition entries required.

        Returns:
            Dictionary with R² scores for each target.
        """
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        # Build training data: for each lifter with enough entries,
        # use features from entries[:-1] to predict the last entry's lifts
        x_rows: list[list[float]] = []
        y_total: list[float] = []
        y_squat: list[float] = []
        y_bench: list[float] = []
        y_deadlift: list[float] = []

        for lifter in lifters:
            if lifter.competition_count < min_entries:
                continue

            # Use all-but-last entries for features, last entry as target
            history = lifter.history()
            target = history[-1]

            if target.total_kg is None:
                continue

            # Create a sub-lifter with entries[:-1] for feature extraction
            sub_lifter = Lifter(name=lifter.name, entries=history[:-1])
            try:
                features = extract_features(sub_lifter)
            except (ValueError, ZeroDivisionError):
                continue

            feat_dict = features.to_dict()
            # Convert to numeric array, handling None and str
            row = _features_to_array(feat_dict)
            x_rows.append(row)
            y_total.append(target.total_kg)
            y_squat.append(target.best3_squat_kg or 0.0)
            y_bench.append(target.best3_bench_kg or 0.0)
            y_deadlift.append(target.best3_deadlift_kg or 0.0)

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

    def predict(self, lifter: Lifter) -> TrajectoryPrediction:
        """Predict future performance for a lifter.

        Args:
            lifter: A Lifter object with competition history.

        Returns:
            TrajectoryPrediction with predicted next-competition lifts.
        """
        import numpy as np

        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        features = extract_features(lifter)
        x = np.array([_features_to_array(features.to_dict())], dtype=np.float64)

        next_total = float(self._total_model.predict(x)[0])  # type: ignore[union-attr]
        next_squat = float(self._squat_model.predict(x)[0])  # type: ignore[union-attr]
        next_bench = float(self._bench_model.predict(x)[0])  # type: ignore[union-attr]
        next_deadlift = float(self._deadlift_model.predict(x)[0])  # type: ignore[union-attr]

        # Simple confidence interval: ±5% of predicted total
        margin = next_total * 0.05
        ci = (round(next_total - margin, 1), round(next_total + margin, 1))

        # Generate trajectory curve (6 projected points, monthly intervals)
        trajectory = _project_trajectory(lifter, next_total, months=12, points=6)

        return TrajectoryPrediction(
            next_total_kg=round(next_total, 1),
            next_squat_kg=round(next_squat, 1),
            next_bench_kg=round(next_bench, 1),
            next_deadlift_kg=round(next_deadlift, 1),
            confidence_interval=ci,
            trajectory_curve=trajectory,
        )

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        import joblib  # type: ignore[import-untyped]

        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        joblib.dump(
            {
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
        self._total_model = data["total"]
        self._squat_model = data["squat"]
        self._bench_model = data["bench"]
        self._deadlift_model = data["deadlift"]
        self._is_trained = True


def predict_trajectory(
    lifter: Lifter, model: TrajectoryModel | None = None
) -> TrajectoryPrediction:
    """Convenience function to predict a lifter's trajectory.

    If no model is provided, raises an error instructing the user to train one first.
    """
    if model is None:
        raise ValueError(
            "A trained TrajectoryModel is required. "
            "Create one with TrajectoryModel() and call .train() first."
        )
    return model.predict(lifter)


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
