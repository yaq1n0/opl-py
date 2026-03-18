"""Gradient Boosted Trees trajectory prediction (HistGradientBoostingRegressor)."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from opl.analytics.trajectory.base import (
    BaseTrajectoryModel,
    TrajectoryPrediction,
    build_feature_row,
    build_training_data,
    project_trajectory_with_model,
    resolve_prediction_context,
)
from opl.analytics.trajectory.registry import register
from opl.core.models import Lifter

if TYPE_CHECKING:
    from sklearn.ensemble import HistGradientBoostingRegressor

_MODEL_VERSION = 3


@register
class GradientBoostingModel(BaseTrajectoryModel):
    """Trajectory prediction using scikit-learn HistGradientBoostingRegressor.

    Handles missing values natively, captures non-linear relationships,
    and is scale-invariant. The default and most battle-tested approach.
    """

    name = "gradient_boosting"
    display_name = "Gradient Boosted Trees"
    description = (
        "HistGradientBoostingRegressor — handles missing values natively, "
        "captures non-linear relationships. The default approach."
    )

    def __init__(self) -> None:
        self._total_model: HistGradientBoostingRegressor | None = None
        self._squat_model: HistGradientBoostingRegressor | None = None
        self._bench_model: HistGradientBoostingRegressor | None = None
        self._deadlift_model: HistGradientBoostingRegressor | None = None
        self._total_residual_std: float = 0.0
        self._is_trained = False

    def train(self, lifters: list[Lifter], min_entries: int = 3) -> dict[str, float]:
        """Train the trajectory model from a list of lifters."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        x_rows, y_total, y_squat, y_bench, y_deadlift = build_training_data(lifters, min_entries)

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

            if name == "total":
                residuals = y_test - model.predict(x_test)
                self._total_residual_std = float(np.std(residuals))

        self._is_trained = True
        return scores

    def predict(
        self,
        lifter: Lifter,
        target_date: datetime.date | None = None,
        target_bodyweight_kg: float | None = None,
    ) -> TrajectoryPrediction:
        """Predict performance for a lifter."""
        from opl.analytics.features import extract_features

        self._check_trained()

        features = extract_features(lifter)
        days_to_target, resolved_date, resolved_bw = resolve_prediction_context(
            lifter, target_date, target_bodyweight_kg
        )

        row = build_feature_row(features.to_dict(), days_to_target, resolved_bw)
        x = np.array([row], dtype=np.float64)

        next_total = float(self._total_model.predict(x)[0])  # type: ignore[union-attr]
        next_squat = float(self._squat_model.predict(x)[0])  # type: ignore[union-attr]
        next_bench = float(self._bench_model.predict(x)[0])  # type: ignore[union-attr]
        next_deadlift = float(self._deadlift_model.predict(x)[0])  # type: ignore[union-attr]

        margin = (
            1.96 * self._total_residual_std if self._total_residual_std > 0 else next_total * 0.05
        )
        ci = (round(next_total - margin, 1), round(next_total + margin, 1))

        history = lifter.history()
        latest = history[-1]
        trajectory = project_trajectory_with_model(
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

    def _predict_total(self, x: np.ndarray) -> float:
        """Predict total from feature array (used by trajectory projection)."""
        return float(self._total_model.predict(x)[0])  # type: ignore[union-attr]

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        import joblib  # type: ignore[import-untyped]

        self._check_trained()
        joblib.dump(
            {
                "version": _MODEL_VERSION,
                "total": self._total_model,
                "squat": self._squat_model,
                "bench": self._bench_model,
                "deadlift": self._deadlift_model,
                "total_residual_std": self._total_residual_std,
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
        self._total_residual_std = data.get("total_residual_std", 0.0)
        self._is_trained = True
