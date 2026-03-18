"""Multi-approach trajectory prediction for powerlifting performance.

This package provides multiple ML approaches for predicting lifter trajectories.
Each approach implements the BaseTrajectoryModel interface and registers itself
in the approach registry.
"""

from __future__ import annotations

import datetime

# Import approach modules to trigger registration.
# Each module uses @register on its model class.
from opl.analytics.trajectory import (
    gradient_boosting as _gbt,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from opl.analytics.trajectory import (
    quantile_gbt as _qgbt,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from opl.analytics.trajectory.base import (
    BaseTrajectoryModel,
    TrajectoryPrediction,
    build_feature_row,
    build_training_data,
    features_to_array,
    project_trajectory_linear,
)
from opl.analytics.trajectory.gradient_boosting import GradientBoostingModel
from opl.analytics.trajectory.registry import (
    get_all_approaches,
    get_approach,
    list_approaches,
    register,
)
from opl.core.models import Lifter

# Backward compatibility alias
TrajectoryModel = GradientBoostingModel


def predict_trajectory(
    lifter: Lifter,
    model: BaseTrajectoryModel | None = None,
    target_date: datetime.date | None = None,
    target_bodyweight_kg: float | None = None,
) -> TrajectoryPrediction:
    """Convenience function to predict a lifter's trajectory.

    Args:
        lifter: A Lifter object with competition history.
        model: A trained trajectory model (any approach). Required.
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


# Keep private names accessible for backward-compatible test imports
_features_to_array = features_to_array
_build_feature_row = build_feature_row
_project_trajectory = project_trajectory_linear

__all__ = [
    "BaseTrajectoryModel",
    "GradientBoostingModel",
    "TrajectoryModel",
    "TrajectoryPrediction",
    "build_feature_row",
    "build_training_data",
    "features_to_array",
    "get_all_approaches",
    "get_approach",
    "list_approaches",
    "predict_trajectory",
    "project_trajectory_linear",
    "register",
]
