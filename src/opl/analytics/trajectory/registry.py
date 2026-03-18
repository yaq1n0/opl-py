"""Registry for trajectory prediction approaches."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opl.analytics.trajectory.base import BaseTrajectoryModel

_APPROACHES: dict[str, type[BaseTrajectoryModel]] = {}


def register(cls: type[BaseTrajectoryModel]) -> type[BaseTrajectoryModel]:
    """Class decorator to register a trajectory model approach."""
    _APPROACHES[cls.name] = cls
    return cls


def list_approaches() -> list[str]:
    """Return names of all registered approaches."""
    return list(_APPROACHES.keys())


def get_approach(name: str) -> type[BaseTrajectoryModel]:
    """Get a registered approach by name."""
    if name not in _APPROACHES:
        available = ", ".join(_APPROACHES.keys()) or "(none)"
        raise KeyError(f"Unknown approach '{name}'. Available: {available}")
    return _APPROACHES[name]


def get_all_approaches() -> dict[str, type[BaseTrajectoryModel]]:
    """Return all registered approaches."""
    return dict(_APPROACHES)
