"""Feature engineering for ML models from lifter competition history."""

from collections import Counter
from dataclasses import dataclass

from opl.core.models import Entry, Lifter


@dataclass
class LifterFeatures:
    """ML-ready features extracted from a lifter's competition history."""

    career_length_days: int
    competition_count: int
    competition_frequency: float  # meets per year
    best_total_kg: float | None
    best_squat_kg: float | None
    best_bench_kg: float | None
    best_deadlift_kg: float | None
    latest_total_kg: float | None
    latest_bodyweight_kg: float | None
    total_progression_rate: float | None  # kg per year
    squat_to_total_ratio: float | None
    bench_to_total_ratio: float | None
    deadlift_to_total_ratio: float | None
    age_at_latest: float | None
    age_at_first: float | None
    weight_class_numeric: float | None
    is_tested: bool | None
    equipment_mode: str
    days_since_last_comp: int

    def to_dict(self) -> dict[str, object]:
        """Convert to a plain dictionary for use with ML frameworks."""
        return {
            "career_length_days": self.career_length_days,
            "competition_count": self.competition_count,
            "competition_frequency": self.competition_frequency,
            "best_total_kg": self.best_total_kg,
            "best_squat_kg": self.best_squat_kg,
            "best_bench_kg": self.best_bench_kg,
            "best_deadlift_kg": self.best_deadlift_kg,
            "latest_total_kg": self.latest_total_kg,
            "latest_bodyweight_kg": self.latest_bodyweight_kg,
            "total_progression_rate": self.total_progression_rate,
            "squat_to_total_ratio": self.squat_to_total_ratio,
            "bench_to_total_ratio": self.bench_to_total_ratio,
            "deadlift_to_total_ratio": self.deadlift_to_total_ratio,
            "age_at_latest": self.age_at_latest,
            "age_at_first": self.age_at_first,
            "weight_class_numeric": self.weight_class_numeric,
            "is_tested": self.is_tested,
            "equipment_mode": self.equipment_mode,
            "days_since_last_comp": self.days_since_last_comp,
        }


def extract_features(lifter: Lifter) -> LifterFeatures:
    """Extract ML-ready features from a lifter's competition history."""
    history = lifter.history()
    if not history:
        raise ValueError(f"Lifter {lifter.name} has no competition entries")

    first = history[0]
    latest = history[-1]

    career_length_days = (latest.date - first.date).days
    career_years = max(career_length_days / 365.25, 0.01)  # avoid division by zero

    competition_frequency = len(history) / career_years if career_length_days > 0 else 0.0

    # Progression rate
    total_progression_rate = _calc_progression_rate(history, career_years)

    # Lift ratios from latest entry with a total
    squat_ratio, bench_ratio, deadlift_ratio = _calc_lift_ratios(latest)

    # Weight class as numeric
    weight_class_numeric = _parse_weight_class(latest.weight_class_kg)

    # Equipment mode
    equipment_counts = Counter(e.equipment.value for e in history)
    equipment_mode = equipment_counts.most_common(1)[0][0]

    return LifterFeatures(
        career_length_days=career_length_days,
        competition_count=len(history),
        competition_frequency=round(competition_frequency, 2),
        best_total_kg=lifter.best_total_kg,
        best_squat_kg=lifter.best_squat_kg,
        best_bench_kg=lifter.best_bench_kg,
        best_deadlift_kg=lifter.best_deadlift_kg,
        latest_total_kg=latest.total_kg,
        latest_bodyweight_kg=latest.bodyweight_kg,
        total_progression_rate=total_progression_rate,
        squat_to_total_ratio=squat_ratio,
        bench_to_total_ratio=bench_ratio,
        deadlift_to_total_ratio=deadlift_ratio,
        age_at_latest=latest.age,
        age_at_first=first.age,
        weight_class_numeric=weight_class_numeric,
        is_tested=latest.tested,
        equipment_mode=equipment_mode,
        days_since_last_comp=(latest.date - history[-2].date).days if len(history) > 1 else 0,
    )


def _calc_progression_rate(history: list[Entry], career_years: float) -> float | None:
    """Calculate total kg gained per year of competing."""
    totals = [(e.date, e.total_kg) for e in history if e.total_kg is not None]
    if len(totals) < 2:
        return None
    first_total = totals[0][1]
    latest_total = totals[-1][1]
    return round((latest_total - first_total) / career_years, 2)


def _calc_lift_ratios(entry: Entry) -> tuple[float | None, float | None, float | None]:
    """Calculate squat:bench:deadlift ratios relative to total."""
    total = entry.total_kg
    if total is None or total == 0:
        return None, None, None

    squat = round(entry.best3_squat_kg / total, 3) if entry.best3_squat_kg else None
    bench = round(entry.best3_bench_kg / total, 3) if entry.best3_bench_kg else None
    deadlift = round(entry.best3_deadlift_kg / total, 3) if entry.best3_deadlift_kg else None

    return squat, bench, deadlift


def _parse_weight_class(weight_class: str | None) -> float | None:
    """Parse weight class string to a numeric value."""
    if not weight_class:
        return None
    # Remove '+' suffix for SHW classes
    cleaned = weight_class.rstrip("+")
    try:
        return float(cleaned)
    except ValueError:
        return None
