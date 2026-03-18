from datetime import date

from pydantic import BaseModel, computed_field

from opl.core.enums import Equipment, Event, Sex


class Entry(BaseModel):
    """A single competition entry (one row in the OPL dataset)."""

    name: str
    sex: Sex
    event: Event
    equipment: Equipment
    age: float | None = None
    age_class: str | None = None
    birth_year_class: str | None = None
    division: str | None = None
    bodyweight_kg: float | None = None
    weight_class_kg: str | None = None
    squat1_kg: float | None = None
    squat2_kg: float | None = None
    squat3_kg: float | None = None
    best3_squat_kg: float | None = None
    bench1_kg: float | None = None
    bench2_kg: float | None = None
    bench3_kg: float | None = None
    best3_bench_kg: float | None = None
    deadlift1_kg: float | None = None
    deadlift2_kg: float | None = None
    deadlift3_kg: float | None = None
    best3_deadlift_kg: float | None = None
    total_kg: float | None = None
    place: str
    dots: float | None = None
    wilks: float | None = None
    glossbrenner: float | None = None
    goodlift: float | None = None
    tested: bool | None = None
    country: str | None = None
    state: str | None = None
    federation: str
    parent_federation: str | None = None
    date: date
    meet_country: str | None = None
    meet_state: str | None = None
    meet_town: str | None = None
    meet_name: str
    sanctioned: bool | None = None


class Lifter(BaseModel):
    """Aggregated view of a lifter across all competitions."""

    name: str
    entries: list[Entry]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def competition_count(self) -> int:
        return len(self.entries)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_total_kg(self) -> float | None:
        totals = [e.total_kg for e in self.entries if e.total_kg is not None]
        return max(totals) if totals else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_squat_kg(self) -> float | None:
        vals = [e.best3_squat_kg for e in self.entries if e.best3_squat_kg is not None]
        return max(vals) if vals else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_bench_kg(self) -> float | None:
        vals = [e.best3_bench_kg for e in self.entries if e.best3_bench_kg is not None]
        return max(vals) if vals else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_deadlift_kg(self) -> float | None:
        vals = [e.best3_deadlift_kg for e in self.entries if e.best3_deadlift_kg is not None]
        return max(vals) if vals else None

    def history(self) -> list[Entry]:
        """Entries sorted chronologically."""
        return sorted(self.entries, key=lambda e: e.date)


class Meet(BaseModel):
    """A competition meet."""

    name: str
    date: date
    federation: str
    country: str | None = None
    state: str | None = None
    town: str | None = None
    entries: list[Entry]
