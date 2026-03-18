from datetime import date

from opl.core.enums import Equipment, Event, Sex
from opl.core.models import Entry, Lifter


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = {
        "name": "Test Lifter",
        "sex": Sex.MALE,
        "event": Event.SBD,
        "equipment": Equipment.RAW,
        "place": "1",
        "federation": "USAPL",
        "date": date(2024, 1, 1),
        "meet_name": "Test Meet",
    }
    defaults.update(overrides)
    return Entry.model_validate(defaults)


def test_entry_creation():
    e = _make_entry(total_kg=500.0, best3_squat_kg=200.0)
    assert e.name == "Test Lifter"
    assert e.total_kg == 500.0
    assert e.best3_squat_kg == 200.0
    assert e.age is None


def test_lifter_computed_fields():
    entries = [
        _make_entry(
            date=date(2023, 1, 1),
            total_kg=500.0,
            best3_squat_kg=200.0,
            best3_bench_kg=130.0,
            best3_deadlift_kg=170.0,
        ),
        _make_entry(
            date=date(2024, 1, 1),
            total_kg=520.0,
            best3_squat_kg=210.0,
            best3_bench_kg=135.0,
            best3_deadlift_kg=175.0,
        ),
    ]
    lifter = Lifter(name="Test Lifter", entries=entries)

    assert lifter.competition_count == 2
    assert lifter.best_total_kg == 520.0
    assert lifter.best_squat_kg == 210.0
    assert lifter.best_bench_kg == 135.0
    assert lifter.best_deadlift_kg == 175.0


def test_lifter_history_sorted():
    entries = [
        _make_entry(date=date(2024, 6, 1)),
        _make_entry(date=date(2023, 1, 1)),
        _make_entry(date=date(2024, 1, 1)),
    ]
    lifter = Lifter(name="Test Lifter", entries=entries)
    history = lifter.history()
    assert history[0].date == date(2023, 1, 1)
    assert history[-1].date == date(2024, 6, 1)


def test_lifter_no_totals():
    entries = [_make_entry()]
    lifter = Lifter(name="Test Lifter", entries=entries)
    assert lifter.best_total_kg is None
    assert lifter.best_squat_kg is None
