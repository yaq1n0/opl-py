from datetime import date
from pathlib import Path

from opl.analytics.features import extract_features
from opl.core.client import OPL
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


def test_extract_features_basic():
    entries = [
        _make_entry(
            date=date(2023, 1, 1),
            age=25.0,
            bodyweight_kg=93.0,
            weight_class_kg="93",
            total_kg=500.0,
            best3_squat_kg=200.0,
            best3_bench_kg=130.0,
            best3_deadlift_kg=170.0,
            tested=True,
        ),
        _make_entry(
            date=date(2023, 7, 1),
            age=25.5,
            bodyweight_kg=92.0,
            weight_class_kg="93",
            total_kg=520.0,
            best3_squat_kg=210.0,
            best3_bench_kg=135.0,
            best3_deadlift_kg=175.0,
            tested=True,
        ),
        _make_entry(
            date=date(2024, 1, 1),
            age=26.0,
            bodyweight_kg=93.0,
            weight_class_kg="93",
            total_kg=540.0,
            best3_squat_kg=220.0,
            best3_bench_kg=140.0,
            best3_deadlift_kg=180.0,
            tested=True,
        ),
    ]
    lifter = Lifter(name="Test Lifter", entries=entries)
    features = extract_features(lifter)

    assert features.competition_count == 3
    assert features.career_length_days == 365
    assert features.best_total_kg == 540.0
    assert features.latest_total_kg == 540.0
    assert features.latest_bodyweight_kg == 93.0
    assert features.age_at_latest == 26.0
    assert features.age_at_first == 25.0
    assert features.weight_class_numeric == 93.0
    assert features.is_tested is True
    assert features.equipment_mode == "Raw"
    assert features.total_progression_rate is not None
    assert features.squat_to_total_ratio is not None


def test_extract_features_from_db(test_db: Path):
    client = OPL(db_path=test_db)
    lifter = client.lifter("John Smith")
    assert lifter is not None
    features = extract_features(lifter)

    assert features.competition_count == 3
    assert features.best_total_kg == 665.0
    assert features.equipment_mode == "Raw"


def test_features_to_dict():
    entries = [
        _make_entry(date=date(2023, 1, 1), total_kg=500.0),
        _make_entry(date=date(2024, 1, 1), total_kg=520.0),
    ]
    lifter = Lifter(name="Test", entries=entries)
    features = extract_features(lifter)
    d = features.to_dict()

    assert isinstance(d, dict)
    assert "competition_count" in d
    assert "best_total_kg" in d
