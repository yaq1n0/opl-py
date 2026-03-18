from pathlib import Path

from opl.analytics.normative import percentile
from opl.core.enums import Equipment, Sex


def test_percentile_basic(test_db: Path):
    # A 220kg squat should be high percentile among our test data
    p = percentile(
        lift="squat",
        weight=220.0,
        sex=Sex.MALE,
        equipment=Equipment.RAW,
        db_path=test_db,
    )
    assert isinstance(p, float)
    assert 0 <= p <= 100


def test_percentile_bench(test_db: Path):
    p = percentile(
        lift="bench",
        weight=150.0,
        sex=Sex.MALE,
        equipment=Equipment.RAW,
        db_path=test_db,
    )
    assert isinstance(p, float)
    assert 0 <= p <= 100


def test_percentile_total(test_db: Path):
    p = percentile(
        lift="total",
        weight=600.0,
        sex=Sex.MALE,
        equipment=Equipment.RAW,
        db_path=test_db,
    )
    assert isinstance(p, float)
    assert p > 0


def test_percentile_with_weight_class(test_db: Path):
    p = percentile(
        lift="total",
        weight=650.0,
        sex=Sex.MALE,
        equipment=Equipment.RAW,
        weight_class="93",
        db_path=test_db,
    )
    assert isinstance(p, float)


def test_percentile_invalid_lift():
    import pytest

    with pytest.raises(ValueError, match="lift must be one of"):
        percentile(
            lift="curl",
            weight=100.0,
            sex=Sex.MALE,
            equipment=Equipment.RAW,
        )
