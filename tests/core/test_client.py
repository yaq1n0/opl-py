from pathlib import Path

from opl.core.client import OPL
from opl.core.enums import Equipment, Event, Sex


def test_lifter_lookup(test_db: Path):
    client = OPL(db_path=test_db)
    lifter = client.lifter("John Smith")
    assert lifter is not None
    assert lifter.name == "John Smith"
    assert lifter.competition_count == 3
    assert lifter.best_total_kg == 665.0


def test_lifter_not_found(test_db: Path):
    client = OPL(db_path=test_db)
    assert client.lifter("Nobody Here") is None


def test_search_lifters(test_db: Path):
    client = OPL(db_path=test_db)
    results = client.search_lifters("Smith")
    assert len(results) == 1
    assert results[0].name == "John Smith"


def test_meet_lookup(test_db: Path):
    client = OPL(db_path=test_db)
    meet = client.meet("Summer Classic")
    assert meet is not None
    assert meet.federation == "USAPL"
    assert len(meet.entries) > 0


def test_meets_filter(test_db: Path):
    client = OPL(db_path=test_db)
    meets = client.meets(federation="USAPL")
    assert len(meets) > 0
    for m in meets:
        assert m.federation == "USAPL"


def test_rankings(test_db: Path):
    client = OPL(db_path=test_db)
    results = client.rankings(sex=Sex.MALE, equipment=Equipment.RAW, limit=5)
    assert len(results) > 0
    for entry in results:
        assert entry.sex == Sex.MALE
        assert entry.equipment == Equipment.RAW


def test_rankings_by_event(test_db: Path):
    client = OPL(db_path=test_db)
    results = client.rankings(event=Event.B)
    assert len(results) > 0
    for entry in results:
        assert entry.event == Event.B


def test_federations(test_db: Path):
    client = OPL(db_path=test_db)
    feds = client.federations()
    assert "USAPL" in feds
    assert "IPF" in feds


def test_raw_query(test_db: Path):
    client = OPL(db_path=test_db)
    results = client.query("SELECT COUNT(*) as n FROM entries")
    assert results[0]["n"] == 49


def test_stats(test_db: Path):
    client = OPL(db_path=test_db)
    stats = client.stats()
    assert stats["row_count"] == 49
