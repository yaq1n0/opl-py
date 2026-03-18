from pathlib import Path

import duckdb
import pytest

from opl.core.db import get_connection, get_db_info


def test_ingest_creates_tables(test_db: Path):
    con = duckdb.connect(str(test_db))
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    con.close()
    assert "entries" in tables
    assert "metadata" in tables


def test_row_count(test_db: Path):
    con = duckdb.connect(str(test_db))
    row = con.execute("SELECT COUNT(*) FROM entries").fetchone()
    count = row[0] if row else 0
    con.close()
    assert count == 49


def test_metadata(test_db: Path):
    info = get_db_info(test_db)
    assert info["row_count"] == 49
    assert info["source_url"] == "https://data.openpowerlifting.org"


def test_get_connection_missing_db():
    with pytest.raises(RuntimeError, match="Run `opl init` first"):
        get_connection(Path("/tmp/nonexistent_opl_test.duckdb"))


def test_indexes_created(test_db: Path):
    con = duckdb.connect(str(test_db))
    indexes = con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    index_names = [row[0] for row in indexes]
    con.close()
    assert "idx_name" in index_names
    assert "idx_federation" in index_names
    assert "idx_date" in index_names
    assert "idx_equipment" in index_names
