from datetime import UTC, datetime
from pathlib import Path

import duckdb
import platformdirs


def default_db_path() -> Path:
    """Return the default DuckDB file path."""
    return Path(platformdirs.user_data_dir("opl-py")) / "opl.duckdb"


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection. Raises RuntimeError if the DB doesn't exist."""
    path = db_path or default_db_path()
    if not path.exists():
        raise RuntimeError(
            f"Database not found at {path}. Run `opl init` first to download and ingest data."
        )
    return duckdb.connect(str(path))


def ingest_csv(csv_path: Path, db_path: Path | None = None) -> Path:
    """Create or replace the DuckDB database from an OPL CSV file.

    Returns the database path.
    """
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(path))
    try:
        _create_entries_table(con, csv_path)
        _create_metadata(con, csv_path)
        _create_indexes(con)
    finally:
        con.close()

    return path


def update_db(csv_path: Path, db_path: Path | None = None) -> Path:
    """Re-ingest CSV into an existing database (drop and recreate entries)."""
    path = db_path or default_db_path()
    if not path.exists():
        raise RuntimeError(f"Database not found at {path}. Run `opl init` first.")

    con = duckdb.connect(str(path))
    try:
        con.execute("DROP TABLE IF EXISTS entries")
        con.execute("DROP INDEX IF EXISTS idx_name")
        con.execute("DROP INDEX IF EXISTS idx_federation")
        con.execute("DROP INDEX IF EXISTS idx_date")
        con.execute("DROP INDEX IF EXISTS idx_equipment")
        _create_entries_table(con, csv_path)
        _update_metadata(con, csv_path)
        _create_indexes(con)
    finally:
        con.close()

    return path


def get_db_info(db_path: Path | None = None) -> dict[str, str | int]:
    """Return metadata about the database."""
    con = get_connection(db_path)
    try:
        row = con.execute("SELECT * FROM metadata LIMIT 1").fetchone()
        if row is None:
            return {}
        cols = [desc[0] for desc in con.description]  # type: ignore[union-attr]
        return dict(zip(cols, row, strict=True))
    finally:
        con.close()


def _create_entries_table(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Create the entries table from a CSV file."""
    con.execute(f"""
        CREATE TABLE entries AS
        SELECT * FROM read_csv_auto(
            '{csv_path}', header=true, ignore_errors=true, null_padding=true
        )
    """)


def _create_metadata(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Create the metadata table."""
    row_count = con.execute("SELECT COUNT(*) FROM entries").fetchone()
    count = row_count[0] if row_count else 0
    csv_date = csv_path.stem  # e.g. "openpowerlifting-2025-01-01"
    now = datetime.now(UTC).isoformat()

    con.execute("DROP TABLE IF EXISTS metadata")
    con.execute("""
        CREATE TABLE metadata (
            last_updated VARCHAR,
            source_url VARCHAR,
            row_count BIGINT,
            csv_date VARCHAR
        )
    """)
    con.execute(
        "INSERT INTO metadata VALUES (?, ?, ?, ?)",
        [now, "https://data.openpowerlifting.org", count, csv_date],
    )


def _update_metadata(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Update metadata after a re-ingest."""
    row_count = con.execute("SELECT COUNT(*) FROM entries").fetchone()
    count = row_count[0] if row_count else 0
    csv_date = csv_path.stem
    now = datetime.now(UTC).isoformat()

    con.execute(
        "UPDATE metadata SET last_updated = ?, row_count = ?, csv_date = ?",
        [now, count, csv_date],
    )


def _create_indexes(con: duckdb.DuckDBPyConnection) -> None:
    """Create indexes on commonly queried columns."""
    con.execute("CREATE INDEX IF NOT EXISTS idx_name ON entries (Name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_federation ON entries (Federation)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_date ON entries (Date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_equipment ON entries (Equipment)")
