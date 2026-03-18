import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from opl.core.db import ingest_csv


@pytest.fixture
def fixture_csv() -> Path:
    return Path(__file__).parent / "fixtures" / "test_data.csv"


@pytest.fixture
def test_db(fixture_csv: Path) -> Generator[Path, None, None]:
    """Create a temporary DuckDB from the test fixture CSV."""
    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "test.duckdb"
    ingest_csv(fixture_csv, db_path)
    yield db_path
    shutil.rmtree(tmp_dir, ignore_errors=True)
