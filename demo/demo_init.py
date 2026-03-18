#!/usr/bin/env python
"""E2E demo: Download OPL data and create DuckDB via CLI.

Usage:
    python -m demo.demo_init [--db-path /tmp/demo.duckdb]

Demonstrates:
    - opl init (download ZIP, extract CSV, ingest into DuckDB)
    - opl info (print database metadata)
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_init(db_path: Path) -> None:
    print(f"=== opl init (db: {db_path}) ===")
    subprocess.run(["opl", "init", "--db-path", str(db_path)], check=True)


def run_info(db_path: Path, label: str = "opl info") -> None:
    print(f"=== {label} ===")
    subprocess.run(["opl", "info", "--db-path", str(db_path)], check=True)


def run_update(db_path: Path) -> None:
    print("=== opl update ===")
    subprocess.run(["opl", "update", "--db-path", str(db_path)], check=True)


def main(db_path: Path | None = None) -> None:
    tmp_dir = None
    if db_path is None:
        tmp_dir = Path(tempfile.mkdtemp())
        db_path = tmp_dir / "demo.duckdb"

    try:
        run_init(db_path)
        print()
        run_info(db_path)
        print()
        run_update(db_path)
        print()
        run_info(db_path, label="opl info (post-update)")
        print()
        print("SUCCESS: init, info, and update all completed.")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    custom_path = None
    if "--db-path" in sys.argv:
        idx = sys.argv.index("--db-path")
        custom_path = Path(sys.argv[idx + 1])
    main(custom_path)
