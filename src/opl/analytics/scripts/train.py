#!/usr/bin/env python
"""Train the trajectory model on the full OPL dataset.

Usage:
    python -m opl.analytics.scripts.train
    python -m opl.analytics.scripts.train --db-path /path/to/opl.duckdb
    python -m opl.analytics.scripts.train --output-dir /path/to/pretrained
    python -m opl.analytics.scripts.train --min-meets 4 --limit 5000  # for testing

Output:
    {output_dir}/{csv_date}/model.joblib

The csv_date is read from the database metadata (derived from the OPL CSV filename,
e.g. "2025-01-01"). This lets you version pretrained models by dataset release.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the OPL trajectory model on the full dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db-path", type=Path, default=None, help="Path to opl.duckdb")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained"),
        help="Root directory for saved models (default: ./pretrained)",
    )
    parser.add_argument(
        "--min-meets",
        type=int,
        default=4,
        help="Minimum number of meets a lifter must have to be included (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of training lifters (omit for full dataset)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import opl
    from opl.analytics import TrajectoryModel

    client = opl.OPL(db_path=args.db_path)

    # Resolve dataset date from DB metadata
    stats = client.stats()
    csv_date_raw: str = str(stats.get("csv_date", "unknown"))
    # Strip "openpowerlifting-" prefix to get just the date, e.g. "2025-01-01"
    dataset_date = csv_date_raw.removeprefix("openpowerlifting-")

    output_dir = args.output_dir / dataset_date
    model_path = output_dir / "model.joblib"

    print(f"Dataset date : {dataset_date}")
    print(f"DB rows      : {stats.get('row_count', 'unknown'):,}")
    print(f"Output       : {model_path}")
    print()

    # --- Load lifters (single bulk query) ---
    print(f"Loading lifters with {args.min_meets}+ meets (bulk query)...")
    t0 = time.monotonic()
    training_lifters = client.lifters_bulk(min_meets=args.min_meets, limit=args.limit)
    elapsed = time.monotonic() - t0
    print(f"  {len(training_lifters):,} lifters loaded in {elapsed:.1f}s")
    print()

    # --- Train ---
    print("Training model...")
    t0 = time.monotonic()
    model = TrajectoryModel()
    scores = model.train(training_lifters, min_entries=args.min_meets)
    print(f"  Training complete in {time.monotonic() - t0:.1f}s")
    print(f"  R² scores: {scores}")
    print()

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    size_mb = model_path.stat().st_size / 1_000_000
    print(f"Saved: {model_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
