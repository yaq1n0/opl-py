"""Train trajectory models on the full OPL dataset."""

from __future__ import annotations

import time
from pathlib import Path


def train(
    db_path: Path | None = None,
    output_dir: Path = Path("pretrained"),
    min_meets: int = 4,
    approach: str | None = None,
    limit: int | None = None,
) -> None:
    import opl
    from opl.analytics.trajectory import get_all_approaches, get_approach

    client = opl.OPL(db_path=db_path)

    stats = client.stats()
    csv_date_raw: str = str(stats.get("csv_date", "unknown"))
    dataset_date = csv_date_raw.removeprefix("openpowerlifting-")

    print(f"Dataset date : {dataset_date}")
    print(f"DB rows      : {stats.get('row_count', 'unknown'):,}")
    print()

    print(f"Loading lifters with {min_meets}+ meets (bulk query)...")
    t0 = time.monotonic()
    training_lifters = client.lifters_bulk(min_meets=min_meets, limit=limit)
    elapsed = time.monotonic() - t0
    print(f"  {len(training_lifters):,} lifters loaded in {elapsed:.1f}s")
    print()

    if approach:
        approach_cls = get_approach(approach)
        approaches = {approach: approach_cls}
    else:
        approaches = get_all_approaches()

    if not approaches:
        print("No approaches registered!")
        return

    print(f"Training {len(approaches)} approach(es): {', '.join(approaches.keys())}")
    print()

    all_scores: dict[str, dict[str, float]] = {}

    for approach_name, approach_cls in approaches.items():
        print(f"{'=' * 60}")
        print(f"  Approach: {approach_cls.display_name} ({approach_name})")
        print(f"{'=' * 60}")

        model_dir = output_dir / approach_name / dataset_date
        model_path = model_dir / "model.joblib"
        print(f"  Output: {model_path}")

        try:
            model = approach_cls()
            t0 = time.monotonic()
            scores = model.train(training_lifters, min_entries=min_meets)
            elapsed = time.monotonic() - t0

            print(f"  Training complete in {elapsed:.1f}s")
            print(f"  R² scores: {scores}")

            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            size_mb = model_path.stat().st_size / 1_000_000
            print(f"  Saved: {model_path} ({size_mb:.2f} MB)")

            all_scores[approach_name] = scores
        except Exception as exc:
            print(f"  FAILED: {exc}")
            all_scores[approach_name] = {}

        print()

    if len(all_scores) > 1:
        print(f"{'=' * 60}")
        print("  Model Comparison (R² scores)")
        print(f"{'=' * 60}")
        header = f"  {'Approach':<25} {'Total':>8} {'Squat':>8} {'Bench':>8} {'Deadlift':>8}"
        print(header)
        print(f"  {'-' * 57}")
        for approach_name, scores in all_scores.items():
            if scores:
                row = (
                    f"  {approach_name:<25} "
                    f"{scores.get('total', 0):>8.4f} "
                    f"{scores.get('squat', 0):>8.4f} "
                    f"{scores.get('bench', 0):>8.4f} "
                    f"{scores.get('deadlift', 0):>8.4f}"
                )
            else:
                row = f"  {approach_name:<25} {'FAILED':>8}"
            print(row)
        print()
