#!/usr/bin/env python
"""E2E demo: Analytics module — percentiles, features, and trajectory prediction.

Usage:
    python -m demo.demo_analytics [--db-path /path/to/opl.duckdb]

Demonstrates:
    - Normative percentile calculations
    - Feature extraction from a real lifter
    - Multi-approach trajectory model training and comparison
    - Trajectory prediction for a specific lifter (next meet + targeted)
    - Model save and load
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

import opl
from opl.analytics import (
    TrajectoryModel,
    extract_features,
    get_all_approaches,
    percentile,
    predict_trajectory,
)
from opl.core.models import Lifter


def show_percentiles(db_path: Path | None) -> None:
    print("=== Normative Percentiles ===")
    p1 = percentile(
        lift="squat",
        weight=200.0,
        sex=opl.Sex.MALE,
        equipment=opl.Equipment.RAW,
        weight_class="93",
        tested=True,
        db_path=db_path,
    )
    print(f"  200kg raw squat, tested M93: {p1}th percentile")

    p2 = percentile(
        lift="total",
        weight=600.0,
        sex=opl.Sex.MALE,
        equipment=opl.Equipment.RAW,
        tested=True,
        db_path=db_path,
    )
    print(f"  600kg raw total, tested M (all classes): {p2}th percentile")

    p3 = percentile(
        lift="bench",
        weight=100.0,
        sex=opl.Sex.FEMALE,
        equipment=opl.Equipment.RAW,
        db_path=db_path,
    )
    print(f"  100kg raw bench, F (all classes): {p3}th percentile")


def resolve_lifter(client: opl.OPL) -> Lifter | None:
    """Return John Haack or fall back to any lifter with 3+ meets."""
    lifter = client.lifter("John Haack")
    if not lifter or lifter.competition_count < 3:
        fallback_name = _find_lifter_with_entries(client, min_entries=3)
        lifter = client.lifter(fallback_name) if fallback_name else None
    return lifter


def show_features(lifter: Lifter | None) -> None:
    lifter_label = lifter.name if lifter else "unknown"
    print(f"=== Feature Extraction: {lifter_label} ===")
    if lifter:
        features = extract_features(lifter)
        print(f"  Lifter: {lifter.name}")
        print(f"  Career length: {features.career_length_days} days")
        print(f"  Competition count: {features.competition_count}")
        print(f"  Frequency: {features.competition_frequency} meets/year")
        print(f"  Best total: {features.best_total_kg} kg")
        print(f"  Latest bodyweight: {features.latest_bodyweight_kg} kg")
        print(f"  Progression rate: {features.total_progression_rate} kg/year")
        print(
            f"  S/B/D ratios: {features.squat_to_total_ratio}/"
            f"{features.bench_to_total_ratio}/{features.deadlift_to_total_ratio}"
        )
        print(f"  Equipment mode: {features.equipment_mode}")
    else:
        print("  No suitable lifter found for feature extraction")


def show_pretrained_prediction(lifter: Lifter | None) -> None:
    print("=== Trajectory Prediction (Pretrained Model) ===")
    pretrained_root = Path(__file__).parent.parent / "pretrained"

    # Check new structure first (pretrained/{approach}/{date}/model.joblib)
    gbt_dir = pretrained_root / "gradient_boosting"
    if gbt_dir.exists():
        candidates = sorted(gbt_dir.glob("*/model.joblib"))
    else:
        # Legacy flat structure
        candidates = sorted(pretrained_root.glob("*/model.joblib"))

    if not candidates:
        print("  No pretrained models found in pretrained/")
        return

    pretrained_path = candidates[-1]
    print(f"  Using pretrained model: {pretrained_path}")
    pretrained_model = TrajectoryModel()
    pretrained_model.load(pretrained_path)
    print(f"  Loaded model, is_trained={pretrained_model._is_trained}")  # pyright: ignore[reportPrivateUsage]

    if lifter and lifter.competition_count >= 3:
        pred = predict_trajectory(lifter, model=pretrained_model)
        print("  --- Default (next meet) ---")
        print(f"  Target date:             {pred.target_date}")
        print(f"  Target bodyweight:       {pred.target_bodyweight_kg} kg")
        print(f"  Predicted total:         {pred.next_total_kg} kg")
        print(f"  Predicted squat:         {pred.next_squat_kg} kg")
        print(f"  Predicted bench:         {pred.next_bench_kg} kg")
        print(f"  Predicted deadlift:      {pred.next_deadlift_kg} kg")
        print(f"  Confidence interval:     {pred.confidence_interval}")

        targeted = predict_trajectory(
            lifter,
            model=pretrained_model,
            target_date=date(2027, 1, 1),
            target_bodyweight_kg=93.0,
        )
        print("  --- Targeted (2027-01-01 @ 93kg) ---")
        print(f"  Predicted total:         {targeted.next_total_kg} kg")
    else:
        print("  No suitable lifter found for pretrained prediction")


def train_all_approaches(client: opl.OPL) -> dict[str, object]:
    """Train all registered approaches and return models + scores."""
    print("=== Multi-Approach Training ===")
    print("Fetching training lifters (4+ meets with totals)...")
    training_names = client.query("""
        SELECT "Name", COUNT(*) as n
        FROM entries
        WHERE "TotalKg" IS NOT NULL AND "TotalKg" > 0
        GROUP BY "Name"
        HAVING COUNT(*) >= 4
        ORDER BY RANDOM()
        LIMIT 500
    """)
    print(f"  Found {len(training_names)} candidate lifters")

    training_lifters = []
    for row in training_names:
        candidate = client.lifter(str(row["Name"]))
        if candidate and candidate.competition_count >= 4:
            training_lifters.append(candidate)

    print(f"  Loaded {len(training_lifters)} lifter objects")

    approaches = get_all_approaches()
    results: dict[str, object] = {"lifters": training_lifters}

    for approach_name, approach_cls in approaches.items():
        print(f"\n  --- {approach_cls.display_name} ({approach_name}) ---")
        try:
            model = approach_cls()
            scores = model.train(training_lifters)
            print(f"  R² scores: {scores}")
            results[approach_name] = model
        except Exception as exc:
            print(f"  FAILED: {exc}")

    # Comparison table
    print("\n  === Comparison ===")
    print(f"  {'Approach':<25} {'Total':>8} {'Squat':>8} {'Bench':>8} {'Deadlift':>8}")
    for approach_name, _approach_cls in approaches.items():
        model = results.get(approach_name)
        if model and hasattr(model, "predict"):
            # Re-train scores aren't stored, but we can note that training succeeded
            print(f"  {approach_name:<25} (trained)")

    return results


def show_prediction(lifter: Lifter | None, results: dict[str, object]) -> None:
    if not lifter or lifter.competition_count < 3:
        return

    approaches = get_all_approaches()
    for approach_name in approaches:
        model = results.get(approach_name)
        if model is None or not hasattr(model, "predict"):
            continue

        prediction = predict_trajectory(lifter, model=model)  # type: ignore[arg-type]
        print(f"\n=== Prediction for {lifter.name} ({approach_name}) ===")
        print(f"  Predicted total:      {prediction.next_total_kg} kg")
        print(f"  Predicted squat:      {prediction.next_squat_kg} kg")
        print(f"  Predicted bench:      {prediction.next_bench_kg} kg")
        print(f"  Predicted deadlift:   {prediction.next_deadlift_kg} kg")
        print(f"  Confidence interval:  {prediction.confidence_interval}")


def show_save_load(lifter: Lifter | None, results: dict[str, object]) -> None:
    print("\n=== Model Save/Load ===")
    model = results.get("gradient_boosting")
    if model is None or not hasattr(model, "save"):
        print("  No gradient_boosting model to save")
        return

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        model_path = Path(tmp.name)

    model.save(model_path)  # type: ignore[union-attr]
    print(f"  Saved model to {model_path} ({model_path.stat().st_size} bytes)")

    loaded_model = TrajectoryModel()
    loaded_model.load(model_path)
    print(f"  Loaded model, is_trained={loaded_model._is_trained}")  # pyright: ignore[reportPrivateUsage]

    if lifter and lifter.competition_count >= 3:
        original = predict_trajectory(lifter, model=model)  # type: ignore[arg-type]
        loaded_pred = predict_trajectory(lifter, model=loaded_model)
        print(f"  Original prediction: {original.next_total_kg} kg")
        print(f"  Loaded prediction:   {loaded_pred.next_total_kg} kg")
        assert loaded_pred.next_total_kg == original.next_total_kg, "Mismatch!"
        print("  Save/load produces identical predictions.")

    model_path.unlink(missing_ok=True)


def main(db_path: Path | None = None) -> None:
    client = opl.OPL(db_path=db_path)

    show_percentiles(db_path)
    print()

    lifter = resolve_lifter(client)
    show_features(lifter)
    print()

    show_pretrained_prediction(lifter)
    print()

    results = train_all_approaches(client)
    show_prediction(lifter, results)
    print()

    show_save_load(lifter, results)
    print()
    print("SUCCESS: All analytics operations completed.")


def _find_lifter_with_entries(client: opl.OPL, min_entries: int = 3) -> str | None:
    """Find any lifter with enough entries for testing."""
    result = client.query(f"""
        SELECT "Name" FROM entries
        WHERE "TotalKg" IS NOT NULL AND "TotalKg" > 0
        GROUP BY "Name"
        HAVING COUNT(*) >= {min_entries}
        LIMIT 1
    """)
    return str(result[0]["Name"]) if result else None


if __name__ == "__main__":
    custom_path = None
    if "--db-path" in sys.argv:
        idx = sys.argv.index("--db-path")
        custom_path = Path(sys.argv[idx + 1])
    main(custom_path)
