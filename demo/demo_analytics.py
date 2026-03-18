#!/usr/bin/env python
"""E2E demo: Analytics module — percentiles, features, and trajectory prediction.

Usage:
    python -m demo.demo_analytics [--db-path /path/to/opl.duckdb]

Demonstrates:
    - Normative percentile calculations
    - Feature extraction from a real lifter
    - Trajectory model training on sampled lifters
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
    TrajectoryPrediction,
    extract_features,
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
    pretrained_dirs = sorted(
        [d for d in pretrained_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not pretrained_dirs:
        print("  No pretrained models found in pretrained/")
        return

    latest_dir = pretrained_dirs[-1]
    pretrained_path = latest_dir / "model.joblib"
    print(f"  Using pretrained model: {latest_dir.name}")
    pretrained_model = TrajectoryModel()
    pretrained_model.load(pretrained_path)
    print(f"  Loaded model, is_trained={pretrained_model._is_trained}")  # pyright: ignore[reportPrivateUsage]

    if lifter and lifter.competition_count >= 3:
        # Default prediction (next expected meet)
        pred = predict_trajectory(lifter, model=pretrained_model)
        print(f"  --- Default (next meet) ---")
        print(f"  Target date:             {pred.target_date}")
        print(f"  Target bodyweight:       {pred.target_bodyweight_kg} kg")
        print(f"  Predicted total:         {pred.next_total_kg} kg")
        print(f"  Predicted squat:         {pred.next_squat_kg} kg")
        print(f"  Predicted bench:         {pred.next_bench_kg} kg")
        print(f"  Predicted deadlift:      {pred.next_deadlift_kg} kg")
        print(f"  Confidence interval:     {pred.confidence_interval}")

        # Targeted prediction: 1 year from now at 93kg
        targeted = predict_trajectory(
            lifter,
            model=pretrained_model,
            target_date=date(2027, 1, 1),
            target_bodyweight_kg=93.0,
        )
        print(f"  --- Targeted (2027-01-01 @ 93kg) ---")
        print(f"  Predicted total:         {targeted.next_total_kg} kg")
        print(f"  Predicted squat:         {targeted.next_squat_kg} kg")
        print(f"  Predicted bench:         {targeted.next_bench_kg} kg")
        print(f"  Predicted deadlift:      {targeted.next_deadlift_kg} kg")
        print(f"  Trajectory curve:        {targeted.trajectory_curve}")
    else:
        print("  No suitable lifter found for pretrained prediction")


def train_model(client: opl.OPL) -> tuple[TrajectoryModel, list[Lifter]]:
    print("=== Trajectory Prediction ===")
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

    model = TrajectoryModel()
    scores = model.train(training_lifters)
    print(f"  Model R² scores: {scores}")

    return model, training_lifters


def show_prediction(lifter: Lifter | None, model: TrajectoryModel) -> TrajectoryPrediction | None:
    if not lifter or lifter.competition_count < 3:
        return None
    # Default prediction
    prediction = predict_trajectory(lifter, model=model)
    print()
    print(f"=== Prediction for {lifter.name} ===")
    print(f"  Target date:          {prediction.target_date}")
    print(f"  Target bodyweight:    {prediction.target_bodyweight_kg} kg")
    print(f"  Predicted total:      {prediction.next_total_kg} kg")
    print(f"  Predicted squat:      {prediction.next_squat_kg} kg")
    print(f"  Predicted bench:      {prediction.next_bench_kg} kg")
    print(f"  Predicted deadlift:   {prediction.next_deadlift_kg} kg")
    print(f"  Confidence interval:  {prediction.confidence_interval}")
    print(f"  Trajectory curve:     {prediction.trajectory_curve}")

    # Targeted: what would they total at 83kg?
    targeted = predict_trajectory(lifter, model=model, target_bodyweight_kg=83.0)
    print(f"  --- At 83kg bodyweight ---")
    print(f"  Predicted total:      {targeted.next_total_kg} kg")
    return prediction


def show_save_load(
    lifter: Lifter | None, model: TrajectoryModel, prediction: TrajectoryPrediction | None
) -> None:
    print("=== Model Save/Load ===")
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        model_path = Path(tmp.name)

    model.save(model_path)
    print(f"  Saved model to {model_path} ({model_path.stat().st_size} bytes)")

    loaded_model = TrajectoryModel()
    loaded_model.load(model_path)
    print(f"  Loaded model, is_trained={loaded_model._is_trained}")  # pyright: ignore[reportPrivateUsage]

    if lifter and lifter.competition_count >= 3 and prediction is not None:
        loaded_pred = predict_trajectory(lifter, model=loaded_model)
        print(f"  Loaded model prediction: {loaded_pred.next_total_kg} kg")
        assert loaded_pred.next_total_kg == prediction.next_total_kg, "Mismatch!"
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

    model, _ = train_model(client)
    prediction = show_prediction(lifter, model)
    print()

    show_save_load(lifter, model, prediction)
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
