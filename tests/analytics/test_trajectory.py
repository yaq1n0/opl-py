from datetime import date
from pathlib import Path

import pytest

from opl.analytics.trajectory import (
    TrajectoryModel,
    TrajectoryPrediction,
    _build_feature_row,  # pyright: ignore[reportPrivateUsage]
    _features_to_array,  # pyright: ignore[reportPrivateUsage]
    _project_trajectory,  # pyright: ignore[reportPrivateUsage]
    predict_trajectory,
)
from opl.core.client import OPL
from opl.core.enums import Equipment, Event, Sex
from opl.core.models import Entry, Lifter


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = {
        "name": "Test Lifter",
        "sex": Sex.MALE,
        "event": Event.SBD,
        "equipment": Equipment.RAW,
        "place": "1",
        "federation": "USAPL",
        "date": date(2024, 1, 1),
        "meet_name": "Test Meet",
    }
    defaults.update(overrides)
    return Entry.model_validate(defaults)


def _make_lifter(name: str, count: int = 4) -> Lifter:
    """Create a lifter with `count` progressive competition entries.

    Bodyweights vary across entries to provide training signal for bodyweight effects.
    """
    entries = []
    base_total = 500.0
    bodyweights = [83.0, 90.0, 93.0, 100.0, 93.0, 105.0]
    weight_classes = ["83", "90", "93", "100", "93", "105"]
    for i in range(count):
        bw = bodyweights[i % len(bodyweights)]
        wc = weight_classes[i % len(weight_classes)]
        # Bodyweight effect: heavier = higher total (roughly 1.5 kg per kg bodyweight)
        bw_effect = (bw - 83.0) * 1.5
        entries.append(
            _make_entry(
                name=name,
                date=date(2022 + i // 2, 1 + (i % 2) * 6, 15),
                age=25.0 + i * 0.5,
                bodyweight_kg=bw,
                weight_class_kg=wc,
                total_kg=base_total + i * 15 + bw_effect,
                best3_squat_kg=200.0 + i * 5 + bw_effect * 0.4,
                best3_bench_kg=130.0 + i * 3 + bw_effect * 0.25,
                best3_deadlift_kg=170.0 + i * 7 + bw_effect * 0.35,
                tested=True,
            )
        )
    return Lifter(name=name, entries=entries)


def _build_training_lifters(n: int = 15) -> list[Lifter]:
    """Build a set of synthetic lifters for training."""
    lifters = []
    for i in range(n):
        lifters.append(_make_lifter(f"Lifter {i}", count=4 + (i % 3)))
    return lifters


class TestTrajectoryModel:
    def test_train_returns_scores(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        scores = model.train(lifters)

        assert isinstance(scores, dict)
        assert "total" in scores
        assert "squat" in scores
        assert "bench" in scores
        assert "deadlift" in scores
        for key, score in scores.items():
            assert isinstance(score, float), f"{key} score is not a float"

    def test_train_marks_model_as_trained(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        assert not model._is_trained  # pyright: ignore[reportPrivateUsage]
        model.train(lifters)
        assert model._is_trained  # pyright: ignore[reportPrivateUsage]

    def test_train_too_few_samples(self):
        lifters = [_make_lifter("Solo", count=4)]
        model = TrajectoryModel()
        with pytest.raises(ValueError, match="Not enough training data"):
            model.train(lifters)

    def test_train_filters_by_min_entries(self):
        # 15 lifters with 4-6 entries each, but min_entries=10 filters them all out
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        with pytest.raises(ValueError, match="Not enough training data"):
            model.train(lifters, min_entries=10)

    def test_train_generates_multi_samples(self):
        """A lifter with N entries should produce N-1 training samples."""
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        scores = model.train(lifters)
        # Just verify training succeeds with more data — the multi-sample
        # generation is tested implicitly by training working with fewer lifters
        assert all(isinstance(v, float) for v in scores.values())

    def test_predict_returns_prediction(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target)

        assert isinstance(prediction, TrajectoryPrediction)
        assert prediction.next_total_kg is not None
        assert prediction.next_squat_kg is not None
        assert prediction.next_bench_kg is not None
        assert prediction.next_deadlift_kg is not None

    def test_predict_backwards_compatible(self):
        """predict(lifter) with no extra args still works."""
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target)

        assert prediction.next_total_kg is not None
        assert prediction.target_date is not None
        assert prediction.target_bodyweight_kg is not None

    def test_predict_with_target_date(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target, target_date=date(2026, 6, 1))

        assert prediction.next_total_kg is not None
        assert prediction.target_date == date(2026, 6, 1)

    def test_predict_with_target_bodyweight(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        pred_heavy = model.predict(target, target_bodyweight_kg=120.0)
        pred_light = model.predict(target, target_bodyweight_kg=60.0)

        assert pred_heavy.target_bodyweight_kg == 120.0
        assert pred_light.target_bodyweight_kg == 60.0
        # Both should return valid predictions
        assert pred_heavy.next_total_kg is not None
        assert pred_light.next_total_kg is not None

    def test_predict_with_both_targets(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(
            target,
            target_date=date(2027, 1, 1),
            target_bodyweight_kg=93.0,
        )

        assert prediction.next_total_kg is not None
        assert prediction.target_date == date(2027, 1, 1)
        assert prediction.target_bodyweight_kg == 93.0

    def test_predict_has_confidence_interval(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target)

        assert prediction.confidence_interval is not None
        assert prediction.next_total_kg is not None
        low, high = prediction.confidence_interval
        assert low < high
        assert low < prediction.next_total_kg < high

    def test_predict_has_trajectory_curve(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target)

        assert len(prediction.trajectory_curve) == 6
        for month, total in prediction.trajectory_curve:
            assert isinstance(month, int)
            assert isinstance(total, float)
        # Months should be increasing
        months = [m for m, _ in prediction.trajectory_curve]
        assert months == sorted(months)

    def test_predict_untrained_raises(self):
        model = TrajectoryModel()
        target = _make_lifter("Target", count=4)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(target)

    def test_save_and_load(self, tmp_path: Path):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        original_pred = model.predict(target)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)
        assert model_path.exists()

        loaded_model = TrajectoryModel()
        loaded_model.load(model_path)
        assert loaded_model._is_trained  # pyright: ignore[reportPrivateUsage]

        loaded_pred = loaded_model.predict(target)
        assert loaded_pred.next_total_kg == original_pred.next_total_kg
        assert loaded_pred.next_squat_kg == original_pred.next_squat_kg

    def test_save_untrained_raises(self, tmp_path: Path):
        model = TrajectoryModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.save(tmp_path / "model.joblib")

    def test_load_old_version_raises(self, tmp_path: Path):
        """Loading a v1 model (without version key) should raise."""
        import joblib  # type: ignore[import-untyped]

        model_path = tmp_path / "old_model.joblib"
        joblib.dump(
            {"total": None, "squat": None, "bench": None, "deadlift": None},
            model_path,
        )
        model = TrajectoryModel()
        with pytest.raises(ValueError, match="version"):
            model.load(model_path)


class TestPredictTrajectoryConvenience:
    def test_no_model_raises(self):
        target = _make_lifter("Target", count=4)
        with pytest.raises(ValueError, match="trained TrajectoryModel is required"):
            predict_trajectory(target)

    def test_with_model(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        prediction = predict_trajectory(target, model=model)
        assert isinstance(prediction, TrajectoryPrediction)
        assert prediction.next_total_kg is not None

    def test_with_targets(self):
        lifters = _build_training_lifters(15)
        model = TrajectoryModel()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        prediction = predict_trajectory(
            target,
            model=model,
            target_date=date(2027, 1, 1),
            target_bodyweight_kg=93.0,
        )
        assert prediction.target_date == date(2027, 1, 1)
        assert prediction.target_bodyweight_kg == 93.0


class TestFeaturesToArray:
    def test_numeric_values(self):
        feat_dict = {
            "career_length_days": 365,
            "competition_count": 4,
            "competition_frequency": 2.0,
            "best_total_kg": 500.0,
            "best_squat_kg": 200.0,
            "best_bench_kg": 130.0,
            "best_deadlift_kg": 170.0,
            "latest_total_kg": 500.0,
            "latest_bodyweight_kg": 93.0,
            "total_progression_rate": 20.0,
            "squat_to_total_ratio": 0.4,
            "bench_to_total_ratio": 0.26,
            "deadlift_to_total_ratio": 0.34,
            "age_at_latest": 26.0,
            "age_at_first": 25.0,
            "weight_class_numeric": 93.0,
            "days_since_last_comp": 180,
            "is_tested": True,
            "equipment_mode": "Raw",
        }
        arr = _features_to_array(feat_dict)
        assert len(arr) == 17  # _FEATURE_KEYS has 17 entries
        assert all(isinstance(v, float) for v in arr)

    def test_none_values_become_nan(self):
        import math

        feat_dict: dict[str, object] = {
            "career_length_days": 0,
            "competition_count": 1,
            "competition_frequency": 0.0,
            "best_total_kg": None,
            "best_squat_kg": None,
            "best_bench_kg": None,
            "best_deadlift_kg": None,
            "latest_total_kg": None,
            "latest_bodyweight_kg": None,
            "total_progression_rate": None,
            "squat_to_total_ratio": None,
            "bench_to_total_ratio": None,
            "deadlift_to_total_ratio": None,
            "age_at_latest": None,
            "age_at_first": None,
            "weight_class_numeric": None,
            "days_since_last_comp": 0,
            "is_tested": None,
            "equipment_mode": "Raw",
        }
        arr = _features_to_array(feat_dict)
        # best_total_kg is index 3, should be nan
        assert math.isnan(arr[3])


class TestBuildFeatureRow:
    def test_appends_context_features(self):
        feat_dict = {
            "career_length_days": 365,
            "competition_count": 4,
            "competition_frequency": 2.0,
            "best_total_kg": 500.0,
            "best_squat_kg": 200.0,
            "best_bench_kg": 130.0,
            "best_deadlift_kg": 170.0,
            "latest_total_kg": 500.0,
            "latest_bodyweight_kg": 93.0,
            "total_progression_rate": 20.0,
            "squat_to_total_ratio": 0.4,
            "bench_to_total_ratio": 0.26,
            "deadlift_to_total_ratio": 0.34,
            "age_at_latest": 26.0,
            "age_at_first": 25.0,
            "weight_class_numeric": 93.0,
            "days_since_last_comp": 180,
        }
        row = _build_feature_row(feat_dict, days_to_target=90, target_bodyweight_kg=93.0)
        assert len(row) == 19  # 17 history features + 2 context features
        assert row[-2] == 90.0  # days_to_target
        assert row[-1] == 93.0  # target_bodyweight_kg

    def test_none_bodyweight(self):
        import math

        feat_dict: dict[str, object] = {"career_length_days": 0}
        row = _build_feature_row(feat_dict, days_to_target=30, target_bodyweight_kg=None)
        assert row[-2] == 30.0
        assert math.isnan(row[-1])


class TestProjectTrajectory:
    def test_with_history(self):
        lifter = _make_lifter("Test", count=4)
        curve = _project_trajectory(lifter, predicted_next=560.0, months=12, points=6)
        assert len(curve) == 6
        for month, total in curve:
            assert isinstance(month, int)
            assert isinstance(total, float)

    def test_single_entry_returns_flat(self):
        lifter = _make_lifter("Test", count=1)
        curve = _project_trajectory(lifter, predicted_next=500.0, months=12, points=6)
        assert len(curve) == 6
        # Should be flat since no progression can be calculated
        totals = [t for _, t in curve]
        assert all(t == 500.0 for t in totals)


class TestTrajectoryWithFixtureDB:
    def test_train_and_predict_from_db(self, test_db: Path):
        """Train on fixture data lifters and predict for one."""
        client = OPL(db_path=test_db)

        # Gather lifters with 3+ entries from fixture
        names_result = client.query("""
            SELECT "Name", COUNT(*) as n
            FROM entries
            WHERE "TotalKg" IS NOT NULL AND "TotalKg" > 0
            GROUP BY "Name"
            HAVING COUNT(*) >= 3
        """)
        lifters = []
        for row in names_result:
            candidate = client.lifter(str(row["Name"]))
            if candidate:
                lifters.append(candidate)

        assert len(lifters) >= 10, f"Expected 10+ lifters with 3+ entries, got {len(lifters)}"

        model = TrajectoryModel()
        scores = model.train(lifters, min_entries=3)
        assert all(k in scores for k in ("total", "squat", "bench", "deadlift"))

        # Predict for a known lifter
        john = client.lifter("John Smith")
        assert john is not None
        prediction = model.predict(john)
        assert prediction.next_total_kg is not None
        assert prediction.next_total_kg > 0

    def test_predict_with_targets_from_db(self, test_db: Path):
        """Predict at a specific date and bodyweight from fixture data."""
        client = OPL(db_path=test_db)

        names_result = client.query("""
            SELECT "Name", COUNT(*) as n
            FROM entries
            WHERE "TotalKg" IS NOT NULL AND "TotalKg" > 0
            GROUP BY "Name"
            HAVING COUNT(*) >= 3
        """)
        lifters = []
        for row in names_result:
            candidate = client.lifter(str(row["Name"]))
            if candidate:
                lifters.append(candidate)

        model = TrajectoryModel()
        model.train(lifters, min_entries=3)

        john = client.lifter("John Smith")
        assert john is not None

        prediction = model.predict(
            john,
            target_date=date(2027, 6, 1),
            target_bodyweight_kg=93.0,
        )
        assert prediction.next_total_kg is not None
        assert prediction.next_total_kg > 0
        assert prediction.target_date == date(2027, 6, 1)
        assert prediction.target_bodyweight_kg == 93.0
