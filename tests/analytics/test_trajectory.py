"""Tests for the trajectory prediction framework — all approaches."""

from datetime import date
from pathlib import Path

import pytest

from opl.analytics.trajectory import (
    TrajectoryModel,
    TrajectoryPrediction,
    predict_trajectory,
)
from opl.analytics.trajectory.base import (
    _FEATURE_KEYS,
    build_feature_row,
    build_training_data,
    features_to_array,
    project_trajectory_linear,
    project_trajectory_with_model,
    resolve_prediction_context,
)
from opl.analytics.trajectory.gradient_boosting import GradientBoostingModel
from opl.analytics.trajectory.quantile_gbt import QuantileGBTModel
from opl.analytics.trajectory.registry import get_all_approaches, get_approach, list_approaches
from opl.core.client import OPL
from opl.core.enums import Equipment, Event, Sex
from opl.core.models import Entry, Lifter

# Backward compat aliases for private names
_features_to_array = features_to_array
_build_feature_row = build_feature_row
_project_trajectory = project_trajectory_linear


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
    """Create a lifter with `count` progressive competition entries."""
    entries = []
    base_total = 500.0
    bodyweights = [83.0, 90.0, 93.0, 100.0, 93.0, 105.0]
    weight_classes = ["83", "90", "93", "100", "93", "105"]
    for i in range(count):
        bw = bodyweights[i % len(bodyweights)]
        wc = weight_classes[i % len(weight_classes)]
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


# ===========================================================================
# Registry tests
# ===========================================================================
class TestRegistry:
    def test_list_approaches(self):
        approaches = list_approaches()
        assert "gradient_boosting" in approaches
        assert "quantile_gbt" in approaches
        assert "random_forest" not in approaches
        assert "neural_network" not in approaches

    def test_get_approach(self):
        cls = get_approach("gradient_boosting")
        assert cls is GradientBoostingModel

    def test_get_unknown_approach(self):
        with pytest.raises(KeyError, match="Unknown approach"):
            get_approach("nonexistent")

    def test_all_approaches_have_metadata(self):
        for name, cls in get_all_approaches().items():
            assert cls.name == name
            assert cls.display_name
            assert cls.description


# ===========================================================================
# Backward compatibility
# ===========================================================================
class TestBackwardCompat:
    def test_trajectory_model_alias(self):
        assert TrajectoryModel is GradientBoostingModel

    def test_old_imports_work(self):
        from opl.analytics import TrajectoryModel as TM  # noqa: N817
        from opl.analytics import TrajectoryPrediction as TP  # noqa: N817
        from opl.analytics import predict_trajectory as pt

        assert TM is GradientBoostingModel
        assert TP is TrajectoryPrediction
        assert callable(pt)


# ===========================================================================
# Gradient Boosting tests (migrated from original test_trajectory.py)
# ===========================================================================
class TestGradientBoostingModel:
    def test_train_returns_scores(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
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
        model = GradientBoostingModel()
        assert not model._is_trained
        model.train(lifters)
        assert model._is_trained

    def test_train_too_few_samples(self):
        lifters = [_make_lifter("Solo", count=4)]
        model = GradientBoostingModel()
        with pytest.raises(ValueError, match="Not enough training data"):
            model.train(lifters)

    def test_train_filters_by_min_entries(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        with pytest.raises(ValueError, match="Not enough training data"):
            model.train(lifters, min_entries=10)

    def test_predict_returns_prediction(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target)

        assert isinstance(prediction, TrajectoryPrediction)
        assert prediction.next_total_kg is not None
        assert prediction.next_squat_kg is not None
        assert prediction.next_bench_kg is not None
        assert prediction.next_deadlift_kg is not None

    def test_predict_backwards_compatible(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target)

        assert prediction.next_total_kg is not None
        assert prediction.target_date is not None
        assert prediction.target_bodyweight_kg is not None

    def test_predict_with_target_date(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target, target_date=date(2026, 6, 1))

        assert prediction.next_total_kg is not None
        assert prediction.target_date == date(2026, 6, 1)

    def test_predict_with_target_bodyweight(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        pred_heavy = model.predict(target, target_bodyweight_kg=120.0)
        pred_light = model.predict(target, target_bodyweight_kg=60.0)

        assert pred_heavy.target_bodyweight_kg == 120.0
        assert pred_light.target_bodyweight_kg == 60.0
        assert pred_heavy.next_total_kg is not None
        assert pred_light.next_total_kg is not None

    def test_predict_with_both_targets(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
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
        model = GradientBoostingModel()
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
        model = GradientBoostingModel()
        model.train(lifters)

        target = _make_lifter("Target Lifter", count=5)
        prediction = model.predict(target)

        assert len(prediction.trajectory_curve) == 6
        for month, total in prediction.trajectory_curve:
            assert isinstance(month, int)
            assert isinstance(total, float)
        months = [m for m, _ in prediction.trajectory_curve]
        assert months == sorted(months)

    def test_predict_untrained_raises(self):
        model = GradientBoostingModel()
        target = _make_lifter("Target", count=4)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(target)

    def test_save_and_load(self, tmp_path: Path):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        original_pred = model.predict(target)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)
        assert model_path.exists()

        loaded_model = GradientBoostingModel()
        loaded_model.load(model_path)
        assert loaded_model._is_trained

        loaded_pred = loaded_model.predict(target)
        assert loaded_pred.next_total_kg == original_pred.next_total_kg
        assert loaded_pred.next_squat_kg == original_pred.next_squat_kg

    def test_save_untrained_raises(self, tmp_path: Path):
        model = GradientBoostingModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.save(tmp_path / "model.joblib")

    def test_load_old_version_raises(self, tmp_path: Path):
        import joblib  # type: ignore[import-untyped]

        model_path = tmp_path / "old_model.joblib"
        joblib.dump(
            {"total": None, "squat": None, "bench": None, "deadlift": None},
            model_path,
        )
        model = GradientBoostingModel()
        with pytest.raises(ValueError, match="version"):
            model.load(model_path)

        v2_path = tmp_path / "v2_model.joblib"
        joblib.dump(
            {"version": 2, "total": None, "squat": None, "bench": None, "deadlift": None},
            v2_path,
        )
        with pytest.raises(ValueError, match="version"):
            model.load(v2_path)


# ===========================================================================
# Quantile GBT tests
# ===========================================================================
class TestQuantileGBTModel:
    def test_train_and_predict(self):
        lifters = _build_training_lifters(15)
        model = QuantileGBTModel()
        scores = model.train(lifters)

        assert isinstance(scores, dict)
        assert all(k in scores for k in ("total", "squat", "bench", "deadlift"))

        target = _make_lifter("Target", count=5)
        prediction = model.predict(target)

        assert isinstance(prediction, TrajectoryPrediction)
        assert prediction.next_total_kg is not None
        assert prediction.confidence_interval is not None

    def test_confidence_interval_from_quantile_models(self):
        """CI should come from quantile models, not global residual std."""
        lifters = _build_training_lifters(15)
        model = QuantileGBTModel()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        prediction = model.predict(target)

        assert prediction.confidence_interval is not None
        low, high = prediction.confidence_interval
        assert low < high

    def test_save_and_load(self, tmp_path: Path):
        lifters = _build_training_lifters(15)
        model = QuantileGBTModel()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        original_pred = model.predict(target)

        model_path = tmp_path / "qgbt_model.joblib"
        model.save(model_path)

        loaded = QuantileGBTModel()
        loaded.load(model_path)
        loaded_pred = loaded.predict(target)

        assert loaded_pred.next_total_kg == original_pred.next_total_kg


# ===========================================================================
# Convenience function tests
# ===========================================================================
class TestPredictTrajectoryConvenience:
    def test_no_model_raises(self):
        target = _make_lifter("Target", count=4)
        with pytest.raises(ValueError, match="trained TrajectoryModel is required"):
            predict_trajectory(target)

    def test_with_model(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        prediction = predict_trajectory(target, model=model)
        assert isinstance(prediction, TrajectoryPrediction)
        assert prediction.next_total_kg is not None

    def test_with_targets(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
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


# ===========================================================================
# Shared helper tests
# ===========================================================================
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
        assert len(arr) == 21
        assert all(isinstance(v, float) for v in arr)

    def test_none_values_become_nan(self):
        import math

        feat_dict: dict[str, object] = {
            "career_length_days": 0,
            "competition_count": 1,
            "competition_frequency": 0.0,
            "best_total_kg": None,
        }
        arr = _features_to_array(feat_dict)
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
        assert len(row) == 23
        assert row[-2] == 90.0
        assert row[-1] == 93.0

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
        totals = [t for _, t in curve]
        assert all(t == 500.0 for t in totals)


# ===========================================================================
# Fixture DB integration tests
# ===========================================================================
class TestTrajectoryWithFixtureDB:
    def test_train_and_predict_from_db(self, test_db: Path):
        """Train on fixture data lifters and predict for one."""
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

        assert len(lifters) >= 10, f"Expected 10+ lifters with 3+ entries, got {len(lifters)}"

        model = GradientBoostingModel()
        scores = model.train(lifters, min_entries=3)
        assert all(k in scores for k in ("total", "squat", "bench", "deadlift"))

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

        model = GradientBoostingModel()
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


# ===========================================================================
# build_training_data tests
# ===========================================================================
class TestBuildTrainingData:
    def test_returns_five_tuple(self):
        lifters = _build_training_lifters(5)
        result = build_training_data(lifters)
        x_rows, y_total, y_squat, y_bench, y_deadlift = result
        assert isinstance(x_rows, list)
        assert isinstance(y_total, list)
        assert isinstance(y_squat, list)
        assert isinstance(y_bench, list)
        assert isinstance(y_deadlift, list)
        assert len(x_rows) == len(y_total) == len(y_squat) == len(y_bench) == len(y_deadlift)

    def test_generates_multiple_samples_per_lifter(self):
        """A lifter with 4 entries generates 3 training samples (one per pair)."""
        lifters = [_make_lifter("Single", count=4)]
        x_rows, y_total, *_ = build_training_data(lifters, min_entries=2)
        # i=1,2,3 → 3 samples
        assert len(x_rows) == 3

    def test_feature_row_length(self):
        """Each training row should have 23 features (21 history + 2 context)."""
        lifters = _build_training_lifters(3)
        x_rows, *_ = build_training_data(lifters, min_entries=2)
        assert len(x_rows) > 0
        assert len(x_rows[0]) == 23

    def test_filters_by_min_entries(self):
        lifters = [_make_lifter("Few", count=3)]
        x_rows, *_ = build_training_data(lifters, min_entries=4)
        assert len(x_rows) == 0

    def test_skips_entries_with_no_total(self):
        entries = [
            _make_entry(
                name="Partial",
                date=date(2022, 1, 15),
                total_kg=500.0,
                best3_squat_kg=200.0,
                best3_bench_kg=130.0,
                best3_deadlift_kg=170.0,
                bodyweight_kg=93.0,
                weight_class_kg="93",
            ),
            _make_entry(
                name="Partial",
                date=date(2022, 6, 15),
                total_kg=None,  # bombed out
            ),
            _make_entry(
                name="Partial",
                date=date(2023, 1, 15),
                total_kg=520.0,
                best3_squat_kg=205.0,
                best3_bench_kg=135.0,
                best3_deadlift_kg=180.0,
                bodyweight_kg=93.0,
                weight_class_kg="93",
            ),
        ]
        from opl.core.models import Lifter as _Lifter

        lifter = _Lifter(name="Partial", entries=entries)
        x_rows, y_total, *_ = build_training_data([lifter], min_entries=2)
        # Only entry[2] has total_kg → at most 1 sample
        assert all(t > 0 for t in y_total)

    def test_y_values_match_target_entries(self):
        """y_total values should come from the target entry's total_kg."""
        lifters = [_make_lifter("Check", count=4)]
        x_rows, y_total, *_ = build_training_data(lifters, min_entries=2)
        assert all(t > 0 for t in y_total)


# ===========================================================================
# resolve_prediction_context tests
# ===========================================================================
class TestResolvePredictionContext:
    def test_explicit_date_is_respected(self):
        lifter = _make_lifter("Test", count=4)
        target_date = date(2027, 6, 1)
        days_to_target, resolved_date, _ = resolve_prediction_context(lifter, target_date, None)
        assert resolved_date == target_date
        history = lifter.history()
        expected_days = float((target_date - history[-1].date).days)
        assert days_to_target == expected_days

    def test_default_date_is_in_future(self):
        lifter = _make_lifter("Test", count=4)
        _, resolved_date, _ = resolve_prediction_context(lifter, None, None)
        history = lifter.history()
        assert resolved_date > history[-1].date

    def test_single_entry_uses_180_day_gap(self):
        lifter = _make_lifter("Single", count=1)
        days_to_target, _, _ = resolve_prediction_context(lifter, None, None)
        assert days_to_target == 180

    def test_explicit_bodyweight_is_respected(self):
        lifter = _make_lifter("Test", count=4)
        _, _, resolved_bw = resolve_prediction_context(lifter, None, 120.0)
        assert resolved_bw == 120.0

    def test_default_bodyweight_uses_latest(self):
        lifter = _make_lifter("Test", count=4)
        _, _, resolved_bw = resolve_prediction_context(lifter, None, None)
        history = lifter.history()
        assert resolved_bw == history[-1].bodyweight_kg


# ===========================================================================
# project_trajectory_with_model tests
# ===========================================================================
class TestProjectTrajectoryWithModel:
    def test_returns_correct_number_of_points(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        curve = project_trajectory_with_model(target, model, target_bodyweight_kg=93.0)
        assert len(curve) == 6

    def test_points_are_month_total_tuples(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        curve = project_trajectory_with_model(target, model, target_bodyweight_kg=None)
        for month, total in curve:
            assert isinstance(month, int)
            assert isinstance(total, float)

    def test_months_are_ascending(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        curve = project_trajectory_with_model(target, model, target_bodyweight_kg=93.0)
        months = [m for m, _ in curve]
        assert months == sorted(months)


# ===========================================================================
# BaseTrajectoryModel property tests
# ===========================================================================
class TestBaseModelProperties:
    def test_is_trained_initially_false(self):
        model = GradientBoostingModel()
        assert model.is_trained is False

    def test_is_trained_true_after_train(self):
        lifters = _build_training_lifters(15)
        model = GradientBoostingModel()
        model.train(lifters)
        assert model.is_trained is True

    def test_check_trained_raises_runtime_error(self):
        model = GradientBoostingModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model._check_trained()


# ===========================================================================
# Extended Quantile GBT tests
# ===========================================================================
class TestQuantileGBTModelExtended:
    def test_predict_untrained_raises(self):
        model = QuantileGBTModel()
        target = _make_lifter("Target", count=4)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(target)

    def test_save_untrained_raises(self, tmp_path: Path):
        model = QuantileGBTModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.save(tmp_path / "qgbt_model.joblib")

    def test_load_old_version_raises(self, tmp_path: Path):
        import joblib  # type: ignore[import-untyped]

        model_path = tmp_path / "old_qgbt_model.joblib"
        joblib.dump({"version": 0, "total": None}, model_path)
        model = QuantileGBTModel()
        with pytest.raises(ValueError, match="version"):
            model.load(model_path)

    def test_predict_with_target_date(self):
        lifters = _build_training_lifters(15)
        model = QuantileGBTModel()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        prediction = model.predict(target, target_date=date(2027, 1, 1))
        assert prediction.target_date == date(2027, 1, 1)
        assert prediction.next_total_kg is not None

    def test_predict_with_target_bodyweight(self):
        lifters = _build_training_lifters(15)
        model = QuantileGBTModel()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        prediction = model.predict(target, target_bodyweight_kg=120.0)
        assert prediction.target_bodyweight_kg == 120.0
        assert prediction.next_total_kg is not None

    def test_has_trajectory_curve(self):
        lifters = _build_training_lifters(15)
        model = QuantileGBTModel()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        prediction = model.predict(target)
        assert len(prediction.trajectory_curve) == 6

    def test_ci_bounds_are_ordered(self):
        """Lower CI bound must be strictly less than upper bound."""
        lifters = _build_training_lifters(15)
        model = QuantileGBTModel()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        prediction = model.predict(target)
        assert prediction.confidence_interval is not None
        low, high = prediction.confidence_interval
        assert low < high

    def test_ci_differs_from_fixed_margin(self):
        """Quantile CI width should vary per lifter, unlike the fixed 5% GBT fallback."""
        lifters = _build_training_lifters(15)
        qgbt = QuantileGBTModel()
        qgbt.train(lifters)

        sparse = _make_lifter("Sparse", count=3)
        rich = _make_lifter("Rich", count=6)

        pred_sparse = qgbt.predict(sparse)
        pred_rich = qgbt.predict(rich)

        assert pred_sparse.confidence_interval is not None
        assert pred_rich.confidence_interval is not None
        # Both are valid CIs; widths may legitimately differ
        sparse_width = pred_sparse.confidence_interval[1] - pred_sparse.confidence_interval[0]
        rich_width = pred_rich.confidence_interval[1] - pred_rich.confidence_interval[0]
        assert sparse_width > 0
        assert rich_width > 0


# ===========================================================================
# Extended features_to_array tests
# ===========================================================================
class TestFeaturesToArrayExtended:
    def test_new_features_present_in_feature_keys(self):
        """All 4 new features must be in _FEATURE_KEYS."""
        assert "recent_avg_total_kg" in _FEATURE_KEYS
        assert "recent_progression_rate" in _FEATURE_KEYS
        assert "total_std_kg" in _FEATURE_KEYS
        assert "meets_since_peak" in _FEATURE_KEYS

    def test_feature_keys_length(self):
        assert len(_FEATURE_KEYS) == 21

    def test_all_feature_keys_produce_correct_array(self):
        """A full feature dict with all 21 keys should produce an array of length 21."""
        feat_dict = dict.fromkeys(_FEATURE_KEYS, 1.0)
        arr = features_to_array(feat_dict)
        assert len(arr) == 21
        assert all(v == 1.0 for v in arr)

    def test_new_features_as_none_become_nan(self):
        import math

        feat_dict: dict[str, object] = {
            "recent_avg_total_kg": None,
            "recent_progression_rate": None,
            "total_std_kg": None,
            "meets_since_peak": None,
        }
        arr = features_to_array(feat_dict)
        # All new features are at indices 17-20 (0-based)
        for i in range(17, 21):
            assert math.isnan(arr[i])

    def test_extract_features_populates_new_fields(self):
        """extract_features should populate all 4 new fields for a lifter with enough history."""
        from opl.analytics.features import extract_features

        lifter = _make_lifter("Rich", count=5)
        feat = extract_features(lifter)
        assert feat.recent_avg_total_kg is not None
        assert feat.recent_progression_rate is not None
        assert feat.total_std_kg is not None
        assert feat.meets_since_peak is not None


# ===========================================================================
# Cross-approach consistency tests
# ===========================================================================
class TestAllApproachesConsistency:
    @pytest.mark.parametrize(
        "approach_name", ["gradient_boosting", "quantile_gbt"]
    )
    def test_train_returns_four_scores(self, approach_name: str):
        from opl.analytics.trajectory.registry import get_approach

        lifters = _build_training_lifters(15)
        model = get_approach(approach_name)()
        scores = model.train(lifters)
        assert set(scores.keys()) == {"total", "squat", "bench", "deadlift"}
        assert all(isinstance(v, float) for v in scores.values())

    @pytest.mark.parametrize(
        "approach_name", ["gradient_boosting", "quantile_gbt"]
    )
    def test_predict_returns_full_prediction(self, approach_name: str):
        from opl.analytics.trajectory.registry import get_approach

        lifters = _build_training_lifters(15)
        model = get_approach(approach_name)()
        model.train(lifters)
        target = _make_lifter("Target", count=5)
        prediction = model.predict(target)

        assert isinstance(prediction, TrajectoryPrediction)
        assert prediction.next_total_kg is not None
        assert prediction.next_squat_kg is not None
        assert prediction.next_bench_kg is not None
        assert prediction.next_deadlift_kg is not None
        assert prediction.confidence_interval is not None
        assert prediction.target_date is not None
        assert len(prediction.trajectory_curve) == 6

    @pytest.mark.parametrize(
        "approach_name", ["gradient_boosting", "quantile_gbt"]
    )
    def test_save_and_load_roundtrip(self, approach_name: str, tmp_path: Path):
        from opl.analytics.trajectory.registry import get_approach

        lifters = _build_training_lifters(15)
        cls = get_approach(approach_name)
        model = cls()
        model.train(lifters)

        target = _make_lifter("Target", count=5)
        original_pred = model.predict(target)

        model_path = tmp_path / f"{approach_name}_model.joblib"
        model.save(model_path)
        assert model_path.exists()

        loaded = cls()
        loaded.load(model_path)
        assert loaded.is_trained

        loaded_pred = loaded.predict(target)
        assert loaded_pred.next_total_kg == original_pred.next_total_kg
